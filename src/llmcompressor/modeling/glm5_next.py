"""GLM-5.3-Flash calibration and MTP support.

GLM-5.3 uses the ``glm5_next`` Transformers model.  It is not compatible with
the older ``glm_moe_dsa`` model implementation: the text stack is nested below
``model.language_model``, the main stack mixes KDA and DSA blocks, and the MTP
block is a regular DSA+MoE decoder without the main-stack mHC connections.
Its MoE calibration is registered through the shared ``MoECalibrationModule``
compatibility layer; the adapter below only supplies GLM-5.3's MLP class and
configuration field names.
"""

from __future__ import annotations

import copy
import json
import os
import re
import types

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

from llmcompressor.modeling.glm_moe_dsa import CalibrationGlmMoeDsaMoE
from llmcompressor.modeling.moe_context import MoECalibrationModule
from llmcompressor.utils.dev import skip_weights_initialize

try:
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextTextAttention,
        Glm5NextTextMLP,
        Glm5NextTextMoE,
        Glm5NextTextRMSNorm,
    )
except ImportError:  # pragma: no cover - depends on the installed Transformers
    Glm5NextTextAttention = None
    Glm5NextTextMLP = None
    Glm5NextTextMoE = None
    Glm5NextTextRMSNorm = None


def _require_transformers_glm5() -> None:
    if Glm5NextTextAttention is None:
        raise ImportError(
            "GLM-5.3 support requires Transformers with the glm5_next model "
            "implementation (Transformers main or a release containing it)."
        )


class Glm5NextTextRoutedExpertMLP(Glm5NextTextMLP):
    """Unpacked routed expert matching ``Glm5NextTextExperts._apply_gate``."""

    def forward(self, x):
        gate = self.gate_proj(x).clamp(min=None, max=self.swiglu_limit)
        up = self.up_proj(x).clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        # HF's packed routed experts intentionally use SiLU regardless of
        # config.hidden_act; shared/dense MLPs continue to use the config value.
        return self.down_proj(F.silu(gate) * up)


class SequentialGlm5NextExperts(nn.ModuleList):
    """Unpack packed 3-D tensors for the shared MoE calibration adapter."""

    def __init__(self, config, original) -> None:
        _require_transformers_glm5()
        self.num_experts = original.gate_up_proj.shape[0]
        with skip_weights_initialize():
            super().__init__(
                [
                    Glm5NextTextRoutedExpertMLP(
                        config,
                        intermediate_size=config.moe_intermediate_size,
                    )
                    for _ in range(self.num_experts)
                ]
            )

        for index in range(self.num_experts):
            gate_up = original.gate_up_proj[index]
            gate, up = gate_up.chunk(2, dim=0)
            self[index].gate_proj.weight.data = gate.contiguous()
            self[index].up_proj.weight.data = up.contiguous()
            self[index].down_proj.weight.data = original.down_proj[index].contiguous()


@MoECalibrationModule.register("Glm5NextTextMoE")
class CalibrationGlm5NextTextMoE(CalibrationGlmMoeDsaMoE):
    """GLM-5.3 MoE replacement used when a packed expert block is calibrated."""

    def _get_num_experts(self, config) -> int:
        """Use GLM-5.3's ``n_routed_experts`` config spelling."""
        return config.n_routed_experts

    def _make_experts(self, config, original_experts) -> nn.ModuleList:
        return SequentialGlm5NextExperts(config, original_experts)

    def __init__(self, original, config, calibrate_all_experts: bool = True):
        text_config = (
            config.get_text_config() if hasattr(config, "get_text_config") else config
        )
        super().__init__(original, text_config, calibrate_all_experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Match ``Glm5NextTextMoE.forward`` with unpacked routed experts."""
        residuals = hidden_states
        orig_shape = hidden_states.shape
        _, topk_weights, topk_indices = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                topk_indices, num_classes=self.num_experts
            ).permute(2, 1, 0)
            hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if self.calibrate_all_experts:
                current = self.experts[expert_idx](hidden_states)[token_idx]
            else:
                current = self.experts[expert_idx](hidden_states[token_idx])
            current = current * topk_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current.to(final_hidden_states.dtype))

        hidden_states = final_hidden_states.view(*orig_shape)
        return hidden_states + self.shared_experts(residuals)


def _load_mtp_tensors(model_path: str, prefix: str) -> dict[str, torch.Tensor]:
    """Load one GLM-5.3 MTP layer from an index/shard checkpoint."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as file:
            weight_map = json.load(file)["weight_map"]
        shards = sorted(
            {
                shard
                for name, shard in weight_map.items()
                if name.startswith(prefix + ".")
            }
        )
    else:
        shards = ["model.safetensors"]

    tensors: dict[str, torch.Tensor] = {}
    for shard in shards:
        shard_path = os.path.join(model_path, shard)
        if not os.path.exists(shard_path):
            raise FileNotFoundError(
                f"GLM-5.3 MTP shard is missing: {shard_path}. "
                "The index is present, but the safetensors data must also be available."
            )
        with safe_open(shard_path, framework="pt") as file:
            for name in file.keys():
                if name.startswith(prefix + "."):
                    tensors[name[len(prefix) + 1 :]] = file.get_tensor(name)

    if not tensors:
        raise ValueError(
            f"No GLM-5.3 MTP tensors found under {prefix!r} in {model_path!r}."
        )
    return tensors


_EXPERT_WEIGHT_RE = re.compile(
    r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


def _fuse_mtp_experts(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert MTP checkpoint keys to the temporary Transformers layout.

    This is a checkpoint-loading conversion only.  The model is immediately
    passed through ``MoECalibrationModule`` before quantization; no
    ``LinearExperts2D`` path is used.
    """
    per_expert: dict[int, dict[str, torch.Tensor]] = {}
    result: dict[str, torch.Tensor] = {}
    for name, tensor in raw.items():
        match = _EXPERT_WEIGHT_RE.match(name)
        if match:
            per_expert.setdefault(int(match.group(1)), {})[match.group(2)] = tensor
        else:
            result[name] = tensor

    if not per_expert:
        return result

    expected = set(range(max(per_expert) + 1))
    if set(per_expert) != expected or any(
        set(parts) != {"gate_proj", "up_proj", "down_proj"}
        for parts in per_expert.values()
    ):
        raise ValueError(
            "GLM-5.3 MTP expert weights are incomplete; cannot fuse them safely."
        )

    result["mlp.experts.gate_up_proj"] = torch.stack(
        [
            torch.cat([per_expert[i]["gate_proj"], per_expert[i]["up_proj"]], dim=0)
            for i in sorted(per_expert)
        ]
    )
    result["mlp.experts.down_proj"] = torch.stack(
        [per_expert[i]["down_proj"] for i in sorted(per_expert)]
    )
    return result


def _extended_text_config(config, layer_idx: int):
    """Add the regular DSA/sparse MTP slot to a 45-layer text config."""
    text_config = (
        config.get_text_config() if hasattr(config, "get_text_config") else config
    )
    extended = copy.copy(text_config)
    for field, value in (
        ("layer_types", "deepseek_sparse_attention"),
        ("mlp_layer_types", "sparse"),
        ("indexer_types", "full"),
    ):
        values = list(getattr(text_config, field))
        if len(values) <= layer_idx:
            values.extend([value] * (layer_idx + 1 - len(values)))
        setattr(extended, field, values)
    return extended


class Glm5NextMTPLayer(nn.Module):
    """One regular DSA+MoE MTP decoder used by GLM-5.3-Flash.

    The main GLM-5.3 stack uses mHC and KDA layers, but the MTP checkpoint has
    only ``input_layernorm``/``post_attention_layernorm`` and DSA projections.
    This intentionally mirrors the non-mHC decoder path used by
    ``vllm/models/glm5next/nvidia/mtp.py`` and SGLang's
    ``glm5_next_nextn.py``/``deepseek_nextn.py``.  Building the Transformers
    decoder class directly would incorrectly allocate mHC parameters that are
    absent from the MTP checkpoint, so the plain residual path is assembled
    explicitly here.
    """

    def __init__(self, model, layer_idx: int, raw_tensors: dict[str, torch.Tensor]):
        super().__init__()
        _require_transformers_glm5()
        if not any(name.startswith("self_attn.indexer.") for name in raw_tensors):
            raise ValueError(
                "GLM-5.3 MTP requires its own full DSA indexer, but no "
                "self_attn.indexer.* tensors were found in the checkpoint."
            )
        config = _extended_text_config(model.config, layer_idx)
        reference = next(model.parameters())

        self.enorm = Glm5NextTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = Glm5NextTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.input_layernorm = Glm5NextTextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn = Glm5NextTextAttention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = Glm5NextTextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = Glm5NextTextMoE(config)
        self.mlp.experts.config = config
        self.shared_head = nn.Module()
        self.shared_head.norm = Glm5NextTextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.to(device=reference.device, dtype=reference.dtype)
        state_dict = {
            name: tensor.to(device=reference.device, dtype=reference.dtype)
            for name, tensor in _fuse_mtp_experts(raw_tensors).items()
        }
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        missing = [name for name in missing if "inv_freq" not in name]
        if missing or unexpected:
            raise ValueError(
                "GLM-5.3 MTP state mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )

    def forward(
        self,
        previous_hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
        embed_tokens: nn.Embedding,
        input_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            if input_ids is None:
                raise ValueError("GLM-5.3 MTP needs input_ids or input_embeds")
            input_embeds = embed_tokens(input_ids)

        # A one-token target pass has no t+1 continuation to predict.  Avoid
        # entering the DSA indexer with a zero-length sequence; its view(...,
        # -1, ...) shape is undefined for that case.  The result is discarded
        # by the calibration wrapper, so returning the aligned hidden states is
        # sufficient and keeps generation-style calls valid.
        if input_embeds.shape[1] == 0:
            return previous_hidden_states

        hidden_states = self.eh_proj(
            torch.cat(
                [self.enorm(input_embeds), self.hnorm(previous_hidden_states)], dim=-1
            )
        )
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=None,
            prev_topk_indices=None,
        )
        hidden_states = residual + (
            attn_out[0] if isinstance(attn_out, tuple) else attn_out
        )

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return self.shared_head.norm(residual + hidden_states)


def attach_mtp_layer(model, model_path: str) -> None:
    """Attach and load ``model.language_model.layers.45`` for calibration."""
    _require_transformers_glm5()
    text_config = model.config.get_text_config()
    mtp_count = getattr(text_config, "num_nextn_predict_layers", 1)
    if mtp_count != 1:
        raise ValueError(
            f"GLM-5.3 adapter currently expects one MTP layer, got {mtp_count}."
        )

    layer_idx = text_config.num_hidden_layers
    prefix = f"model.language_model.layers.{layer_idx}"
    raw = _load_mtp_tensors(model_path, prefix)
    mtp_layer = Glm5NextMTPLayer(model, layer_idx, raw)
    language_model = model.model.language_model
    if len(language_model.layers) != layer_idx:
        raise ValueError(
            f"Expected {layer_idx} loaded GLM-5.3 text layers, "
            f"got {len(language_model.layers)}."
        )
    language_model.layers.append(mtp_layer)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        output_router_logits=None,
        mm_token_type_ids=None,
        logits_to_keep=0,
        cache_position=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            output_router_logits=output_router_logits,
            mm_token_type_ids=mm_token_type_ids,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )

        if position_ids is None:
            if cache_position is not None:
                position_ids = cache_position
                if position_ids.ndim == 1:
                    position_ids = position_ids.unsqueeze(0)
            else:
                position_ids = torch.arange(
                    hidden_states.shape[1], device=hidden_states.device
                ).unsqueeze(0)
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)

        mtp_hidden_states = hidden_states[:, :-1, :]
        mtp_position_ids = position_ids[:, 1:]
        mtp_input_ids = input_ids[:, 1:] if input_ids is not None else None
        mtp_input_embeds = inputs_embeds[:, 1:] if inputs_embeds is not None else None
        if attention_mask is None:
            mtp_attention_mask = torch.ones(
                mtp_hidden_states.shape[:2],
                dtype=torch.bool,
                device=mtp_hidden_states.device,
            )
        elif torch.is_tensor(attention_mask) and attention_mask.ndim == 2:
            mtp_attention_mask = attention_mask[:, 1:].bool()
        else:
            raise ValueError(
                "GLM-5.3 MTP calibration expects a 2D padding mask; the HF "
                "KPool indexer builds its own causal sparse-attention mask."
            )

        # Online vLLM keeps an initial position-0 placeholder and zeros its
        # embedding in fused_eh_norm.  Offline teacher-forced calibration drops
        # that unpaired slot instead: token t+1 is aligned with target hidden t,
        # so this slice starts at position 1 for an ordinary text sequence.
        self.model.language_model.layers[self.config.text_config.num_hidden_layers](
            previous_hidden_states=mtp_hidden_states,
            input_ids=mtp_input_ids,
            input_embeds=mtp_input_embeds,
            embed_tokens=self.model.language_model.embed_tokens,
            attention_mask=mtp_attention_mask,
            position_ids=mtp_position_ids,
        )

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    model.forward = types.MethodType(forward, model)


try:
    from transformers.models.glm5_next.modeling_glm5_next import (
        MoeCausalLMOutputWithPast,
    )
except ImportError:  # pragma: no cover - imported only by the attach path
    MoeCausalLMOutputWithPast = None


__all__ = [
    "CalibrationGlm5NextTextMoE",
    "Glm5NextMTPLayer",
    "attach_mtp_layer",
]
