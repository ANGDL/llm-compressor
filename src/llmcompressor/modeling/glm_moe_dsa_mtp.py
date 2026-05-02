"""
End-to-end MTP (Multi-Token Prediction) layer support for GLM-5 (GlmMoeDsa) quantization.

GLM-5 has 78 main decoder layers (0–77) plus one MTP layer at index 78
(``model.layers.78.*`` in the checkpoint).  The HuggingFace transformers
``GlmMoeDsaPreTrainedModel`` has::

    _keys_to_ignore_on_load_unexpected = [r"model\\.layers\\.78.*"]

so layer 78 is silently dropped during ``from_pretrained`` and never quantized.

This module provides :func:`attach_mtp_layer` which:

1. Reads the MTP weights directly from the safetensors shards.
2. Builds a :class:`GlmMoeDsaMTPLayer` using the same
   ``GlmMoeDsaDecoderLayer`` / ``GlmMoeDsaRMSNorm`` classes from transformers,
   so all sub-modules (attention projections, MoE experts, norms) are identical
   in structure to the main model layers.
3. Attaches the MTP layer as ``model.mtp`` so that llmcompressor's GPTQ pass
    can discover and quantize the decoder ``Linear`` layers inside it while
    keeping ``eh_proj`` in BF16.
4. Monkey-patches ``model.forward`` to run a second forward through the MTP
   layer after the main model forward.  The MTP logits are discarded; only
   the activations matter for Hessian accumulation.

MTP forward formula (mirrors sglang / DeepSeek-V3 NextN design)::

    mtp_hidden = eh_proj(concat(enorm(embed_tokens(input_ids)), hnorm(main_hidden)))
    mtp_out    = decoder(mtp_hidden, ...)          # GlmMoeDsaDecoderLayer
    mtp_out    = shared_head_norm(mtp_out)

Usage in a quantization script::

    from llmcompressor.modeling.glm_moe_dsa_mtp import attach_mtp_layer

    model = AutoModelForCausalLM.from_pretrained(model_id, ...)
    attach_mtp_layer(model, model_id)
    # Pass model to oneshot() as usual – MTP decoder Linear layers will be quantized.
"""

from __future__ import annotations

import copy
import json
import os
import types
from typing import Optional

import torch
import torch.nn as nn
from safetensors import safe_open

from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaDecoderLayer,
    GlmMoeDsaRMSNorm,
)
from transformers.modeling_outputs import CausalLMOutputWithPast


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_mtp_tensors(model_path: str, mtp_prefix: str) -> dict[str, torch.Tensor]:
    """Return every tensor whose name starts with ``<mtp_prefix>.`` ."""
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]
        shard_files = sorted(
            {v for k, v in weight_map.items() if k.startswith(mtp_prefix + ".")}
        )
    else:
        shard_files = ["model.safetensors"]

    tensors: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        path = os.path.join(model_path, shard)
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if key.startswith(mtp_prefix + "."):
                    tensors[key] = f.get_tensor(key)

    if not tensors:
        raise ValueError(
            f"No tensors found with prefix '{mtp_prefix}' in {model_path}. "
            "Make sure this is a GLM-5 checkpoint that contains MTP weights."
        )
    return tensors


def _strip_prefix(
    tensors: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    dot = prefix + "."
    return {k[len(dot):]: v for k, v in tensors.items() if k.startswith(dot)}


# ---------------------------------------------------------------------------
# MTP-specific weight names (not part of the decoder sub-module)
# ---------------------------------------------------------------------------
# Checkpoint layout (relative to ``model.layers.78.``):
#   enorm.weight                   -> enorm
#   hnorm.weight                   -> hnorm
#   eh_proj.weight                 -> eh_proj
#   shared_head.norm.weight        -> shared_head_norm
#   self_attn.*                    -> decoder.self_attn.*
#   mlp.*                          -> decoder.mlp.*
#   input_layernorm.*              -> decoder.input_layernorm.*
#   post_attention_layernorm.*     -> decoder.post_attention_layernorm.*

_MTP_SPEC_PREFIXES = ("enorm.", "hnorm.", "eh_proj.", "shared_head.")


def _fuse_moe_expert_weights(
    decoder_sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Convert per-expert Linear keys in *decoder_sd* into the fused 3-D tensors
    that ``GlmMoeDsaNaiveMoe`` expects.

    Checkpoint format (one entry per expert):
        mlp.experts.<i>.gate_proj.weight  [moe_intermediate, hidden]
        mlp.experts.<i>.up_proj.weight    [moe_intermediate, hidden]
        mlp.experts.<i>.down_proj.weight  [hidden, moe_intermediate]

    GlmMoeDsaNaiveMoe format:
        mlp.experts.gate_up_proj  [num_experts, 2*moe_intermediate, hidden]
        mlp.experts.down_proj     [num_experts, hidden, moe_intermediate]

    Keys that do not match the per-expert pattern are passed through unchanged.
    """
    import re

    _EXPERT_RE = re.compile(
        r"^(mlp\.experts\.)(\d+)\.(gate_proj|up_proj|down_proj)(\.weight)$"
    )

    per_expert: dict[int, dict[str, torch.Tensor]] = {}
    passthrough: dict[str, torch.Tensor] = {}

    for key, tensor in decoder_sd.items():
        m = _EXPERT_RE.match(key)
        if m:
            idx = int(m.group(2))
            proj = m.group(3)
            per_expert.setdefault(idx, {})[proj] = tensor
        else:
            passthrough[key] = tensor

    if not per_expert:
        return decoder_sd  # no individual expert keys – nothing to fuse

    num_experts = max(per_expert) + 1
    gate_up_list: list[torch.Tensor] = []
    down_list: list[torch.Tensor] = []
    for i in range(num_experts):
        exp = per_expert[i]
        # cat gate+up along dim-0 → [2*moe_intermediate, hidden]
        gate_up_list.append(torch.cat([exp["gate_proj"], exp["up_proj"]], dim=0))
        down_list.append(exp["down_proj"])

    fused = dict(passthrough)
    fused["mlp.experts.gate_up_proj"] = torch.stack(gate_up_list, dim=0)
    fused["mlp.experts.down_proj"] = torch.stack(down_list, dim=0)
    return fused


# ---------------------------------------------------------------------------
# MTP layer module
# ---------------------------------------------------------------------------

class GlmMoeDsaMTPLayer(nn.Module):
    """
    One MTP (Multi-Token Prediction) layer for GLM-5.

    Sub-modules
    -----------
    enorm           : RMSNorm applied to the main model's ``hidden_states``.
    hnorm           : RMSNorm applied to the token embedding branch.
    eh_proj         : Linear projection kept in BF16 and excluded from quantization.
    shared_head_norm: Final RMSNorm before the shared ``lm_head``.
    decoder         : ``GlmMoeDsaDecoderLayer`` – full attention + MoE/MLP block.
    """

    def __init__(
        self,
        config,
        mtp_layer_idx: int,
        raw_tensors: dict[str, torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        hidden_size = config.hidden_size

        self.enorm = GlmMoeDsaRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.hnorm = GlmMoeDsaRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.shared_head_norm = GlmMoeDsaRMSNorm(hidden_size, eps=config.rms_norm_eps)

        # GlmMoeDsaDecoderLayer / GlmMoeDsaAttention access config list fields
        # like mlp_layer_types[mtp_layer_idx] and indexer_types[mtp_layer_idx].
        # The main-model config only has entries for layers 0..num_hidden_layers-1,
        # so we extend them (on a shallow copy) before constructing the decoder.
        needs_extend = (
            len(config.mlp_layer_types) <= mtp_layer_idx
            or (
                hasattr(config, "indexer_types")
                and len(config.indexer_types) <= mtp_layer_idx
            )
        )
        if needs_extend:
            extended = copy.copy(config)
            if len(config.mlp_layer_types) <= mtp_layer_idx:
                extended.mlp_layer_types = (
                    list(config.mlp_layer_types)
                    + ["sparse"] * (mtp_layer_idx - len(config.mlp_layer_types) + 1)
                )
            if hasattr(config, "indexer_types") and len(config.indexer_types) <= mtp_layer_idx:
                # Layer 78 has the same self_attn structure (including indexer.*)
                # as layer 77, so reuse the last entry rather than guessing.
                fill = config.indexer_types[-1] if config.indexer_types else "sparse"
                extended.indexer_types = (
                    list(config.indexer_types)
                    + [fill] * (mtp_layer_idx - len(config.indexer_types) + 1)
                )
        else:
            extended = config

        # Reuse the same decoder layer class as the main stack.
        self.decoder = GlmMoeDsaDecoderLayer(extended, layer_idx=mtp_layer_idx)

        self.to(device=device, dtype=dtype)

        self._load_weights(raw_tensors, dtype, device)

    def _load_weights(
        self,
        raw: dict[str, torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        own_sd: dict[str, torch.Tensor] = {}
        decoder_sd: dict[str, torch.Tensor] = {}

        for name, tensor in raw.items():
            t = tensor.to(dtype=dtype, device=device)
            if any(name.startswith(p) for p in _MTP_SPEC_PREFIXES):
                # "shared_head.norm.weight" -> "shared_head_norm.weight"
                mapped = name.replace("shared_head.norm.", "shared_head_norm.")
                own_sd[mapped] = t
            else:
                decoder_sd[name] = t

        # Load enorm / hnorm / eh_proj / shared_head_norm
        missing, _ = self.load_state_dict(own_sd, strict=False)
        own_missing = [k for k in missing if not k.startswith("decoder.")]
        if own_missing:
            raise ValueError(
                f"[GlmMoeDsaMTPLayer] Missing MTP-specific weights: {own_missing}\n"
                f"Available keys in checkpoint: {list(own_sd.keys())}"
            )

        # Fuse per-expert Linear keys into 3-D tensors for GlmMoeDsaNaiveMoe.
        # The checkpoint stores experts individually (mlp.experts.<i>.gate_proj …)
        # while GlmMoeDsaNaiveMoe expects fused (mlp.experts.gate_up_proj …).
        decoder_sd = _fuse_moe_expert_weights(decoder_sd)

        # Load decoder weights
        dec_missing, _ = self.decoder.load_state_dict(decoder_sd, strict=False)
        if dec_missing:
            # rotary inv_freq buffers are typically absent from the checkpoint
            non_rotary = [k for k in dec_missing if "inv_freq" not in k]
            if non_rotary:
                print(
                    f"[GlmMoeDsaMTPLayer] Decoder missing weights "
                    f"(non-rotary): {non_rotary}"
                )

    def forward(
        self,
        token_hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        embed_tokens: nn.Embedding,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple] = None,
        **kwargs,
    ) -> torch.Tensor:
        token_embeds = embed_tokens(input_ids)          # (B, T, H)
        mtp_input = torch.cat(
            [self.enorm(token_embeds), self.hnorm(token_hidden_states)],
            dim=-1,
        )

        # DeepSeek/sglang MTP design: concat then project.
        # The token branch flows through ``enorm`` and the speculative branch
        # flows through ``hnorm`` before the shared projection.
        #   eh_proj weight: [H, 2H]  (in=2H, out=H)
        mtp_hidden = self.eh_proj(mtp_input)

        mtp_out = self.decoder(
            mtp_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # GlmMoeDsaDecoderLayer.forward returns (hidden_states, ...) tuple;
        # extract the tensor before passing to the norm.
        if isinstance(mtp_out, tuple):
            mtp_out = mtp_out[0]

        return self.shared_head_norm(mtp_out)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def attach_mtp_layer(model, model_path: str) -> None:
    """
    Load the MTP layer (``model.layers.{num_hidden_layers}.*``) from *model_path*
    and attach it to *model* so that the MTP decoder participates in the GPTQ
    calibration forward pass and gets quantized end-to-end.

    The function mutates *model* in-place:

    * Adds ``model.mtp`` (:class:`GlmMoeDsaMTPLayer`).
    * Replaces ``model.forward`` with a patched version that runs an additional
      MTP forward after every main-model forward.  The MTP logits are discarded;
      only the activations matter for Hessian accumulation.

    Parameters
    ----------
    model :
        A ``GlmMoeDsaForCausalLM`` instance already loaded via
        ``AutoModelForCausalLM.from_pretrained``.
    model_path :
        Directory containing the original (BF16) safetensors shards.
    """
    config = model.config
    mtp_layer_idx = config.num_hidden_layers   # layer 78 for GLM-5

    try:
        ref_param = next(iter(model.parameters()))
        dtype, device = ref_param.dtype, ref_param.device
    except StopIteration:
        dtype, device = torch.bfloat16, torch.device("cpu")

    mtp_prefix = f"model.layers.{mtp_layer_idx}"
    print(f"[attach_mtp_layer] Loading MTP weights from '{mtp_prefix}' ...")

    raw = _load_mtp_tensors(model_path, mtp_prefix)
    raw = _strip_prefix(raw, mtp_prefix)
    print(f"[attach_mtp_layer] Found {len(raw)} MTP tensors.")

    model.mtp = GlmMoeDsaMTPLayer(
        config=config,
        mtp_layer_idx=mtp_layer_idx,
        raw_tensors=raw,
        dtype=dtype,
        device=device,
    )

    # ------------------------------------------------------------------
    # Patch model.forward to include MTP in a single backbone pass.
    #
    # Design constraints imposed by the sequential pipeline:
    #
    # 1. `inspect.getsource(model.forward)` fetches the source and `exec`s it
    #    in the namespace of `GlmMoeDsaForCausalLM`'s module.  The exec'd
    #    code must define a name ``"forward"`` in that namespace, so the
    #    inner function must be named exactly ``forward``.
    #
    # 2. The exec'd function runs with only the transformers module's globals
    #    in scope – no closures allowed.  Every symbol must be either:
    #    - a name already present in ``transformers.models.glm_moe_dsa``'s
    #      module dict  (e.g. ``CausalLMOutputWithPast``, ``torch``), or
    #    - an attribute access via ``self`` (the bound model instance).
    #
    # 3. The function must be bound with ``types.MethodType`` so that
    #    ``model.forward.__func__`` exists (required by compressed_tensors'
    #    ``offload_module``).
    #
    # 4. We must NOT call the backbone (``self.model``) twice.  The pipeline
    #    uses ``torch.fx.symbolic_trace``; two calls to the same sub-module
    #    would produce a malformed computation graph.  Therefore we inline
    #    the CausalLM logic here and call ``self.model`` exactly once.
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        logits_to_keep=0,
        **kwargs,
    ):
        # ---- backbone (single pass) ----
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        # ---- lm_head logits (mirrors GlmMoeDsaForCausalLM.forward) ----
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
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        # ---- MTP forward ----
        # Activations flow through MTP Linear layers for GPTQ Hessian
        # accumulation.  The MTP logits are intentionally discarded.
        if position_ids is None:
            seq_len = hidden_states.shape[1]
            position_ids = torch.arange(
                seq_len, device=hidden_states.device
            ).unsqueeze(0)
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        self.mtp(
            token_hidden_states=hidden_states,
            input_ids=input_ids,
            embed_tokens=self.model.embed_tokens,
            # Do NOT pass the raw attention_mask here: the main model's forward
            # converts the Long 0/1 padding mask to a 4D float causal mask
            # internally, but we bypass that conversion.  SDPA rejects a Long
            # mask, and for Hessian accumulation exact masking is irrelevant.
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Bind as a proper method:
    #   - model.forward.__func__ exists  → compressed_tensors offload_module ✓
    #   - model.forward.__name__ == "forward"  → namespace["forward"] after exec ✓
    model.forward = types.MethodType(forward, model)

    # ------------------------------------------------------------------
    # Patch model.state_dict() so the MTP layer is saved with the same
    # key prefix as the original checkpoint:
    #
    #   mtp.decoder.self_attn.q_a_proj.weight
    #       → model.layers.{N}.self_attn.q_a_proj.weight
    #
    # Remapping rules (applied to the part after "mtp."):
    #   1. shared_head_norm. → shared_head.norm.   (reverse the load mapping)
    #   2. decoder.          → ""                  (strip the decoder. infix)
    #   3. prepend model.layers.{N}.
    # ------------------------------------------------------------------
    ckpt_prefix = f"model.layers.{mtp_layer_idx}."
    _orig_state_dict = model.state_dict

    def _patched_state_dict(*args, **kwargs):
        sd = _orig_state_dict(*args, **kwargs)
        remapped = {}
        for k, v in sd.items():
            if k.startswith("mtp."):
                rel = k[len("mtp."):]                          # strip "mtp."
                rel = rel.replace("shared_head_norm.", "shared_head.norm.")
                if rel.startswith("decoder."):
                    rel = rel[len("decoder."):]                # strip "decoder."
                remapped[ckpt_prefix + rel] = v
            else:
                remapped[k] = v
        return remapped

    model.state_dict = _patched_state_dict

    print(
        "[attach_mtp_layer] Done. MTP sub-modules: "
        + ", ".join(n for n, _ in model.mtp.named_modules() if n)
    )
