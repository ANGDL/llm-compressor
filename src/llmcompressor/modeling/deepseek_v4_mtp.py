"""
MTP (Multi-Token Prediction) layer support for DeepSeek V4 quantization.

DeepSeek V4 stores MTP weights under ``mtp.0.*`` in the checkpoint using the
raw/internal naming convention (``attn.wq_a``, ``ffn.experts.{i}.w1``, etc.).
The HuggingFace ``DeepseekV4PreTrainedModel`` ignores these via::

    _keys_to_ignore_on_load_unexpected = [r"(^|\\.)mtp\\..*"]

This module provides :func:`attach_mtp_layer` which:

1. Reads MTP weights from safetensors using the ``mtp.0.*`` prefix.
2. Converts raw checkpoint keys to HuggingFace format (same conversion as
   ``transformers.conversion_mapping`` applies to main layers).
3. Fuses per-expert weights into 3D tensors for ``DeepseekV4Experts``.
4. Builds a :class:`DeepseekV4MTPLayer` using the same HF module classes.
5. Attaches as ``model.mtp`` and patches ``model.forward`` for calibration.

Usage::

    from llmcompressor.modeling.deepseek_v4_mtp import attach_mtp_layer

    model = AutoModelForCausalLM.from_pretrained(model_id, ...)
    attach_mtp_layer(model, model_id)
"""

from __future__ import annotations

import copy
import json
import os
import re
import types
from typing import Optional

import torch
import torch.nn as nn
from safetensors import safe_open

from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4DecoderLayer,
    DeepseekV4HyperHead,
    DeepseekV4RMSNorm,
)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_mtp_tensors(model_path: str, mtp_prefix: str = "mtp.0") -> dict[str, torch.Tensor]:
    """Load all bf16 weight tensors with the given prefix, skipping .scale keys."""
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]
        shard_files = sorted(
            {v for k, v in weight_map.items()
             if k.startswith(mtp_prefix + ".") and not k.endswith(".scale")}
        )
    else:
        shard_files = ["model.safetensors"]

    tensors: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        path = os.path.join(model_path, shard)
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if key.startswith(mtp_prefix + ".") and not key.endswith(".scale"):
                    tensors[key] = f.get_tensor(key)

    if not tensors:
        raise ValueError(
            f"No MTP tensors found with prefix '{mtp_prefix}' in {model_path}."
        )
    return tensors


def _strip_prefix(tensors: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    dot = prefix + "."
    return {k[len(dot):]: v for k, v in tensors.items() if k.startswith(dot)}


# ---------------------------------------------------------------------------
# Key conversion: raw checkpoint format → HuggingFace module format
# ---------------------------------------------------------------------------
# Raw checkpoint (mtp.0.*):          HF module format:
#   attn.wq_a.weight              →  self_attn.q_a_proj.weight
#   attn.wq_b.weight              →  self_attn.q_b_proj.weight
#   attn.wkv.weight               →  self_attn.kv_proj.weight
#   attn.wo_a.weight              →  self_attn.o_a_proj.weight
#   attn.wo_b.weight              →  self_attn.o_b_proj.weight
#   attn.q_norm.weight            →  self_attn.q_a_norm.weight
#   attn.kv_norm.weight           →  self_attn.kv_norm.weight
#   attn.attn_sink                →  self_attn.sinks
#   attn_norm.weight              →  input_layernorm.weight
#   ffn_norm.weight               →  post_attention_layernorm.weight
#   ffn.gate.weight               →  mlp.gate.weight
#   ffn.gate.bias                 →  mlp.gate.e_score_correction_bias
#   ffn.experts.{i}.w1.weight     →  (fused into mlp.experts.gate_up_proj)
#   ffn.experts.{i}.w3.weight     →  (fused into mlp.experts.gate_up_proj)
#   ffn.experts.{i}.w2.weight     →  (fused into mlp.experts.down_proj)
#   ffn.shared_experts.w1.weight  →  mlp.shared_experts.gate_proj.weight
#   ffn.shared_experts.w2.weight  →  mlp.shared_experts.down_proj.weight
#   ffn.shared_experts.w3.weight  →  mlp.shared_experts.up_proj.weight
#   hc_attn_fn                    →  attn_hc.fn
#   hc_attn_base                  →  attn_hc.base
#   hc_attn_scale                 →  attn_hc.scale
#   hc_ffn_fn                     →  ffn_hc.fn
#   hc_ffn_base                   →  ffn_hc.base
#   hc_ffn_scale                  →  ffn_hc.scale
#   hc_head_fn                    →  (MTP-specific: hc_head.hc_fn)
#   hc_head_base                  →  (MTP-specific: hc_head.hc_base)
#   hc_head_scale                 →  (MTP-specific: hc_head.hc_scale)
#   enorm.weight                  →  (MTP-specific: enorm)
#   hnorm.weight                  →  (MTP-specific: hnorm)
#   e_proj.weight                 →  (MTP-specific: e_proj)
#   h_proj.weight                 →  (MTP-specific: h_proj)
#   norm.weight                   →  (MTP-specific: shared_head_norm)

_MTP_SPEC_PREFIXES = (
    "e_proj.", "h_proj.", "enorm.", "hnorm.",
    "hc_head_", "norm.",
)

_RENAME_RULES = [
    # Structural prefix renames
    (r"^attn_norm\.", "input_layernorm."),
    (r"^ffn_norm\.", "post_attention_layernorm."),
    (r"^hc_attn_fn$", "attn_hc.fn"),
    (r"^hc_attn_base$", "attn_hc.base"),
    (r"^hc_attn_scale$", "attn_hc.scale"),
    (r"^hc_ffn_fn$", "ffn_hc.fn"),
    (r"^hc_ffn_base$", "ffn_hc.base"),
    (r"^hc_ffn_scale$", "ffn_hc.scale"),
    (r"^attn\.", "self_attn."),
    (r"^ffn\.", "mlp."),
    # Attention leaf renames (after attn→self_attn)
    (r"^self_attn\.wq_a\.", "self_attn.q_a_proj."),
    (r"^self_attn\.wq_b\.", "self_attn.q_b_proj."),
    (r"^self_attn\.wkv\.", "self_attn.kv_proj."),
    (r"^self_attn\.wo_a\.", "self_attn.o_a_proj."),
    (r"^self_attn\.wo_b\.", "self_attn.o_b_proj."),
    (r"^self_attn\.q_norm\.", "self_attn.q_a_norm."),
    (r"^self_attn\.attn_sink$", "self_attn.sinks"),
    # MLP leaf renames (after ffn→mlp)
    (r"^mlp\.gate\.bias$", "mlp.gate.e_score_correction_bias"),
    (r"^mlp\.shared_experts\.w1\.", "mlp.shared_experts.gate_proj."),
    (r"^mlp\.shared_experts\.w2\.", "mlp.shared_experts.down_proj."),
    (r"^mlp\.shared_experts\.w3\.", "mlp.shared_experts.up_proj."),
]


def _convert_key(name: str) -> str:
    """Apply rename rules to convert a raw checkpoint key to HF format.
    Rules are applied in two passes: structural prefixes first, then leaf renames."""
    for pattern, replacement in _RENAME_RULES:
        new_name = re.sub(pattern, replacement, name)
        if new_name != name:
            name = new_name
    return name


_EXPERT_RE = re.compile(r"^mlp\.experts\.(\d+)\.(w1|w2|w3)\.weight$")


def _convert_and_fuse_weights(
    raw: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Convert raw MTP tensors into two dicts:
    - own_sd: MTP-specific weights (enorm, hnorm, e_proj, h_proj, hc_head, norm)
    - decoder_sd: decoder weights in HF format (with experts fused)
    """
    own_sd: dict[str, torch.Tensor] = {}
    decoder_sd: dict[str, torch.Tensor] = {}
    # Collect per-expert weights for fusion
    experts: dict[int, dict[str, torch.Tensor]] = {}

    for name, tensor in raw.items():
        # MTP-specific weights
        if any(name.startswith(p) for p in _MTP_SPEC_PREFIXES):
            if name == "hc_head_fn":
                own_sd["hc_head.hc_fn"] = tensor.float()
            elif name == "hc_head_base":
                own_sd["hc_head.hc_base"] = tensor.float()
            elif name == "hc_head_scale":
                own_sd["hc_head.hc_scale"] = tensor.float()
            elif name == "norm.weight":
                own_sd["shared_head_norm.weight"] = tensor
            else:
                own_sd[name] = tensor
            continue

        # Convert key to HF format
        hf_name = _convert_key(name)

        # Check if this is a per-expert weight that needs fusion
        m = _EXPERT_RE.match(hf_name)
        if m:
            idx, proj = int(m.group(1)), m.group(2)
            experts.setdefault(idx, {})[proj] = tensor
        else:
            decoder_sd[hf_name] = tensor

    # Fuse experts: w1+w3 → gate_up_proj [N, 2*inter, hidden], w2 → down_proj [N, hidden, inter]
    if experts:
        num_experts = max(experts) + 1
        gate_up_list = []
        down_list = []
        for i in range(num_experts):
            exp = experts[i]
            gate_up_list.append(torch.cat([exp["w1"], exp["w3"]], dim=0))
            down_list.append(exp["w2"])
        decoder_sd["mlp.experts.gate_up_proj"] = torch.stack(gate_up_list, dim=0)
        decoder_sd["mlp.experts.down_proj"] = torch.stack(down_list, dim=0)

    return own_sd, decoder_sd


# ---------------------------------------------------------------------------
# MTP layer module
# ---------------------------------------------------------------------------


class DeepseekV4MTPLayer(nn.Module):
    """
    One MTP layer for DeepSeek V4.

    Key architectural differences from main decoder layers:
    - No compressor/indexer in attention (sliding_attention type)
    - Uses standard top-k MoE routing (not hash routing)
    - Has e_proj/h_proj for input projection instead of direct embedding
    - Has HyperHead for collapsing hc_mult streams at the end
    """

    def __init__(
        self,
        config,
        raw_tensors: dict[str, torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult

        self.enorm = DeepseekV4RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.hnorm = DeepseekV4RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.e_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.h_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hc_head = DeepseekV4HyperHead(config)
        self.shared_head_norm = DeepseekV4RMSNorm(hidden_size, eps=config.rms_norm_eps)

        # MTP decoder uses sliding_attention (no compressor) and moe (not hash_moe).
        # Create a config copy with appropriate layer_types/mlp_layer_types for idx 0.
        mtp_config = copy.copy(config)
        mtp_config.layer_types = ["sliding_attention"]
        mtp_config.mlp_layer_types = ["moe"]
        self.decoder = DeepseekV4DecoderLayer(mtp_config, layer_idx=0)

        self.to(device=device, dtype=dtype)
        self._load_weights(raw_tensors, dtype, device)

    def _load_weights(
        self,
        raw: dict[str, torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        own_sd, decoder_sd = _convert_and_fuse_weights(raw)

        # Move to target dtype/device
        for k in own_sd:
            if own_sd[k].is_floating_point() and "hc_head" not in k:
                own_sd[k] = own_sd[k].to(dtype=dtype, device=device)
            else:
                own_sd[k] = own_sd[k].to(device=device)
        for k in decoder_sd:
            # HyperConnection fn/base/scale must stay float32 for Sinkhorn precision
            if "attn_hc." in k or "ffn_hc." in k:
                decoder_sd[k] = decoder_sd[k].to(device=device)
            else:
                decoder_sd[k] = decoder_sd[k].to(dtype=dtype, device=device)

        missing, _ = self.load_state_dict(own_sd, strict=False)
        own_missing = [k for k in missing if not k.startswith("decoder.")]
        if own_missing:
            raise ValueError(
                f"[DeepseekV4MTPLayer] Missing MTP-specific weights: {own_missing}\n"
                f"Available keys: {list(own_sd.keys())}"
            )

        dec_missing, _ = self.decoder.load_state_dict(decoder_sd, strict=False)
        if dec_missing:
            non_trivial = [k for k in dec_missing if "inv_freq" not in k]
            if non_trivial:
                print(f"[DeepseekV4MTPLayer] Decoder missing weights: {non_trivial}")

    def forward(
        self,
        token_hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        embed_tokens: nn.Embedding,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        position_embeddings: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        token_embeds = embed_tokens(input_ids)
        e_hidden = self.e_proj(self.enorm(token_embeds))
        h_hidden = self.h_proj(self.hnorm(token_hidden_states))

        # Expand to hc_mult parallel streams: [B, S, hc_mult, D]
        mtp_hidden = (e_hidden + h_hidden).unsqueeze(2).expand(
            -1, -1, self.hc_mult, -1
        ).contiguous()

        mtp_out = self.decoder(
            mtp_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            input_ids=input_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if isinstance(mtp_out, tuple):
            mtp_out = mtp_out[0]

        collapsed = self.hc_head(mtp_out)
        return self.shared_head_norm(collapsed)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def attach_mtp_layer(model, model_path: str) -> None:
    """
    Load the MTP layer from *model_path* and attach it to *model* for
    end-to-end GPTQ calibration and quantization.

    Mutates *model* in-place:
    * Adds ``model.mtp`` (:class:`DeepseekV4MTPLayer`).
    * Replaces ``model.forward`` to run MTP after the main forward.
    * Patches ``model.state_dict()`` to remap MTP keys back to checkpoint format.
    """
    config = model.config
    mtp_prefix = "mtp.0"

    try:
        ref_param = next(iter(model.parameters()))
        dtype, device = ref_param.dtype, ref_param.device
    except StopIteration:
        dtype, device = torch.bfloat16, torch.device("cpu")

    print(f"[attach_mtp_layer] Loading DeepSeek-V4 MTP weights from '{mtp_prefix}' ...")

    raw = _load_mtp_tensors(model_path, mtp_prefix)
    raw = _strip_prefix(raw, mtp_prefix)
    print(f"[attach_mtp_layer] Found {len(raw)} MTP tensors.")

    model.mtp = DeepseekV4MTPLayer(
        config=config,
        raw_tensors=raw,
        dtype=dtype,
        device=device,
    )

    # ------------------------------------------------------------------
    # Patch model.forward to include MTP in a single backbone pass.
    # Constraints: no closures, function named "forward", bound with
    # types.MethodType, single call to self.model.
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
        output_router_logits=None,
        logits_to_keep=0,
        **kwargs,
    ):
        from transformers.modeling_outputs import MoeCausalLMOutputWithPast

        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
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
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        # ---- MTP forward (activations for Hessian, logits discarded) ----
        if position_ids is None:
            seq_len = hidden_states.shape[1]
            position_ids = torch.arange(
                seq_len, device=hidden_states.device
            ).unsqueeze(0)

        position_embeddings = {
            "main": self.model.rotary_emb(
                hidden_states, position_ids=position_ids, layer_type="main"
            ),
            "compress": self.model.rotary_emb(
                hidden_states, position_ids=position_ids, layer_type="compress"
            ),
        }

        self.mtp(
            token_hidden_states=hidden_states,
            input_ids=input_ids,
            embed_tokens=self.model.embed_tokens,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            position_embeddings=position_embeddings,
        )

        return MoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=getattr(outputs, "router_logits", None),
        )

    model.forward = types.MethodType(forward, model)

    # ------------------------------------------------------------------
    # Patch state_dict() to remap MTP keys back to raw checkpoint format.
    # Reverse of _convert_key + expert unfusion for save_pretrained.
    # ------------------------------------------------------------------
    _orig_state_dict = model.state_dict

    def _patched_state_dict(*args, **kwargs):
        sd = _orig_state_dict(*args, **kwargs)
        remapped = {}
        for k, v in sd.items():
            if k.startswith("mtp."):
                rel = k[len("mtp."):]
                # MTP-specific reverse mappings
                rel = rel.replace("shared_head_norm.", "norm.")
                rel = rel.replace("hc_head.hc_fn", "hc_head_fn")
                rel = rel.replace("hc_head.hc_base", "hc_head_base")
                rel = rel.replace("hc_head.hc_scale", "hc_head_scale")
                rel = rel.replace("hc_head.input_norm.", "hc_head_input_norm.")
                if rel.startswith("decoder."):
                    rel = rel[len("decoder."):]
                    # Reverse decoder key conversions
                    rel = _reverse_convert_key(rel)
                remapped["mtp.0." + rel] = v
            else:
                remapped[k] = v
        return remapped

    model.state_dict = _patched_state_dict

    print(
        "[attach_mtp_layer] Done. MTP sub-modules: "
        + ", ".join(n for n, _ in model.mtp.named_modules() if n)
    )


# Reverse rename rules for state_dict save
_REVERSE_RENAME_RULES = [
    # Attention leaf renames (before self_attn→attn)
    (r"^self_attn\.q_a_proj\.", "self_attn.wq_a."),
    (r"^self_attn\.q_b_proj\.", "self_attn.wq_b."),
    (r"^self_attn\.kv_proj\.", "self_attn.wkv."),
    (r"^self_attn\.o_a_proj\.", "self_attn.wo_a."),
    (r"^self_attn\.o_b_proj\.", "self_attn.wo_b."),
    (r"^self_attn\.q_a_norm\.", "self_attn.q_norm."),
    (r"^self_attn\.sinks$", "self_attn.attn_sink"),
    # MLP leaf renames (before mlp→ffn)
    (r"^mlp\.gate\.e_score_correction_bias$", "mlp.gate.bias"),
    (r"^mlp\.shared_experts\.gate_proj\.", "mlp.shared_experts.w1."),
    (r"^mlp\.shared_experts\.down_proj\.", "mlp.shared_experts.w2."),
    (r"^mlp\.shared_experts\.up_proj\.", "mlp.shared_experts.w3."),
    # Structural prefix renames
    (r"^input_layernorm\.", "attn_norm."),
    (r"^post_attention_layernorm\.", "ffn_norm."),
    (r"^attn_hc\.fn$", "hc_attn_fn"),
    (r"^attn_hc\.base$", "hc_attn_base"),
    (r"^attn_hc\.scale$", "hc_attn_scale"),
    (r"^ffn_hc\.fn$", "hc_ffn_fn"),
    (r"^ffn_hc\.base$", "hc_ffn_base"),
    (r"^ffn_hc\.scale$", "hc_ffn_scale"),
    (r"^self_attn\.", "attn."),
    (r"^mlp\.", "ffn."),
]


def _reverse_convert_key(name: str) -> str:
    """Reverse HF key back to raw checkpoint format."""
    for pattern, replacement in _REVERSE_RENAME_RULES:
        new_name = re.sub(pattern, replacement, name)
        if new_name != name:
            name = new_name
    return name
