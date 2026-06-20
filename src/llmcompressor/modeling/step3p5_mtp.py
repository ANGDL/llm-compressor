"""
End-to-end MTP (Multi-Token Prediction) layer support for Step-3.5 quantization.

Step-3.5 stores three MTP layers under ``model.layers.45.*`` / ``46.*`` /
``47.*`` in the checkpoint while the HuggingFace remote model only instantiates
``config.num_hidden_layers`` main decoder blocks (0-44) and ignores the extra
MTP layers via ``_keys_to_ignore_on_load_unexpected``.

This module provides :func:`attach_mtp_layer` which:

1. Reads the MTP weights directly from the safetensors shards.
2. Builds MTP layers with the same decoder / norm classes as the loaded model.
3. Attaches them as ``model.mtp`` so llmcompressor can discover and quantize
   the MTP decoder ``Linear`` layers end-to-end.
4. Monkey-patches ``model.forward`` so calibration runs the main backbone once,
   then flows activations through all MTP layers sequentially.
5. Monkey-patches ``model.state_dict`` so saved checkpoints use the original
   ``model.layers.{45,46,47}.*`` naming.
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
from transformers.modeling_outputs import CausalLMOutputWithPast


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _load_mtp_tensors(model_path: str, mtp_prefix: str) -> dict[str, torch.Tensor]:
    """Return every tensor whose name starts with ``<mtp_prefix>.``."""
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
            "Make sure this checkpoint contains Step-3.5 MTP weights."
        )
    return tensors


def _strip_prefix(
    tensors: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    dot = prefix + "."
    return {k[len(dot):]: v for k, v in tensors.items() if k.startswith(dot)}


# ---------------------------------------------------------------------------
# MTP layer module
# ---------------------------------------------------------------------------


class Step3p5MTPLayer(nn.Module):
    """One Step-3.5 MTP layer.

    Checkpoint layout relative to ``model.layers.<idx>.``::

        enorm.weight                         -> enorm
        hnorm.weight                         -> hnorm
        eh_proj.weight                       -> eh_proj
        transformer.shared_head.norm.weight  -> shared_head_norm
        transformer.shared_head.output.weight-> shared_head_output
        self_attn.*                          -> decoder.self_attn.*
        mlp.*                                -> decoder.mlp.*
        input_layernorm.*                    -> decoder.input_layernorm.*
        post_attention_layernorm.*           -> decoder.post_attention_layernorm.*
    """

    def __init__(
        self,
        model,
        mtp_layer_idx: int,
        raw_tensors: dict[str, torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        config = model.config
        hidden_size = config.hidden_size

        decoder_cls = model.model.layers[0].__class__
        norm_cls = model.model.norm.__class__

        self.enorm = norm_cls(hidden_size, eps=config.rms_norm_eps)
        self.hnorm = norm_cls(hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.shared_head_norm = norm_cls(hidden_size, eps=config.rms_norm_eps)
        self.shared_head_output = nn.Linear(hidden_size, config.vocab_size, bias=False)

        mtp_config = copy.copy(config)
        self.decoder = decoder_cls(mtp_config, layer_idx=mtp_layer_idx)

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
            if name.startswith("transformer.shared_head.norm."):
                own_sd[name.replace("transformer.shared_head.norm.", "shared_head_norm.")] = t
            elif name.startswith("transformer.shared_head.output."):
                own_sd[
                    name.replace("transformer.shared_head.output.", "shared_head_output.")
                ] = t
            elif name.startswith(("enorm.", "hnorm.", "eh_proj.")):
                own_sd[name] = t
            else:
                decoder_sd[name] = t

        missing, _ = self.load_state_dict(own_sd, strict=False)
        own_missing = [k for k in missing if not k.startswith("decoder.")]
        if own_missing:
            raise ValueError(
                f"[Step3p5MTPLayer] Missing MTP-specific weights: {own_missing}\n"
                f"Available keys in checkpoint: {list(own_sd.keys())}"
            )

        dec_missing, _ = self.decoder.load_state_dict(decoder_sd, strict=False)
        if dec_missing:
            non_rotary = [k for k in dec_missing if "inv_freq" not in k]
            if non_rotary:
                print(f"[Step3p5MTPLayer] Decoder missing weights: {non_rotary}")

    def forward(
        self,
        token_hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        embed_tokens: nn.Embedding,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        token_embeds = embed_tokens(input_ids)
        mtp_input = torch.cat(
            [self.enorm(token_embeds), self.hnorm(token_hidden_states)],
            dim=-1,
        )
        mtp_hidden = self.eh_proj(mtp_input)

        mtp_out = self.decoder(
            mtp_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        if isinstance(mtp_out, tuple):
            mtp_out = mtp_out[0]

        shared_head_hidden = self.shared_head_norm(mtp_out)
        self.shared_head_output(shared_head_hidden)
        return shared_head_hidden


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def attach_mtp_layer(model, model_path: str) -> None:
    """
    Load all Step-3.5 MTP layers (``model.layers.{num_hidden_layers + i}.*``)
    from *model_path* and attach them to *model* so the MTP stack participates
    in calibration and gets quantized end-to-end.
    """
    config = model.config
    num_mtp_layers = getattr(config, "num_nextn_predict_layers", 0)
    if num_mtp_layers <= 0:
        return

    mtp_start_idx = config.num_hidden_layers

    try:
        ref_param = next(iter(model.parameters()))
        dtype, device = ref_param.dtype, ref_param.device
    except StopIteration:
        dtype, device = torch.bfloat16, torch.device("cpu")

    mtp_layers = []
    for offset in range(num_mtp_layers):
        mtp_layer_idx = mtp_start_idx + offset
        mtp_prefix = f"model.layers.{mtp_layer_idx}"
        print(f"[attach_mtp_layer] Loading Step3.5 MTP weights from '{mtp_prefix}' ...")
        raw = _load_mtp_tensors(model_path, mtp_prefix)
        raw = _strip_prefix(raw, mtp_prefix)
        print(f"[attach_mtp_layer] Found {len(raw)} MTP tensors for layer {mtp_layer_idx}.")
        mtp_layers.append(
            Step3p5MTPLayer(
                model=model,
                mtp_layer_idx=mtp_layer_idx,
                raw_tensors=raw,
                dtype=dtype,
                device=device,
            )
        )

    model.mtp = nn.ModuleList(mtp_layers)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
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
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        if position_ids is None:
            if cache_position is None:
                seq_len = hidden_states.shape[1]
                position_ids = torch.arange(
                    seq_len, device=hidden_states.device
                ).unsqueeze(0)
            else:
                position_ids = cache_position.unsqueeze(0)

        mtp_hidden_states = hidden_states
        for mtp_layer in self.mtp:
            mtp_hidden_states = mtp_layer(
                token_hidden_states=mtp_hidden_states,
                input_ids=input_ids,
                embed_tokens=self.model.embed_tokens,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=None,
                cache_position=cache_position,
            )

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    model.forward = types.MethodType(forward, model)

    _orig_state_dict = model.state_dict

    def _patched_state_dict(*args, **kwargs):
        sd = _orig_state_dict(*args, **kwargs)
        remapped = {}
        for k, v in sd.items():
            if not k.startswith("mtp."):
                remapped[k] = v
                continue

            parts = k.split(".")
            layer_offset = int(parts[1])
            layer_idx = mtp_start_idx + layer_offset
            rel = ".".join(parts[2:])
            if rel.startswith("shared_head_norm."):
                rel = rel.replace("shared_head_norm.", "transformer.shared_head.norm.", 1)
            elif rel.startswith("shared_head_output."):
                rel = rel.replace(
                    "shared_head_output.", "transformer.shared_head.output.", 1
                )
            elif rel.startswith("decoder."):
                rel = rel[len("decoder."):]
            remapped[f"model.layers.{layer_idx}.{rel}"] = v
        return remapped

    model.state_dict = _patched_state_dict

    print(
        "[attach_mtp_layer] Done. MTP sub-modules: "
        + ", ".join(name for name, _ in model.mtp.named_modules() if name)
    )
