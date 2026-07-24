"""End-to-end MTP support for MiMo-V2.5 quantization.

MiMo-V2.5 checkpoints contain three MTP blocks under
``model.mtp.layers.{0,1,2}``, but the remote Hugging Face model intentionally
does not construct them. ``attach_mtp_layer`` restores those blocks before
calibration so their decoder Linear layers are quantized and saved under the
same checkpoint names.
"""

from __future__ import annotations

import json
import os
import types
from typing import Optional

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers.modeling_outputs import CausalLMOutputWithPast


def _load_mtp_tensors(model_path: str, mtp_prefix: str) -> dict[str, torch.Tensor]:
    """Load all tensors belonging to one MTP block."""
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file, encoding="utf-8") as file:
            weight_map: dict[str, str] = json.load(file)["weight_map"]
        shard_files = sorted(
            {
                shard
                for name, shard in weight_map.items()
                if name.startswith(mtp_prefix + ".")
            }
        )
    else:
        shard_files = ["model.safetensors"]

    tensors: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        with safe_open(os.path.join(model_path, shard), framework="pt") as file:
            for name in file.keys():
                if name.startswith(mtp_prefix + "."):
                    tensors[name[len(mtp_prefix) + 1 :]] = file.get_tensor(name)

    if not tensors:
        raise ValueError(
            f"No tensors found with prefix '{mtp_prefix}' in {model_path}."
        )
    return tensors


def _count_mtp_layers(model_path: str) -> int:
    """Count contiguous ``model.mtp.layers.<index>`` blocks in a checkpoint."""
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file, encoding="utf-8") as file:
            names = json.load(file)["weight_map"]
    else:
        checkpoint_path = os.path.join(model_path, "model.safetensors")
        with safe_open(checkpoint_path, framework="pt") as file:
            names = file.keys()

    indices = {
        int(name[len("model.mtp.layers.") :].split(".", 1)[0])
        for name in names
        if name.startswith("model.mtp.layers.")
        and name[len("model.mtp.layers.") :].split(".", 1)[0].isdigit()
    }
    return len(indices)


class MiMoV2MTPLayer(nn.Module):
    """One MiMo-V2.5 SWA-attention, dense-MLP MTP decoder block."""

    def __init__(
        self,
        model,
        layer_idx: int,
        tensors: dict[str, torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        config = model.config
        hidden_size = config.hidden_size
        norm_cls = model.model.norm.__class__
        attention_cls = model.model.layers[0].self_attn.__class__
        mlp_cls = model.model.layers[0].mlp.__class__

        self.enorm = norm_cls(hidden_size, eps=config.layernorm_epsilon)
        self.hnorm = norm_cls(hidden_size, eps=config.layernorm_epsilon)
        self.eh_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.final_layernorm = norm_cls(hidden_size, eps=config.layernorm_epsilon)
        self.input_layernorm = norm_cls(hidden_size, eps=config.layernorm_epsilon)
        self.pre_mlp_layernorm = norm_cls(hidden_size, eps=config.layernorm_epsilon)
        self.self_attn = attention_cls(
            config,
            is_swa=True,
            layer_idx=layer_idx,
            projection_layout="split",
        )
        self.mlp = mlp_cls(config)

        self.to(device=device, dtype=dtype)
        state_dict = {
            name: tensor.to(device=device, dtype=dtype)
            for name, tensor in tensors.items()
        }
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        missing = [name for name in missing if "inv_freq" not in name]
        if missing:
            raise ValueError(f"Missing MTP weights for block {layer_idx}: {missing}")
        if unexpected:
            raise ValueError(
                f"Unexpected MTP weights for block {layer_idx}: {unexpected}"
            )

    def forward(
        self,
        previous_hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        embed_tokens: nn.Embedding,
        position_ids: Optional[torch.LongTensor],
        cache_position: Optional[torch.LongTensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        token_embeddings = embed_tokens(input_ids)
        hidden_states = self.eh_proj(
            torch.cat(
                [self.hnorm(previous_hidden_states), self.enorm(token_embeddings)],
                dim=-1,
            )
        )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_output, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.pre_mlp_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return self.final_layernorm(hidden_states)


class _MiMoV2MTPContainer(nn.Module):
    """Passive container that preserves ``model.mtp.layers`` checkpoint paths."""

    def __init__(self, layers: list[MiMoV2MTPLayer]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "MTP container is invoked through the patched model forward."
        )


def attach_mtp_layer(model, model_path: str) -> None:
    """Attach MiMo-V2.5 MTP blocks and include them in calibration forwards."""
    num_mtp_layers = _count_mtp_layers(model_path)
    if num_mtp_layers == 0:
        print(f"[attach_mtp_layer] No MTP weights found in {model_path!r}; skipping.")
        return

    reference_parameter = next(iter(model.parameters()))
    layers: list[MiMoV2MTPLayer] = []
    for offset in range(num_mtp_layers):
        prefix = f"model.mtp.layers.{offset}"
        tensors = _load_mtp_tensors(model_path, prefix)
        layers.append(
            MiMoV2MTPLayer(
                model=model,
                layer_idx=model.config.num_hidden_layers + offset,
                tensors=tensors,
                dtype=reference_parameter.dtype,
                device=reference_parameter.device,
            )
        )
        print(f"[attach_mtp_layer] Loaded {len(tensors)} tensors from '{prefix}'.")

    model.model.mtp = _MiMoV2MTPContainer(layers)

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

        if position_ids is None:
            position_ids = torch.arange(
                hidden_states.shape[1], device=hidden_states.device
            ).unsqueeze(0)
        position_embeddings = self.model.swa_rotary_emb(hidden_states, position_ids)

        mtp_hidden_states = hidden_states
        for mtp_layer in self.model.mtp.layers:
            mtp_hidden_states = mtp_layer(
                previous_hidden_states=mtp_hidden_states,
                input_ids=input_ids,
                embed_tokens=self.model.embed_tokens,
                position_ids=position_ids,
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

    model.forward = types.MethodType(forward, model)
