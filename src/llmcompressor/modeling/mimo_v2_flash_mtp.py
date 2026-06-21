"""
End-to-end MTP (Multi-Token Prediction) layer support for MiMo-V2-Flash quantization.

MiMo-V2-Flash ships 3 MTP blocks under ``model.mtp.layers.{0,1,2}.*`` in the
checkpoint while ``MiMoV2FlashForCausalLM`` itself does not instantiate an
``mtp`` submodule, so those keys are silently dropped during ``from_pretrained``
and never quantized.

This module provides :func:`attach_mtp_layer` which:

1. Reads each MTP block's weights directly from the safetensors shards.
2. Builds a :class:`MiMoV2FlashMTPLayer` per MTP block, reusing
   ``MiMoV2RMSNorm`` / ``MiMoV2Attention`` / ``MiMoV2MLP`` from the
   trust-remote-code modeling module so the Linear layers carry the same
   class identity as the main stack and llmcompressor's recipe matchers
   discover them automatically.
3. Attaches the stack as ``model.model.mtp.layers`` (a ``nn.ModuleList`` of
   length 3) so ``named_modules()`` produces names that exactly match the
   on-disk checkpoint naming (``model.mtp.layers.<i>.{enorm,hnorm,...}``).
   This is critical because HF's ``save_pretrained`` builds an offload
   ``module_map`` keyed on the in-memory attribute path; if we instead
   patched ``state_dict()`` to rename keys, that map would be out of sync
   and saving an offloaded model would crash with ``KeyError``.
4. Monkey-patches ``model.forward`` so calibration runs the main backbone
   once and then chains all MTP blocks sequentially. The MTP logits are
   discarded — only the activations matter for Hessian accumulation.

MTP forward formula (mirrors sglang ``mimo_v2_nextn`` and DeepSeek-V3 NextN)::

    h          = eh_proj(concat(enorm(embed_tokens(input_ids)),
                                hnorm(prev_hidden_states)))
    # MiMoV2DecoderLayer-equivalent (SWA + dense MLP, with the source
    # checkpoint's pre_mlp_layernorm naming for the post-attention norm):
    h          = h + self_attn(input_layernorm(h), ...)
    h          = h + mlp(pre_mlp_layernorm(h))
    mtp_out    = final_layernorm(h)

Sub-module layout (matches checkpoint key naming exactly, so no state_dict
remapping is required):

    model.mtp.layers.<i>.enorm
    model.mtp.layers.<i>.hnorm
    model.mtp.layers.<i>.eh_proj                ← Linear(2H, H), kept BF16
    model.mtp.layers.<i>.final_layernorm
    model.mtp.layers.<i>.input_layernorm
    model.mtp.layers.<i>.pre_mlp_layernorm
    model.mtp.layers.<i>.self_attn.{q,k,v,o}_proj
    model.mtp.layers.<i>.self_attn.attention_sink_bias
    model.mtp.layers.<i>.mlp.{gate,up,down}_proj

Usage in a quantization script::

    from llmcompressor.modeling.mimo_v2_flash_mtp import attach_mtp_layer

    model = AutoModelForCausalLM.from_pretrained(model_id, ..., trust_remote_code=True)
    attach_mtp_layer(model, model_id)
    # Pass model to oneshot() — MTP self_attn / mlp Linear layers will be quantized.

Notes vs. ``glm_moe_dsa_mtp`` (GLM-5):
- MiMo-V2-Flash has THREE MTP blocks (vs. one for GLM-5), exposed under
  ``model.mtp.layers.{0,1,2}`` in the checkpoint — fundamentally a different
  layout than GLM-5's ``model.layers.{N}`` flat continuation.
- MTP attention is sliding-window (``is_swa=True``) per the sglang reference.
- MTP MLP is a dense ``MiMoV2MLP`` sized with ``config.intermediate_size``
  (16384 — the same width as the dense layer 0), NOT the MoE expert width.
- Sub-modules are attached at depths matching the source checkpoint, so we
  don't need to rename ``post_attention_layernorm`` ↔ ``pre_mlp_layernorm``
  (the GLM-5 / Step-3.5 helpers do this via a state_dict patch).

============================================================================
Lessons learned (read this BEFORE adding similar attach_mtp helpers).

We hit four distinct bugs while integrating these MTP blocks; each one is
worth understanding because the same trade-offs surface for any "post-load
attached subgraph" (MTP, NextN, draft heads, …).

1. **Match checkpoint key naming with in-memory attribute paths, not via
   ``state_dict()`` monkey-patches.**
   transformers ≥ 4.57's ``save_pretrained`` (offloaded path) builds
   ``module_map`` from ``named_modules()`` and looks up *every* state_dict
   key in it ([modeling_utils.py:3960-3979], [4148-4163]). Renaming via a
   patched ``state_dict`` desynchronises the two and crashes with
   ``KeyError: 'model.mtp.layers.0.enorm.weight'`` near the end of a
   multi-hour save. Lay attributes out so ``named_modules()`` *is* the
   canonical naming. Here we attach as ``model.model.mtp.layers.<i>`` and
   inline the decoder fields (``self_attn`` / ``mlp`` / ``input_layernorm``
   / ``pre_mlp_layernorm``) directly into ``MiMoV2FlashMTPLayer``, instead
   of an inner ``decoder`` sub-module that would need ``decoder.`` stripped
   on save.

2. **The wrapper container's ``forward`` must satisfy two opposing
   constraints — pick a plain ``nn.Module`` with an explicit ``forward``,
   not ``ModuleDict``.**
   - llmcompressor's sequential pipeline calls
     ``autowrap_forward(module)`` on every ancestor of a sequential target
     ([pipelines/sequential/ast_helpers.py:48]); if
     ``module.forward.__name__ == "_forward_unimplemented"`` it raises
     ``ValueError("Cannot calibrate model which does not implement forward
     method")``. A bare ``nn.Module`` subclass without a ``forward``
     definition fails here. So define one.
   - But you cannot dodge that check by subclassing ``ModuleDict``
     (``ast_helpers.py`` skips ``ModuleList``/``ModuleDict``), because
     ``observe(list(model.modules()), "weight")`` in
     ``modifiers/quantization/calibration.py:97`` does
     ``isinstance(module, Iterable)`` and recurses on every iterable. A
     ``ModuleDict`` iterates over its STRING keys (``"layers"``); strings
     are themselves iterable, so the recursion goes ``"layers" → 'l', 'a',
     'y', ... → 'l', 'a', ...`` until ``RecursionError`` (982 frames
     observed in practice). ``ModuleList`` would dodge ``observe`` (it
     yields child Modules, not strings), but the simplest answer is plain
     ``nn.Module`` with an explicit (raising) ``forward``: it's
     non-iterable, so ``observe`` falls through to the no-observer branch,
     and ``autowrap`` is happy because the forward is defined.

3. **The patched ``model.forward`` has hard sequential-pipeline
   constraints (same as ``glm_moe_dsa_mtp``):**
   - Must be named exactly ``forward`` — ``autowrap_forward`` re-execs the
     source via ``inspect.getsource`` then ``namespace["forward"]``.
   - No closure variables; every symbol must come from the modeling
     module's globals or via ``self`` (the bound model).
   - Must be bound with ``types.MethodType(...)`` so
     ``model.forward.__func__`` exists (compressed_tensors offload path
     needs it).
   - Must call ``self.model(...)`` exactly once. Two backbone calls confuse
     ``torch.fx.symbolic_trace``, which is what the sequential pipeline
     uses.
   - For Hessian accumulation, MTP logits are intentionally discarded —
     calling ``self.lm_head`` a second time would double-count Hessians on
     the shared head.

4. **Beware of FP8-block scale shapes that don't match ``weight.shape //
   block``.** Source quantizers often pad each TP rank's slice to a block
   boundary before computing scales while storing the weight unpadded. The
   stock ``FP8BlockDequantizer`` reshapes weights with ``weight.shape //
   block`` and broadcasts against the scale, which fails on K/V (and SWA
   K) projections where ``rows / TP`` isn't a multiple of 128. Treat the
   *scale* shape as authoritative: pad the weight to ``Sb*bh × Cb*bw``,
   dequantize, trim back. See ``MiMoV2FlashFP8BlockDequantizer`` in the
   example script. (Round-trip is exact because the padded zeros stay
   zero.)

The validation scaffold in the corresponding tests / sanity-check scripts
exercises all four constraints simultaneously: ``autowrap_forwards`` over
the ancestors, ``observe(list(model.modules()), "weight")`` for recursion
safety, the HF ``module_map`` build, and a state_dict comparison against
the source key naming. If you add a new attach helper, run those four
checks before you trust it.
============================================================================
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
            "Make sure this MiMo-V2-Flash checkpoint contains MTP weights."
        )
    return tensors


def _strip_prefix(
    tensors: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    dot = prefix + "."
    return {k[len(dot):]: v for k, v in tensors.items() if k.startswith(dot)}


# ---------------------------------------------------------------------------
# Single-block MTP module
# ---------------------------------------------------------------------------


class MiMoV2FlashMTPLayer(nn.Module):
    """One MiMo-V2-Flash MTP block.

    Sub-modules are attached so the attribute path matches the source
    checkpoint's key naming exactly — see this module's docstring.
    """

    def __init__(
        self,
        model,
        layer_idx: int,
        raw_tensors: dict[str, torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        config = model.config
        H = config.hidden_size
        eps = config.layernorm_epsilon

        # Reuse the exact same classes the main stack uses. We pull them off
        # the loaded model so we don't have to import the trust_remote_code
        # module by name.
        norm_cls = model.model.norm.__class__                  # MiMoV2RMSNorm
        attn_cls = model.model.layers[0].self_attn.__class__   # MiMoV2Attention
        # MiMoV2MLP – grab from any MiMoV2MLP instance. Layer 0 is the dense
        # MLP layer (moe_layer_freq[0] == 0); use it to get the class.
        mlp_cls = model.model.layers[0].mlp.__class__          # MiMoV2MLP

        # ---- MTP-specific ----
        self.enorm = norm_cls(H, eps=eps)
        self.hnorm = norm_cls(H, eps=eps)
        self.eh_proj = nn.Linear(2 * H, H, bias=False)
        self.final_layernorm = norm_cls(H, eps=eps)

        # ---- Decoder-equivalent (SWA attention + dense MLP) ----
        self.input_layernorm = norm_cls(H, eps=eps)
        self.pre_mlp_layernorm = norm_cls(H, eps=eps)
        # MTP attention is sliding-window with sink_bias (per sglang
        # ``mimo_v2_nextn``), regardless of where the synthetic layer_idx
        # would land in hybrid_layer_pattern.
        self.self_attn = attn_cls(config, is_swa=True, layer_idx=layer_idx)
        # MTP MLP is dense, sized with config.intermediate_size (16384), NOT
        # config.moe_intermediate_size — matches sglang ``mimo_v2_nextn``.
        self.mlp = mlp_cls(config)

        self.to(device=device, dtype=dtype)
        self._load_weights(raw_tensors, dtype, device)

    def _load_weights(
        self,
        raw: dict[str, torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        sd = {k: v.to(dtype=dtype, device=device) for k, v in raw.items()}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if missing:
            non_rotary = [k for k in missing if "inv_freq" not in k]
            if non_rotary:
                print(
                    f"[MiMoV2FlashMTPLayer] Missing weights "
                    f"(non-rotary): {non_rotary}"
                )
        if unexpected:
            print(
                f"[MiMoV2FlashMTPLayer] Unexpected weights "
                f"(skipped on load): {unexpected}"
            )

    def forward(
        self,
        prev_hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        embed_tokens: nn.Embedding,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # ---- MTP input projection ----
        # Concat order matches sglang ``mimo_v2_nextn`` (token branch first):
        #     eh_proj weight [H, 2H] splits its input as
        #         [:H]  ← enorm(token_embeds)
        #         [H:]  ← hnorm(prev_hidden_states)
        token_embeds = embed_tokens(input_ids)
        h = self.eh_proj(
            torch.cat(
                [self.enorm(token_embeds), self.hnorm(prev_hidden_states)],
                dim=-1,
            )
        )

        # ---- MiMoV2DecoderLayer-equivalent forward ----
        # Mirrors MiMoV2DecoderLayer.forward verbatim, except that the
        # post-attention norm is named ``pre_mlp_layernorm`` here so the
        # checkpoint key naming round-trips cleanly.
        residual = h
        h = self.input_layernorm(h)
        attn_out, _ = self.self_attn(
            hidden_states=h,
            attention_mask=None,        # mask preparation is bypassed; OK for Hessians
            position_ids=position_ids,
            past_key_values=None,       # no KV cache during calibration
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        h = residual + attn_out

        residual = h
        h = self.pre_mlp_layernorm(h)
        h = self.mlp(h)
        h = residual + h

        return self.final_layernorm(h)


class _MiMoV2FlashMTPContainer(nn.Module):
    """Tiny wrapper so the attribute path becomes ``model.mtp.layers.<i>``.

    Attached as ``model.model.mtp`` (not ``model.mtp``), so ``named_modules()``
    on the top-level CausalLM yields ``model.mtp.layers.<i>.*`` — exactly
    matching the on-disk checkpoint naming and HF's offloaded
    ``save_pretrained`` ``module_map`` keys.

    Why a plain ``nn.Module`` (not ``ModuleDict``):
    - ``ModuleDict`` is iterable over its KEYS (strings). llmcompressor's
      observer dispatch in ``modifiers/quantization/calibration.observe`` does
      ``isinstance(module, Iterable)`` and recurses; it would treat the
      string keys as Iterable too, hitting infinite recursion.
    - ``ModuleList`` would dodge ``observe`` (iterating yields child
      ``nn.Module`` instances, not strings), but plain ``nn.Module`` is the
      simplest semantic fit and also sidesteps the autowrap path: we provide
      an explicit ``forward`` method below so ``autowrap_forward`` doesn't
      hit the ``_forward_unimplemented`` guard. The body is never executed
      because the patched ``model.forward`` iterates ``self.model.mtp.layers``
      directly, bypassing the container.
    """

    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "_MiMoV2FlashMTPContainer is a passive container and should not "
            "be called directly. The patched model.forward iterates "
            "model.mtp.layers explicitly."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _count_mtp_layers(model_path: str) -> int:
    """Read the safetensors index to count ``model.mtp.layers.<i>.*`` blocks."""
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        # Single-shard fallback — rely on the safetensors header.
        candidates: set[int] = set()
        path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(path):
            with safe_open(path, framework="pt") as f:
                for key in f.keys():
                    if key.startswith("model.mtp.layers."):
                        idx_str = key[len("model.mtp.layers."):].split(".", 1)[0]
                        if idx_str.isdigit():
                            candidates.add(int(idx_str))
        return len(candidates)

    with open(index_file) as f:
        weight_map: dict[str, str] = json.load(f)["weight_map"]
    indices: set[int] = set()
    for k in weight_map:
        if k.startswith("model.mtp.layers."):
            idx_str = k[len("model.mtp.layers."):].split(".", 1)[0]
            if idx_str.isdigit():
                indices.add(int(idx_str))
    return len(indices)


def attach_mtp_layer(model, model_path: str) -> None:
    """
    Load all MiMo-V2-Flash MTP blocks (``model.mtp.layers.{0,1,2}.*``) from
    *model_path* and attach them as ``model.model.mtp.layers`` so the MTP
    stack participates in calibration and gets quantized end-to-end.

    The function mutates *model* in-place:

    * Adds ``model.model.mtp`` (a tiny container with ``layers: nn.ModuleList``
      of :class:`MiMoV2FlashMTPLayer`).
    * Replaces ``model.forward`` with a patched version that runs the main
      backbone once and then chains all MTP blocks sequentially. The MTP
      logits are discarded; only activations matter for Hessian accumulation.

    No ``state_dict`` patch is applied — the in-memory attribute path is
    designed to match the source checkpoint key naming exactly.

    Parameters
    ----------
    model :
        A ``MiMoV2FlashForCausalLM`` instance already loaded via
        ``AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)``.
    model_path :
        Directory containing the (BF16) safetensors shards with MTP weights
        under the ``model.mtp.layers.<i>.*`` namespace.
    """
    config = model.config
    num_mtp = _count_mtp_layers(model_path)
    if num_mtp <= 0:
        print(
            f"[attach_mtp_layer] No 'model.mtp.layers.*' tensors found under "
            f"{model_path!r}; skipping MTP attach."
        )
        return

    try:
        ref_param = next(iter(model.parameters()))
        dtype, device = ref_param.dtype, ref_param.device
    except StopIteration:
        dtype, device = torch.bfloat16, torch.device("cpu")

    mtp_layers: list[nn.Module] = []
    for offset in range(num_mtp):
        # Synthetic main-stack layer index — only used by MiMoV2Attention to
        # tag layer_idx for KV cache; we don't use the cache during calibration
        # so the value just needs to be unique past the main stack.
        layer_idx = config.num_hidden_layers + offset
        mtp_prefix = f"model.mtp.layers.{offset}"

        print(
            f"[attach_mtp_layer] Loading MiMo-V2-Flash MTP weights from "
            f"'{mtp_prefix}' ..."
        )
        raw = _load_mtp_tensors(model_path, mtp_prefix)
        raw = _strip_prefix(raw, mtp_prefix)
        print(
            f"[attach_mtp_layer] Found {len(raw)} MTP tensors for block {offset}."
        )

        mtp_layers.append(
            MiMoV2FlashMTPLayer(
                model=model,
                layer_idx=layer_idx,
                raw_tensors=raw,
                dtype=dtype,
                device=device,
            )
        )

    # Attach under model.model so named_modules() produces "model.mtp.layers.<i>".
    model.model.mtp = _MiMoV2FlashMTPContainer(mtp_layers)

    # ------------------------------------------------------------------
    # Patch model.forward to include the MTP chain after a single backbone
    # pass. Same constraints as in glm_moe_dsa_mtp.attach_mtp_layer:
    #
    # 1. The function must be named exactly ``forward`` (autowrap_forward
    #    re-exec's its source and looks up ``namespace["forward"]``).
    # 2. The function runs in the modeling module's globals — no closures.
    #    Every symbol must be either present in that module or accessed via
    #    ``self`` (the bound model).
    # 3. The function must be bound with ``types.MethodType`` so that
    #    ``model.forward.__func__`` exists (compressed_tensors offload path).
    # 4. The backbone (``self.model``) must be called exactly once — multiple
    #    calls confuse torch.fx symbolic tracing.
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

        # ---- main lm_head (mirrors MiMoV2FlashForCausalLM.forward) ----
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

        # ---- MTP chain ----
        # Activations flow through MTP Linear layers for GPTQ Hessian
        # accumulation. The final MTP outputs are intentionally discarded
        # (we do NOT call lm_head again — that would double-count the head).
        if position_ids is None:
            seq_len = hidden_states.shape[1]
            position_ids = torch.arange(
                seq_len, device=hidden_states.device
            ).unsqueeze(0)

        # MTP attention is SWA, so use the SWA rotary embeddings.
        mtp_position_embeddings = self.model.swa_rotary_emb(
            hidden_states, position_ids
        )

        prev_hidden = hidden_states
        for mtp_block in self.model.mtp.layers:
            prev_hidden = mtp_block(
                prev_hidden_states=prev_hidden,
                input_ids=input_ids,
                embed_tokens=self.model.embed_tokens,
                position_ids=position_ids,
                cache_position=cache_position,
                position_embeddings=mtp_position_embeddings,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    model.forward = types.MethodType(forward, model)

    print(
        "[attach_mtp_layer] Done. MTP sub-modules: "
        + ", ".join(
            name for name, _ in model.model.mtp.named_modules() if name
        )
    )
