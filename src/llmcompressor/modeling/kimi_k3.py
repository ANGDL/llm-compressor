"""Calibration replacement for the checkpoint-native Kimi K3 MoE block."""

from __future__ import annotations

import inspect
import sys

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.utils import generic as transformers_generic
from transformers.utils import import_utils

from llmcompressor.modeling.moe_context import MoECalibrationModule

__all__ = [
    "CalibrationKimiK3SparseMoeBlock",
    "patch_kimi_k3_transformers_compat",
]


def _patch_flash_attn_varlen_func() -> None:
    """Drop ``deterministic`` for flash-attn versions that do not accept it."""
    try:
        import flash_attn
    except ImportError:
        return

    function = getattr(flash_attn, "flash_attn_varlen_func", None)
    if function is None or getattr(function, "_kimi_k3_compat_patched", False):
        return
    try:
        supports_deterministic = (
            "deterministic" in inspect.signature(function).parameters
        )
    except (TypeError, ValueError):
        supports_deterministic = False
    if supports_deterministic:
        return

    def compatible_flash_attn_varlen_func(*args, **kwargs):
        kwargs.pop("deterministic", None)
        return function(*args, **kwargs)

    compatible_flash_attn_varlen_func._kimi_k3_compat_patched = True
    compatible_flash_attn_varlen_func.__wrapped__ = function
    flash_attn.flash_attn_varlen_func = compatible_flash_attn_varlen_func


def _patch_transformers_v5_for_checkpoint_code() -> None:
    """Restore Transformers APIs and capability flags used by K3."""
    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: hasattr(torch, "fx")

    if getattr(PreTrainedModel, "_kimi_k3_fa2_compat_patched", False):
        return
    original_flash_attn_can_dispatch = PreTrainedModel._flash_attn_can_dispatch

    def flash_attn_can_dispatch(self, flash_attn_version, is_init_check=False):
        supports_fa2 = getattr(self, "_supports_flash_attn_2", False)
        if flash_attn_version != 2 or not supports_fa2 or self._supports_flash_attn:
            return original_flash_attn_can_dispatch(
                self, flash_attn_version, is_init_check
            )

        self._supports_flash_attn = True
        try:
            return original_flash_attn_can_dispatch(
                self, flash_attn_version, is_init_check
            )
        finally:
            self._supports_flash_attn = False

    PreTrainedModel._flash_attn_can_dispatch = flash_attn_can_dispatch
    PreTrainedModel._kimi_k3_fa2_compat_patched = True


def _patch_kimi_k3_remote_model(model_class: type) -> None:
    """Patch APIs captured as globals by K3's dynamically loaded text model."""
    model_module = sys.modules.get(model_class.__module__)
    linear_lm_class = getattr(model_module, "KimiLinearForCausalLM", None)
    linear_module = (
        sys.modules.get(linear_lm_class.__module__)
        if linear_lm_class is not None
        else None
    )
    if linear_module is None:
        return

    create_causal_mask = getattr(linear_module, "create_causal_mask", None)
    if create_causal_mask is not None and not getattr(
        create_causal_mask, "_kimi_k3_compat_patched", False
    ):
        parameters = inspect.signature(create_causal_mask).parameters
        rename_input = (
            "inputs_embeds" in parameters and "input_embeds" not in parameters
        )
        drop_cache_position = "cache_position" not in parameters
        if rename_input or drop_cache_position:

            def compatible_create_causal_mask(*args, **kwargs):
                if rename_input and "input_embeds" in kwargs:
                    kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
                if drop_cache_position:
                    kwargs.pop("cache_position", None)
                return create_causal_mask(*args, **kwargs)

            compatible_create_causal_mask._kimi_k3_compat_patched = True
            compatible_create_causal_mask.__wrapped__ = create_causal_mask
            linear_module.create_causal_mask = compatible_create_causal_mask

    linear_model_class = getattr(linear_module, "KimiLinearModel", None)
    if linear_model_class is None:
        return
    original_init = linear_model_class.__init__
    if getattr(original_init, "_kimi_k3_compat_patched", False):
        return

    def compatible_linear_model_init(self, config, *args, **kwargs):
        requested_attention = getattr(config, "_attn_implementation", None)
        original_init(self, config, *args, **kwargs)
        if requested_attention not in (None, "flash_attention_2"):
            # K3 unconditionally selects FA2 after constructing its layers. Keep
            # an explicit eager/SDPA request usable for CPU calibration and tests.
            config._attn_implementation = requested_attention
            self._use_flash_attention_2 = False

    compatible_linear_model_init._kimi_k3_compat_patched = True
    compatible_linear_model_init.__wrapped__ = original_init
    linear_model_class.__init__ = compatible_linear_model_init


def patch_kimi_k3_transformers_compat(model_class: type | None = None) -> None:
    """Adapt K3 checkpoint code to the installed Transformers API."""
    _patch_transformers_v5_for_checkpoint_code()
    _patch_flash_attn_varlen_func()

    if not hasattr(transformers_generic, "OutputRecorder"):
        # Transformers 5.13 moved OutputRecorder out of utils.generic, while the
        # K3 checkpoint from the same API generation still imports the old path.
        from transformers.utils.output_capturing import OutputRecorder

        transformers_generic.OutputRecorder = OutputRecorder

    if model_class is None:
        return

    _patch_kimi_k3_remote_model(model_class)

    tie_weights = model_class.tie_weights
    if "recompute_mapping" in inspect.signature(tie_weights).parameters or getattr(
        tie_weights, "_kimi_k3_compat_patched", False
    ):
        return

    def compatible_tie_weights(self, *args, **kwargs):
        # K3's override takes no arguments and delegates to language_model. New
        # Transformers versions pass their tied-weight bookkeeping arguments.
        return tie_weights(self)

    compatible_tie_weights._kimi_k3_compat_patched = True
    model_class.tie_weights = compatible_tie_weights


@MoECalibrationModule.register("KimiSparseMoeBlock")
class CalibrationKimiK3SparseMoeBlock(MoECalibrationModule):
    """Run every K3 routed expert while preserving the original top-k output.

    K3 first projects the hidden state into a latent routed-expert dimension,
    then applies the expert, optional latent RMSNorm, and a latent up-projection.
    The calibration wrapper retains those projections and the shared experts so
    their input statistics are collected exactly as in the reference model.
    """

    is_permanent = True

    def __init__(
        self,
        original: torch.nn.Module,
        config,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.config = getattr(original, "config", config)
        self.hidden_dim = original.hidden_dim
        self.num_experts = original.num_experts
        self.top_k = original.top_k
        self.experts = original.experts
        self.gate = original.gate
        self.shared_experts = getattr(original, "shared_experts", None)
        self.use_latent_moe = getattr(original, "use_latent_moe", False)
        self.routed_expert_down_proj = getattr(
            original, "routed_expert_down_proj", None
        )
        self.routed_expert_norm = getattr(original, "routed_expert_norm", None)
        self.routed_expert_up_proj = getattr(original, "routed_expert_up_proj", None)
        self.calibrate_all_experts = calibrate_all_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        residual = hidden_states
        topk_indices, topk_weights = self.gate(hidden_states)
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])

        if self.use_latent_moe:
            flat_hidden = self.routed_expert_down_proj(flat_hidden)

        output = torch.zeros_like(flat_hidden, dtype=topk_weights.dtype)
        expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).permute(
            2, 0, 1
        )

        for expert_index, expert in enumerate(self.experts):
            token_indices, weight_indices = torch.where(expert_mask[expert_index])
            has_tokens = token_indices.numel() > 0
            if self.calibrate_all_experts:
                expert_output = expert(flat_hidden)
                if has_tokens:
                    weights = topk_weights[token_indices, weight_indices]
                    output.index_add_(
                        0,
                        token_indices,
                        expert_output[token_indices] * weights.unsqueeze(-1),
                    )
            elif has_tokens:
                expert_output = expert(flat_hidden[token_indices])
                weights = topk_weights[token_indices, weight_indices]
                output.index_add_(
                    0,
                    token_indices,
                    expert_output * weights.unsqueeze(-1),
                )

        output = output.to(flat_hidden.dtype)
        if self.use_latent_moe:
            if self.routed_expert_norm is not None:
                output = self.routed_expert_norm(output)
            output = self.routed_expert_up_proj(output)

        output = output.view(original_shape)
        if self.shared_experts is not None:
            output = output + self.shared_experts(residual)
        return output
