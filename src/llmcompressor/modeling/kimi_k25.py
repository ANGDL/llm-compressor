"""Compatibility helpers for Kimi K2.5 checkpoint model code."""

import inspect
from os import PathLike
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.utils import import_utils

from llmcompressor.modeling.moe_context import MoECalibrationModule


def _patch_flash_attn_varlen_func() -> None:
    """Adapt checkpoint FA2 calls to flash-attn versions without ``deterministic``.

    The checkpoint imports ``flash_attn_varlen_func`` directly while its attention
    wrapper always passes ``deterministic``. Older flash-attn releases reject that
    keyword, so patch the module export before dynamic checkpoint code is imported.
    """
    try:
        import flash_attn
    except ImportError:
        return

    function = getattr(flash_attn, "flash_attn_varlen_func", None)
    if function is None or getattr(function, "_kimi_k25_compat_patched", False):
        return
    try:
        supports_deterministic = "deterministic" in inspect.signature(function).parameters
    except (TypeError, ValueError):
        supports_deterministic = False
    if supports_deterministic:
        return

    def compatible_flash_attn_varlen_func(*args, **kwargs):
        kwargs.pop("deterministic", None)
        return function(*args, **kwargs)

    compatible_flash_attn_varlen_func._kimi_k25_compat_patched = True
    compatible_flash_attn_varlen_func.__wrapped__ = function
    flash_attn.flash_attn_varlen_func = compatible_flash_attn_varlen_func


def _patch_transformers_v5_for_checkpoint_code() -> None:
    # Older DeepSeek remote code imports this helper, which was removed in v5.
    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: hasattr(torch, "fx")
    # Transformers v5 renamed the capability flag. The checkpoint implements its
    # own varlen FA2 path and advertises it using the pre-v5 flag. Mirror that flag
    # during dispatch instead of changing the attention implementation.
    if not getattr(PreTrainedModel, "_kimi_k25_fa2_compat_patched", False):
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
        PreTrainedModel._kimi_k25_fa2_compat_patched = True


def _patch_checkpoint_tie_weights(model_class: type) -> None:
    if "recompute_mapping" in inspect.signature(model_class.tie_weights).parameters:
        return

    original_tie_weights = model_class.tie_weights

    def tie_weights(self, *args, **kwargs):
        kwargs.pop("missing_keys", None)
        kwargs.pop("recompute_mapping", None)
        return original_tie_weights(self, *args, **kwargs)

    model_class.tie_weights = tie_weights


def _prepare_checkpoint_model_class(
    pretrained_model_name_or_path: str | PathLike[str],
    kwargs: dict[str, Any],
):
    config = kwargs.get("config")
    if config is None:
        config_kwargs = {
            key: kwargs[key]
            for key in ("cache_dir", "revision", "token", "local_files_only", "code_revision")
            if key in kwargs
        }
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True,
            **config_kwargs,
        )
        kwargs["config"] = config

    model_ref = getattr(config, "auto_map", {}).get("AutoModelForCausalLM")
    if model_ref is not None:
        class_kwargs = {
            key: kwargs[key]
            for key in ("cache_dir", "revision", "token", "local_files_only", "code_revision")
            if key in kwargs
        }
        model_class = get_class_from_dynamic_module(
            model_ref,
            pretrained_model_name_or_path,
            **class_kwargs,
        )
        _patch_checkpoint_tie_weights(model_class)
        return model_class
    return None


@MoECalibrationModule.register("DeepseekV3MoE")
class CalibrationKimiK25DeepseekV3MoE(MoECalibrationModule):
    """Calibration wrapper for the checkpoint's unpacked DeepSeek-V3 experts."""

    is_permanent = True

    def __init__(
        self,
        original: torch.nn.Module,
        config: Any,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.config = getattr(original, "config", config)
        self.experts = original.experts
        self.gate = original.gate
        self.shared_experts = original.shared_experts
        self.calibrate_all_experts = calibrate_all_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        original_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        output = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=len(self.experts)
        ).permute(2, 0, 1)

        for expert_index, expert in enumerate(self.experts):
            if expert is None:
                continue
            token_indices, weight_indices = torch.where(expert_mask[expert_index])
            has_tokens = token_indices.numel() > 0

            if self.calibrate_all_experts:
                expert_output = expert(hidden_states)
                if has_tokens:
                    weights = topk_weights[token_indices, weight_indices]
                    output.index_add_(
                        0,
                        token_indices,
                        expert_output[token_indices] * weights.unsqueeze(-1),
                    )
            elif has_tokens:
                expert_output = expert(hidden_states[token_indices])
                weights = topk_weights[token_indices, weight_indices]
                output.index_add_(
                    0, token_indices, expert_output * weights.unsqueeze(-1)
                )

        output = output.to(hidden_states.dtype).view(original_shape)
        return output + self.shared_experts(residuals)


def load_kimi_k25_model(
    pretrained_model_name_or_path: str | PathLike[str],
    **kwargs: Any,
) -> PreTrainedModel:
    """Load Kimi K2.5 using the implementation bundled with the checkpoint.

    The checkpoint implementation keeps each MoE expert as an independent module,
    matching the weight layout on disk. The native Transformers implementation packs
    those weights into 3D expert tensors during loading, which is prohibitively slow
    and memory intensive for the 1.9 TB Kimi K2.6 checkpoint.
    """
    _patch_transformers_v5_for_checkpoint_code()
    _patch_flash_attn_varlen_func()
    kwargs.pop("trust_remote_code", None)
    model_class = _prepare_checkpoint_model_class(pretrained_model_name_or_path, kwargs)
    if model_class is not None:
        return model_class.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )
    return AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=True,
        **kwargs,
    )


__all__ = [
    "CalibrationKimiK25DeepseekV3MoE",
    "load_kimi_k25_model",
]
