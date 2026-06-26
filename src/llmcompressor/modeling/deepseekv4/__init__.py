"""
Native DeepSeek-V4 model implementation for LLM Compressor.

Starting in transformers >= 5.12, upstream ships its own
`transformers.models.deepseek_v4.DeepseekV4ForCausalLM` for the HF-style
DeepSeek-V4 architecture. To avoid AutoConfig / class-name collisions while
keeping this reference implementation usable, we expose ours under a
distinct model_type ("deepseek_v4_native") and class name
(`DeepseekV4NativeForCausalLM`). The legacy `DeepseekV4ForCausalLM` symbol
is preserved as a deprecated alias for back-compat with existing scripts.
"""

import warnings

from transformers import AutoConfig, AutoModelForCausalLM

from .config import ModelConfig
from .model import DeepseekV4NativeForCausalLM, DeepseekV4NativePreTrainedModel

# Register under the LC-native model_type. This will not clash with HF's
# upstream "deepseek_v4" registration since we use a different key.
AutoConfig.register(ModelConfig.model_type, ModelConfig, exist_ok=True)
AutoModelForCausalLM.register(ModelConfig, DeepseekV4NativeForCausalLM, exist_ok=True)


def __getattr__(name):
    # Deprecated aliases: old code referenced `DeepseekV4ForCausalLM` /
    # `DeepseekV4PreTrainedModel` from this package. Keep them working but
    # emit a DeprecationWarning so callers migrate to the *Native* names.
    if name == "DeepseekV4ForCausalLM":
        warnings.warn(
            "`llmcompressor.modeling.deepseekv4.DeepseekV4ForCausalLM` is "
            "deprecated and will be removed; import "
            "`DeepseekV4NativeForCausalLM` instead. The new name avoids "
            "collision with the upstream `transformers.models.deepseek_v4."
            "DeepseekV4ForCausalLM` shipped since transformers 5.12.",
            DeprecationWarning,
            stacklevel=2,
        )
        return DeepseekV4NativeForCausalLM
    if name == "DeepseekV4PreTrainedModel":
        warnings.warn(
            "`llmcompressor.modeling.deepseekv4.DeepseekV4PreTrainedModel` is "
            "deprecated; import `DeepseekV4NativePreTrainedModel` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return DeepseekV4NativePreTrainedModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ModelConfig",
    "DeepseekV4NativeForCausalLM",
    "DeepseekV4NativePreTrainedModel",
]
