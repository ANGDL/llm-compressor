"""Register native V4.1 and an original-layout fallback when upstream is absent."""

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

from .cache import DeepseekV41Cache
from .config import (
    DeepseekV41Config,
    DeepseekV41TextConfig,
    DeepseekV41VisionConfig,
    ModelConfig,
)
from .model import (
    DeepseekV41ForCausalLM,
    DeepseekV41ForConditionalGeneration,
    DeepseekV41NativeForCausalLM,
    DeepseekV41NativeForConditionalGeneration,
)

AutoConfig.register(ModelConfig.model_type, ModelConfig, exist_ok=True)
AutoModelForCausalLM.register(ModelConfig, DeepseekV41NativeForCausalLM, exist_ok=True)
AutoModelForImageTextToText.register(
    ModelConfig, DeepseekV41NativeForConditionalGeneration, exist_ok=True
)
if "deepseek_v41" not in CONFIG_MAPPING_NAMES:
    AutoConfig.register(DeepseekV41Config.model_type, DeepseekV41Config, exist_ok=True)
    AutoModelForCausalLM.register(
        DeepseekV41Config, DeepseekV41ForCausalLM, exist_ok=True
    )
    AutoModelForImageTextToText.register(
        DeepseekV41Config, DeepseekV41ForConditionalGeneration, exist_ok=True
    )

__all__ = [
    "ModelConfig",
    "DeepseekV41Config",
    "DeepseekV41Cache",
    "DeepseekV41TextConfig",
    "DeepseekV41VisionConfig",
    "DeepseekV41ForCausalLM",
    "DeepseekV41NativeForCausalLM",
    "DeepseekV41ForConditionalGeneration",
    "DeepseekV41NativeForConditionalGeneration",
]
