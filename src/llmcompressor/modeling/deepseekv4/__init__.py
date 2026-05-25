from transformers import AutoConfig, AutoModelForCausalLM

from .config import ModelConfig
from .model import DeepseekV4ForCausalLM

try:
    AutoConfig.register("deepseek_v4", ModelConfig)
    AutoModelForCausalLM.register(ModelConfig, DeepseekV4ForCausalLM)
except ValueError:
    pass

__all__ = ["ModelConfig", "DeepseekV4ForCausalLM"]
