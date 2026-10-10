"""Conversion of checkpoint-native weights into computation tensors."""

from .base import WeightMaterializer, materialize_weights
from .deepseek_v4 import DeepSeekV4WeightMaterializer, DeepSeekV4WeightSource
from .default import CastWeightMaterializer
from .dtype import StreamingDTypePolicy
from .kimi_k3 import KimiK3WeightMaterializer, KimiK3WeightSource
from .transformers import (
    TransformersWeightMaterializer,
    apply_transformers_weight_conversions,
)

__all__ = [
    "CastWeightMaterializer",
    "DeepSeekV4WeightMaterializer",
    "DeepSeekV4WeightSource",
    "KimiK3WeightMaterializer",
    "KimiK3WeightSource",
    "StreamingDTypePolicy",
    "TransformersWeightMaterializer",
    "WeightMaterializer",
    "apply_transformers_weight_conversions",
    "materialize_weights",
]
