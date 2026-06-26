# ruff: noqa

"""
Model preparation and fusion utilities for compression workflows.

Provides tools for preparing models for compression including
layer fusion, module preparation, and model structure optimization.
Handles pre-compression transformations and architectural modifications
needed for efficient compression.
"""

# trigger registration
from .deepseek_v32 import CalibrationDeepseekV32MoE  # noqa: F401
from .deepseek_v4 import CalibrationDeepseekV4MoE  # noqa: F401
from .glm_moe_dsa import CalibrationGlmMoeDsaMoE  # noqa: F401
from .qwen3_5_moe import CalibrationQwen3_5MoeSparseMoeBlock  # noqa: F401
from .offset_norm import CalibrationOffsetNorm  # noqa: F401
from .step3p5 import CalibrationStep3p5MoEMLP  # noqa: F401

from .fuse import *
