"""
HIGGS: Heuristic ILP-Guided Grouped Scheme Mixed-Precision Quantization
"""

from llmcompressor.entrypoints.higgs.base import (
    HiggsMSECollectorConverter,
    get_higgs_config,
)
from llmcompressor.entrypoints.higgs.ilp_solver import (
    solve_ilp_mixed_precision,
)
from llmcompressor.entrypoints.higgs.utils import (
    compute_fused_layer_mse,
    compute_heuristic_alphas,
    compute_layer_mse,
    generate_config_groups,
)


def get_higgs_config_streaming(*args, **kwargs):
    """Build a HIGGS config while keeping checkpoint residency bounded."""

    from llmcompressor.streaming.higgs import get_higgs_config_streaming as _get

    return _get(*args, **kwargs)


__all__ = [
    "get_higgs_config",
    "get_higgs_config_streaming",
    "HiggsMSECollectorConverter",
    "compute_layer_mse",
    "compute_fused_layer_mse",
    "solve_ilp_mixed_precision",
    "generate_config_groups",
    "compute_heuristic_alphas",
]
