from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from .configuration_kimi_k3 import KimiK3Config, KimiK3VisionConfig
from .modeling_kimi_k3 import KimiK3ForConditionalGeneration

# Keep the pre-existing calibration adapter reachable after the checkpoint
# implementation became a package with the same import stem.
_legacy_path = Path(__file__).resolve().parent.parent / "kimi_k3.py"
_legacy_spec = spec_from_file_location("llmcompressor.modeling._kimi_k3_legacy", _legacy_path)
if _legacy_spec is None or _legacy_spec.loader is None:
    raise ImportError(f"Unable to load Kimi K3 calibration adapter: {_legacy_path}")
_legacy_module = module_from_spec(_legacy_spec)
_legacy_spec.loader.exec_module(_legacy_module)
CalibrationKimiK3SparseMoeBlock = _legacy_module.CalibrationKimiK3SparseMoeBlock
# The checkpoint-native implementation lives in this package, while the
# calibration/Transformers compatibility adapter predates the package split in
# the sibling ``kimi_k3.py`` module.  Re-export the adapter here as well: this
# package shadows the sibling module during normal imports.
patch_kimi_k3_transformers_compat = _legacy_module.patch_kimi_k3_transformers_compat

# Keep the private helpers reachable for the focused compatibility tests and
# for downstream callers that imported them from the old same-stem module.
_patch_flash_attn_varlen_func = _legacy_module._patch_flash_attn_varlen_func
_patch_transformers_v5_for_checkpoint_code = (
    _legacy_module._patch_transformers_v5_for_checkpoint_code
)
_patch_kimi_k3_remote_model = _legacy_module._patch_kimi_k3_remote_model

__all__ = [
    "KimiK3ForConditionalGeneration",
    "KimiK3Config",
    "KimiK3VisionConfig",
    "CalibrationKimiK3SparseMoeBlock",
    "patch_kimi_k3_transformers_compat",
]
