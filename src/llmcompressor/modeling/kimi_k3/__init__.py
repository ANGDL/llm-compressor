from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

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

__all__ = [
    "KimiK3ForConditionalGeneration",
    "CalibrationKimiK3SparseMoeBlock",
]
