from loguru import logger
from pydantic import Field

from llmcompressor.core import State
from llmcompressor.modifiers import Modifier

__all__ = ["IMatrixGatherer"]


class IMatrixGatherer(Modifier):
    """Compatibility wrapper for legacy two-modifier iMatrix recipes.

    ``imatrix_mse`` observers now collect activation importance directly while the
    quantization modifier calibrates. This modifier intentionally performs no hook
    registration, but remains importable so existing recipes continue to load.

    The legacy fields are retained because they may be present in serialized recipes.
    """

    requires_calibration_data: bool = True

    targets: str | list[str] = Field(default_factory=lambda: ["Linear"])
    ignore: list[str] = Field(default_factory=lambda: ["lm_head"])
    weight_observer: str = "imatrix_mse"
    attach_by_initialize: bool = True

    def on_initialize(self, state: State, **kwargs) -> bool:
        logger.warning(
            "IMatrixGatherer is deprecated and no longer registers hooks. "
            "Configure weights.observer='imatrix_mse' on the quantization "
            "modifier instead.",
            log_once=True,
        )
        return True
