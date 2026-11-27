"""Resolve calibration output reuse independently of expert placement."""

import warnings
from collections.abc import Sequence
from typing import Any

from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

from .modifier_contract import calibration_outputs_reusable


def resolve_propagate_error(
    modifiers: Sequence[Any], propagate_error: bool | None
) -> bool:
    """Choose a second forward or reuse outputs from the calibration pass.

    ``propagate_error=True`` is only meaningful when the modifier changes
    floating weights before propagation. GPTQ compresses weights during its
    update callback. RTN only updates qparams; its floating weights are
    compressed after the propagation pass, so an explicit request for RTN is
    downgraded to calibration output reuse.
    """
    if propagate_error is not None:
        if not isinstance(propagate_error, bool):
            raise ValueError("propagate_error must be True, False, or None")
        if len(modifiers) != 1 or type(modifiers[0]) not in (
            QuantizationModifier,
            GPTQModifier,
        ):
            raise ValueError(
                "Explicit propagate_error requires one QuantizationModifier or "
                "GPTQModifier; other recipes use their output-reuse contracts"
            )
        if propagate_error is True and type(modifiers[0]) is QuantizationModifier:
            warnings.warn(
                "propagate_error=True has no effect for QuantizationModifier "
                "(RTN) in streaming: RTN updates quantization parameters but "
                "leaves floating weights unchanged until compression, which "
                "happens after the propagation pass. Falling back to "
                "propagate_error=False (calibration output reuse).",
                UserWarning,
                stacklevel=2,
            )
            return False
        return propagate_error
    return not calibration_outputs_reusable(modifiers)


def propagation_mode(propagate_error: bool) -> str:
    return "post-update" if propagate_error else "calibration-outputs"
