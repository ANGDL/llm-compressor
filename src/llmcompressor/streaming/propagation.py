"""Resolve calibration output reuse independently of expert placement."""

from collections.abc import Sequence
from typing import Any

from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

from .modifier_contract import calibration_outputs_reusable


def resolve_propagate_error(
    modifiers: Sequence[Any], propagate_error: bool | None
) -> bool:
    """Choose a second forward or reuse outputs from the calibration pass."""
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
        return propagate_error
    return not calibration_outputs_reusable(modifiers)


def propagation_mode(propagate_error: bool) -> str:
    return "post-update" if propagate_error else "calibration-outputs"
