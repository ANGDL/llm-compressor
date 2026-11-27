"""Resource adapters independent of modifier lifecycle ownership."""

from collections.abc import Iterable
from contextlib import ExitStack, contextmanager
from typing import Any

from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

from .gptq_adapter import GPTQStreamingMemoryAdapter
from .modifier_adapters import get_modifier_adapter
from .quantization_adapter import QuantizationStreamingMemoryAdapter


def get_modifier_memory_adapter(modifier: Any):
    if type(modifier) is GPTQModifier:
        return GPTQStreamingMemoryAdapter(modifier)
    if type(modifier) is QuantizationModifier:
        return QuantizationStreamingMemoryAdapter(modifier)
    return get_modifier_adapter(modifier)


@contextmanager
def modifier_calibration_context(modifiers: Iterable[Any]):
    """Scope optional hook compatibility to one streaming modifier session."""
    with ExitStack() as stack:
        for modifier in modifiers:
            adapter = get_modifier_memory_adapter(modifier)
            context = getattr(adapter, "calibration_context", None)
            if callable(context):
                stack.enter_context(context())
        yield
