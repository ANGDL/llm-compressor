"""Use the existing modifier lifecycle within one expert residency window."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from copy import deepcopy

import torch
from compressed_tensors.offload.dist_utils import is_distributed
from torch import nn

from llmcompressor.core import LifecycleCallbacks, create_session
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier


class ExpertGroupModifierAdapter:
    """Reconstruct a fresh modifier per group without touching its internals.

    This reference adapter supports GPTQ and weight-only RTN. Transforms and
    modifiers with dependencies outside an expert require separate adapters.
    """

    def __init__(self, modifier: GPTQModifier | QuantizationModifier):
        if type(modifier) not in (GPTQModifier, QuantizationModifier):
            raise TypeError("expert groups currently support GPTQ and RTN modifiers")
        if modifier.kv_cache_scheme is not None:
            raise ValueError("expert groups do not support KV-cache quantization")
        self.modifier_type = type(modifier)
        self.arguments = modifier.model_dump(mode="python")

    @contextmanager
    def calibration(self, model: nn.Module) -> Generator[None, None, None]:
        if is_distributed():
            raise ValueError("expert group execution currently requires a single rank")
        modifier = self.modifier_type(**deepcopy(self.arguments))
        modules = list(model.modules())
        with create_session() as session:
            session.state.layerwise_calibration = True
            try:
                session.initialize(
                    model=model, recipe=[modifier], start=-1, copy_data=False
                )
                LifecycleCallbacks.calibration_start()
                modifier.start_layerwise_calibration(model, modules)
                with torch.no_grad():
                    yield
                LifecycleCallbacks.sequential_epoch_end(modules)
                modifier.end_layerwise_calibration(modules)
                LifecycleCallbacks.calibration_end()
                session.finalize()
            finally:
                modifier.end_layerwise_calibration(modules)
                modifier.remove_hooks()
