"""GPTQ resource and coalesced-call compatibility owned by streaming."""

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps

import torch

from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.gptq.helpers import (
    GPTQ_PRECISION,
    batch_key,
    quantization_peak_bytes,
)

from .calibration import calibration_sample_count


@dataclass
class GPTQStreamingMemoryAdapter:
    modifier: GPTQModifier

    def estimate_update_memory(self, modules: Sequence[torch.nn.Module]) -> int:
        """Bound the largest compatible compression batch before admission."""
        batches = {}
        for module in modules:
            if not self.modifier._is_calibrated(module):
                continue
            key = batch_key(module)
            batches.setdefault(
                key if key is not None else id(module), []
            ).append(module)
        scratch = 0
        for members in batches.values():
            batch_size = (
                len(members)
                if self.modifier.batched_quantization == "auto"
                else min(len(members), self.modifier.batched_quantization or 1)
            )
            scratch = max(
                scratch,
                batch_size
                * quantization_peak_bytes(members[0], self.modifier.block_size),
            )
        return scratch

    def estimate_batch_memory(
        self, module: torch.nn.Module, dispatch_tokens: int, phase: str
    ) -> int:
        """Bound FP32 input conversion and the temporary Hessian product."""
        if phase != "calibration" or not self.modifier._is_calibrated(module):
            return 0
        columns = module.weight.shape[1]
        return (2 * dispatch_tokens * columns + columns * columns) * (
            GPTQ_PRECISION.itemsize
        )

    @contextmanager
    def calibration_context(self):
        """Preserve original call counts while retaining GPTQ's original hook."""
        modifier = self.modifier
        original = modifier.calibrate_module
        had_override = "calibrate_module" in vars(modifier)
        previous_override = vars(modifier).get("calibrate_module")

        @wraps(original)
        def calibrate(module, args, output):
            result = original(module, args, output)
            expected = calibration_sample_count(args[0].ndim)
            if expected is not None:
                observed = 1 if args[0].ndim == 2 else args[0].shape[0]
                modifier._num_samples[module].add_(expected - observed)
            return result

        object.__setattr__(modifier, "calibrate_module", calibrate)
        try:
            yield
        finally:
            if had_override:
                object.__setattr__(modifier, "calibrate_module", previous_override)
            else:
                object.__delattr__(modifier, "calibrate_module")
