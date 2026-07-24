"""
Utility functions for metrics logging and GPU memory monitoring.

This module provides functions for tracking device memory usage, loss, and runtime
during module compression (optimization). Supports both NVIDIA and AMD GPU monitoring
"""

import time
from collections.abc import Iterable

import torch
from compressed_tensors.offload import is_distributed
from loguru import logger

from llmcompressor._torch_accelerator_compat import (
    accelerator_device_count,
    accelerator_get_memory_info,
    accelerator_is_available,
    accelerator_max_memory_allocated,
    current_device_index,
)

__all__ = ["CompressionLogger"]


class CompressionLogger:
    """
    Log metrics related to compression algorithms.

    :param module: module associated with the compression stage
    :param device_ids: accelerator devices to inspect. ``None`` preserves the
        legacy behavior of inspecting every visible device; an empty iterable
        disables accelerator memory queries.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        device_ids: Iterable[int] | None = None,
    ):
        self.module = module
        self.device_ids = (
            None if device_ids is None else tuple(dict.fromkeys(device_ids))
        )
        self.start_tick = None

        self._name = None
        self._loss = None

    def set_results(
        self,
        name: str | None = None,
        loss: float | None = None,
    ):
        self._name = name
        self._loss = loss

    def __enter__(self) -> "CompressionLogger":
        self.start_tick = time.time()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        stop_tick = time.time()

        patch = logger.patch(lambda r: r.update(function=(self._name or "compress")))

        patch.log("METRIC", f"time {(stop_tick - self.start_tick):.2f}s")
        if self._loss is not None:
            patch.log("METRIC", f"error {self._loss:.2f}")

        if not accelerator_is_available() or torch.mps.is_available():
            return

        device_ids = (
            _get_visible_devices() if self.device_ids is None else self.device_ids
        )
        for device_id in device_ids:
            used_memory = accelerator_max_memory_allocated(device_id) / 1e9
            max_memory = accelerator_get_memory_info(device_id)[1] / 1e9
            if max_memory == 0:
                continue
            perc_used = 100 * used_memory / max_memory
            patch.log(
                "METRIC",
                (
                    f"Accelerator {device_id} | usage: {perc_used:.2f}%"
                    f" | total memory: {max_memory:.1f} Gb"
                ),
            )


def _get_visible_devices() -> Iterable:
    if is_distributed():
        return [current_device_index()]

    else:
        return range(accelerator_device_count())
