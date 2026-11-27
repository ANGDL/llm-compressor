"""Bounded CPU batch production for streaming calibration."""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from ._logging import streaming_logger
from .activations.store import _tensor_bytes, _tensors
from .loading.host_memory import (
    HostMemoryBudget,
    HostMemoryError,
    HostMemoryReservation,
)


def initial_boundary_async_enabled() -> bool:
    value = os.getenv("LLMCOMPRESSOR_STREAMING_ASYNC_INITIAL_BOUNDARY", "1")
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass
class PreparedCalibrationBatch(AbstractContextManager):
    """Own one CPU batch and its admission until the consumer releases it."""

    index: int
    value: Any
    reservation: HostMemoryReservation
    prepare_seconds: float

    def close(self) -> None:
        self.value = None
        self.reservation.close()

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()
        return False


class CalibrationDataWorker(AbstractContextManager):
    """Produce an independently owned batch while the caller consumes the current."""

    def __init__(
        self,
        count: int,
        produce: Callable[[int], Any],
        estimate_bytes: Callable[[int], int],
        *,
        host_budget: HostMemoryBudget,
        prepare: Callable[[Any], Any] | None = None,
        enabled: bool = True,
    ):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")
        self.count = count
        self.produce = produce
        self.estimate_bytes = estimate_bytes
        self.host_budget = host_budget
        self.prepare = prepare
        self.enabled = enabled
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="streaming-data")
            if enabled
            else None
        )
        self._pending: Future[PreparedCalibrationBatch] | None = None
        self._index = 0
        self._closed = False
        self._fallbacks = 0
        self._wait_seconds = 0.0
        self._prepare_seconds = 0.0

    def _produce(
        self, index: int, reservation: HostMemoryReservation
    ) -> PreparedCalibrationBatch:
        started = perf_counter()
        value = None
        try:
            value = self.produce(index)
            if any(tensor.device.type != "cpu" for tensor in _tensors(value)):
                raise TypeError("Calibration data workers require CPU batches")
            initial_bytes = _tensor_bytes(value)
            reservation.commit(initial_bytes)
            if self.prepare is not None:
                value = self.prepare(value)
                if any(tensor.device.type != "cpu" for tensor in _tensors(value)):
                    raise TypeError("Calibration data workers require CPU batches")
                prepared_bytes = _tensor_bytes(value)
                if prepared_bytes >= initial_bytes:
                    reservation.commit(prepared_bytes - initial_bytes)
                else:
                    reservation.release_committed(initial_bytes - prepared_bytes)
            reservation.release_reserved(reservation.reserved_bytes)
            return PreparedCalibrationBatch(
                index, value, reservation, perf_counter() - started
            )
        except BaseException:
            value = None
            reservation.close()
            raise

    def prefetch(self) -> None:
        """Reserve capacity before submitting the next read; defer under pressure."""
        if (
            self._closed
            or self._executor is None
            or self._pending is not None
            or self._index == self.count
        ):
            return
        try:
            reservation = self.host_budget.reserve(
                f"calibration input batch {self._index}",
                self.estimate_bytes(self._index),
            )
        except HostMemoryError:
            self._fallbacks += 1
            return
        try:
            self._pending = self._executor.submit(
                self._produce, self._index, reservation
            )
        except BaseException:
            reservation.close()
            raise

    def take(self) -> PreparedCalibrationBatch:
        """Acquire the next ordered batch without hiding production failures."""
        if self._closed:
            raise RuntimeError("Calibration data worker is closed")
        if self._index == self.count:
            raise StopIteration
        started = perf_counter()
        if self._pending is None:
            reservation = self.host_budget.reserve(
                f"calibration input batch {self._index}",
                self.estimate_bytes(self._index),
            )
            batch = self._produce(self._index, reservation)
        else:
            pending, self._pending = self._pending, None
            try:
                batch = pending.result()
            except BaseException as error:
                error.add_note(f"Calibration data worker failed at batch {self._index}")
                raise
        self._index += 1
        self._prepare_seconds += batch.prepare_seconds
        self._wait_seconds += perf_counter() - started
        return batch

    def discard_prefetch(self) -> None:
        """Release speculative input capacity without advancing the batch index."""
        pending, self._pending = self._pending, None
        if pending is not None:
            pending.result().close()

    def close(self, *, suppress_errors: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        pending, self._pending = self._pending, None
        error = None
        if pending is not None:
            try:
                pending.result().close()
            except BaseException as caught:
                error = caught
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        streaming_logger.info(
            "initial boundary | input worker | "
            f"batches={self._index} | prepare_s={self._prepare_seconds:.2f} | "
            f"wait_s={self._wait_seconds:.2f} | deferred={self._fallbacks}"
        )
        if error is not None and not suppress_errors:
            raise error

    def __exit__(self, exc_type, _exc_value, _traceback):
        self.close(suppress_errors=exc_type is not None)
        return False
