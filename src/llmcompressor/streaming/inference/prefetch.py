"""One CPU-only lookahead group for native inference offload hooks."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter

import torch

from llmcompressor.streaming.loading import HostMemoryError, HostMemoryReservation

from .weights import StreamingWeights


@dataclass
class _PreparedGroup:
    values: dict[str, torch.Tensor]
    reservation: HostMemoryReservation

    def close(self):
        self.values.clear()
        self.reservation.close()


class InferencePrefetcher:
    """Prepare at most one group without installing weights into the model."""

    def __init__(self, weights: StreamingWeights, max_bytes: int | None):
        self.weights = weights
        self.max_bytes = max_bytes
        self.submitted_groups = 0
        self.hits = 0
        self.skipped_groups = 0
        self.discarded_groups = 0
        self.peak_estimated_bytes = 0
        self.prepare_seconds = 0.0
        self.wait_seconds = 0.0
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="inference-weight-prefetch"
        )
        self._pending: Future[_PreparedGroup] | None = None

    def _prepare(self, names, reservation):
        prepared = _PreparedGroup({}, reservation)
        started = perf_counter()
        try:
            with torch.inference_mode():
                for name in names:
                    value = self.weights._materialize(name)
                    prepared.values[name] = value
                    reservation.commit(value.numel() * value.element_size())
                    del value
            reservation.release_reserved(reservation.reserved_bytes)
            return prepared
        except BaseException:
            prepared.close()
            raise
        finally:
            self.prepare_seconds += perf_counter() - started

    def submit(self, names: tuple[str, ...]):
        self.clear()
        if not names:
            return
        sizes = [self.weights._weights[name] for name in names]
        required = sum(weight.nbytes for weight in sizes) + max(
            weight.preparation_bytes for weight in sizes
        )
        if self.max_bytes is not None and required > self.max_bytes:
            self.skipped_groups += 1
            return
        try:
            reservation = self.weights.host_memory_budget.reserve(
                "inference next-module preparation", required
            )
        except HostMemoryError:
            self.skipped_groups += 1
            return
        try:
            self._pending = self._executor.submit(self._prepare, names, reservation)
        except BaseException:
            reservation.close()
            raise
        self.submitted_groups += 1
        self.peak_estimated_bytes = max(self.peak_estimated_bytes, required)

    def take(self, name: str) -> torch.Tensor | None:
        if self._pending is None:
            return None
        started = perf_counter()
        try:
            prepared = self._pending.result()
        except BaseException:
            self._pending = None
            raise
        finally:
            self.wait_seconds += perf_counter() - started
        if name not in prepared.values:
            self.clear()
            return None
        value = prepared.values.pop(name)
        prepared.reservation.release_committed(value.numel() * value.element_size())
        if not prepared.values:
            prepared.close()
            self._pending = None
        self.hits += 1
        return value

    def clear(self):
        pending, self._pending = self._pending, None
        if pending is not None:
            try:
                prepared = pending.result()
            except BaseException:
                return
            prepared.close()
            self.discarded_groups += 1

    def close(self):
        self.clear()
        self._executor.shutdown(wait=True)

    def report(self):
        return {
            "submitted_groups": self.submitted_groups,
            "hits": self.hits,
            "skipped_groups": self.skipped_groups,
            "discarded_groups": self.discarded_groups,
            "peak_estimated_bytes": self.peak_estimated_bytes,
            "prepare_seconds": self.prepare_seconds,
            "wait_seconds": self.wait_seconds,
            "pending_groups": int(self._pending is not None),
        }
