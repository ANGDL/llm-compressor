"""Bounded background materialization for the next streaming subgraph."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager
from time import perf_counter

import torch

from .._logging import streaming_logger
from .host_memory import HostMemoryReservation
from .session import (
    PreparedSubgraphWeights,
    SubgraphLoadPlan,
    SubgraphWeightSession,
)


class SubgraphPrefetcher(AbstractContextManager):
    """Own at most one background subgraph materialization task."""

    def __init__(self, session: SubgraphWeightSession):
        self.session = session
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="streaming-weight-prefetch"
        )
        self._pending: tuple[
            str, Future[PreparedSubgraphWeights]
        ] | None = None
        self._closed = False

    def _prepare(
        self,
        label: str,
        plan: SubgraphLoadPlan,
        dtype: torch.dtype,
        reservation: HostMemoryReservation,
    ) -> PreparedSubgraphWeights:
        started_at = perf_counter()
        streaming_logger.info(f"{label} | started | device=cpu")
        prepared = self.session.prepare(
            plan, dtype=dtype, reservation=reservation
        )
        streaming_logger.info(
            f"{label} | complete | device=cpu | "
            f"time={perf_counter() - started_at:.2f}s"
        )
        return prepared

    def submit(
        self,
        plan: SubgraphLoadPlan,
        *,
        dtype: torch.dtype,
        reservation: HostMemoryReservation,
        label: str,
    ) -> None:
        """Schedule one plan after the previous result has been consumed."""

        if self._closed:
            reservation.close()
            raise RuntimeError("Subgraph prefetcher is closed")
        if self._pending is not None:
            reservation.close()
            raise RuntimeError("A subgraph prefetch is already pending")
        try:
            future = self._executor.submit(
                self._prepare, label, plan, dtype, reservation
            )
        except Exception:
            reservation.close()
            raise
        self._pending = (label, future)
        streaming_logger.info(f"{label} | queued | device=cpu")

    def take(self) -> PreparedSubgraphWeights:
        """Wait for and transfer ownership of the pending prepared weights."""

        if self._pending is None:
            raise RuntimeError("No subgraph prefetch is pending")
        label, future = self._pending
        self._pending = None
        try:
            return future.result()
        except Exception as error:
            raise RuntimeError(
                f"Subgraph weight prefetch failed for {label!r}"
            ) from error

    def close(self, *, suppress_errors: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        error = None
        if self._pending is not None:
            try:
                self.take().close()
            except Exception as caught:
                error = caught
        self._executor.shutdown(wait=True)
        if error is not None and not suppress_errors:
            raise error

    def __exit__(self, exc_type, _exc_val, _exc_tb):
        self.close(suppress_errors=exc_type is not None)
        return False
