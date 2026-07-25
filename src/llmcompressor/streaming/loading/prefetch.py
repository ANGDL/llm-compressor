"""Bounded background materialization for the next streaming subgraph."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext

import torch
from loguru import logger

from llmcompressor.utils.metric_logging import CompressionLogger

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

    @staticmethod
    def _device_ids(device: torch.device) -> tuple[int, ...]:
        if device.type != "cuda":
            return ()
        if device.index is None:
            raise ValueError("CUDA prefetch devices must include an explicit index")
        return (device.index,)

    def _prepare(
        self,
        label: str,
        plan: SubgraphLoadPlan,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PreparedSubgraphWeights:
        device_context = (
            torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        )
        with device_context:
            metrics = CompressionLogger(
                self.session.model, device_ids=self._device_ids(device)
            )
            metrics.set_results(name=label)
            with metrics:
                return self.session.prepare(plan, device=device, dtype=dtype)

    def submit(
        self,
        plan: SubgraphLoadPlan,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        label: str,
    ) -> None:
        """Schedule one plan after the previous result has been consumed."""

        if self._closed:
            raise RuntimeError("Subgraph prefetcher is closed")
        if self._pending is not None:
            raise RuntimeError("A subgraph prefetch is already pending")
        resolved = torch.device(device)
        self._pending = (
            label,
            self._executor.submit(
                self._prepare, label, plan, resolved, dtype
            ),
        )
        logger.info(
            f"streaming pipeline: queued weight prefetch {label!r} "
            f"on {resolved}"
        )

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
