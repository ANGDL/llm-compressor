"""Bounded background materialization for the next streaming subgraph."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager
from threading import Lock
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
        self._transfer_streams: dict[torch.device, torch.cuda.Stream] = {}
        self._transfer_stream_lock = Lock()
        self._closed = False

    def _transfer_stream_for(self, device: torch.device | str) -> torch.cuda.Stream:
        """Return the persistent copy stream for one execution device.

        The stream is created on the caller thread (the pipeline thread) and
        reused for every target on that device. Creating a fresh stream inside
        the prefetch worker for every target makes stream lifetime depend on
        worker scheduling and defeats the single dedicated-copy-stream design.
        """

        resolved = torch.device(device)
        if resolved.type == "cuda" and resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        with self._transfer_stream_lock:
            stream = self._transfer_streams.get(resolved)
            if stream is None:
                stream = torch.cuda.Stream(device=resolved)
                self._transfer_streams[resolved] = stream
            return stream

    def _prepare(
        self,
        label: str,
        plan: SubgraphLoadPlan,
        dtype: torch.dtype,
        reservation: HostMemoryReservation,
        device: torch.device | str | None,
        device_reservation,
        transfer_reservation: HostMemoryReservation | None,
        transfer_stream: torch.cuda.Stream | None,
    ) -> PreparedSubgraphWeights:
        started_at = perf_counter()
        streaming_logger.info(f"{label} | started | device=cpu")
        try:
            prepared = self.session.prepare(
                plan, dtype=dtype, reservation=reservation
            )
            prepared_at = perf_counter()
            if device is not None and transfer_reservation is not None:
                prepared.move_to_async(
                    device,
                    device_reservation=device_reservation,
                    transfer_reservation=transfer_reservation,
                    transfer_stream=transfer_stream,
                )
                device_reservation = None
                transfer_reservation = None
            elif device_reservation is not None:
                prepared.device_reservation = device_reservation
                device_reservation = None
                if transfer_reservation is not None:
                    transfer_reservation.close()
                    transfer_reservation = None
        except BaseException:
            if device_reservation is not None:
                device_reservation.close()
            if transfer_reservation is not None:
                transfer_reservation.close()
            raise
        transfer_seconds = perf_counter() - prepared_at
        stream_id = (
            getattr(transfer_stream, "stream_id", None)
            if transfer_stream is not None
            else None
        )
        streaming_logger.info(
            f"{label} | complete | device={prepared.device.type} | "
            f"prepare={prepared_at - started_at:.2f}s | "
            f"transfer_enqueue={transfer_seconds:.2f}s | "
            f"copy_stream={stream_id}"
        )
        return prepared

    def submit(
        self,
        plan: SubgraphLoadPlan,
        *,
        dtype: torch.dtype,
        reservation: HostMemoryReservation,
        label: str,
        device: torch.device | str | None = None,
        device_reservation=None,
        transfer_reservation: HostMemoryReservation | None = None,
    ) -> None:
        """Schedule one plan after the previous result has been consumed."""

        if self._closed:
            reservation.close()
            if device_reservation is not None:
                device_reservation.close()
            if transfer_reservation is not None:
                transfer_reservation.close()
            raise RuntimeError("Subgraph prefetcher is closed")
        if self._pending is not None:
            reservation.close()
            if device_reservation is not None:
                device_reservation.close()
            if transfer_reservation is not None:
                transfer_reservation.close()
            raise RuntimeError("A subgraph prefetch is already pending")
        transfer_stream = None
        if device is not None and transfer_reservation is not None:
            resolved = torch.device(device)
            if resolved.type == "cuda":
                transfer_stream = self._transfer_stream_for(resolved)
        try:
            future = self._executor.submit(
                self._prepare,
                label,
                plan,
                dtype,
                reservation,
                device,
                device_reservation,
                transfer_reservation,
                transfer_stream,
            )
        except Exception:
            reservation.close()
            if device_reservation is not None:
                device_reservation.close()
            if transfer_reservation is not None:
                transfer_reservation.close()
            raise
        self._pending = (label, future)
        streaming_logger.info(f"{label} | queued | device=cpu")

    def take(self) -> PreparedSubgraphWeights:
        """Wait for and transfer ownership of the pending prepared weights."""

        if self._pending is None:
            raise RuntimeError("No subgraph prefetch is pending")
        label, future = self._pending
        self._pending = None
        started_at = perf_counter()
        try:
            prepared = future.result()
        except Exception as error:
            raise RuntimeError(
                f"Subgraph weight prefetch failed for {label!r}"
            ) from error
        streaming_logger.info(
            f"{label} | acquired | wait_s={perf_counter() - started_at:.2f}"
        )
        return prepared

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
        for stream in self._transfer_streams.values():
            try:
                stream.synchronize()
            except Exception as caught:
                if error is None:
                    error = caught
        self._transfer_streams.clear()
        if error is not None and not suppress_errors:
            raise error

    def __exit__(self, exc_type, _exc_val, _exc_tb):
        self.close(suppress_errors=exc_type is not None)
        return False
