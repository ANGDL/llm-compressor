"""Ordered asynchronous CPU commits for initial calibration boundaries."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import fields, is_dataclass
from threading import Event
from time import perf_counter
from typing import Any

import torch

from ._logging import streaming_logger
from .activations.store import BoundaryActivationStore, _tensor_bytes, _tensors
from .activations.transfer import transfer_tree_to_cpu
from .loading.host_memory import HostMemoryBudget, HostMemoryError


@contextmanager
def initial_boundary_cleanup(store: BoundaryActivationStore):
    """Release partial initial snapshots when collection fails."""
    try:
        yield
    except BaseException:
        try:
            store.delete(0)
        except Exception as error:
            streaming_logger.warning(f"initial boundary cleanup failed: {error}")
        raise


def _snapshot_cpu_leaves(value, memo=None):
    memo = memo if memo is not None else {}
    if isinstance(value, torch.Tensor):
        identity = id(value)
        if identity not in memo:
            memo[identity] = (
                value.detach().clone() if value.device.type == "cpu" else value
            )
        return memo[identity]
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{
                item.name: _snapshot_cpu_leaves(getattr(value, item.name), memo)
                for item in fields(value)
            }
        )
    if isinstance(value, dict):
        return {name: _snapshot_cpu_leaves(item, memo) for name, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_cpu_leaves(item, memo) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_cpu_leaves(item, memo) for item in value)
    return value


class InitialBoundaryWriter(AbstractContextManager):
    """Keep at most two output batches in flight, committing in source order."""

    def __init__(
        self,
        store: BoundaryActivationStore,
        *,
        host_budget: HostMemoryBudget,
        device: torch.device,
        enabled: bool,
        release_input_prefetch: Callable[[], None] | None = None,
    ):
        self.store = store
        self.host_budget = host_budget
        self.device = device
        self.enabled = enabled
        self.release_input_prefetch = release_input_prefetch
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="streaming-boundary")
            if enabled
            else None
        )
        self._stream = (
            torch.cuda.Stream(device=device)
            if enabled and device.type == "cuda"
            else None
        )
        self._pending: deque[Future] = deque()
        self._closed = False
        self._wait_seconds = 0.0
        self._fallbacks = 0
        self._failed = Event()

    def wait_for_slot(self) -> None:
        """Apply backpressure before the model creates another output batch."""
        if self._closed:
            raise RuntimeError("Initial boundary writer is closed")
        while self._pending and (len(self._pending) >= 2 or self._pending[0].done()):
            self._join_one()

    def _join_one(self) -> None:
        started = perf_counter()
        self._pending.popleft().result()
        self._wait_seconds += perf_counter() - started

    def _write(self, index, transfer, reservation, cpu_bytes, resources):
        value = None
        try:
            if self._failed.is_set():
                raise RuntimeError("A previous initial boundary commit failed")
            started = perf_counter()
            value = transfer.wait()
            ready = perf_counter()
            self.store.put(0, index, value, reservation=reservation)
            if not reservation.closed:
                reservation.release_committed(cpu_bytes)
            streaming_logger.info(
                f"initial boundary | output batch={index} | "
                f"copy_wait_s={ready - started:.2f} | "
                f"commit_s={perf_counter() - ready:.2f}"
            )
        except BaseException:
            self._failed.set()
            reservation.close()
            raise
        finally:
            value = None
            resources.close()

    def submit(self, index: int, value: Any) -> None:
        """Snapshot CPU leaves before the model can reuse them in a later batch."""
        if self._closed:
            raise RuntimeError("Initial boundary writer is closed")
        if self._executor is None:
            self.store.put(0, index, value)
            return
        self.wait_for_slot()
        cpu_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in {id(item): item for item in _tensors(value)}.values()
            if tensor.device.type == "cpu"
        )
        reservation = None
        snapshot = None
        with ExitStack() as resources:
            try:
                reservation = self.host_budget.reserve(
                    f"initial boundary output batch {index}",
                    _tensor_bytes(value) + cpu_bytes,
                )
                snapshot = _snapshot_cpu_leaves(value)
                reservation.commit(cpu_bytes)
                transfer = transfer_tree_to_cpu(
                    snapshot,
                    stream=self._stream,
                    wait_stream=(
                        torch.cuda.current_stream(self.device)
                        if self._stream is not None
                        else None
                    ),
                    host_budget=self.host_budget,
                    owner=f"initial boundary D2H batch {index}",
                )
                resources.callback(transfer.close)
            except HostMemoryError:
                snapshot = None
                if reservation is not None:
                    reservation.close()
                while self._pending:
                    self._join_one()
                if self.release_input_prefetch is not None:
                    self.release_input_prefetch()
                self._fallbacks += 1
                self.store.put(0, index, value)
                return
            except BaseException:
                if reservation is not None:
                    reservation.close()
                raise
            if transfer.event is not None:
                torch.cuda.current_stream(self.device).wait_event(transfer.event)
            ownership = resources.pop_all()
            try:
                self._pending.append(
                    self._executor.submit(
                        self._write, index, transfer, reservation, cpu_bytes, ownership
                    )
                )
            except BaseException:
                ownership.close()
                reservation.close()
                raise

    def close(self, *, suppress_errors: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        error = None
        while self._pending:
            try:
                self._join_one()
            except BaseException as caught:
                if error is None:
                    error = caught
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        streaming_logger.info(
            "initial boundary | output worker | "
            f"wait_s={self._wait_seconds:.2f} | serial_fallbacks={self._fallbacks}"
        )
        if error is not None and not suppress_errors:
            raise error

    def __exit__(self, exc_type, _exc_value, _traceback):
        self.close(suppress_errors=exc_type is not None)
        return False
