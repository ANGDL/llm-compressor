"""One CPU-only lookahead group for native inference offload hooks."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from queue import Empty
from time import perf_counter

import torch

from llmcompressor.streaming.loading import HostMemoryError, HostMemoryReservation

from .weights import StreamingWeights, prefetch_materializer


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


@dataclass
class _ProcessGroup:
    generation: int
    requested: set[str]
    buffers: dict[str, torch.Tensor]
    reservation: HostMemoryReservation
    ready: set[str]
    layout: tuple
    buffer: torch.Tensor | None = None
    done: bool = False
    error: str | None = None


def _process_worker(weights, commands, results, cancel, threads: int) -> None:
    """Decode groups in a separate interpreter so decoding never holds the GIL."""

    torch.set_num_threads(threads)
    while True:
        command = commands.get()
        if command is None:
            return
        generation, layout, flat = command
        started = perf_counter()
        try:
            for name, offset, nbytes, dtype_name, shape in layout:
                if cancel.value >= generation:
                    break
                value = weights._materialize(name)
                dtype = getattr(torch, dtype_name.rsplit(".", 1)[-1])
                target = flat.narrow(0, offset, nbytes).view(dtype).view(shape)
                target.copy_(value)
                del value, target
            results.put(
                (
                    generation,
                    None,
                    perf_counter() - started,
                    None,
                    None,
                )
            )
        except BaseException as error:  # pragma: no cover - defensive transport
            results.put(
                (
                    generation,
                    None,
                    perf_counter() - started,
                    f"{type(error).__name__}: {error}",
                    None,
                )
            )


class ProcessInferencePrefetcher:
    """Prepare one layer in a forked process with shared-memory output buffers.

    The worker is forked before the execution device is used, inherits the
    checkpoint mappings, and never initializes CUDA. Decoded tensors are written
    directly into preallocated shared buffers owned by the parent process.
    """

    def __init__(
        self,
        weights: StreamingWeights,
        max_bytes: int | None,
        *,
        threads: int | None = None,
    ):
        self.weights = weights
        self.max_bytes = max_bytes
        self.threads = threads or min(16, torch.get_num_threads())
        self._decodes_float = prefetch_materializer(weights.materializer) is not None
        self.submitted_groups = 0
        self.hits = 0
        self.skipped_groups = 0
        self.discarded_groups = 0
        self.peak_estimated_bytes = 0
        self.prepare_seconds = 0.0
        self.wait_seconds = 0.0
        self._generation = 0
        self._pending: _ProcessGroup | None = None
        self._retired: dict[int, _ProcessGroup] = {}
        self._buffers: list[torch.Tensor | None] = [None, None]
        self._context = mp.get_context("spawn")
        self._commands = self._context.Queue()
        self._results = self._context.Queue()
        self._cancel = self._context.Value("i", 0)
        self._process = self._context.Process(
            target=_process_worker,
            args=(
                weights,
                self._commands,
                self._results,
                self._cancel,
                self.threads,
            ),
            daemon=True,
        )
        self._process.start()

    def _drain(self, timeout: float = 0) -> None:
        while True:
            try:
                if timeout:
                    message = self._results.get(timeout=timeout)
                else:
                    message = self._results.get_nowait()
            except Empty:
                return
            generation, name, seconds, error, value = message
            timeout = 0
            if name is None:
                self.prepare_seconds += seconds
                group = self._pending
                if group is not None and group.generation == generation:
                    group.done = True
                    group.error = error
                    if error is not None:
                        self._release(group)
                        self._pending = None
                    elif group.buffer is not None:
                        flat = group.buffer
                        committed = 0
                        for (
                            name,
                            offset,
                            nbytes,
                            dtype_name,
                            shape,
                        ) in group.layout:
                            dtype = getattr(
                                torch, dtype_name.rsplit(".", 1)[-1]
                            )
                            view = (
                                flat.narrow(0, offset, nbytes)
                                .view(dtype)
                                .view(shape)
                            )
                            group.buffers[name] = view
                            group.ready.add(name)
                            committed += nbytes
                        group.reservation.commit(committed)
                retired = self._retired.pop(generation, None)
                if retired is not None:
                    self._release(retired)
                continue
            group = self._pending
            if group is not None and group.generation == generation:
                group.ready.add(name)
                if error is not None:
                    group.error = error
            elif generation in self._retired and error is not None:
                self._retired[generation].error = error

    @staticmethod
    def _release(group: _ProcessGroup) -> None:
        group.buffers.clear()
        group.reservation.close()

    def submit(self, names: tuple[str, ...]) -> None:
        self.clear()
        if not names:
            return
        decode_names = tuple(
            name
            for name in names
            if self._decodes_float
            or self.weights._weights[name].weight_bits is not None
        )
        if not decode_names:
            self.skipped_groups += 1
            return
        layout = []
        offset = 0
        for name in decode_names:
            weight = self.weights._weights[name]
            offset = (offset + 7) // 8 * 8
            layout.append(
                (name, offset, weight.nbytes, str(weight.dtype), weight.shape)
            )
            offset += weight.nbytes
        required = offset + max(
            self.weights._weights[name].preparation_bytes
            for name in decode_names
        )
        if self.max_bytes is not None and required > self.max_bytes:
            self.skipped_groups += 1
            return
        try:
            reservation = self.weights.host_memory_budget.reserve(
                "inference next-module process preparation", required
            )
        except HostMemoryError:
            self.skipped_groups += 1
            return
        self._generation += 1
        index = self._generation % len(self._buffers)
        buffer = self._buffers[index]
        if buffer is not None and any(
            group.buffer is buffer for group in self._retired.values()
        ):
            buffer = None
        if buffer is None or buffer.numel() < offset:
            buffer = torch.empty(offset, dtype=torch.uint8)
            buffer.share_memory_()
            self._buffers[index] = buffer
        try:
            self._commands.put((self._generation, tuple(layout), buffer))
        except BaseException:
            reservation.close()
            raise
        self._pending = _ProcessGroup(
            generation=self._generation,
            requested=set(decode_names),
            buffers={},
            reservation=reservation,
            ready=set(),
            layout=tuple(layout),
            buffer=buffer,
        )
        self.submitted_groups += 1
        self.peak_estimated_bytes = max(self.peak_estimated_bytes, required)

    def take(self, name: str) -> torch.Tensor | None:
        group = self._pending
        if group is None:
            return None
        if name not in group.requested:
            return None
        started = perf_counter()
        try:
            while name not in group.ready and not group.done:
                self._drain(timeout=0.05)
                if group.error is not None:
                    raise RuntimeError(
                        f"Inference process prefetch failed: {group.error}"
                    )
                if not self._process.is_alive():
                    raise RuntimeError("Inference prefetch worker exited")
        finally:
            self.wait_seconds += perf_counter() - started
        if name not in group.ready:
            self.clear()
            return None
        value = group.buffers.pop(name)
        group.requested.discard(name)
        group.ready.discard(name)
        group.reservation.release_committed(
            value.numel() * value.element_size()
        )
        if not group.requested:
            group.reservation.close()
            self._pending = None
        self.hits += 1
        return value

    def clear(self) -> None:
        if self._pending is None:
            self._drain()
            return
        self._cancel.value = max(self._cancel.value, self._generation)
        retired = self._pending
        self._pending = None
        self._retired[retired.generation] = retired
        self.discarded_groups += 1
        self._drain()
        for generation in tuple(self._retired):
            group = self._retired[generation]
            if group.done:
                self._retired.pop(generation)
                self._release(group)

    def close(self) -> None:
        self.clear()
        try:
            self._commands.put(None)
        except (BrokenPipeError, EOFError):
            pass
        self._process.join(timeout=30)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=10)
        for group in self._retired.values():
            self._release(group)
        self._retired.clear()
        self._buffers.clear()
        self._commands.close()
        self._results.close()

    def report(self) -> dict:
        self._drain()
        return {
            "submitted_groups": self.submitted_groups,
            "hits": self.hits,
            "skipped_groups": self.skipped_groups,
            "discarded_groups": self.discarded_groups,
            "peak_estimated_bytes": self.peak_estimated_bytes,
            "prepare_seconds": self.prepare_seconds,
            "wait_seconds": self.wait_seconds,
            "pending_groups": int(self._pending is not None),
            "mode": "process",
            "worker_pid": self._process.pid,
            "worker_threads": self.threads,
        }
