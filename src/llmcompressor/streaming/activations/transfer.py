"""Packed host/device transfers for nested streaming tensor values."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, fields, is_dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

import torch

from .store import _tensors

if TYPE_CHECKING:
    from ..admission import DeviceMemoryBudget
    from ..loading.host_memory import HostMemoryBudget

_TRANSFER_ALIGNMENT = 256


def _align(offset: int) -> int:
    remainder = offset % _TRANSFER_ALIGNMENT
    return offset if remainder == 0 else offset + _TRANSFER_ALIGNMENT - remainder


@dataclass(frozen=True)
class _TensorSlot:
    offset: int
    nbytes: int
    dtype: torch.dtype
    shape: tuple[int, ...]


@dataclass
class _TransferPlan:
    slots: dict[int, _TensorSlot]
    total_bytes: int = 0


@dataclass
class TreeTransfer:
    """A potentially asynchronous packed transfer and its reconstructed value."""

    value: Any
    cpu_buffer: torch.Tensor | None
    device_buffer: torch.Tensor | None
    event: torch.cuda.Event | None
    tensor_count: int
    total_bytes: int
    allocate_seconds: float = 0.0
    pack_seconds: float = 0.0
    submit_seconds: float = 0.0
    _reservations: ExitStack | None = None

    def wait(self) -> Any:
        """Order the consumer stream after the packed copy and return the value."""

        if self.event is not None:
            stream = torch.cuda.current_stream(self.device_buffer.device)
            stream.wait_event(self.event)
            self.device_buffer.record_stream(stream)
        return self.value

    def close(self) -> None:
        """Release pinned and device staging after the consumer has consumed it."""

        try:
            if self.event is not None:
                self.event.synchronize()
        finally:
            self.value = None
            self.cpu_buffer = None
            self.device_buffer = None
            self.event = None
            if self._reservations is not None:
                self._reservations.close()
                self._reservations = None

    def __enter__(self) -> TreeTransfer:
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


@dataclass
class TreeTransferToCPU:
    """A coalesced device-to-host transfer for one nested tensor value."""

    value: Any
    source_value: Any
    cpu_buffer: torch.Tensor | None
    event: torch.cuda.Event | None
    tensor_count: int
    total_bytes: int
    _reservations: ExitStack | None = None

    def wait(self) -> Any:
        if self.event is not None:
            self.event.synchronize()
            self.event = None
        self.source_value = None
        return self.value

    def close(self) -> None:
        try:
            self.wait()
        finally:
            self.value = None
            self.source_value = None
            self.cpu_buffer = None
            self.event = None
            if self._reservations is not None:
                self._reservations.close()
                self._reservations = None


@dataclass
class PreparedTreeTransfer:
    """A CPU-only packed input that can be prepared by a data worker."""

    value: Any
    cpu_buffer: torch.Tensor
    plan: _TransferPlan
    allocate_seconds: float
    pack_seconds: float

    def transfer(
        self,
        device: torch.device,
        *,
        stream: torch.cuda.Stream | None,
        device_budget: DeviceMemoryBudget,
    ) -> TreeTransfer:
        reservations = ExitStack()
        try:
            reservations.enter_context(
                device_budget.reserve(
                    "initial boundary packed input",
                    self.plan.total_bytes,
                    device=device,
                )
            )
            started = perf_counter()
            device_buffer = torch.empty(
                self.plan.total_bytes, dtype=torch.uint8, device=device
            )
            reconstructed = _unpack(self.value, device_buffer, self.plan, {})
            event = None
            if stream is not None and self.cpu_buffer.is_pinned():
                with torch.cuda.stream(stream):
                    event = torch.cuda.Event()
                    try:
                        device_buffer.copy_(self.cpu_buffer, non_blocking=True)
                        event.record(stream)
                    except BaseException:
                        stream.synchronize()
                        raise
            else:
                device_buffer.copy_(self.cpu_buffer)
            return TreeTransfer(
                reconstructed,
                self.cpu_buffer,
                device_buffer,
                event,
                len(self.plan.slots),
                self.plan.total_bytes,
                self.allocate_seconds,
                self.pack_seconds,
                perf_counter() - started,
                reservations.pop_all(),
            )
        finally:
            reservations.close()


def packed_transfer_bytes(value: Any) -> int:
    """Estimate aligned CPU staging without reading tensor contents."""
    plan = _TransferPlan({})
    _collect(value, plan, expected_device=None)
    return plan.total_bytes


def prepare_tree_transfer(value: Any) -> PreparedTreeTransfer:
    """Pack CPU tensors into independently owned pinned staging."""
    plan = _TransferPlan({})
    _collect(value, plan, expected_device=torch.device("cpu"))
    started = perf_counter()
    try:
        cpu_buffer = torch.empty(
            plan.total_bytes,
            dtype=torch.uint8,
            pin_memory=torch.cuda.is_available(),
        )
    except RuntimeError:
        cpu_buffer = torch.empty(plan.total_bytes, dtype=torch.uint8)
    allocated = perf_counter()
    _pack(value, cpu_buffer, plan, set())
    return PreparedTreeTransfer(
        value, cpu_buffer, plan, allocated - started, perf_counter() - allocated
    )


def _collect(
    value: Any,
    plan: _TransferPlan,
    *,
    expected_device: torch.device | None,
) -> None:
    if isinstance(value, torch.Tensor):
        identity = id(value)
        if identity in plan.slots:
            return
        if expected_device is not None and value.device != expected_device:
            raise TypeError("coalesced transfers require tensors on one common device")
        nbytes = value.numel() * value.element_size()
        offset = _align(plan.total_bytes)
        plan.slots[identity] = _TensorSlot(
            offset=offset,
            nbytes=nbytes,
            dtype=value.dtype,
            shape=tuple(value.shape),
        )
        plan.total_bytes = offset + nbytes
        return
    if value is None or isinstance(
        value, (bool, int, float, str, bytes, torch.dtype, torch.device, torch.layout)
    ):
        return
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            _collect(
                getattr(value, item.name),
                plan,
                expected_device=expected_device,
            )
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect(item, plan, expected_device=expected_device)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect(item, plan, expected_device=expected_device)
        return
    raise TypeError(
        f"Unsupported coalesced transfer value type {type(value).__qualname__}"
    )


def _pack(
    value: Any, buffer: torch.Tensor, plan: _TransferPlan, seen: set[int]
) -> None:
    if isinstance(value, torch.Tensor):
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        slot = plan.slots[identity]
        if slot.nbytes == 0:
            return
        target = (
            buffer.narrow(0, slot.offset, slot.nbytes).view(slot.dtype).view(slot.shape)
        )
        target.copy_(value.detach())
        return
    if value is None or isinstance(
        value, (bool, int, float, str, bytes, torch.dtype, torch.device, torch.layout)
    ):
        return
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            _pack(getattr(value, item.name), buffer, plan, seen)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _pack(item, buffer, plan, seen)
        return
    if isinstance(value, dict):
        for item in value.values():
            _pack(item, buffer, plan, seen)
        return
    raise TypeError(
        f"Unsupported coalesced transfer value type {type(value).__qualname__}"
    )


def _pack_async(
    tensors: list[torch.Tensor],
    buffer: torch.Tensor,
    plan: _TransferPlan,
    stream: torch.cuda.Stream | None,
) -> None:
    seen = set()
    for tensor in tensors:
        identity = id(tensor)
        if identity in seen:
            continue
        seen.add(identity)
        slot = plan.slots[identity]
        if slot.nbytes == 0:
            continue
        target = (
            buffer.narrow(0, slot.offset, slot.nbytes).view(slot.dtype).view(slot.shape)
        )
        target.copy_(tensor.detach(), non_blocking=stream is not None)
        if stream is not None:
            tensor.record_stream(stream)


def _unpack(
    value: Any,
    device_buffer: torch.Tensor,
    plan: _TransferPlan,
    memo: dict[int, torch.Tensor],
) -> Any:
    if isinstance(value, torch.Tensor):
        identity = id(value)
        if identity not in plan.slots:
            return value
        if identity not in memo:
            slot = plan.slots[identity]
            if slot.nbytes == 0:
                memo[identity] = torch.empty(
                    slot.shape,
                    device=device_buffer.device,
                    dtype=slot.dtype,
                )
            else:
                memo[identity] = (
                    device_buffer.narrow(0, slot.offset, slot.nbytes)
                    .view(slot.dtype)
                    .view(slot.shape)
                )
        return memo[identity]
    if value is None or isinstance(
        value, (bool, int, float, str, bytes, torch.dtype, torch.device, torch.layout)
    ):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{
                item.name: _unpack(getattr(value, item.name), device_buffer, plan, memo)
                for item in fields(value)
            }
        )
    if isinstance(value, list):
        return [_unpack(item, device_buffer, plan, memo) for item in value]
    if isinstance(value, tuple):
        return tuple(_unpack(item, device_buffer, plan, memo) for item in value)
    if isinstance(value, dict):
        return {
            key: _unpack(item, device_buffer, plan, memo) for key, item in value.items()
        }
    raise TypeError(
        f"Unsupported coalesced transfer value type {type(value).__qualname__}"
    )


def transfer_tree(
    value: Any,
    device: torch.device | str,
    *,
    stream: torch.cuda.Stream | None = None,
    host_budget: HostMemoryBudget | None = None,
    device_budget: DeviceMemoryBudget | None = None,
    owner: str = "packed H2D",
) -> TreeTransfer:
    """Pack one CPU tensor tree into a pinned buffer and transfer it in one copy."""

    resolved = torch.device(device)
    plan = _TransferPlan({})
    _collect(value, plan, expected_device=torch.device("cpu"))
    if not plan.slots:
        return TreeTransfer(value, None, None, None, 0, 0)

    with ExitStack() as reservations:
        if device_budget is not None:
            reservations.enter_context(
                device_budget.reserve(owner, plan.total_bytes, device=resolved)
            )
        host_reservation = (
            reservations.enter_context(host_budget.reserve(owner, plan.total_bytes))
            if host_budget is not None
            else None
        )
        allocate_started = perf_counter()
        try:
            cpu_buffer = torch.empty(
                plan.total_bytes,
                dtype=torch.uint8,
                pin_memory=resolved.type != "cpu",
            )
        except RuntimeError:
            cpu_buffer = torch.empty(plan.total_bytes, dtype=torch.uint8)
        if host_reservation is not None:
            host_reservation.commit(plan.total_bytes)
        allocate_elapsed = perf_counter() - allocate_started
        pack_started = perf_counter()
        _pack(value, cpu_buffer, plan, set())
        pack_elapsed = perf_counter() - pack_started

        submit_started = perf_counter()
        event = None
        if resolved.type == "cpu":
            device_buffer = cpu_buffer
            reconstructed = _unpack(value, device_buffer, plan, {})
        elif stream is not None:
            with torch.cuda.stream(stream):
                device_buffer = torch.empty(
                    plan.total_bytes, dtype=torch.uint8, device=resolved
                )
                reconstructed = _unpack(value, device_buffer, plan, {})
                event = torch.cuda.Event()
                try:
                    device_buffer.copy_(cpu_buffer, non_blocking=True)
                    event.record(stream)
                except BaseException:
                    stream.synchronize()
                    raise
        else:
            device_buffer = torch.empty(
                plan.total_bytes, dtype=torch.uint8, device=resolved
            )
            reconstructed = _unpack(value, device_buffer, plan, {})
            device_buffer.copy_(cpu_buffer, non_blocking=False)
        submit_elapsed = perf_counter() - submit_started

        return TreeTransfer(
            reconstructed,
            cpu_buffer,
            device_buffer,
            event,
            len(plan.slots),
            plan.total_bytes,
            allocate_elapsed,
            pack_elapsed,
            submit_elapsed,
            reservations.pop_all(),
        )


def transfer_tree_to_cpu(
    value: Any,
    *,
    stream: torch.cuda.Stream | None = None,
    wait_stream: torch.cuda.Stream | None = None,
    host_budget: HostMemoryBudget | None = None,
    owner: str = "packed D2H",
) -> TreeTransferToCPU:
    """Copy tensors from one device into one pinned buffer, retaining CPU leaves.

    ``wait_stream`` orders the copy after work already queued on that stream.
    Producers that run on the current stream must pass it when ``stream`` is a
    separate copy stream; CUDA streams do not order each other implicitly.
    """

    tensors = list(_tensors(value))
    device_tensors = [tensor for tensor in tensors if tensor.device.type != "cpu"]
    if not device_tensors:
        return TreeTransferToCPU(value, value, None, None, len(tensors), 0)

    plan = _TransferPlan({})
    _collect(device_tensors, plan, expected_device=device_tensors[0].device)

    with ExitStack() as reservations:
        host_reservation = (
            reservations.enter_context(host_budget.reserve(owner, plan.total_bytes))
            if host_budget is not None
            else None
        )
        cpu_buffer = torch.empty(plan.total_bytes, dtype=torch.uint8, pin_memory=True)
        if host_reservation is not None:
            host_reservation.commit(plan.total_bytes)
        reconstructed = _unpack(value, cpu_buffer, plan, {})
        if stream is None:
            _pack_async(device_tensors, cpu_buffer, plan, None)
            event = None
        else:
            with torch.cuda.stream(stream):
                if wait_stream is not None and wait_stream != stream:
                    stream.wait_stream(wait_stream)
                event = torch.cuda.Event()
                try:
                    _pack_async(device_tensors, cpu_buffer, plan, stream)
                    event.record(stream)
                except BaseException:
                    stream.synchronize()
                    raise
        return TreeTransferToCPU(
            reconstructed,
            value,
            cpu_buffer,
            event,
            len(plan.slots),
            plan.total_bytes,
            reservations.pop_all(),
        )
