"""Resource admission for streaming targets and modifier workspaces."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from shutil import disk_usage
from threading import RLock
from typing import Any, Callable

import torch
from compressed_tensors.quantization import QuantizationStrategy

from .modifier_contract import ModifierContractError, build_modifier_contracts

__all__ = [
    "DeviceMemoryBudget",
    "DeviceMemoryError",
    "DiskSpaceBudget",
    "DiskSpaceError",
    "ModifierMemoryEstimate",
    "estimate_modifier_memory",
]


class DeviceMemoryError(MemoryError):
    """Raised before a target exceeds the available execution-device budget."""


class DiskSpaceError(OSError):
    """A target cannot fit its temporary output on the work filesystem."""


class DiskSpaceBudget:
    """Check temporary disk capacity independently of host and device capacity."""

    def __init__(
        self,
        root: str | Path,
        *,
        available_bytes: Callable[[Path], int] | None = None,
        safety_reserve_fraction: float = 0.05,
    ):
        if not 0 <= safety_reserve_fraction < 1:
            raise ValueError("safety_reserve_fraction must be in [0, 1)")
        self.root = Path(root)
        self._available_bytes = available_bytes or (lambda path: disk_usage(path).free)
        self.safety_reserve_fraction = safety_reserve_fraction

    def check(self, owner: str, required_bytes: int) -> None:
        if required_bytes < 0:
            raise ValueError("required_bytes must be non-negative")
        root = self.root
        while not root.exists() and root != root.parent:
            root = root.parent
        available = self._available_bytes(root)
        if available < 0:
            raise ValueError("Available disk bytes must be non-negative")
        usable = int(available * (1 - self.safety_reserve_fraction))
        if required_bytes > usable:
            raise DiskSpaceError(
                f"Insufficient temporary disk space for {owner}: requested "
                f"{required_bytes} bytes, available {usable} bytes on {root}"
            )


@dataclass(frozen=True)
class ModifierMemoryEstimate:
    """Aggregated memory required by modifier statistics and scratch."""

    held_bytes: int
    scratch_bytes: int
    details: tuple[dict[str, Any], ...] = ()
    host_bytes: int = 0

    @property
    def peak_bytes(self) -> int:
        return self.held_bytes + self.scratch_bytes


def _estimate_observer_memory(
    modules: Iterable[torch.nn.Module], num_tokens: int
) -> tuple[int, int]:
    held = scratch = 0
    supported = {
        "minmax", "memoryless_minmax", "static_minmax", "strict_symmetric_minmax",
        "mse", "memoryless_mse", "imatrix_mse",
    }
    for module in modules:
        scheme = getattr(module, "quantization_scheme", None)
        weight = getattr(module, "weight", None)
        if scheme is None or scheme.weights is None or weight is None:
            continue
        args = scheme.weights
        observer = args.observer
        if observer not in supported:
            raise ModifierContractError(
                f"Weight observer {observer!r} has no streaming memory adapter"
            )
        if args.strategy == QuantizationStrategy.TENSOR:
            groups = 1
        elif args.strategy == QuantizationStrategy.CHANNEL:
            groups = weight.shape[0]
        elif args.group_size is not None and args.group_size > 0:
            groups = weight.shape[0] * (
                (weight.numel() // weight.shape[0] + args.group_size - 1)
                // args.group_size
            )
        else:
            groups = weight.numel()
        held += 8 * groups * torch.float32.itemsize
        module_scratch = 8 * weight.numel() * torch.float32.itemsize
        if observer == "imatrix_mse":
            in_features = weight.shape[-1]
            held += in_features * torch.float32.itemsize + torch.int64.itemsize
            module_scratch += 2 * num_tokens * in_features * torch.float32.itemsize
        scratch = max(scratch, module_scratch)
    return held, scratch


def estimate_modifier_memory(
    modifiers: Iterable[Any],
    modules: Iterable[torch.nn.Module],
    *,
    num_tokens: int,
    max_batch_tokens: int | None = None,
) -> ModifierMemoryEstimate:
    """Evaluate modifier estimates against the actual target modules.

    The modifier owns the estimate implementation. No policy is inferred from
    modifier class names. A calibration modifier that cannot provide an estimate
    is rejected before its target is loaded.
    """
    if num_tokens < 0 or (max_batch_tokens is not None and max_batch_tokens < 0):
        raise ValueError("token counts must be non-negative")
    modules = tuple(dict.fromkeys(modules))
    modifiers = tuple(modifiers)
    contracts = build_modifier_contracts(modifiers)
    held = scratch = host = 0
    details = []
    for modifier, contract in zip(modifiers, contracts):
        estimate_fn = getattr(modifier, "estimate_sequential_memory", None)
        modifier_held = modifier_scratch = 0
        modifier_host = 0
        host_estimate_fn = getattr(modifier, "estimate_sequential_host_memory", None)
        if callable(host_estimate_fn):
            for module in modules:
                value = host_estimate_fn(module, num_tokens)
                if not isinstance(value, Integral) or value < 0:
                    raise ModifierContractError(
                        f"{contract.modifier_type} must provide non-negative "
                        "host memory bytes"
                    )
                modifier_host += int(value)
        unknown = False
        if contract.memory_estimate_is_static:
            modifier_held, modifier_scratch = contract.memory_estimate
        elif estimate_fn is None:
            unknown = contract.requires_calibration_data
        else:
            for module in modules:
                try:
                    estimate = estimate_fn(module, num_tokens)
                except NotImplementedError:
                    estimate = None
                if estimate is None:
                    unknown = True
                    continue
                if (
                    not isinstance(estimate, tuple)
                    or len(estimate) != 2
                    or any(
                        not isinstance(value, Integral) or value < 0
                        for value in estimate
                    )
                ):
                    raise ModifierContractError(
                        f"{type(modifier).__qualname__}.estimate_sequential_memory "
                        "must return non-negative (held, scratch) or None"
                    )
                modifier_held += int(estimate[0])
                modifier_scratch = max(modifier_scratch, int(estimate[1]))
        if unknown:
            raise ModifierContractError(
                f"{contract.modifier_type} does not provide a bounded memory "
                "estimate for the current streaming target"
            )
        held += modifier_held
        scratch += modifier_scratch
        host += modifier_host
        details.append(
            {
                "modifier_type": contract.modifier_type,
                "held_bytes": modifier_held,
                "scratch_bytes": modifier_scratch,
                "host_bytes": modifier_host,
            }
        )
    observer_held, observer_scratch = _estimate_observer_memory(
        modules, num_tokens if max_batch_tokens is None else max_batch_tokens
    )
    if observer_held or observer_scratch:
        held += observer_held
        scratch += observer_scratch
        details.append(
            {
                "modifier_type": "quantization_observers",
                "held_bytes": observer_held,
                "scratch_bytes": observer_scratch,
            }
        )
    return ModifierMemoryEstimate(held, scratch, tuple(details), host)


class DeviceMemoryReservation:
    def __init__(
        self,
        budget: DeviceMemoryBudget,
        owner: str,
        reserved_bytes: int,
        device: torch.device,
    ):
        self._budget = budget
        self.owner = owner
        self.device = device
        self._reserved_bytes = reserved_bytes
        self._closed = False

    @property
    def reserved_bytes(self) -> int:
        with self._budget._lock:
            return self._reserved_bytes

    def close(self) -> None:
        with self._budget._lock:
            if self._closed:
                return
            self._reserved_bytes = 0
            self._closed = True
            self._budget._reservations.discard(self)

    def __enter__(self) -> DeviceMemoryReservation:
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        self.close()


class DeviceMemoryBudget:
    """Reserve execution-device capacity independently from host capacity."""

    def __init__(
        self,
        *,
        available_bytes: (
            int | Callable[[torch.device], int | None] | None
        ) = None,
        safety_reserve_fraction: float = 0.1,
    ):
        if not 0 <= safety_reserve_fraction < 1:
            raise ValueError("safety_reserve_fraction must be in [0, 1)")
        if available_bytes is None:
            self._available_bytes = self._query_available
        elif callable(available_bytes):
            self._available_bytes = available_bytes
        else:
            if available_bytes < 0:
                raise ValueError("available_bytes must be non-negative")
            self._available_bytes = lambda _device: available_bytes
        self.safety_reserve_fraction = safety_reserve_fraction
        self._reservations: set[DeviceMemoryReservation] = set()
        self._lock = RLock()

    @staticmethod
    def _query_available(device: torch.device) -> int | None:
        if device.type == "cuda" and torch.cuda.is_available():
            return int(torch.cuda.mem_get_info(device)[0])
        device_module = getattr(torch, device.type, None)
        mem_get_info = getattr(device_module, "mem_get_info", None)
        if callable(mem_get_info):
            return int(mem_get_info(device)[0])
        return None

    def reserve(
        self,
        owner: str,
        required_bytes: int,
        *,
        device: torch.device | str,
    ) -> DeviceMemoryReservation:
        if not owner:
            raise ValueError("A device-memory reservation requires an owner")
        if required_bytes < 0:
            raise ValueError("required_bytes must be non-negative")
        resolved = torch.device(device)
        if resolved.index is None and resolved.type == "cuda":
            resolved = torch.device("cuda", torch.cuda.current_device())
        with self._lock:
            available = self._available_bytes(resolved)
            if available is None and resolved.type != "cpu":
                raise DeviceMemoryError(
                    f"Cannot determine available device memory for {owner} "
                    f"on {resolved}; provide an explicit available_bytes budget"
                )
            if available is not None:
                if available < 0:
                    raise ValueError("Available device memory must be non-negative")
                promised = sum(
                    item._reserved_bytes
                    for item in self._reservations
                    if item.device == resolved
                )
                usable = max(
                    0,
                    available
                    - int(available * self.safety_reserve_fraction)
                    - promised,
                )
                if required_bytes > usable:
                    raise DeviceMemoryError(
                        f"Insufficient device memory for {owner}: requested "
                        f"{required_bytes} bytes, available {usable} bytes "
                        f"after safety reserve on {resolved}"
                    )
            reservation = DeviceMemoryReservation(self, owner, required_bytes, resolved)
            self._reservations.add(reservation)
            return reservation
