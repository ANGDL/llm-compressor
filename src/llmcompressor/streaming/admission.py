"""Resource admission for streaming targets and modifier workspaces."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable

import torch

from .modifier_contract import ModifierContractError, build_modifier_contracts

__all__ = [
    "DeviceMemoryBudget",
    "DeviceMemoryError",
    "ModifierMemoryEstimate",
    "estimate_modifier_memory",
]


class DeviceMemoryError(MemoryError):
    """Raised before a target exceeds the available execution-device budget."""


@dataclass(frozen=True)
class ModifierMemoryEstimate:
    """Aggregated memory required by modifier statistics and scratch."""

    held_bytes: int
    scratch_bytes: int
    details: tuple[dict[str, Any], ...] = ()

    @property
    def peak_bytes(self) -> int:
        return self.held_bytes + self.scratch_bytes


def estimate_modifier_memory(
    modifiers: Iterable[Any],
    modules: Iterable[torch.nn.Module],
    *,
    num_tokens: int,
) -> ModifierMemoryEstimate:
    """Evaluate modifier estimates against the actual target modules.

    The modifier owns the estimate implementation. No policy is inferred from
    modifier class names. A calibration modifier that cannot provide an estimate
    is rejected before its target is loaded.
    """
    modules = tuple(modules)
    modifiers = tuple(modifiers)
    contracts = build_modifier_contracts(modifiers)
    if len(contracts) != len(modifiers):
        raise ModifierContractError("modifier contract count does not match modifiers")

    held = scratch = 0
    details = []
    for modifier, contract in zip(modifiers, contracts):
        estimate_fn = getattr(modifier, "estimate_sequential_memory", None)
        modifier_held = modifier_scratch = 0
        unknown = False
        if contract.memory_estimate is not None:
            modifier_held, modifier_scratch = contract.memory_estimate
        elif estimate_fn is None:
            unknown = contract.requires_calibration_data
        else:
            for module in modules:
                estimate = estimate_fn(module, num_tokens)
                if estimate is None:
                    unknown = True
                    continue
                if (
                    not isinstance(estimate, tuple)
                    or len(estimate) != 2
                    or any(int(value) < 0 for value in estimate)
                ):
                    raise ModifierContractError(
                        f"{type(modifier).__qualname__}.estimate_sequential_memory "
                        "must return non-negative (held, scratch) or None"
                    )
                modifier_held += int(estimate[0])
                modifier_scratch += int(estimate[1])
        if unknown:
            raise ModifierContractError(
                f"{contract.modifier_type} does not provide a bounded memory "
                "estimate for the current streaming target"
            )
        held += modifier_held
        scratch += modifier_scratch
        details.append(
            {
                "modifier_type": contract.modifier_type,
                "held_bytes": modifier_held,
                "scratch_bytes": modifier_scratch,
            }
        )
    return ModifierMemoryEstimate(held, scratch, tuple(details))


class DeviceMemoryReservation:
    def __init__(self, budget: DeviceMemoryBudget, owner: str, reserved_bytes: int):
        self._budget = budget
        self.owner = owner
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
        with self._lock:
            available = self._available_bytes(resolved)
            if available is not None:
                promised = sum(item._reserved_bytes for item in self._reservations)
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
            reservation = DeviceMemoryReservation(self, owner, required_bytes)
            self._reservations.add(reservation)
            return reservation


def estimate_target_device_bytes(
    session,
    plan,
    *,
    dtype: torch.dtype,
) -> int:
    """Return materialized target bytes without including CPU decode workspace."""
    return session.estimate_materialized_bytes(plan, dtype=dtype)
