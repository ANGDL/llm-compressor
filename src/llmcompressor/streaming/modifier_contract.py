"""Explicit capability contracts for modifiers used by streaming PTQ."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch

__all__ = [
    "ModifierContract",
    "ModifierContractError",
    "build_modifier_contracts",
    "calibration_outputs_reusable",
]


class ModifierContractError(ValueError):
    """A modifier does not make its streaming behavior explicit."""


@dataclass(frozen=True)
class ModifierContract:
    """The execution promises a streaming pipeline may rely on."""

    modifier_type: str
    calibration_outputs_reusable: bool
    requires_calibration_data: bool
    modifies_weights: bool
    modifies_norms: bool
    statistics_lifetime: str
    propagation_mode: str
    recovery_state: str
    output_ownership: str
    memory_estimate: tuple[int, int] | None
    memory_estimate_is_static: bool = False
    parameter_placement: str = "single-device"
    statistics_sync_scope: str = "single-participant"
    collective_owner: str = "modifier"
    communication_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.modifier_type:
            raise ValueError("modifier_type cannot be empty")
        for name in (
            "statistics_lifetime",
            "propagation_mode",
            "recovery_state",
            "output_ownership",
            "parameter_placement",
            "statistics_sync_scope",
            "collective_owner",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} cannot be empty")
        if self.memory_estimate is not None and any(
            value < 0 for value in self.memory_estimate
        ):
            raise ValueError("memory_estimate values must be non-negative")
        if self.communication_bytes < 0:
            raise ValueError("communication_bytes must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "modifier_type": self.modifier_type,
            "calibration_outputs_reusable": self.calibration_outputs_reusable,
            "requires_calibration_data": self.requires_calibration_data,
            "modifies_weights": self.modifies_weights,
            "modifies_norms": self.modifies_norms,
            "statistics_lifetime": self.statistics_lifetime,
            "propagation_mode": self.propagation_mode,
            "recovery_state": self.recovery_state,
            "output_ownership": self.output_ownership,
            "memory_estimate": self.memory_estimate,
            "memory_estimate_is_static": self.memory_estimate_is_static,
            "parameter_placement": self.parameter_placement,
            "statistics_sync_scope": self.statistics_sync_scope,
            "collective_owner": self.collective_owner,
            "communication_bytes": self.communication_bytes,
        }


def _declared(modifier: Any, name: str, default: Any) -> Any:
    """Read a declaration without using the modifier class name as policy."""

    return type(modifier).__dict__.get(name, default)


def _memory_estimate(modifier: Any) -> tuple[int, int] | None:
    declared = _declared(modifier, "memory_estimate", None)
    if declared is not None:
        if (
            not isinstance(declared, tuple)
            or len(declared) != 2
            or any(not isinstance(value, int) for value in declared)
        ):
            raise ModifierContractError(
                f"{type(modifier).__qualname__}.memory_estimate must "
                "be a tuple of two integers"
            )
        return declared

    estimate = getattr(modifier, "estimate_sequential_memory", None)
    if estimate is None:
        return None
    try:
        value = estimate(torch.nn.Identity(), 0)
    except (AttributeError, NotImplementedError, TypeError):
        return None
    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) != 2:
        raise ModifierContractError(
            f"{type(modifier).__qualname__}.estimate_sequential_memory must "
            "return (held, scratch) or None"
        )
    return int(value[0]), int(value[1])


def build_modifier_contracts(
    modifiers: Iterable[Any],
) -> tuple[ModifierContract, ...]:
    """Build explicit contracts and reject invalid capability declarations."""

    contracts = []
    for modifier in modifiers:
        contracts.append(
            ModifierContract(
                modifier_type=(
                    f"{type(modifier).__module__}.{type(modifier).__qualname__}"
                ),
                calibration_outputs_reusable=bool(
                    _declared(modifier, "calibration_outputs_reusable", False)
                ),
                requires_calibration_data=bool(
                    getattr(modifier, "requires_calibration_data", False)
                ),
                modifies_weights=bool(
                    _declared(modifier, "modifies_weights", True)
                ),
                modifies_norms=bool(
                    _declared(modifier, "modifies_norms", False)
                ),
                statistics_lifetime=str(
                    _declared(modifier, "statistics_lifetime", "subgraph")
                ),
                propagation_mode=str(
                    _declared(
                        modifier,
                        "propagation_mode",
                        "sequential-compatible",
                    )
                ),
                recovery_state=str(
                    _declared(modifier, "recovery_state", "replay-subgraph")
                ),
                output_ownership=str(
                    _declared(modifier, "output_ownership", "pipeline")
                ),
                memory_estimate=_memory_estimate(modifier),
                memory_estimate_is_static=(
                    _declared(modifier, "memory_estimate", None) is not None
                ),
                parameter_placement=str(
                    _declared(modifier, "parameter_placement", "single-device")
                ),
                statistics_sync_scope=str(
                    _declared(modifier, "statistics_sync_scope", "single-participant")
                ),
                collective_owner=str(
                    _declared(modifier, "collective_owner", "modifier")
                ),
                communication_bytes=int(_declared(modifier, "communication_bytes", 0)),
            )
        )
    if not contracts:
        raise ModifierContractError("Streaming requires at least one modifier")
    return tuple(contracts)


def calibration_outputs_reusable(modifiers: Iterable[Any]) -> bool:
    """Return whether every modifier explicitly permits output reuse."""

    contracts = build_modifier_contracts(modifiers) if modifiers else ()
    return bool(contracts) and all(
        contract.calibration_outputs_reusable
        for contract in contracts
    )
