"""Reusable router outputs for bounded expert execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ..artifacts import fingerprint_json

__all__ = ["ExpertRoute", "RoutePlan", "RoutePlanCollection"]


@dataclass(frozen=True)
class ExpertRoute:
    """Token rows and gate weights for one expert in one route plan."""

    token_indices: torch.Tensor
    slot_indices: torch.Tensor
    weights: torch.Tensor

    def __post_init__(self) -> None:
        if self.token_indices.ndim != 1 or self.slot_indices.ndim != 1:
            raise ValueError("route indices must be one-dimensional")
        if self.token_indices.shape != self.slot_indices.shape:
            raise ValueError("token and slot indices must have the same shape")
        if self.weights.ndim != 1 or self.weights.shape != self.token_indices.shape:
            raise ValueError("route weights must match route indices")
        if self.token_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("token_indices must use an integer dtype")
        if self.slot_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("slot_indices must use an integer dtype")
        if not self.weights.is_floating_point():
            raise ValueError("route weights must use a floating point dtype")
        if not (
            self.token_indices.device
            == self.slot_indices.device
            == self.weights.device
        ):
            raise ValueError("route tensors must share one device")

    @property
    def token_count(self) -> int:
        return int(self.token_indices.numel())


@dataclass(frozen=True)
class RoutePlan:
    """A frozen, reusable mapping from routed tokens to expert inputs."""

    num_tokens: int
    num_experts: int
    top_k: int
    routes: tuple[ExpertRoute, ...]
    original_shape: tuple[int, ...] | None = None
    batch_index: int = 0
    sequence_ids: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if len(self.routes) != self.num_experts:
            raise ValueError("routes must contain one entry per expert")
        if self.batch_index < 0:
            raise ValueError("batch_index must be non-negative")
        if self.sequence_ids is not None:
            if self.sequence_ids.ndim != 1:
                raise ValueError("sequence_ids must be one-dimensional")
            if self.sequence_ids.numel() != self.num_tokens:
                raise ValueError("sequence_ids must cover every routed token")
            if self.sequence_ids.device != self.device:
                raise ValueError("sequence_ids must share the route device")
        for route in self.routes:
            if route.token_count:
                if int(route.token_indices.min().item()) < 0:
                    raise ValueError("token_indices cannot be negative")
                if int(route.token_indices.max().item()) >= self.num_tokens:
                    raise ValueError("token_indices exceed num_tokens")
                if int(route.slot_indices.min().item()) < 0:
                    raise ValueError("slot_indices cannot be negative")
                if int(route.slot_indices.max().item()) >= self.top_k:
                    raise ValueError("slot_indices exceed top_k")

    @classmethod
    def from_tensors(
        cls,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        *,
        num_experts: int,
        original_shape: Sequence[int] | None = None,
        batch_index: int = 0,
        sequence_ids: torch.Tensor | None = None,
    ) -> "RoutePlan":
        """Build a plan from the standard ``[tokens, top_k]`` router output."""

        if top_k_index.ndim != 2:
            raise ValueError("top_k_index must have shape [tokens, top_k]")
        if top_k_weights.shape != top_k_index.shape:
            raise ValueError("top_k_weights must match top_k_index")
        if top_k_index.dtype not in (torch.int32, torch.int64):
            raise ValueError("top_k_index must use an integer dtype")
        if not top_k_weights.is_floating_point():
            raise ValueError("top_k_weights must use a floating point dtype")
        if top_k_index.device != top_k_weights.device:
            raise ValueError("router outputs must share one device")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if top_k_index.numel() and (
            int(top_k_index.min().item()) < 0
            or int(top_k_index.max().item()) >= num_experts
        ):
            raise ValueError("router output contains an unknown expert")

        routes = []
        expert_mask_source = top_k_index.transpose(0, 1)
        for expert_id in range(num_experts):
            slot_indices, token_indices = torch.where(
                expert_mask_source == expert_id
            )
            routes.append(
                ExpertRoute(
                    token_indices=token_indices,
                    slot_indices=slot_indices,
                    weights=top_k_weights[token_indices, slot_indices],
                )
            )
        return cls(
            num_tokens=int(top_k_index.shape[0]),
            num_experts=num_experts,
            top_k=int(top_k_index.shape[1]),
            routes=tuple(routes),
            original_shape=(
                tuple(int(size) for size in original_shape)
                if original_shape is not None
                else None
            ),
            batch_index=batch_index,
            sequence_ids=sequence_ids,
        )

    @property
    def device(self) -> torch.device:
        for route in self.routes:
            return route.token_indices.device
        if self.sequence_ids is not None:
            return self.sequence_ids.device
        return torch.device("cpu")

    @property
    def token_counts(self) -> tuple[int, ...]:
        return tuple(route.token_count for route in self.routes)

    @property
    def total_assignments(self) -> int:
        return sum(self.token_counts)

    def route(self, expert_id: int) -> ExpertRoute:
        if expert_id < 0 or expert_id >= self.num_experts:
            raise IndexError(expert_id)
        return self.routes[expert_id]

    def work_units(
        self,
        token_work_units: Mapping[int, float] | Sequence[float] | None = None,
    ) -> tuple[float, ...]:
        """Return routed work per expert without moving route tensors."""

        if token_work_units is None:
            costs = (1.0,) * self.num_experts
        elif isinstance(token_work_units, Mapping):
            costs = tuple(
                float(token_work_units.get(expert_id, 1.0))
                for expert_id in range(self.num_experts)
            )
        else:
            costs = tuple(float(value) for value in token_work_units)
            if len(costs) != self.num_experts:
                raise ValueError("token_work_units must match num_experts")
        if any(value <= 0 for value in costs):
            raise ValueError("token_work_units must be positive")
        return tuple(
            float(token_count) * costs[expert_id]
            for expert_id, token_count in enumerate(self.token_counts)
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return a tensor-free identity suitable for manifests and logs."""

        return {
            "batch_index": self.batch_index,
            "num_tokens": self.num_tokens,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "original_shape": self.original_shape,
            "token_counts": list(self.token_counts),
            "total_assignments": self.total_assignments,
            "route_device": str(self.device),
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_metadata())


@dataclass(frozen=True)
class RoutePlanCollection:
    """Route plans for multiple calibration boundaries."""

    plans: tuple[RoutePlan, ...]

    def __post_init__(self) -> None:
        if not self.plans:
            raise ValueError("RoutePlanCollection requires at least one plan")
        first = self.plans[0]
        if any(
            plan.num_experts != first.num_experts or plan.top_k != first.top_k
            for plan in self.plans[1:]
        ):
            raise ValueError("all route plans must use the same expert topology")
        batch_indices = [plan.batch_index for plan in self.plans]
        if len(batch_indices) != len(set(batch_indices)):
            raise ValueError("route plan batch indices must be unique")

    @property
    def num_experts(self) -> int:
        return self.plans[0].num_experts

    @property
    def token_counts(self) -> tuple[int, ...]:
        totals = [0] * self.num_experts
        for plan in self.plans:
            for expert_id, count in enumerate(plan.token_counts):
                totals[expert_id] += count
        return tuple(totals)

    @property
    def total_tokens(self) -> int:
        return sum(plan.num_tokens for plan in self.plans)

    @property
    def total_assignments(self) -> int:
        return sum(plan.total_assignments for plan in self.plans)

    def work_units(
        self,
        token_work_units: Mapping[int, float] | Sequence[float] | None = None,
    ) -> tuple[float, ...]:
        if token_work_units is None:
            costs = (1.0,) * self.num_experts
        elif isinstance(token_work_units, Mapping):
            costs = tuple(
                float(token_work_units.get(expert_id, 1.0))
                for expert_id in range(self.num_experts)
            )
        else:
            costs = tuple(float(value) for value in token_work_units)
            if len(costs) != self.num_experts:
                raise ValueError("token_work_units must match num_experts")
        if any(value <= 0 for value in costs):
            raise ValueError("token_work_units must be positive")
        return tuple(
            float(count) * costs[expert_id]
            for expert_id, count in enumerate(self.token_counts)
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "plans": [plan.to_metadata() for plan in self.plans],
            "token_counts": list(self.token_counts),
            "total_tokens": self.total_tokens,
            "total_assignments": self.total_assignments,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_metadata())
