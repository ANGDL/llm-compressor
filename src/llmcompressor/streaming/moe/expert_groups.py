"""Bounded-residency execution for linearized MoE experts.

The executor in this module owns expert grouping and weight residency only. It
does not implement a quantization algorithm. A caller can attach existing
modifier collectors to the loaded expert modules and run the same group
multiple times for calibration before evicting it.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..loading import LoadedSubgraph, SubgraphWeightSession

__all__ = [
    "ExpertGroup",
    "ExpertGroupExecutor",
    "ExpertGroupPlan",
    "execute_expert_group",
]


@dataclass(frozen=True)
class ExpertGroup:
    """One ordered set of experts sharing a residency window."""

    experts_name: str
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.experts_name:
            raise ValueError("experts_name must not be empty")
        if not self.indices:
            raise ValueError("an expert group must contain at least one expert")
        if tuple(sorted(set(self.indices))) != self.indices:
            raise ValueError("expert indices must be unique and sorted")
        if self.indices[0] < 0:
            raise ValueError("expert indices must be non-negative")

    @property
    def module_names(self) -> tuple[str, ...]:
        return tuple(
            f"{self.experts_name}.{expert_index}" for expert_index in self.indices
        )


@dataclass(frozen=True)
class ExpertGroupPlan:
    """Deterministic partition of one experts module."""

    experts_name: str
    num_experts: int
    group_size: int
    groups: tuple[ExpertGroup, ...]

    def __post_init__(self) -> None:
        if not self.experts_name:
            raise ValueError("experts_name must not be empty")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        expected = tuple(range(self.num_experts))
        actual = tuple(index for group in self.groups for index in group.indices)
        if actual != expected:
            raise ValueError(
                "expert groups must cover every expert exactly once in index order"
            )
        if any(len(group.indices) > self.group_size for group in self.groups):
            raise ValueError("an expert group exceeds group_size")
        if any(group.experts_name != self.experts_name for group in self.groups):
            raise ValueError("all groups must belong to experts_name")

    @classmethod
    def from_num_experts(
        cls,
        experts_name: str,
        num_experts: int,
        group_size: int,
    ) -> "ExpertGroupPlan":
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        groups = tuple(
            ExpertGroup(
                experts_name,
                tuple(range(start, min(start + group_size, num_experts))),
            )
            for start in range(0, num_experts, group_size)
        )
        return cls(experts_name, num_experts, group_size, groups)

    @classmethod
    def from_module(
        cls,
        experts_name: str,
        experts: nn.Module,
        group_size: int,
    ) -> "ExpertGroupPlan":
        num_experts = getattr(experts, "num_experts", None)
        if num_experts is None:
            try:
                num_experts = len(experts)
            except TypeError as error:
                raise TypeError(
                    f"Cannot infer expert count from {type(experts).__name__}"
                ) from error
        return cls.from_num_experts(experts_name, int(num_experts), group_size)


def _validate_routing(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    num_experts: int,
) -> None:
    if hidden_states.ndim != 2:
        raise ValueError(
            "hidden_states must have shape [num_tokens, hidden_size]; "
            f"got {tuple(hidden_states.shape)}"
        )
    if top_k_index.ndim != 2 or top_k_weights.shape != top_k_index.shape:
        raise ValueError(
            "top_k_index and top_k_weights must have matching shape [tokens, top_k]"
        )
    if top_k_index.shape[0] != hidden_states.shape[0]:
        raise ValueError("routing token count must match hidden_states")
    if top_k_index.dtype not in (torch.int32, torch.int64):
        raise ValueError("top_k_index must use an integer dtype")
    if top_k_index.numel() and (
        top_k_index.min().item() < 0 or top_k_index.max().item() >= num_experts
    ):
        raise ValueError(
            "routing contains an expert index outside the routed expert range"
        )


def _expert_count(experts: nn.Module) -> int:
    num_experts = getattr(experts, "num_experts", None)
    if num_experts is not None:
        return int(num_experts)
    try:
        return len(experts)
    except TypeError as error:
        raise TypeError(f"{type(experts).__name__} is not an expert container") from error


def execute_expert_group(
    experts: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    group: ExpertGroup,
    *,
    output: torch.Tensor | None = None,
    calibrate_all_experts: bool = False,
) -> torch.Tensor:
    """Execute one group and add its routed results into ``output``.

    The expert loop and ``torch.where`` layout match ``LinearExperts2D.forward``.
    Groups must be executed in ascending expert order when reproducible
    accumulation order is required. ``calibrate_all_experts`` preserves the
    existing calibration behavior by running every expert on all hidden states
    while retaining only its routed output.
    """
    num_experts = _expert_count(experts)
    _validate_routing(hidden_states, top_k_index, top_k_weights, num_experts)
    if any(index >= num_experts for index in group.indices):
        raise ValueError("expert group contains an out-of-range expert")
    if getattr(experts, "novices", None) is not None:
        raise ValueError(
            "expert group execution does not support novice experts; "
            "handle novice outputs in a model-specific adapter"
        )

    result = torch.zeros_like(hidden_states) if output is None else output
    if result.shape != hidden_states.shape or result.device != hidden_states.device:
        raise ValueError("output must match hidden_states shape and device")

    for expert_index in group.indices:
        expert_mask = top_k_index == expert_index
        top_k_pos, token_indices = torch.where(expert_mask.transpose(0, 1))
        expert = experts[expert_index]
        if calibrate_all_experts:
            expert_output = expert(hidden_states)[token_indices]
        else:
            expert_output = expert(hidden_states[token_indices])
        expert_weights = top_k_weights[token_indices, top_k_pos, None]
        weighted_output = expert_output * expert_weights
        result.index_add_(
            0,
            token_indices,
            weighted_output.to(result.dtype),
        )
    return result


class ExpertGroupExecutor:
    """Load, execute, and evict one bounded group of experts at a time."""

    def __init__(
        self,
        model: nn.Module,
        weight_session: SubgraphWeightSession,
        plan: ExpertGroupPlan,
    ) -> None:
        self.model = model
        self.weight_session = weight_session
        self.plan = plan
        try:
            experts = model.get_submodule(plan.experts_name)
        except AttributeError as error:
            raise ValueError(
                f"Unknown experts module {plan.experts_name!r}"
            ) from error
        actual_plan = ExpertGroupPlan.from_module(
            plan.experts_name, experts, plan.group_size
        )
        if actual_plan != plan:
            raise ValueError(
                f"Expert group plan does not match {plan.experts_name!r}"
            )

    @contextmanager
    def loaded_group(
        self,
        group: ExpertGroup,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Generator[LoadedSubgraph, None, None]:
        """Install exactly one expert group and restore it on exit."""
        if group not in self.plan.groups:
            raise ValueError("group does not belong to this executor plan")
        load_plan = self.weight_session.plan_modules(group.module_names)
        with self.weight_session.loaded_plan(
            load_plan,
            device=device,
            dtype=dtype,
        ) as loaded:
            yield loaded

    @torch.no_grad()
    def execute(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        calibrate_all_experts: bool = False,
    ) -> torch.Tensor:
        """Execute all groups while keeping only one group resident."""
        resolved_device = torch.device(device)
        if hidden_states.device != resolved_device:
            raise ValueError(
                "hidden_states must already be on the executor device; "
                f"got {hidden_states.device}, expected {resolved_device}"
            )
        result = torch.zeros_like(hidden_states)
        experts = self.model.get_submodule(self.plan.experts_name)
        for group in self.plan.groups:
            with self.loaded_group(group, device=resolved_device, dtype=dtype):
                execute_expert_group(
                    experts,
                    hidden_states,
                    top_k_index,
                    top_k_weights,
                    group,
                    output=result,
                    calibrate_all_experts=calibrate_all_experts,
                )
        return result
