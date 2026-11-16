"""Generic process topology and placement contracts for streaming EP.

The topology is model independent. Model-specific code only describes expert
containers and checkpoint name parsing; execution, ownership, and validation
remain here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence

import torch
import torch.distributed as dist

from ..artifacts import fingerprint_json

__all__ = [
    "EPConfig",
    "EPMode",
    "EPTopology",
    "ExpertModuleSpec",
    "ExpertPlacementPlan",
    "build_expert_placement",
    "ensure_streaming_ep_process_group",
    "make_expert_state_filter",
]


class EPMode(str, Enum):
    DISABLED = "disabled"
    ALL_EXPERT = "all_expert"
    ROUTED = "routed"


@dataclass(frozen=True)
class EPConfig:
    """Static EP/DP dimensions for one streaming run."""

    ep_size: int = 1
    dp_size: int = 1
    mode: EPMode | str = EPMode.DISABLED

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            try:
                object.__setattr__(self, "mode", EPMode(self.mode))
            except ValueError as error:
                allowed = ", ".join(item.value for item in EPMode)
                raise ValueError(
                    f"mode must be one of {allowed}; got {self.mode!r}"
                ) from error
        elif not isinstance(self.mode, EPMode):
            raise ValueError("mode must be an EPMode or string")
        for name in ("ep_size", "dp_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.mode is EPMode.DISABLED and (self.ep_size != 1 or self.dp_size != 1):
            raise ValueError("disabled EP requires ep_size=dp_size=1")

    @property
    def world_size(self) -> int:
        return self.ep_size * self.dp_size

    @property
    def enabled(self) -> bool:
        return self.mode is not EPMode.DISABLED

    def validate_world_size(self, world_size: int) -> None:
        if world_size != self.world_size:
            raise ValueError(
                f"EP requires world_size={self.world_size}; got {world_size}"
            )


@dataclass(frozen=True)
class EPTopology:
    """Rank coordinates and optional EP/DP process groups."""

    config: EPConfig
    rank: int = 0
    ep_rank: int = 0
    dp_rank: int = 0
    ep_group: dist.ProcessGroup | None = None
    dp_group: dist.ProcessGroup | None = None

    @classmethod
    def from_config(
        cls,
        config: EPConfig,
        *,
        create_groups: bool = True,
    ) -> "EPTopology":
        if not dist.is_available() or not dist.is_initialized():
            config.validate_world_size(1)
            return cls(config)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        config.validate_world_size(world_size)
        ep_rank = rank % config.ep_size
        dp_rank = rank // config.ep_size
        if config.world_size == 1 or not create_groups:
            return cls(config, rank, ep_rank, dp_rank)

        # Every rank creates every group in the same order. This is required by
        # torch.distributed even when a rank is not a member of a given group.
        ep_groups = {}
        for dp_index in range(config.dp_size):
            ranks = list(
                range(
                    dp_index * config.ep_size,
                    (dp_index + 1) * config.ep_size,
                )
            )
            ep_groups[dp_index] = dist.new_group(ranks=ranks)
        dp_groups = {}
        for ep_index in range(config.ep_size):
            ranks = [
                dp_index * config.ep_size + ep_index
                for dp_index in range(config.dp_size)
            ]
            dp_groups[ep_index] = dist.new_group(ranks=ranks)
        return cls(
            config,
            rank,
            ep_rank,
            dp_rank,
            ep_groups[dp_rank],
            dp_groups[ep_rank],
        )


def ensure_streaming_ep_process_group(config: EPConfig) -> bool:
    """Initialize ``torch.distributed`` for a torchrun EP invocation."""

    if config.world_size == 1:
        return False
    if dist.is_initialized():
        config.validate_world_size(dist.get_world_size())
        return False
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    config.validate_world_size(world_size)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    return True


@dataclass(frozen=True)
class ExpertModuleSpec:
    """External description of one expert container."""

    module_name: str
    num_experts: int
    expert_id_from_state_name: Callable[[str], int | None]

    def __post_init__(self) -> None:
        if not self.module_name:
            raise ValueError("module_name cannot be empty")
        if (
            isinstance(self.num_experts, bool)
            or not isinstance(self.num_experts, int)
            or self.num_experts < 1
        ):
            raise ValueError("num_experts must be a positive integer")


@dataclass(frozen=True)
class ExpertPlacementPlan:
    """Stable, byte-balanced mapping from logical expert ids to EP ranks."""

    ep_size: int
    expert_bytes: tuple[int, ...]
    owner_by_expert: tuple[int, ...]
    local_bytes_by_rank: tuple[int, ...]

    @property
    def num_experts(self) -> int:
        return len(self.owner_by_expert)

    def owner(self, expert_id: int) -> int:
        if expert_id < 0 or expert_id >= self.num_experts:
            raise IndexError(expert_id)
        return self.owner_by_expert[expert_id]

    def local_experts(self, ep_rank: int) -> tuple[int, ...]:
        if ep_rank < 0 or ep_rank >= self.ep_size:
            raise ValueError("ep_rank is outside the placement")
        return tuple(
            expert_id
            for expert_id, owner in enumerate(self.owner_by_expert)
            if owner == ep_rank
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "ep_size": self.ep_size,
            "expert_bytes": list(self.expert_bytes),
            "owner_by_expert": list(self.owner_by_expert),
            "local_bytes_by_rank": list(self.local_bytes_by_rank),
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())

    def validate_capacity(
        self,
        capacity_bytes: int | Sequence[int],
        *,
        replicated_bytes: int = 0,
    ) -> None:
        capacities = (
            (capacity_bytes,) * self.ep_size
            if isinstance(capacity_bytes, int)
            else tuple(capacity_bytes)
        )
        if len(capacities) != self.ep_size:
            raise ValueError("capacity_bytes must match ep_size")
        for rank, required in enumerate(self.local_bytes_by_rank):
            if required + replicated_bytes > capacities[rank]:
                raise MemoryError(
                    f"EP rank {rank} requires {required + replicated_bytes} bytes"
                )


def build_expert_placement(
    *,
    num_experts: int,
    ep_size: int,
    expert_bytes: Sequence[int] | None = None,
) -> ExpertPlacementPlan:
    if num_experts < 1 or ep_size < 1:
        raise ValueError("num_experts and ep_size must be positive")
    sizes = (
        tuple(0 for _ in range(num_experts))
        if expert_bytes is None
        else tuple(expert_bytes)
    )
    if len(sizes) != num_experts or any(size < 0 for size in sizes):
        raise ValueError("expert_bytes must contain non-negative values per expert")
    owners = [-1] * num_experts
    totals = [0] * ep_size
    order = (
        range(num_experts)
        if not any(sizes)
        else sorted(range(num_experts), key=lambda index: (-sizes[index], index))
    )
    for expert_id in order:
        owner = min(range(ep_size), key=lambda rank: (totals[rank], rank))
        owners[expert_id] = owner
        totals[owner] += sizes[expert_id]
    return ExpertPlacementPlan(ep_size, sizes, tuple(owners), tuple(totals))


def make_expert_state_filter(
    placement: ExpertPlacementPlan,
    ep_rank: int,
    expert_id_from_state_name: Callable[[str], int | None],
) -> Callable[[str], bool]:
    if ep_rank < 0 or ep_rank >= placement.ep_size:
        raise ValueError("ep_rank is outside the placement")

    def state_filter(name: str) -> bool:
        expert_id = expert_id_from_state_name(name)
        return expert_id is None or placement.owner(expert_id) == ep_rank

    return state_filter
