"""Parallel execution contracts used by streaming PTQ."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelContext:
    """Describe the fixed process topology for one streaming run.

    The initial implementation is deliberately usable on one process.  A
    single process is represented as EP size one instead of relying on an
    implicit rank-zero convention, so transaction metadata can use the same
    shape when reference EP is added.
    """

    rank: int = 0
    world_size: int = 1
    ep_rank: int = 0
    ep_size: int = 1
    dp_rank: int = 0
    dp_size: int = 1
    tp_rank: int = 0
    tp_size: int = 1
    pp_rank: int = 0
    pp_size: int = 1
    coordinator_rank: int = 0

    def __post_init__(self) -> None:
        for name in (
            "rank",
            "world_size",
            "ep_rank",
            "ep_size",
            "dp_rank",
            "dp_size",
            "tp_rank",
            "tp_size",
            "pp_rank",
            "pp_size",
            "coordinator_rank",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be within world_size")
        for rank_name, size_name in (
            ("ep_rank", "ep_size"),
            ("dp_rank", "dp_size"),
            ("tp_rank", "tp_size"),
            ("pp_rank", "pp_size"),
        ):
            size = getattr(self, size_name)
            rank = getattr(self, rank_name)
            if size <= 0:
                raise ValueError(f"{size_name} must be positive")
            if not 0 <= rank < size:
                raise ValueError(f"{rank_name} must be within {size_name}")
        if not 0 <= self.coordinator_rank < self.world_size:
            raise ValueError("coordinator_rank must be within world_size")

    @classmethod
    def single_device(cls) -> ParallelContext:
        """Return the explicit EP=1 reference topology."""

        return cls()

    @property
    def participant_id(self) -> str:
        return f"rank-{self.rank:05d}"

    @property
    def is_coordinator(self) -> bool:
        return self.rank == self.coordinator_rank

    def to_dict(self) -> dict[str, int]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "ep_rank": self.ep_rank,
            "ep_size": self.ep_size,
            "dp_rank": self.dp_rank,
            "dp_size": self.dp_size,
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "pp_rank": self.pp_rank,
            "pp_size": self.pp_size,
            "coordinator_rank": self.coordinator_rank,
        }

    @property
    def participants(self) -> tuple[str, ...]:
        return tuple(f"rank-{rank:05d}" for rank in range(self.world_size))


@dataclass(frozen=True)
class StatisticsScope:
    """Declare which logical inputs a rank is allowed to count."""

    input_replica_id: str = "replica-00000"
    logical_owner_group: str = "ep"
    counts_replicated_inputs_once: bool = True

    def __post_init__(self) -> None:
        if not self.input_replica_id:
            raise ValueError("input_replica_id cannot be empty")
        if not self.logical_owner_group:
            raise ValueError("logical_owner_group cannot be empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "input_replica_id": self.input_replica_id,
            "logical_owner_group": self.logical_owner_group,
            "counts_replicated_inputs_once": self.counts_replicated_inputs_once,
        }
