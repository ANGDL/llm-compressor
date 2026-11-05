"""Logical parameter and expert ownership for streaming PTQ."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..artifacts import fingerprint_json


@dataclass(frozen=True)
class PlacementEntry:
    """Ownership of one logical parameter or expert."""

    logical_name: str
    owner_rank: int
    replicated: bool = False
    replica_group: str | None = None
    expert_id: int | None = None

    def __post_init__(self) -> None:
        if not self.logical_name:
            raise ValueError("logical_name cannot be empty")
        if self.owner_rank < 0:
            raise ValueError("owner_rank must be non-negative")
        if self.replicated and not self.replica_group:
            raise ValueError("replicated entries require replica_group")
        if self.expert_id is not None and self.expert_id < 0:
            raise ValueError("expert_id must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "logical_name": self.logical_name,
            "owner_rank": self.owner_rank,
            "replicated": self.replicated,
            "replica_group": self.replica_group,
            "expert_id": self.expert_id,
        }


@dataclass(frozen=True)
class PlacementPlan:
    """Stable mapping from logical names to explicit owners."""

    entries: tuple[PlacementEntry, ...] = ()
    ep_size: int = 1

    def __post_init__(self) -> None:
        if self.ep_size <= 0:
            raise ValueError("ep_size must be positive")
        names = [entry.logical_name for entry in self.entries]
        if len(names) != len(set(names)):
            raise ValueError("PlacementPlan logical names must be unique")
        if any(entry.owner_rank >= self.ep_size for entry in self.entries):
            raise ValueError("Placement owner exceeds ep_size")

    @classmethod
    def from_mapping(
        cls,
        owners: Mapping[str, int],
        *,
        ep_size: int = 1,
    ) -> PlacementPlan:
        return cls(
            tuple(
                PlacementEntry(logical_name=name, owner_rank=owner)
                for name, owner in sorted(owners.items())
            ),
            ep_size=ep_size,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "ep_size": self.ep_size,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())

    def owner(self, logical_name: str) -> int:
        for entry in self.entries:
            if entry.logical_name == logical_name:
                return entry.owner_rank
        raise KeyError(logical_name)
