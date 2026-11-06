"""Recovery error classification and durable transaction inspection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from .artifacts import ArtifactCompatibilityError
from .checkpoint.coordinator import CommittedTransactionCorruptionError

if TYPE_CHECKING:
    from .checkpoint.writer import StreamingCheckpointWriter

__all__ = [
    "RecoveryErrorKind",
    "ResumePlan",
    "ResumeScanStats",
    "build_resume_plan",
    "classify_recovery_error",
]


class RecoveryErrorKind(str, Enum):
    UNCOMMITTED = "uncommitted"
    INCOMPATIBLE = "incompatible"
    COMMITTED_CORRUPTION = "committed_corruption"
    RESOURCE = "resource"


@dataclass(frozen=True)
class ResumeScanStats:
    """Bytes and objects inspected while constructing one resume plan."""

    transaction_count: int = 0
    metadata_bytes: int = 0
    payload_count: int = 0
    payload_bytes: int = 0
    boundary_count: int = 0
    boundary_bytes: int = 0


@dataclass(frozen=True)
class ResumePlan:
    """The validated contiguous prefix of committed subgraph transactions."""

    transaction_ids: tuple[str, ...]
    next_transaction_id: str
    boundary_index: int | None
    metadata: tuple[dict[str, Any], ...] = ()
    scan_stats: ResumeScanStats = ResumeScanStats()
    prefix_metadata: dict[str, Any] | None = None

    @property
    def metadata_by_id(self) -> dict[str, dict[str, Any]]:
        return {value["transaction_id"]: value for value in self.metadata}


def build_resume_plan(writer: StreamingCheckpointWriter) -> ResumePlan:
    """Scan committed transaction metadata once and reject holes or corruption."""

    committed = list(writer.committed_metadata())
    prefix = [value for value in committed if value["transaction_id"] == "prefix-00000"]
    metadata = [
        value
        for value in committed
        if value["transaction_id"].startswith("subgraph-")
    ]
    ordered = sorted(
        metadata,
        key=lambda value: value["transaction_id"],
    )
    expected = []
    boundary_index = None
    for index, value in enumerate(ordered):
        transaction_id = value["transaction_id"]
        expected_id = f"subgraph-{index:05d}"
        if transaction_id != expected_id:
            raise RuntimeError(
                f"Committed transaction chain has a gap: expected {expected_id!r}, "
                f"found {transaction_id!r}"
            )
        expected.append(transaction_id)
        boundary = value.get("boundary")
        if boundary is not None:
            boundary_index = int(boundary["index"])
    next_index = len(expected)
    return ResumePlan(
        transaction_ids=tuple(expected),
        next_transaction_id=f"subgraph-{next_index:05d}",
        boundary_index=boundary_index,
        metadata=tuple(ordered),
        scan_stats=writer.scan_stats,
        prefix_metadata=prefix[0] if prefix else None,
    )


def classify_recovery_error(error: BaseException) -> RecoveryErrorKind:
    if isinstance(error, (ArtifactCompatibilityError, ValueError)):
        return RecoveryErrorKind.INCOMPATIBLE
    if isinstance(error, CommittedTransactionCorruptionError):
        return RecoveryErrorKind.COMMITTED_CORRUPTION
    if isinstance(error, (MemoryError, OSError)):
        return RecoveryErrorKind.RESOURCE
    return RecoveryErrorKind.UNCOMMITTED
