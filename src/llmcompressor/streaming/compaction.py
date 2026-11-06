"""Durable retirement records for committed activation boundaries."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .artifacts import fingerprint_json

__all__ = ["BoundaryCompactor", "RetiredBoundaryError"]


class RetiredBoundaryError(RuntimeError):
    """A boundary was validly committed but has already been compacted."""


class BoundaryCompactor:
    """Retire old transaction boundary payloads after a successor commits."""

    def __init__(
        self,
        staging_dir: str | Path,
        *,
        keep_recent: int = 2,
    ):
        if keep_recent < 1:
            raise ValueError("keep_recent must be positive")
        self.root = Path(staging_dir)
        self.transactions_dir = self.root / "transactions"
        self.retirements_dir = self.root / "retirements"
        self.keep_recent = keep_recent

    def compact(self, transaction_ids: tuple[str, ...]) -> tuple[str, ...]:
        """Retire boundary payloads outside the active recovery window."""

        subgraphs = [
            transaction_id
            for transaction_id in transaction_ids
            if (
                transaction_id.startswith("subgraph-")
                or transaction_id == "prefix-00000"
            )
        ]
        if len(subgraphs) <= self.keep_recent:
            return ()
        retired = []
        for transaction_id in subgraphs[: -self.keep_recent]:
            successor = _next_subgraph_id(transaction_id)
            if not (self.transactions_dir / successor).is_dir():
                continue
            metadata_path = self.transactions_dir / transaction_id / "metadata.json"
            metadata = _read_json(metadata_path)
            boundary = metadata.get("boundary")
            if boundary is None or self.is_retired(transaction_id):
                continue
            records = _boundary_records(boundary)
            retirement = {
                "schema_version": 1,
                "transaction_id": transaction_id,
                "successor_transaction_id": successor,
                "boundary_index": int(boundary["index"]),
                "boundary_sha256": boundary.get("sha256"),
                "batch_count": len(records),
                "records_sha256": fingerprint_json(records),
            }
            retirement_path = self.retirements_dir / f"{transaction_id}.json"
            _atomic_json(retirement_path, retirement)
            _sync_directory(self.retirements_dir)
            boundary_dir = self.transactions_dir / transaction_id / "boundaries"
            if boundary_dir.is_dir():
                shutil.rmtree(boundary_dir)
                _sync_directory(self.transactions_dir / transaction_id)
            retired.append(transaction_id)
        return tuple(retired)

    def is_retired(self, transaction_id: str) -> bool:
        path = self.retirements_dir / f"{transaction_id}.json"
        if not path.is_file():
            return False
        self.validate_retirement(transaction_id)
        return True

    def validate_retirement(self, transaction_id: str) -> None:
        path = self.retirements_dir / f"{transaction_id}.json"
        value = _read_json(path)
        required = {
            "schema_version",
            "transaction_id",
            "successor_transaction_id",
            "boundary_index",
            "boundary_sha256",
            "batch_count",
            "records_sha256",
        }
        if set(value) != required:
            raise RuntimeError(f"Retirement record {path} has invalid fields")
        if value["transaction_id"] != transaction_id:
            raise RuntimeError(f"Retirement record {path} has invalid transaction")
        successor = str(value["successor_transaction_id"])
        if successor != _next_subgraph_id(transaction_id):
            raise RuntimeError(f"Retirement record {path} has invalid successor")
        successor_path = self.transactions_dir / successor / "metadata.json"
        if not successor_path.is_file():
            raise RuntimeError(f"Retirement record {path} has no committed successor")
        successor_metadata = _read_json(successor_path)
        if successor_metadata.get("transaction_id") != successor:
            raise RuntimeError(
                f"Retirement record {path} points to invalid successor metadata"
            )
        boundary = _read_json(
            self.transactions_dir / transaction_id / "metadata.json"
        ).get("boundary")
        if boundary is None:
            raise RuntimeError(f"Retirement record {path} has no boundary")
        records = _boundary_records(boundary)
        if int(value["boundary_index"]) != int(boundary["index"]):
            raise RuntimeError(f"Retirement record {path} has invalid boundary")
        if int(value["batch_count"]) != len(records):
            raise RuntimeError(f"Retirement record {path} has invalid batch count")
        if value["boundary_sha256"] != boundary.get("sha256"):
            raise RuntimeError(f"Retirement record {path} has invalid boundary digest")
        if value["records_sha256"] != fingerprint_json(records):
            raise RuntimeError(f"Retirement record {path} has invalid records digest")


def _next_subgraph_id(transaction_id: str) -> str:
    if transaction_id == "prefix-00000":
        return "subgraph-00000"
    prefix, _, suffix = transaction_id.rpartition("-")
    if prefix != "subgraph" or not suffix.isdigit():
        raise ValueError(f"Invalid subgraph transaction id {transaction_id!r}")
    return f"subgraph-{int(suffix) + 1:05d}"


def _boundary_records(boundary: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = boundary.get("batches", [])
    if not isinstance(records, list):
        raise ValueError("Boundary batches must be a list")
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Boundary batch records must be mappings")
    return [dict(record) for record in records]


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
