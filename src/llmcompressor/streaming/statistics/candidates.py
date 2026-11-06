"""Content-addressed calibration statistics for independent candidates."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

from llmcompressor.streaming.artifacts import fingerprint_json
from llmcompressor.streaming.loading.host_memory import HostMemoryBudget

__all__ = [
    "CandidateStatisticsCache",
    "CandidateStatisticsKey",
]


@dataclass(frozen=True)
class CandidateStatisticsKey:
    source_fingerprint: str
    dataset_fingerprint: str
    target_name: str
    target_index: int
    boundary_fingerprint: str
    prefix_fingerprint: str
    algorithms: tuple[str, ...]
    observer_fingerprint: str
    transform_fingerprint: str
    dtype: str
    implementation: str = "streaming-statistics-cache-v1"

    def __post_init__(self):
        if self.target_index < 0 or not self.target_name or not self.algorithms:
            raise ValueError("Statistics cache requires a target and algorithms")
        for name in (
            "source_fingerprint",
            "boundary_fingerprint",
            "observer_fingerprint",
            "transform_fingerprint",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(
                item not in "0123456789abcdef" for item in value
            ):
                raise ValueError(f"{name} must be a SHA-256 digest")

    @property
    def digest(self) -> str:
        return fingerprint_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_fingerprint": self.source_fingerprint,
            "dataset_fingerprint": self.dataset_fingerprint,
            "target_name": self.target_name,
            "target_index": self.target_index,
            "boundary_fingerprint": self.boundary_fingerprint,
            "prefix_fingerprint": self.prefix_fingerprint,
            "algorithms": list(self.algorithms),
            "observer_fingerprint": self.observer_fingerprint,
            "transform_fingerprint": self.transform_fingerprint,
            "dtype": self.dtype,
            "implementation": self.implementation,
        }


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, sort_keys=True, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync(path: Path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class CandidateStatisticsCache:
    """Store statistics only when every candidate identity component matches."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_bytes: int = 2 * 1024**3,
        max_entry_bytes: int = 512 * 1024**2,
        max_entries: int = 8,
    ):
        if max_bytes < 1 or max_entry_bytes < 1 or max_entries < 1:
            raise ValueError("Statistics cache budgets must be positive")
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.max_entry_bytes = max_entry_bytes
        self.max_entries = max_entries
        self.root.mkdir(parents=True, exist_ok=True)

    def _directory(self, key: CandidateStatisticsKey) -> Path:
        return self.root / key.digest

    def get(
        self,
        key: CandidateStatisticsKey,
        *,
        expected_names: tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor] | None:
        with self.root.joinpath(".cache.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            return self._get_locked(key, expected_names=expected_names)

    def _get_locked(self, key, *, expected_names=None):
        directory = self._directory(key)
        if not directory.exists():
            return None
        metadata_path = directory / "metadata.json"
        stats_path = directory / "stats.safetensors"
        complete = directory / "COMPLETE"
        if not (
            metadata_path.is_file() and stats_path.is_file() and complete.is_file()
        ):
            raise RuntimeError("Committed candidate statistics entry is incomplete")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if complete.read_text().strip() != _sha256(metadata_path):
                raise RuntimeError("Candidate statistics metadata digest changed")
            if metadata.get("key") != key.to_dict():
                raise RuntimeError("Candidate statistics cache identity changed")
            if metadata["size"] != stats_path.stat().st_size:
                raise RuntimeError("Candidate statistics payload size changed")
            if metadata["sha256"] != _sha256(stats_path):
                raise RuntimeError("Candidate statistics payload digest changed")
            names = tuple(metadata["tensor_names"])
            if expected_names is not None and names != tuple(sorted(expected_names)):
                return None
            with safe_open(stats_path, framework="pt", device="cpu") as file:
                if tuple(sorted(file.keys())) != names:
                    raise RuntimeError("Candidate statistics tensor names changed")
                required = int(metadata["tensor_bytes"])
                if required > self.max_entry_bytes:
                    raise MemoryError("Candidate statistics entry exceeds host budget")
                with HostMemoryBudget().reserve("candidate statistics", required):
                    result = {name: file.get_tensor(name).clone() for name in names}
                for name, tensor in result.items():
                    expected = metadata["tensor_metadata"][name]
                    if (
                        list(tensor.shape) != expected["shape"]
                        or str(tensor.dtype) != expected["dtype"]
                    ):
                        raise RuntimeError(
                            "Candidate statistics tensor metadata changed"
                        )
                return result
        except (OSError, KeyError, TypeError, ValueError, SafetensorError) as error:
            raise RuntimeError("Cannot read committed candidate statistics") from error

    def put(
        self,
        key: CandidateStatisticsKey,
        statistics: Mapping[str, torch.Tensor],
    ) -> None:
        if not statistics:
            raise ValueError("Candidate statistics cannot be empty")
        if any(not isinstance(tensor, torch.Tensor) for tensor in statistics.values()):
            raise TypeError("Candidate statistics values must be tensors")
        required = sum(tensor.nbytes for tensor in statistics.values())
        if required > self.max_entry_bytes:
            raise MemoryError("Candidate statistics entry exceeds host budget")
        with self.root.joinpath(".cache.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            self._put_locked(key, statistics, required)

    def _put_locked(self, key, statistics, required):
        for pending in self.root.glob(".*.pending"):
            parts = pending.name.split(".")
            if (
                len(parts) == 4
                and len(parts[1]) == 64
                and len(parts[2]) == 32
                and all(
                    character in "0123456789abcdef" for character in parts[1] + parts[2]
                )
                and pending.is_dir()
                and not pending.is_symlink()
            ):
                shutil.rmtree(pending)
        directory = self._directory(key)
        if directory.exists():
            existing = self._get_locked(key)
            if existing is None:
                raise RuntimeError("Candidate statistics entry is incomplete")
            if existing.keys() != statistics.keys() or any(
                not torch.equal(existing[name], tensor.detach().cpu())
                for name, tensor in statistics.items()
            ):
                raise RuntimeError("Candidate statistics are immutable")
            return
        entries = sum(
            path.is_dir() and not path.name.startswith(".")
            for path in self.root.iterdir()
        )
        if entries >= self.max_entries:
            raise OSError("Candidate statistics cache entry budget exceeded")
        overhead = (
            4096
            + len(json.dumps(key.to_dict()).encode())
            + sum(
                4 * len(name.encode()) + 512 + len(str(tuple(tensor.shape)))
                for name, tensor in statistics.items()
            )
        )
        if self.disk_bytes() + required + overhead > self.max_bytes:
            raise OSError("Candidate statistics cache budget exceeded")
        pending = self.root / f".{key.digest}.{uuid.uuid4().hex}.pending"
        pending.mkdir()
        stats_path = pending / "stats.safetensors"
        try:
            with HostMemoryBudget().reserve("candidate statistics snapshot", required):
                tensors = {
                    name: tensor.detach().to("cpu").contiguous()
                    for name, tensor in statistics.items()
                }
                save_file(tensors, stats_path)
                _sync(stats_path)
            metadata = {
                "key": key.to_dict(),
                "tensor_names": sorted(tensors),
                "sha256": _sha256(stats_path),
                "size": stats_path.stat().st_size,
                "tensor_bytes": required,
                "tensor_metadata": {
                    name: {
                        "dtype": str(tensor.dtype),
                        "shape": list(tensor.shape),
                    }
                    for name, tensor in tensors.items()
                },
            }
            _atomic_json(pending / "metadata.json", metadata)
            complete = pending / "COMPLETE"
            complete.write_text(_sha256(pending / "metadata.json") + "\n")
            _sync(complete)
            _sync(pending)
            if self.disk_bytes() > self.max_bytes:
                raise OSError("Candidate statistics cache budget exceeded")
            os.replace(pending, directory)
            _sync(self.root)
        finally:
            shutil.rmtree(pending, ignore_errors=True)

    def invalidate_prefix(self, prefix_fingerprint: str) -> int:
        with self.root.joinpath(".cache.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            return self._invalidate_prefix_locked(prefix_fingerprint)

    def _invalidate_prefix_locked(self, prefix_fingerprint):
        removed = 0
        for directory in self.root.iterdir():
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            try:
                metadata = json.loads(
                    (directory / "metadata.json").read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                continue
            if metadata.get("key", {}).get("prefix_fingerprint") == prefix_fingerprint:
                shutil.rmtree(directory)
                removed += 1
        if removed:
            _sync(self.root)
        return removed

    def disk_bytes(self) -> int:
        return sum(
            path.stat().st_size for path in self.root.rglob("*") if path.is_file()
        )
