"""Transactional storage for per-target streaming calibration statistics."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

from .manifest import StreamingRunManifest, TargetStatisticsMetadata
from .validation import validate_manifest_compatibility

MANIFEST_FILE = "manifest.json"
RECIPE_FILE = "recipe.json"
TARGETS_FILE = "targets.json"
METADATA_FILE = "metadata.json"
STATISTICS_FILE = "stats.safetensors"
COMPLETE_FILE = "COMPLETE"


def _atomic_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _sync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_path(path)
    try:
        with temporary.open("wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Any) -> None:
    content = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    _atomic_write_bytes(path, content)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


class ArtifactStore:
    """Own the durable boundary between streaming compression stages."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.statistics_dir = self.root / "statistics"

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_FILE

    def initialize(
        self,
        manifest: StreamingRunManifest,
        *,
        normalized_recipe: Mapping[str, Any] | Sequence[Any],
        targets: Sequence[str],
    ) -> None:
        """Create a store or validate that an existing store can be resumed."""

        if self.manifest_path.exists():
            validate_manifest_compatibility(self.load_manifest(), manifest)
            if _read_json(self.root / RECIPE_FILE) != normalized_recipe:
                raise ValueError("Stored normalized recipe content does not match")
            if _read_json(self.root / TARGETS_FILE) != list(targets):
                raise ValueError("Stored target list does not match")
            return

        self.statistics_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.root / RECIPE_FILE, normalized_recipe)
        _atomic_write_json(self.root / TARGETS_FILE, list(targets))
        # The manifest is the commit record for initialization and is written last.
        _atomic_write_json(self.manifest_path, manifest.to_dict())

    def load_manifest(self) -> StreamingRunManifest:
        return StreamingRunManifest.from_dict(_read_json(self.manifest_path))

    def target_dir(self, target_index: int) -> Path:
        if target_index < 0:
            raise ValueError("target_index must be non-negative")
        return self.statistics_dir / f"target-{target_index:05d}"

    def is_target_complete(self, target_index: int) -> bool:
        directory = self.target_dir(target_index)
        required = (METADATA_FILE, STATISTICS_FILE, COMPLETE_FILE)
        if not all((directory / name).is_file() for name in required):
            return False
        try:
            metadata = self.load_target_metadata(target_index)
            if not metadata.completed:
                return False
            with safe_open(
                directory / STATISTICS_FILE, framework="pt", device="cpu"
            ) as file:
                return tuple(sorted(file.keys())) == tuple(
                    sorted(metadata.tensor_names)
                )
        except (OSError, ValueError, RuntimeError, SafetensorError):
            return False

    def commit_target(
        self,
        metadata: TargetStatisticsMetadata,
        statistics: Mapping[str, torch.Tensor],
    ) -> None:
        """Atomically commit tensors, metadata, then the completion marker."""

        if not self.manifest_path.is_file():
            raise RuntimeError("ArtifactStore must be initialized before committing")
        if metadata.target_index < 0:
            raise ValueError("target_index must be non-negative")
        if not statistics:
            raise ValueError("Target statistics cannot be empty")

        tensor_names = tuple(sorted(statistics))
        declared_names = tuple(sorted(metadata.tensor_names))
        if declared_names != tensor_names:
            raise ValueError(
                "Metadata tensor_names do not match statistics: "
                f"declared={declared_names}, actual={tensor_names}"
            )

        tensors = {}
        for name, tensor in statistics.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Statistic {name!r} is not a torch.Tensor")
            tensors[name] = tensor.detach().to("cpu").contiguous()

        directory = self.target_dir(metadata.target_index)
        directory.mkdir(parents=True, exist_ok=True)
        complete_path = directory / COMPLETE_FILE
        complete_path.unlink(missing_ok=True)

        stats_path = directory / STATISTICS_FILE
        temporary_stats = _atomic_path(stats_path)
        try:
            save_file(tensors, temporary_stats)
            with temporary_stats.open("rb") as file:
                os.fsync(file.fileno())
            os.replace(temporary_stats, stats_path)
            _sync_directory(directory)
        finally:
            temporary_stats.unlink(missing_ok=True)

        completed_metadata = replace(metadata, completed=True)
        _atomic_write_json(
            directory / METADATA_FILE, completed_metadata.to_dict()
        )
        _atomic_write_bytes(complete_path, b"")

    def load_target_metadata(
        self, target_index: int
    ) -> TargetStatisticsMetadata:
        path = self.target_dir(target_index) / METADATA_FILE
        return TargetStatisticsMetadata.from_dict(_read_json(path))

    def load_target_statistics(
        self, target_index: int
    ) -> dict[str, torch.Tensor]:
        if not self.is_target_complete(target_index):
            raise RuntimeError(f"Target {target_index} is not completely committed")
        path = self.target_dir(target_index) / STATISTICS_FILE
        with safe_open(path, framework="pt", device="cpu") as file:
            return {name: file.get_tensor(name) for name in file.keys()}
