"""Versioned manifest types for resumable streaming compression runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, TypeVar

CURRENT_SCHEMA_VERSION = 1

_T = TypeVar("_T")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def fingerprint_json(value: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible value."""

    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_sha256(value: str, field_name: str) -> None:
    invalid_character = any(
        character not in "0123456789abcdef" for character in value
    )
    if len(value) != 64 or invalid_character:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _from_mapping(cls: type[_T], value: Mapping[str, Any]) -> _T:
    allowed = {field.name for field in fields(cls)}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unexpected fields for {cls.__name__}: {sorted(unknown)}")
    return cls(**value)


@dataclass(frozen=True)
class CheckpointShardInfo:
    name: str
    size: int
    mtime_ns: int
    sha256: str | None = None


@dataclass(frozen=True)
class SourceCheckpointInfo:
    """Content identity for a local safetensors checkpoint.

    ``location`` is diagnostic only. Compatibility is determined by the config,
    index, and shard fingerprints, so moving an unchanged checkpoint is safe.
    """

    location: str
    config_sha256: str
    index_sha256: str | None
    shards: tuple[CheckpointShardInfo, ...]
    strict: bool = False

    @property
    def content_fingerprint(self) -> str:
        value = asdict(self)
        value.pop("location")
        return fingerprint_json(value)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceCheckpointInfo:
        data = dict(value)
        if not isinstance(data.get("shards"), (list, tuple)):
            raise ValueError("SourceCheckpointInfo.shards must be a sequence")
        data["shards"] = tuple(
            _from_mapping(CheckpointShardInfo, shard) for shard in data["shards"]
        )
        return _from_mapping(cls, data)


@dataclass(frozen=True)
class RecipeInfo:
    normalized_sha256: str


@dataclass(frozen=True)
class CalibrationInfo:
    dataset_fingerprint: str
    num_samples: int
    max_seq_length: int | None = None
    seed: int | None = None


@dataclass(frozen=True)
class SequentialInfo:
    targets: tuple[str, ...]
    targets_per_subgraph: int = 1
    calibration_mode: str = "reference"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SequentialInfo:
        data = dict(value)
        data["targets"] = tuple(data["targets"])
        return _from_mapping(cls, data)


@dataclass(frozen=True)
class MaterializerInfo:
    identifier: str
    config_sha256: str


@dataclass(frozen=True)
class SoftwareInfo:
    versions: tuple[tuple[str, str], ...]

    @classmethod
    def from_versions(cls, versions: Mapping[str, str]) -> SoftwareInfo:
        return cls(tuple(sorted(versions.items())))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SoftwareInfo:
        versions = tuple(
            (str(name), str(version)) for name, version in value["versions"]
        )
        return cls(versions=versions)


@dataclass(frozen=True)
class StreamingRunManifest:
    source: SourceCheckpointInfo
    recipe: RecipeInfo
    calibration: CalibrationInfo
    sequential: SequentialInfo
    materializer: MaterializerInfo
    software: SoftwareInfo
    schema_version: int = CURRENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported streaming artifact schema version "
                f"{self.schema_version!r}; expected {CURRENT_SCHEMA_VERSION}"
            )
        _require_sha256(self.recipe.normalized_sha256, "recipe.normalized_sha256")
        _require_sha256(
            self.materializer.config_sha256, "materializer.config_sha256"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StreamingRunManifest:
        expected_fields = {field.name for field in fields(cls)}
        unknown = set(value) - expected_fields
        missing = expected_fields - set(value)
        if unknown:
            raise ValueError(
                f"Unexpected fields for StreamingRunManifest: {sorted(unknown)}"
            )
        if missing:
            raise ValueError(
                f"Missing fields for StreamingRunManifest: {sorted(missing)}"
            )
        schema_version = value.get("schema_version")
        if schema_version != CURRENT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported streaming artifact schema version "
                f"{schema_version!r}; expected {CURRENT_SCHEMA_VERSION}"
            )

        return cls(
            source=SourceCheckpointInfo.from_dict(value["source"]),
            recipe=_from_mapping(RecipeInfo, value["recipe"]),
            calibration=_from_mapping(CalibrationInfo, value["calibration"]),
            sequential=SequentialInfo.from_dict(value["sequential"]),
            materializer=_from_mapping(MaterializerInfo, value["materializer"]),
            software=SoftwareInfo.from_dict(value["software"]),
            schema_version=schema_version,
        )


@dataclass(frozen=True)
class TargetStatisticsMetadata:
    target_name: str
    target_index: int
    algorithms: tuple[str, ...]
    tensor_names: tuple[str, ...]
    source_tensor_fingerprints: tuple[tuple[str, str], ...]
    completed: bool

    def __post_init__(self) -> None:
        if self.target_index < 0:
            raise ValueError("target_index must be non-negative")
        if len(set(self.tensor_names)) != len(self.tensor_names):
            raise ValueError("tensor_names must be unique")
        source_names = [name for name, _ in self.source_tensor_fingerprints]
        if len(set(source_names)) != len(source_names):
            raise ValueError("source_tensor_fingerprints names must be unique")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TargetStatisticsMetadata:
        data = dict(value)
        data["algorithms"] = tuple(data["algorithms"])
        data["tensor_names"] = tuple(data["tensor_names"])
        data["source_tensor_fingerprints"] = tuple(
            (str(name), str(digest))
            for name, digest in data["source_tensor_fingerprints"]
        )
        return _from_mapping(cls, data)


def fingerprint_checkpoint(
    checkpoint_dir: str | Path, *, strict: bool = False
) -> SourceCheckpointInfo:
    """Fingerprint a local safetensors checkpoint without loading tensors.

    The default mode hashes config/index metadata and records shard stat data.
    Strict mode additionally hashes every shard, which is safer but expensive for
    multi-terabyte checkpoints.
    """

    root = Path(checkpoint_dir).expanduser().resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint config: {config_path}")

    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = _load_json(index_path)
        weight_map = index.get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
        index_sha256 = fingerprint_json(index)
    else:
        shard_names = sorted(path.name for path in root.glob("*.safetensors"))
        index_sha256 = None

    if not shard_names:
        raise FileNotFoundError(f"No safetensors weights found in {root}")

    shards = []
    for name in shard_names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint index references missing shard: {path}"
            )
        stat = path.stat()
        shards.append(
            CheckpointShardInfo(
                name=name,
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                sha256=_hash_file(path) if strict else None,
            )
        )

    return SourceCheckpointInfo(
        location=str(root),
        config_sha256=fingerprint_json(_load_json(config_path)),
        index_sha256=index_sha256,
        shards=tuple(shards),
        strict=strict,
    )
