"""Bounded storage for activations passed between adjacent targets."""

from __future__ import annotations

import os
import pickle
import shutil
import uuid
from abc import ABC, abstractmethod
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

_PRIMITIVE_TYPES = (type(None), bool, int, float, str, bytes)
_KV_CACHE_NAMES = {
    "cache",
    "key_cache",
    "kv_cache",
    "past_key_value",
    "past_key_values",
    "value_cache",
}


def _check_index(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _contains_kv_cache_name(name: object) -> bool:
    return isinstance(name, str) and name.lower() in _KV_CACHE_NAMES


def _snapshot(value: Any, device: torch.device, path: str = "value") -> Any:
    """Detach supported values and reject opaque or KV-cache objects."""

    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device).clone()
    if isinstance(value, _PRIMITIVE_TYPES):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        values = {}
        for field in fields(value):
            field_value = getattr(value, field.name)
            if _contains_kv_cache_name(field.name) and field_value is not None:
                raise TypeError(
                    f"KV cache field {path}.{field.name} is not supported by "
                    "boundary storage"
                )
            values[field.name] = _snapshot(
                field_value, device, f"{path}.{field.name}"
            )
        try:
            return type(value)(**values)
        except TypeError as error:
            raise TypeError(
                f"Dataclass {type(value).__qualname__} at {path} cannot be rebuilt"
            ) from error
    if isinstance(value, list):
        return [
            _snapshot(item, device, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _snapshot(item, device, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, _PRIMITIVE_TYPES):
                raise TypeError(
                    f"Unsupported mapping key {type(key).__qualname__} at {path}"
                )
            if _contains_kv_cache_name(key) and item is not None:
                raise TypeError(
                    f"KV cache field {path}.{key} is not supported by boundary storage"
                )
            result[key] = _snapshot(item, device, f"{path}.{key}")
        return result
    raise TypeError(
        f"Unsupported boundary value type {type(value).__qualname__} at {path}"
    )


def _move(value: Any, device: torch.device) -> Any:
    """Rebuild a stored value with tensors moved to the consumer device."""

    if isinstance(value, torch.Tensor):
        return value.to(device=device).clone()
    if isinstance(value, _PRIMITIVE_TYPES):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{
                field.name: _move(getattr(value, field.name), device)
                for field in fields(value)
            }
        )
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    raise TypeError(f"Stored boundary value has unsupported type {type(value)!r}")


def _tensor_bytes(value: Any, seen: set[int] | None = None) -> int:
    seen = seen if seen is not None else set()
    if isinstance(value, torch.Tensor):
        storage = value.untyped_storage()
        identity = id(storage)
        if identity in seen:
            return 0
        seen.add(identity)
        return storage.nbytes()
    if is_dataclass(value) and not isinstance(value, type):
        return sum(
            _tensor_bytes(getattr(value, field.name), seen)
            for field in fields(value)
        )
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item, seen) for item in value)
    if isinstance(value, dict):
        return sum(_tensor_bytes(item, seen) for item in value.values())
    return 0


class BoundaryActivationStore(ABC):
    """Common batch-oriented interface for adjacent target activations."""

    @abstractmethod
    def put(self, boundary: int, batch: int, value: Any) -> None:
        """Store one complete batch, replacing an existing batch atomically."""

    @abstractmethod
    def get(
        self, boundary: int, batch: int, *, device: torch.device | str = "cpu"
    ) -> Any:
        """Read one batch and move its tensors to the requested device."""

    @abstractmethod
    def batch_indices(self, boundary: int) -> tuple[int, ...]:
        """Return committed batch indices for a boundary."""

    @abstractmethod
    def delete(self, boundary: int) -> None:
        """Delete a consumed boundary and all of its batches."""

    @abstractmethod
    def contains(self, boundary: int, batch: int) -> bool:
        """Return whether a complete batch is available."""

    def iter_boundary(
        self, boundary: int, *, device: torch.device | str = "cpu"
    ) -> Iterator[tuple[int, Any]]:
        for batch in self.batch_indices(boundary):
            yield batch, self.get(boundary, batch, device=device)


class InMemoryBoundaryActivationStore(BoundaryActivationStore):
    """Keep detached activation snapshots on a configured device."""

    def __init__(self, storage_device: torch.device | str = "cpu"):
        self.storage_device = torch.device(storage_device)
        if self.storage_device.type == "meta":
            raise ValueError("Boundary activations cannot be stored on meta")
        self._boundaries: dict[int, dict[int, Any]] = {}

    def put(self, boundary: int, batch: int, value: Any) -> None:
        _check_index(boundary, "boundary")
        _check_index(batch, "batch")
        snapshot = _snapshot(value, self.storage_device)
        self._boundaries.setdefault(boundary, {})[batch] = snapshot

    def get(
        self, boundary: int, batch: int, *, device: torch.device | str = "cpu"
    ) -> Any:
        _check_index(boundary, "boundary")
        _check_index(batch, "batch")
        try:
            value = self._boundaries[boundary][batch]
        except KeyError as error:
            raise KeyError(
                f"Boundary {boundary}, batch {batch} is not committed"
            ) from error
        return _move(value, torch.device(device))

    def batch_indices(self, boundary: int) -> tuple[int, ...]:
        _check_index(boundary, "boundary")
        return tuple(sorted(self._boundaries.get(boundary, {})))

    def delete(self, boundary: int) -> None:
        _check_index(boundary, "boundary")
        self._boundaries.pop(boundary, None)

    def contains(self, boundary: int, batch: int) -> bool:
        _check_index(boundary, "boundary")
        _check_index(batch, "batch")
        return batch in self._boundaries.get(boundary, {})

    def tensor_bytes(self) -> int:
        return sum(
            _tensor_bytes(value)
            for batches in self._boundaries.values()
            for value in batches.values()
        )


class DiskBoundaryActivationStore(BoundaryActivationStore):
    """Store each detached batch in an atomically published local file.

    Files are private intermediate artifacts and must only be loaded from a trusted
    run directory because PyTorch serialization may reconstruct dataclass objects.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _boundary_dir(self, boundary: int) -> Path:
        _check_index(boundary, "boundary")
        return self.root / f"boundary-{boundary:05d}"

    def _batch_path(self, boundary: int, batch: int) -> Path:
        _check_index(batch, "batch")
        return self._boundary_dir(boundary) / f"batch-{batch:05d}.pt"

    def put(self, boundary: int, batch: int, value: Any) -> None:
        path = self._batch_path(boundary, batch)
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = _snapshot(value, torch.device("cpu"))
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            torch.save(snapshot, temporary)
            with temporary.open("rb") as file:
                os.fsync(file.fileno())
            os.replace(temporary, path)
            self._sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def get(
        self, boundary: int, batch: int, *, device: torch.device | str = "cpu"
    ) -> Any:
        path = self._batch_path(boundary, batch)
        if not path.is_file():
            raise KeyError(f"Boundary {boundary}, batch {batch} is not committed")
        try:
            value = torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
            raise RuntimeError(
                f"Cannot read committed boundary batch {path}"
            ) from error
        return _move(value, torch.device(device))

    def batch_indices(self, boundary: int) -> tuple[int, ...]:
        directory = self._boundary_dir(boundary)
        if not directory.is_dir():
            return ()
        batches = []
        for path in directory.glob("batch-*.pt"):
            suffix = path.stem.removeprefix("batch-")
            if suffix.isdigit():
                batches.append(int(suffix))
        return tuple(sorted(batches))

    def delete(self, boundary: int) -> None:
        directory = self._boundary_dir(boundary)
        if directory.exists():
            shutil.rmtree(directory)
            self._sync_directory(self.root)

    def contains(self, boundary: int, batch: int) -> bool:
        return self._batch_path(boundary, batch).is_file()

    def disk_bytes(self) -> int:
        return sum(
            path.stat().st_size
            for path in self.root.rglob("*")
            if path.is_file() and not path.name.endswith(".tmp")
        )

    @staticmethod
    def _sync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
