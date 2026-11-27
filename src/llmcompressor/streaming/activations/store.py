"""Bounded storage for activations passed between adjacent targets."""

from __future__ import annotations

import os
import pickle
import shutil
import uuid
from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import torch

if TYPE_CHECKING:
    from llmcompressor.streaming.loading.host_memory import (
        HostMemoryBudget,
        HostMemoryReservation,
    )

_PRIMITIVE_TYPES = (type(None), bool, int, float, str, bytes, torch.dtype)
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


def _snapshot(
    value: Any,
    device: torch.device,
    path: str = "value",
    *,
    memo: dict[int, torch.Tensor] | None = None,
    content_candidates: dict[tuple, list[torch.Tensor]] | None = None,
    identity_candidates: dict[int, torch.Tensor] | None = None,
) -> Any:
    """Detach supported values and reject opaque or KV-cache objects."""

    memo = memo if memo is not None else {}
    if isinstance(value, torch.Tensor):
        identity = id(value)
        if identity in memo:
            return memo[identity]
        if identity_candidates is not None:
            reused = identity_candidates.get(identity)
            if reused is not None:
                memo[identity] = reused
                return reused
        detached = value.detach()
        snapshot = detached.to(device=device)
        if snapshot.device == detached.device:
            snapshot = snapshot.clone()
        if content_candidates is not None:
            signature = (snapshot.dtype, tuple(snapshot.shape))
            for candidate in content_candidates.get(signature, ()):
                if torch.equal(snapshot, candidate):
                    memo[identity] = candidate
                    return candidate
            content_candidates.setdefault(signature, []).append(snapshot)
        memo[identity] = snapshot
        return snapshot
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
                field_value,
                device,
                f"{path}.{field.name}",
                memo=memo,
                content_candidates=content_candidates,
                identity_candidates=identity_candidates,
            )
        try:
            return type(value)(**values)
        except TypeError as error:
            raise TypeError(
                f"Dataclass {type(value).__qualname__} at {path} cannot be rebuilt"
            ) from error
    if isinstance(value, list):
        return [
            _snapshot(
                item,
                device,
                f"{path}[{index}]",
                memo=memo,
                content_candidates=content_candidates,
                identity_candidates=identity_candidates,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _snapshot(
                item,
                device,
                f"{path}[{index}]",
                memo=memo,
                content_candidates=content_candidates,
                identity_candidates=identity_candidates,
            )
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
            result[key] = _snapshot(
                item,
                device,
                f"{path}.{key}",
                memo=memo,
                content_candidates=content_candidates,
                identity_candidates=identity_candidates,
            )
        return result
    raise TypeError(
        f"Unsupported boundary value type {type(value).__qualname__} at {path}"
    )


def _move(
    value: Any,
    device: torch.device,
    memo: dict[int, torch.Tensor] | None = None,
) -> Any:
    """Rebuild a stored value with tensors moved to the consumer device."""

    memo = memo if memo is not None else {}
    if isinstance(value, torch.Tensor):
        identity = id(value)
        if identity not in memo:
            moved = value.to(device=device)
            memo[identity] = moved.clone() if moved.device == value.device else moved
        return memo[identity]
    if isinstance(value, _PRIMITIVE_TYPES):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{
                field.name: _move(getattr(value, field.name), device, memo)
                for field in fields(value)
            }
        )
    if isinstance(value, list):
        return [_move(item, device, memo) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device, memo) for item in value)
    if isinstance(value, dict):
        return {key: _move(item, device, memo) for key, item in value.items()}
    raise TypeError(f"Stored boundary value has unsupported type {type(value)!r}")


def _tensor_bytes(value: Any, seen: set[int] | None = None) -> int:
    seen = seen if seen is not None else set()
    if isinstance(value, torch.Tensor):
        identity = id(value)
        if identity in seen:
            return 0
        seen.add(identity)
        return value.numel() * value.element_size()
    if is_dataclass(value) and not isinstance(value, type):
        return sum(
            _tensor_bytes(getattr(value, field.name), seen) for field in fields(value)
        )
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item, seen) for item in value)
    if isinstance(value, dict):
        return sum(_tensor_bytes(item, seen) for item in value.values())
    return 0


def _tensors(value: Any) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            yield from _tensors(getattr(value, item.name))
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)


@dataclass(frozen=True)
class ExecutionBoundaryMetadata:
    """Metadata for exactly the tensors transferred by an execution read."""

    values: Any
    tensor_bytes: int


class BoundaryActivationStore(ABC):
    """Common batch-oriented interface for adjacent target activations."""

    routing_inputs_are_borrowed = False

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

    @abstractmethod
    def _delete_batch(self, boundary: int, batch: int) -> None:
        """Delete one committed batch after its replacement is durable."""

    def get_for_execution(
        self,
        boundary: int,
        batch: int,
        input_names: Collection[str],
        *,
        device: torch.device | str = "cpu",
    ) -> Any:
        """Read group inputs and aliases, retaining unused values privately.

        Commit partial execution results with consumed_names to preserve carry.
        Reads isolate tensors from the store, including inputs mutated in place.
        """
        return self.get(boundary, batch, device=device)

    def execution_metadata(
        self, boundary: int, batch: int, input_names: Collection[str]
    ) -> ExecutionBoundaryMetadata:
        values = self.get_for_execution(
            boundary, batch, input_names, device="meta"
        )
        return ExecutionBoundaryMetadata(values, _tensor_bytes(values))

    def read_for_routing(
        self, boundary: int, batch: int, input_names: Collection[str]
    ) -> Any:
        """Read CPU inputs for a verified nonmutating routing graph."""
        return self.get_for_execution(boundary, batch, input_names, device="cpu")

    def commit_propagated_batch(
        self,
        source_boundary: int,
        destination_boundary: int,
        batch: int,
        value: Any,
        *,
        consumed_names: Collection[str] | None = None,
        reuse_source_aliases: bool = False,
    ) -> None:
        """Commit a CPU destination before consuming its matching source batch."""

        del reuse_source_aliases
        if source_boundary == destination_boundary:
            raise ValueError("Propagation boundaries must be distinct")
        if consumed_names is not None:
            source = self.get(source_boundary, batch)
            value = {
                **{
                    name: item for name, item in source.items()
                    if name not in consumed_names
                },
                **value,
            }
        self.put(destination_boundary, batch, value)
        self._delete_batch(source_boundary, batch)

    def iter_boundary(
        self, boundary: int, *, device: torch.device | str = "cpu"
    ) -> Iterator[tuple[int, Any]]:
        for batch in self.batch_indices(boundary):
            yield batch, self.get(boundary, batch, device=device)


class InMemoryBoundaryActivationStore(BoundaryActivationStore):
    """Keep detached activation snapshots on a configured device."""

    routing_inputs_are_borrowed = True

    def __init__(
        self,
        storage_device: torch.device | str = "cpu",
        *,
        deduplicate_tensors: bool = False,
        host_budget: HostMemoryBudget | None = None,
    ):
        self.storage_device = torch.device(storage_device)
        if self.storage_device.type != "cpu":
            raise ValueError("Complete activation boundaries must be stored on CPU")
        self.deduplicate_tensors = deduplicate_tensors
        self.host_budget = host_budget
        self._boundaries: dict[int, dict[int, Any]] = {}
        self._reservations: dict[int, dict[int, HostMemoryReservation]] = {}
        self._execution_names: dict[
            tuple[int, int, frozenset[str]], frozenset[str]
        ] = {}

    def put(self, boundary: int, batch: int, value: Any) -> None:
        self._put(boundary, batch, value)

    def _put(
        self,
        boundary: int,
        batch: int,
        value: Any,
        carried: Mapping[str, Any] | None = None,
        reused: Collection[torch.Tensor] | None = None,
    ) -> None:
        _check_index(boundary, "boundary")
        _check_index(batch, "batch")
        complete_value = value if carried is None else {**carried, **value}
        reservation = (
            self.host_budget.reserve(
                f"activation boundary {boundary} batch {batch}",
                _tensor_bytes(complete_value),
            )
            if self.host_budget is not None
            else None
        )
        try:
            candidates = {} if self.deduplicate_tensors else None
            if candidates is not None and carried is not None:
                seen = set()
                for tensor in _tensors(carried):
                    if id(tensor) not in seen:
                        seen.add(id(tensor))
                        signature = (tensor.dtype, tuple(tensor.shape))
                        candidates.setdefault(signature, []).append(tensor)
            snapshot = _snapshot(
                value,
                self.storage_device,
                memo={},
                content_candidates=candidates,
                identity_candidates=(
                    {id(tensor): tensor for tensor in reused}
                    if reused is not None
                    else None
                ),
            )
            if carried is not None:
                snapshot = {**carried, **snapshot}
            if reservation is not None:
                reservation.commit(_tensor_bytes(snapshot))
                reservation.release_reserved(reservation.reserved_bytes)
        except Exception:
            if reservation is not None:
                reservation.close()
            raise
        previous = self._reservations.get(boundary, {}).pop(batch, None)
        self._boundaries.setdefault(boundary, {})[batch] = snapshot
        self._invalidate_execution_names(boundary, batch)
        if reservation is not None:
            self._reservations.setdefault(boundary, {})[batch] = reservation
        if previous is not None:
            previous.close()

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
        return _move(value, torch.device(device), memo={})

    def get_for_execution(
        self,
        boundary: int,
        batch: int,
        input_names: Collection[str],
        *,
        device: torch.device | str = "cpu",
    ) -> Any:
        _check_index(boundary, "boundary")
        _check_index(batch, "batch")
        try:
            value = self._boundaries[boundary][batch]
        except KeyError as error:
            raise KeyError(
                f"Boundary {boundary}, batch {batch} is not committed"
            ) from error
        if not isinstance(value, Mapping):
            return self.get(boundary, batch, device=device)
        selected = self._select_execution_names(boundary, batch, value, input_names)
        return _move(
            {name: item for name, item in value.items() if name in selected},
            torch.device(device),
            memo={},
        )

    def _select_execution_names(self, boundary, batch, value, input_names):
        key = (boundary, batch, frozenset(input_names))
        if key in self._execution_names:
            return self._execution_names[key]
        selected = set(input_names)
        missing = selected - value.keys()
        if missing:
            raise KeyError(f"Missing boundary inputs: {sorted(missing)}")
        identities = {
            name: {id(tensor) for tensor in _tensors(item)}
            for name, item in value.items()
        }
        active = set().union(*(identities[name] for name in selected))
        while True:
            aliases = {
                name for name, tensors in identities.items()
                if name not in selected and tensors & active
            }
            if not aliases:
                break
            selected.update(aliases)
            active.update(set().union(*(identities[name] for name in aliases)))
        result = frozenset(selected)
        self._execution_names[key] = result
        return result

    def read_for_routing(self, boundary, batch, input_names):
        value = self._boundaries[boundary][batch]
        if not isinstance(value, Mapping):
            return super().read_for_routing(boundary, batch, input_names)
        selected = self._select_execution_names(boundary, batch, value, input_names)
        return {name: item for name, item in value.items() if name in selected}

    def _invalidate_execution_names(self, boundary, batch=None):
        for key in tuple(self._execution_names):
            if key[0] == boundary and (batch is None or key[1] == batch):
                del self._execution_names[key]

    def commit_propagated_batch(
        self,
        source_boundary: int,
        destination_boundary: int,
        batch: int,
        value: Any,
        *,
        consumed_names: Collection[str] | None = None,
        reuse_source_aliases: bool = False,
    ) -> None:
        if consumed_names is None:
            return super().commit_propagated_batch(
                source_boundary,
                destination_boundary,
                batch,
                value,
                reuse_source_aliases=reuse_source_aliases,
            )
        if source_boundary == destination_boundary:
            raise ValueError("Propagation boundaries must be distinct")
        _check_index(source_boundary, "source_boundary")
        _check_index(batch, "batch")
        try:
            source = self._boundaries[source_boundary][batch]
        except KeyError as error:
            raise KeyError(
                f"Boundary {source_boundary}, batch {batch} is not committed"
            ) from error
        carried = {
            name: item for name, item in source.items()
            if name not in consumed_names and name not in value
        }
        reused = tuple(_tensors(source)) if reuse_source_aliases else None
        self._put(destination_boundary, batch, value, carried, reused=reused)
        self._delete_batch(source_boundary, batch)

    def batch_indices(self, boundary: int) -> tuple[int, ...]:
        _check_index(boundary, "boundary")
        return tuple(sorted(self._boundaries.get(boundary, {})))

    def delete(self, boundary: int) -> None:
        _check_index(boundary, "boundary")
        self._boundaries.pop(boundary, None)
        self._invalidate_execution_names(boundary)
        for reservation in self._reservations.pop(boundary, {}).values():
            reservation.close()

    def contains(self, boundary: int, batch: int) -> bool:
        _check_index(boundary, "boundary")
        _check_index(batch, "batch")
        return batch in self._boundaries.get(boundary, {})

    def _delete_batch(self, boundary: int, batch: int) -> None:
        _check_index(boundary, "boundary")
        _check_index(batch, "batch")
        self._invalidate_execution_names(boundary, batch)
        batches = self._boundaries.get(boundary)
        if batches is None:
            return
        batches.pop(batch, None)
        if not batches:
            self._boundaries.pop(boundary, None)
        reservations = self._reservations.get(boundary)
        if reservations is not None:
            reservation = reservations.pop(batch, None)
            if reservation is not None:
                reservation.close()
            if not reservations:
                self._reservations.pop(boundary, None)

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

    def __init__(
        self,
        root: str | Path,
        *,
        host_budget: HostMemoryBudget | None = None,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.host_budget = host_budget

    def _boundary_dir(self, boundary: int) -> Path:
        _check_index(boundary, "boundary")
        return self.root / f"boundary-{boundary:05d}"

    def _batch_path(self, boundary: int, batch: int) -> Path:
        _check_index(batch, "batch")
        return self._boundary_dir(boundary) / f"batch-{batch:05d}.pt"

    def put(self, boundary: int, batch: int, value: Any) -> None:
        path = self._batch_path(boundary, batch)
        path.parent.mkdir(parents=True, exist_ok=True)
        reservation = (
            self.host_budget.reserve(
                f"activation boundary {boundary} batch {batch}",
                _tensor_bytes(value),
            )
            if self.host_budget is not None
            else None
        )
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            snapshot = _snapshot(value, torch.device("cpu"))
            if reservation is not None:
                reservation.commit(_tensor_bytes(snapshot))
                reservation.release_reserved(reservation.reserved_bytes)
            torch.save(snapshot, temporary)
            with temporary.open("rb") as file:
                os.fsync(file.fileno())
            os.replace(temporary, path)
            self._sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
            if reservation is not None:
                reservation.close()

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

    def _delete_batch(self, boundary: int, batch: int) -> None:
        path = self._batch_path(boundary, batch)
        path.unlink(missing_ok=True)
        directory = path.parent
        try:
            directory.rmdir()
        except OSError:
            pass
        self._sync_directory(self.root)

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
