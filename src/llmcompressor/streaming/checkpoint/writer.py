"""Transactional, tensor-at-a-time checkpoint staging."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import uuid
from collections import defaultdict
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from llmcompressor.streaming.artifacts import ArtifactCompatibilityError

__all__ = [
    "DirectSafetensorsWriter",
    "StreamingCheckpointWriter",
    "TensorRecord",
]


_SAFETENSORS_DTYPES = {
    torch.bool: "BOOL",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.uint8: "U8",
}
for _dtype, _name in (
    ("float8_e4m3fn", "F8_E4M3"),
    ("float8_e5m2", "F8_E5M2"),
):
    if hasattr(torch, _dtype):
        _SAFETENSORS_DTYPES[getattr(torch, _dtype)] = _name


def _sync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class TensorRecord:
    """A durable tensor payload owned by a committed transaction."""

    name: str
    output_shard: str
    payload: Path
    dtype: str
    shape: tuple[int, ...]
    size: int
    sha256: str

    def to_dict(self, *, root: Path) -> dict[str, Any]:
        return {
            "name": self.name,
            "output_shard": self.output_shard,
            "payload": str(self.payload.relative_to(root)),
            "dtype": self.dtype,
            "shape": list(self.shape),
            "size": self.size,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, root: Path) -> TensorRecord:
        return cls(
            name=str(value["name"]),
            output_shard=str(value["output_shard"]),
            payload=root / str(value["payload"]),
            dtype=str(value["dtype"]),
            shape=tuple(int(item) for item in value["shape"]),
            size=int(value["size"]),
            sha256=str(value["sha256"]),
        )


class _Transaction(AbstractContextManager["_Transaction"]):
    def __init__(
        self,
        writer: StreamingCheckpointWriter,
        transaction_id: str,
    ):
        self.writer = writer
        self.transaction_id = transaction_id
        self.pending = writer.transactions_dir / f"{transaction_id}.pending"
        self.committed = writer.transactions_dir / transaction_id
        self.records: dict[str, TensorRecord] = {}
        self.quantized_modules: dict[str, str] = {}
        self.omitted_tied_weights: dict[str, str] = {}
        self.config_updates: dict[str, Any] = {}
        self._did_commit = False

    def __enter__(self) -> _Transaction:
        if self.committed.exists():
            raise FileExistsError(
                f"Transaction {self.transaction_id!r} is already committed"
            )
        shutil.rmtree(self.pending, ignore_errors=True)
        (self.pending / "tensors").mkdir(parents=True)
        return self

    def write_tensor(
        self, name: str, tensor: torch.Tensor, *, output_shard: str
    ) -> TensorRecord:
        if not name or name in self.records:
            raise ValueError(f"Duplicate or empty tensor name {name!r}")
        if not output_shard.endswith(".safetensors"):
            raise ValueError(
                f"Output shard must end in .safetensors: {output_shard!r}"
            )
        value = tensor.detach().to("cpu").contiguous()
        try:
            dtype_name = _SAFETENSORS_DTYPES[value.dtype]
        except KeyError as error:
            raise TypeError(f"Unsupported output dtype {value.dtype}") from error
        payload = self.pending / "tensors" / f"{len(self.records):08d}.bin"
        digest = hashlib.sha256()
        # NumPy cannot expose every dtype supported by safetensors (notably
        # bfloat16). Viewing the contiguous tensor as bytes keeps the exact
        # storage representation without a full-size dtype conversion.
        storage = memoryview(
            value.reshape(-1).view(torch.uint8).numpy()
        ).cast("B")
        with payload.open("wb") as file:
            file.write(storage)
            digest.update(storage)
            file.flush()
            os.fsync(file.fileno())
        record = TensorRecord(
            name=name,
            output_shard=output_shard,
            payload=payload,
            dtype=dtype_name,
            shape=tuple(value.shape),
            size=payload.stat().st_size,
            sha256=digest.hexdigest(),
        )
        self.records[name] = record
        return record

    def mark_quantized(self, module_name: str, format_name: str) -> None:
        previous = self.quantized_modules.setdefault(module_name, format_name)
        if previous != format_name:
            raise ValueError(
                f"Conflicting formats for quantized module {module_name!r}"
            )

    def omit_tied_weight(self, alias: str, canonical: str) -> None:
        previous = self.omitted_tied_weights.setdefault(alias, canonical)
        if previous != canonical:
            raise ValueError(f"Conflicting canonical tensor for {alias!r}")

    def update_config(self, updates: Mapping[str, Any]) -> None:
        overlap = self.config_updates.keys() & updates.keys()
        if any(self.config_updates[key] != updates[key] for key in overlap):
            raise ValueError(f"Conflicting config updates: {sorted(overlap)}")
        self.config_updates.update(updates)

    def write_boundary(
        self, batch_index: int, value: Any, *, boundary: int
    ) -> None:
        """Stage one next-boundary batch inside this transaction."""

        if batch_index < 0 or boundary < 0:
            raise ValueError("boundary and batch_index must be non-negative")
        if not hasattr(self, "boundaries"):
            self.boundaries: dict[int, Path] = {}
        if batch_index in self.boundaries:
            raise ValueError(f"Duplicate boundary batch {batch_index}")
        path = self.pending / "boundaries" / f"batch-{batch_index:05d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value, path)
        self.boundaries[batch_index] = path
        self.boundary_index = boundary

    def commit(self) -> None:
        if self._did_commit:
            raise RuntimeError("Transaction has already been committed")
        if not self.records and not self.config_updates:
            raise ValueError("Transaction has no output delta")
        metadata = {
            "schema_version": 1,
            "transaction_id": self.transaction_id,
            "run_fingerprint": self.writer.run_fingerprint,
            "records": [
                record.to_dict(root=self.pending)
                for record in self.records.values()
            ],
            "quantized_modules": self.quantized_modules,
            "omitted_tied_weights": self.omitted_tied_weights,
            "config_updates": self.config_updates,
            "boundary": (
                {
                    "index": self.boundary_index,
                    "batches": [
                        path.relative_to(self.pending).as_posix()
                        for _, path in sorted(self.boundaries.items())
                    ],
                }
                if hasattr(self, "boundaries")
                else None
            ),
        }
        _atomic_json(self.pending / "metadata.json", metadata)
        _sync_directory(self.pending / "tensors")
        _sync_directory(self.pending)
        os.replace(self.pending, self.committed)
        _sync_directory(self.writer.transactions_dir)
        self._did_commit = True

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._did_commit:
            shutil.rmtree(self.pending, ignore_errors=True)
        return None


class StreamingCheckpointWriter:
    """Collect subgraph output deltas and assemble output shards later.

    A transaction may contain tensors for multiple output shards. Conversely, a
    shard may receive tensors from multiple transactions. This keeps the execution
    unit (a subgraph) independent from the checkpoint file layout.
    """

    def __init__(self, root: str | Path, *, run_fingerprint: str):
        if not run_fingerprint:
            raise ValueError("run_fingerprint cannot be empty")
        self.root = Path(root)
        self.run_fingerprint = run_fingerprint
        self.transactions_dir = self.root / "transactions"
        self.shards_dir = self.root / "shards"
        self.states_dir = self.root / "state"
        self.transactions_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.root / "writer.json"
        if manifest.exists():
            value = json.loads(manifest.read_text(encoding="utf-8"))
            if value.get("run_fingerprint") != run_fingerprint:
                raise ArtifactCompatibilityError(
                    "Checkpoint staging belongs to a different quantization run"
                )
        else:
            _atomic_json(
                manifest,
                {"schema_version": 1, "run_fingerprint": run_fingerprint},
            )

    def transaction(self, transaction_id: str) -> _Transaction:
        if not transaction_id or transaction_id.endswith(".pending"):
            raise ValueError(f"Invalid transaction id {transaction_id!r}")
        if Path(transaction_id).name != transaction_id:
            raise ValueError("transaction_id must not contain path separators")
        return _Transaction(self, transaction_id)

    def is_transaction_complete(self, transaction_id: str) -> bool:
        path = self.transactions_dir / transaction_id / "metadata.json"
        if not path.is_file():
            return False
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata.get("run_fingerprint") != self.run_fingerprint:
                raise ArtifactCompatibilityError(
                    f"Transaction {transaction_id!r} belongs to a different run"
                )
            for record in self._records(metadata, path.parent):
                if not self._validate_payload(record):
                    return False
            boundary = metadata.get("boundary")
            if boundary is not None:
                for relative_path in boundary.get("batches", []):
                    boundary_path = path.parent / relative_path
                    if not boundary_path.is_file():
                        return False
                    torch.load(
                        boundary_path, map_location="cpu", weights_only=False
                    )
            return True
        except (
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            EOFError,
            pickle.UnpicklingError,
            json.JSONDecodeError,
        ):
            return False

    def has_transaction(self, transaction_id: str) -> bool:
        """Return whether a committed transaction directory exists.

        This intentionally does not validate the transaction. Callers can use it
        together with :meth:`is_transaction_complete` to distinguish an absent
        transaction (safe to execute) from a corrupt committed transaction (must
        not be silently overwritten).
        """

        return (self.transactions_dir / transaction_id).is_dir()

    def committed_metadata(self) -> list[dict[str, Any]]:
        result = []
        for directory in sorted(self.transactions_dir.iterdir()):
            if not directory.is_dir() or directory.name.endswith(".pending"):
                continue
            path = directory / "metadata.json"
            if not self.is_transaction_complete(directory.name):
                raise RuntimeError(
                    f"Committed transaction {directory.name!r} is corrupt"
                )
            result.append(json.loads(path.read_text(encoding="utf-8")))
        return result

    def committed_boundary(
        self, transaction_id: str
    ) -> tuple[int, list[Any]] | None:
        """Load a boundary committed with a subgraph transaction."""

        if not self.is_transaction_complete(transaction_id):
            return None
        root = self.transactions_dir / transaction_id
        metadata = json.loads((root / "metadata.json").read_text())
        boundary = metadata.get("boundary")
        if boundary is None:
            return None
        values = [
            torch.load(root / path, map_location="cpu", weights_only=False)
            for path in boundary["batches"]
        ]
        return int(boundary["index"]), values

    def assemble_shards(self) -> dict[str, dict[str, Any]]:
        metadata_values = self.committed_metadata()
        records_by_shard: dict[str, dict[str, TensorRecord]] = defaultdict(dict)
        quantized_modules: dict[str, str] = {}
        omitted_tied_weights: dict[str, str] = {}
        for metadata in metadata_values:
            root = self.transactions_dir / metadata["transaction_id"]
            for record in self._records(metadata, root):
                existing = records_by_shard[record.output_shard].get(record.name)
                if existing is not None:
                    raise ValueError(
                        f"Tensor {record.name!r} is produced by multiple transactions"
                    )
                records_by_shard[record.output_shard][record.name] = record
            self._merge_unique(
                quantized_modules, metadata.get("quantized_modules", {})
            )
            self._merge_unique(
                omitted_tied_weights, metadata.get("omitted_tied_weights", {})
            )

        states = {}
        for shard_name, records in sorted(records_by_shard.items()):
            output = self.shards_dir / shard_name
            total_size = self._assemble_safetensors(output, records)
            state = {
                "completed": True,
                "output_shard": shard_name,
                "source_shard": shard_name,
                "quantized_modules": sorted(quantized_modules),
                "module_formats": quantized_modules,
                "omitted_tied_weights": omitted_tied_weights,
                "run_fingerprint": self.run_fingerprint,
                "tensor_names": sorted(records),
                "total_size": total_size,
            }
            _atomic_json(self.states_dir / f"{shard_name}.json", state)
            states[shard_name] = state
        return states

    @staticmethod
    def _records(metadata: Mapping[str, Any], root: Path) -> list[TensorRecord]:
        return [
            TensorRecord.from_dict(value, root=root)
            for value in metadata.get("records", [])
        ]

    @staticmethod
    def _validate_payload(record: TensorRecord) -> bool:
        if not record.payload.is_file() or record.payload.stat().st_size != record.size:
            return False
        digest = hashlib.sha256()
        with record.payload.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == record.sha256

    @staticmethod
    def _merge_unique(target: dict[str, str], values: Mapping[str, str]) -> None:
        for key, value in values.items():
            previous = target.setdefault(key, value)
            if previous != value:
                raise ValueError(f"Conflicting output metadata for {key!r}")

    @staticmethod
    def _assemble_safetensors(
        path: Path, records: Mapping[str, TensorRecord]
    ) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        offset = 0
        header = {}
        for name in sorted(records):
            record = records[name]
            header[name] = {
                "dtype": record.dtype,
                "shape": list(record.shape),
                "data_offsets": [offset, offset + record.size],
            }
            offset += record.size
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        encoded += b" " * ((8 - len(encoded) % 8) % 8)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as output:
                output.write(len(encoded).to_bytes(8, byteorder="little"))
                output.write(encoded)
                for name in sorted(records):
                    with records[name].payload.open("rb") as payload:
                        shutil.copyfileobj(payload, output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            _sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return offset


class DirectSafetensorsWriter:
    """Write each completed execution unit as its final safetensors shard.

    Unlike :class:`StreamingCheckpointWriter`, this writer has no raw tensor
    payload layer and no later shard assembly. State files contain only the
    information needed to build the final model index and compression config.
    """

    def __init__(self, root: str | Path, *, run_fingerprint: str):
        self.root = Path(root)
        self.run_fingerprint = run_fingerprint
        # ``root`` is the directory that will become output_dir. Shards are
        # therefore written at their final relative paths and never copied or
        # reassembled during publication.
        self.shards_dir = self.root
        self.states_dir = self.root / ".streaming-state"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.states_dir.mkdir(parents=True, exist_ok=True)

    def write_shard(
        self,
        shard_id: str,
        tensors: Mapping[str, torch.Tensor],
        *,
        quantized_modules: Mapping[str, str] | None = None,
        omitted_tied_weights: Mapping[str, str] | None = None,
    ) -> Path:
        if not shard_id or Path(shard_id).name != shard_id:
            raise ValueError(f"Invalid shard id {shard_id!r}")
        if not tensors:
            raise ValueError("A direct shard must contain at least one tensor")
        shard_name = f"model-{shard_id}.safetensors"
        output = self.shards_dir / shard_name
        state_path = self.states_dir / f"{shard_name}.json"
        if output.exists() or state_path.exists():
            raise FileExistsError(f"Direct shard {shard_id!r} already exists")
        values = {
            name: tensor.detach().to("cpu").contiguous()
            for name, tensor in tensors.items()
        }
        temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
        try:
            save_file(values, temporary)
            with temporary.open("rb") as file:
                os.fsync(file.fileno())
            os.replace(temporary, output)
            _sync_directory(self.shards_dir)
        finally:
            temporary.unlink(missing_ok=True)
        total_size = sum(
            value.numel() * value.element_size() for value in values.values()
        )
        _atomic_json(
            state_path,
            {
                "completed": True,
                "output_shard": shard_name,
                "quantized_modules": sorted(quantized_modules or {}),
                "module_formats": dict(quantized_modules or {}),
                "omitted_tied_weights": dict(omitted_tied_weights or {}),
                "run_fingerprint": self.run_fingerprint,
                "tensor_names": sorted(values),
                "total_size": total_size,
            },
        )
        return output
