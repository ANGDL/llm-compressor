"""Tensor-to-shard mapping for local safetensors checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open

_SAFETENSORS_DTYPES = {
    "BOOL": torch.bool,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
}
for _safetensors_name, _torch_name in (
    ("BF16", "bfloat16"),
    ("F8_E4M3", "float8_e4m3fn"),
    ("F8_E5M2", "float8_e5m2"),
    ("U16", "uint16"),
    ("U32", "uint32"),
    ("U64", "uint64"),
):
    if hasattr(torch, _torch_name):
        _SAFETENSORS_DTYPES[_safetensors_name] = getattr(torch, _torch_name)

# PyTorch has no unsigned E8M0 scalar dtype. Expose its storage as bytes so a
# checkpoint-specific materializer can decode the exponent-only scale values.
_SAFETENSORS_DTYPES["F8_E8M0"] = torch.uint8


@dataclass(frozen=True)
class TensorMetadata:
    """Header-only information about one checkpoint tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    shard: Path


class WeightMap:
    """Immutable mapping from tensor names to source checkpoint shards."""

    def __init__(self, tensors: dict[str, TensorMetadata]):
        if not tensors:
            raise ValueError("Checkpoint does not contain any tensors")
        self._tensors = dict(tensors)

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path) -> WeightMap:
        path = Path(checkpoint).expanduser().resolve()
        if path.is_dir():
            index_path = path / "model.safetensors.index.json"
            if index_path.is_file():
                return cls._from_index(index_path)
            shards = sorted(path.glob("*.safetensors"))
            if len(shards) != 1:
                raise ValueError(
                    "A checkpoint without model.safetensors.index.json must "
                    f"contain exactly one safetensors file; found {len(shards)}"
                )
            return cls._from_shards({None: shards[0]})
        if path.name.endswith(".safetensors.index.json"):
            return cls._from_index(path)
        if path.suffix == ".safetensors" and path.is_file():
            return cls._from_shards({None: path})
        raise ValueError(
            "checkpoint must be a directory, a safetensors index, or a "
            f"safetensors file: {path}"
        )

    @classmethod
    def _from_index(cls, index_path: Path) -> WeightMap:
        with index_path.open(encoding="utf-8") as file:
            index = json.load(file)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid or empty weight_map in {index_path}")

        root = index_path.parent.resolve()
        tensor_shards = {}
        for tensor_name, shard_name in weight_map.items():
            if not isinstance(tensor_name, str) or not isinstance(shard_name, str):
                raise ValueError(f"Invalid weight_map entry in {index_path}")
            shard = (root / shard_name).resolve()
            if not shard.is_relative_to(root):
                raise ValueError(
                    f"Shard for {tensor_name!r} escapes checkpoint directory"
                )
            tensor_shards[tensor_name] = shard
        return cls._from_shards(tensor_shards)

    @classmethod
    def _from_shards(
        cls, tensor_shards: dict[str | None, Path]
    ) -> WeightMap:
        declared = {name for name in tensor_shards if name is not None}
        tensors = {}
        for shard in sorted(set(tensor_shards.values())):
            if not shard.is_file():
                raise FileNotFoundError(f"Missing checkpoint shard: {shard}")
            with safe_open(shard, framework="pt", device="cpu") as file:
                actual_names = set(file.keys())
                expected_names = {
                    name
                    for name, mapped_shard in tensor_shards.items()
                    if name is not None and mapped_shard == shard
                }
                missing = expected_names - actual_names
                if missing:
                    raise KeyError(
                        f"Tensors declared in the index are missing from {shard}: "
                        f"{sorted(missing)}"
                    )
                names = actual_names if not declared else expected_names
                for name in names:
                    tensor_slice = file.get_slice(name)
                    dtype_name = tensor_slice.get_dtype()
                    try:
                        dtype = _SAFETENSORS_DTYPES[dtype_name]
                    except KeyError as error:
                        raise ValueError(
                            f"Unsupported safetensors dtype {dtype_name!r} for "
                            f"tensor {name!r}"
                        ) from error
                    tensors[name] = TensorMetadata(
                        name=name,
                        shape=tuple(tensor_slice.get_shape()),
                        dtype=dtype,
                        shard=shard,
                    )
        return cls(tensors)

    def __iter__(self) -> Iterator[str]:
        return iter(self._tensors)

    def __len__(self) -> int:
        return len(self._tensors)

    def metadata(self, name: str) -> TensorMetadata:
        try:
            return self._tensors[name]
        except KeyError as error:
            raise KeyError(
                f"Tensor {name!r} is not present in the checkpoint"
            ) from error
