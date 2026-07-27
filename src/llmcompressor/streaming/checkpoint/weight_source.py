"""On-demand reads from sharded safetensors checkpoints."""

from __future__ import annotations

import json
import struct
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Collection, Iterable, Iterator, Protocol

import torch
from safetensors import safe_open

from .weight_map import TensorMetadata, WeightMap


class CheckpointWeightSource(Protocol):
    """Source that owns metadata, but never retains loaded weight tensors."""

    def tensor_names(self) -> Collection[str]: ...

    def metadata(self, name: str) -> TensorMetadata: ...

    def load_tensors(
        self, names: Iterable[str], *, device: torch.device
    ) -> dict[str, torch.Tensor]: ...

    def iter_tensor_groups(
        self, groups: Iterable[Iterable[str]], *, device: torch.device
    ) -> Iterator[dict[str, torch.Tensor]]: ...


class SafetensorsWeightSource:
    """Read only requested tensors, grouping reads by source shard."""

    def __init__(self, checkpoint: str | Path):
        self.weight_map = WeightMap.from_checkpoint(checkpoint)

    def tensor_names(self) -> Collection[str]:
        return tuple(self.weight_map)

    def metadata(self, name: str) -> TensorMetadata:
        return self.weight_map.metadata(name)

    def load_tensors(
        self, names: Iterable[str], *, device: torch.device
    ) -> dict[str, torch.Tensor]:
        device = torch.device(device)
        if device.type == "meta":
            raise ValueError("Cannot load checkpoint tensors onto the meta device")

        grouped = defaultdict(list)
        requested = list(dict.fromkeys(names))
        for name in requested:
            grouped[self.metadata(name).shard].append(name)

        result = {}
        for shard, shard_names in grouped.items():
            with safe_open(shard, framework="pt", device=str(device)) as file:
                for name in shard_names:
                    tensor_slice = file.get_slice(name)
                    if tensor_slice.get_dtype() == "F8_E8M0":
                        result[name] = _read_e8m0(shard, name).to(device)
                    else:
                        result[name] = file.get_tensor(name)
        return result

    def iter_tensor_groups(
        self, groups: Iterable[Iterable[str]], *, device: torch.device
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield bounded tensor groups while reusing open shard handles."""

        device = torch.device(device)
        if device.type == "meta":
            raise ValueError("Cannot load checkpoint tensors onto the meta device")

        requested_groups = [tuple(dict.fromkeys(group)) for group in groups]
        with ExitStack() as stack:
            handles = {}
            for names in requested_groups:
                values = {}
                for name in names:
                    metadata = self.metadata(name)
                    shard = metadata.shard
                    if shard not in handles:
                        handles[shard] = stack.enter_context(
                            safe_open(shard, framework="pt", device=str(device))
                        )
                    file = handles[shard]
                    tensor_slice = file.get_slice(name)
                    if tensor_slice.get_dtype() == "F8_E8M0":
                        values[name] = _read_e8m0(shard, name).to(device)
                    else:
                        values[name] = file.get_tensor(name)
                yield values


def _read_e8m0(shard: Path, name: str) -> torch.Tensor:
    """Read unsupported F8_E8M0 safetensors storage as uint8 bytes."""
    with shard.open("rb") as file:
        header_size = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(header_size))
        info = header[name]
        start, end = info["data_offsets"]
        file.seek(8 + header_size + start)
        storage = bytearray(file.read(end - start))
    return torch.frombuffer(storage, dtype=torch.uint8).reshape(info["shape"])
