"""On-demand reads from sharded safetensors checkpoints."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Collection, Iterable, Protocol

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
                    result[name] = file.get_tensor(name)
        return result
