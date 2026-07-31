"""On-demand reads from sharded safetensors checkpoints."""

from __future__ import annotations

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

    def load_tensors_cpu(
        self, names: Iterable[str]
    ) -> dict[str, torch.Tensor]: ...

    def iter_tensor_groups_cpu(
        self, groups: Iterable[Iterable[str]]
    ) -> Iterator[dict[str, torch.Tensor]]: ...


class SafetensorsWeightSource:
    """Read only requested tensors, grouping reads by source shard."""

    def __init__(self, checkpoint: str | Path):
        self.weight_map = WeightMap.from_checkpoint(checkpoint)

    def tensor_names(self) -> Collection[str]:
        return tuple(self.weight_map)

    def metadata(self, name: str) -> TensorMetadata:
        return self.weight_map.metadata(name)

    def load_tensors_cpu(
        self, names: Iterable[str]
    ) -> dict[str, torch.Tensor]:
        grouped = defaultdict(list)
        requested = list(dict.fromkeys(names))
        for name in requested:
            grouped[self.metadata(name).shard].append(name)

        result = {}
        for shard, shard_names in grouped.items():
            with safe_open(shard, framework="pt", device="cpu") as file:
                for name in shard_names:
                    result[name] = file.get_tensor(name)
        return result

    def iter_tensor_groups_cpu(
        self, groups: Iterable[Iterable[str]]
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield bounded tensor groups while reusing open shard handles."""

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
                            safe_open(shard, framework="pt", device="cpu")
                        )
                    file = handles[shard]
                    values[name] = file.get_tensor(name)
                yield values
