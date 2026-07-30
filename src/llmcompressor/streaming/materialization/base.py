"""Interfaces and validation for checkpoint weight materialization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any, Iterable, Mapping

import torch

from llmcompressor.streaming.artifacts import MaterializerInfo, fingerprint_json
from llmcompressor.streaming.checkpoint import (
    CheckpointWeightSource,
    SafetensorsWeightSource,
    TensorMetadata,
)

from .._logging import streaming_logger

_PROGRESS_INTERVAL = 64

class WeightMaterializer(ABC):
    """Decode a logical floating-point weight and declare its dependencies."""

    @property
    def identifier(self) -> str:
        cls = type(self)
        return f"{cls.__module__}.{cls.__qualname__}"

    def configuration(self) -> Mapping[str, Any]:
        return {}

    def manifest_info(self, *, target_dtype: torch.dtype) -> MaterializerInfo:
        configuration = {
            "configuration": self.configuration(),
            "target_dtype": str(target_dtype),
        }
        return MaterializerInfo(self.identifier, fingerprint_json(configuration))

    def dependencies(
        self, tensor_name: str, metadata: TensorMetadata
    ) -> list[str]:
        return []

    def logical_shape(
        self, tensor_name: str, metadata: TensorMetadata
    ) -> tuple[int, ...]:
        """Return the decoded tensor shape exposed to the model."""
        return metadata.shape

    def deferred_module(
        self, tensor_name: str, metadata: TensorMetadata
    ) -> str | None:
        """Return a module that should be materialized only when invoked.

        The returned name must own ``tensor_name``. Parent target loads omit all
        state under that module, while loading the deferred module itself remains
        eager. This keeps unusually large conditional branches, such as routed
        experts, outside the resident target working set.
        """
        return None

    def deferred_cache_device(self) -> torch.device | None:
        """Return the offload device used to cache decoded deferred weights.

        A cache spans the lifetime of the resident parent target. This mirrors
        ordinary oneshot's parameter offload cache: deferred weights are decoded
        once, kept off the execution device between calls, and released when the
        parent target is unloaded.
        """
        return None

    def create_source(self, checkpoint: str) -> CheckpointWeightSource:
        """Create the checkpoint view consumed by all streaming stages."""
        return SafetensorsWeightSource(checkpoint)

    def output_config_updates(self) -> Mapping[str, Any]:
        """Return config fields required to reload the materialized output."""
        return {}

    def transform_output_config(
        self, config: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Apply checkpoint-specific cleanup after output metadata is merged."""
        return config

    @abstractmethod
    def materialize(
        self,
        tensor_name: str,
        tensors: Mapping[str, torch.Tensor],
        *,
        target_dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return one logical weight in the requested dtype and device."""


def materialize_weights(
    source: CheckpointWeightSource,
    names: Iterable[str],
    materializer: WeightMaterializer,
    *,
    target_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Load requested weights and dependencies, then validate decoded results."""

    device = torch.device(device)
    requested = list(dict.fromkeys(names))
    metadata = {name: source.metadata(name) for name in requested}
    requests = []
    for name in requested:
        group = (
            name,
            *dict.fromkeys(materializer.dependencies(name, metadata[name])),
        )
        group_metadata = [source.metadata(group_name) for group_name in group]
        requests.append((name, group, group_metadata))
    requests.sort(
        key=lambda request: min(
            (item.shard.as_posix(), item.storage_index)
            for item in request[2]
        )
    )
    tensor_groups = [group for _, group, _ in requests]

    results = {}
    iter_groups = getattr(source, "iter_tensor_groups", None)
    raw_groups = (
        iter_groups(tensor_groups, device=device)
        if callable(iter_groups)
        else (
            source.load_tensors(group, device=device) for group in tensor_groups
        )
    )
    raw_groups = iter(raw_groups)
    total = len(requests)
    started_at = perf_counter()
    if total >= _PROGRESS_INTERVAL:
        streaming_logger.info(
            f"weights | materialize | tensors={total} | device={device}"
        )
    for index, (name, _, _) in enumerate(requests, start=1):
        if total >= _PROGRESS_INTERVAL and (index - 1) % _PROGRESS_INTERVAL == 0:
            streaming_logger.debug(
                "weights | loading chunk | tensors="
                f"{index}-{min(index + _PROGRESS_INTERVAL - 1, total)}/{total}; "
                f"first={name!r}, shard={metadata[name].shard.name!r}"
            )
        streaming_logger.debug(
            f"streaming materialization: loading tensor {index}/{total} {name!r} "
            f"from {metadata[name].shard.name!r}"
        )
        try:
            raw_tensors = next(raw_groups)
        except StopIteration as error:
            raise RuntimeError(
                "Checkpoint source returned fewer tensor groups than requested"
            ) from error
        tensor = materializer.materialize(
            name, raw_tensors, target_dtype=target_dtype, device=device
        )
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Materializer returned a non-tensor for {name!r}")
        if not tensor.dtype.is_floating_point:
            raise TypeError(
                f"Materializer returned non-floating dtype {tensor.dtype} for {name!r}"
            )
        if tensor.dtype != target_dtype:
            raise ValueError(
                f"Materializer returned dtype {tensor.dtype} for {name!r}; "
                f"expected {target_dtype}"
            )
        expected_shape = materializer.logical_shape(name, metadata[name])
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Materializer returned shape {tuple(tensor.shape)} for {name!r}; "
                f"expected {expected_shape}"
            )
        if tensor.device != device:
            raise ValueError(
                f"Materializer returned device {tensor.device} for {name!r}; "
                f"expected {device}"
            )
        results[name] = tensor
        if total >= _PROGRESS_INTERVAL and (
            index % _PROGRESS_INTERVAL == 0 or index == total
        ):
            streaming_logger.debug(
                f"weights | progress={index}/{total} | "
                f"time={perf_counter() - started_at:.2f}s"
            )
    return results
