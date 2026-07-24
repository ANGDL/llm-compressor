"""Interfaces and validation for checkpoint weight materialization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping

import torch

from llmcompressor.streaming.artifacts import MaterializerInfo, fingerprint_json
from llmcompressor.streaming.checkpoint import (
    CheckpointWeightSource,
    SafetensorsWeightSource,
    TensorMetadata,
)


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

    def create_source(self, checkpoint: str) -> CheckpointWeightSource:
        """Create the checkpoint view consumed by all streaming stages."""
        return SafetensorsWeightSource(checkpoint)

    def output_config_updates(self) -> Mapping[str, Any]:
        """Return config fields required to reload the materialized output."""
        return {}

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
    dependencies = {
        dependency
        for name in requested
        for dependency in materializer.dependencies(name, metadata[name])
    }
    raw_tensors = source.load_tensors(
        [*requested, *sorted(dependencies)], device=device
    )

    results = {}
    for name in requested:
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
    return results
