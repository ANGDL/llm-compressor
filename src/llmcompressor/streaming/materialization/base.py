"""Interfaces and validation for checkpoint weight materialization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import torch

from llmcompressor.streaming.artifacts import MaterializerInfo, fingerprint_json
from llmcompressor.streaming.checkpoint import (
    CheckpointWeightSource,
    SafetensorsWeightSource,
    TensorMetadata,
)

from .._logging import streaming_logger
from .dtype import StreamingDTypePolicy

if TYPE_CHECKING:
    from llmcompressor.streaming.loading.host_memory import HostMemoryReservation

_PROGRESS_INTERVAL = 64


class WeightMaterializer(ABC):
    """Decode a logical floating-point weight and declare its dependencies."""

    @property
    def identifier(self) -> str:
        cls = type(self)
        return f"{cls.__module__}.{cls.__qualname__}"

    def configuration(self) -> Mapping[str, Any]:
        return {}

    def manifest_info(
        self,
        *,
        target_dtype: torch.dtype,
        dtype_policy: StreamingDTypePolicy | None = None,
    ) -> MaterializerInfo:
        configuration = {
            "configuration": self.configuration(),
            "target_dtype": str(target_dtype),
        }
        if dtype_policy is not None:
            configuration["dtype_policy"] = dtype_policy.configuration()
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

    def estimate_workspace_bytes(
        self,
        tensor_name: str,
        metadata: TensorMetadata,
        target_dtype: torch.dtype,
    ) -> int:
        """Return additional peak CPU workspace beyond inputs and output."""

        return 0

    def create_source(self, checkpoint: str) -> CheckpointWeightSource:
        """Create the checkpoint view consumed by all streaming stages."""
        return SafetensorsWeightSource(checkpoint)

    def output_config_updates(self) -> Mapping[str, Any]:
        """Return config fields required to reload the materialized output."""
        return {}

    def output_tensor_name(self, tensor_name: str) -> str:
        """Map an internal logical tensor name to its serialized name."""
        return tensor_name

    def output_module_name(self, module_name: str) -> str:
        """Map an internal logical module name to its serialized name."""
        output_weight = self.output_tensor_name(f"{module_name}.weight")
        if not output_weight.endswith(".weight"):
            raise ValueError(
                "Materializer output tensor mapping must preserve the '.weight' "
                f"suffix for module names: {module_name!r} -> {output_weight!r}"
            )
        return output_weight.removesuffix(".weight")

    def output_weight_scale_name(self, output_module_name: str) -> str:
        """Return the serialized weight-scale name for an output module."""
        return f"{output_module_name}.weight_scale"

    def transform_output_config(
        self, config: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Apply checkpoint-specific cleanup after output metadata is merged."""
        return config

    @abstractmethod
    def materialize_cpu(
        self,
        tensor_name: str,
        tensors: Mapping[str, torch.Tensor],
        *,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return one logical weight in the requested dtype on CPU."""


def materialize_weights(
    source: CheckpointWeightSource,
    names: Iterable[str],
    materializer: WeightMaterializer,
    *,
    target_dtype: torch.dtype,
    reservation: HostMemoryReservation | None = None,
    dtype_policy: StreamingDTypePolicy | None = None,
) -> dict[str, torch.Tensor]:
    """Load and decode requested weights on CPU in physical storage order."""

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
    iter_groups = getattr(source, "iter_tensor_groups_cpu", None)
    raw_groups = (
        iter_groups(tensor_groups)
        if callable(iter_groups)
        else (
            source.load_tensors_cpu(group) for group in tensor_groups
        )
    )
    raw_groups = iter(raw_groups)
    total = len(requests)
    started_at = perf_counter()
    if total >= _PROGRESS_INTERVAL:
        streaming_logger.info(
            f"weights | materialize | tensors={total} | device=cpu"
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
        raw_bytes = sum(
            value.numel() * value.element_size()
            for value in raw_tensors.values()
        )
        if reservation is not None:
            reservation.commit(raw_bytes)
        materialize_dtype = (
            dtype_policy.resolve(
                name,
                metadata[name].dtype,
                fallback_dtype=target_dtype,
            )
            if dtype_policy is not None
            else target_dtype
        )
        tensor = materializer.materialize_cpu(
            name, raw_tensors, target_dtype=materialize_dtype
        )
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Materializer returned a non-tensor for {name!r}")
        if not tensor.dtype.is_floating_point:
            raise TypeError(
                f"Materializer returned non-floating dtype {tensor.dtype} for {name!r}"
            )
        if tensor.dtype != materialize_dtype:
            raise ValueError(
                f"Materializer returned dtype {tensor.dtype} for {name!r}; "
                f"expected {materialize_dtype}"
            )
        expected_shape = materializer.logical_shape(name, metadata[name])
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Materializer returned shape {tuple(tensor.shape)} for {name!r}; "
                f"expected {expected_shape}"
            )
        if tensor.device.type != "cpu":
            raise ValueError(
                f"Materializer returned device {tensor.device} for {name!r}; "
                "expected cpu"
            )
        if reservation is not None:
            reservation.commit(tensor.numel() * tensor.element_size())
        results[name] = tensor
        del raw_tensors
        if reservation is not None:
            reservation.uncommit(raw_bytes)
        if total >= _PROGRESS_INTERVAL and (
            index % _PROGRESS_INTERVAL == 0 or index == total
        ):
            streaming_logger.debug(
                f"weights | progress={index}/{total} | "
                f"time={perf_counter() - started_at:.2f}s"
            )
    return results
