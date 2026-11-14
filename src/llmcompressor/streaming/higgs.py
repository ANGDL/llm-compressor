"""Bounded HIGGS scheme selection for streaming checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme
from compressed_tensors.utils import match_quantizable_tensors
from loguru import logger

from llmcompressor.entrypoints.higgs.base import HiggsMSECollectorConverter
from llmcompressor.entrypoints.model_free.microscale import get_fused_names

from .artifacts import fingerprint_json
from .checkpoint import CheckpointWeightSource, TensorMetadata
from .materialization import CastWeightMaterializer, WeightMaterializer

__all__ = [
    "StreamingHiggsConfigProvider",
    "get_higgs_config_streaming",
]


@dataclass(frozen=True)
class _LogicalWeight:
    """A logical 2-D HIGGS weight backed by one physical checkpoint tensor."""

    name: str
    physical_name: str
    expert_index: int | None = None
    split_kind: str | None = None


def _metadata_tensor(metadata: TensorMetadata) -> torch.Tensor:
    """Create a metadata-only tensor suitable for target matching."""

    dtype = metadata.dtype
    if not dtype.is_floating_point:
        dtype = torch.float32
    return torch.empty(metadata.shape, dtype=dtype, device="meta")


def _expert_prefix(name: str) -> tuple[str, int] | None:
    marker = ".experts."
    if marker not in name:
        return None
    prefix, _, suffix = name.partition(marker)
    index, separator, _ = suffix.partition(".")
    if not separator or not index.isdigit():
        return None
    return f"{prefix}{marker}{index}", int(index)


def _expanded_logical_weights(
    name: str, metadata: TensorMetadata
) -> tuple[_LogicalWeight, ...]:
    """Expand the bounded set of common stacked expert layouts.

    Native Transformers MoE modules normally expose individual expert names.
    Some checkpoints instead store ``gate_up_proj``/``down_proj`` or
    ``w13``/``w2`` as one stacked tensor.  The metadata expansion lets HIGGS
    match the same per-expert names without materializing a complete shard.
    """

    if not name.endswith(".weight") or len(metadata.shape) != 3:
        return (_LogicalWeight(name=name, physical_name=name),)

    leaf = name.removesuffix(".weight").rsplit(".", 1)[-1]
    if leaf not in {"gate_up_proj", "w13", "w13_weight", "down_proj", "w2"}:
        return (_LogicalWeight(name=name, physical_name=name),)

    prefix = name.removesuffix(f".{leaf}.weight")
    if prefix.endswith(".experts"):
        prefix = prefix.removesuffix(".experts")
    experts, width, depth = metadata.shape
    if experts <= 0:
        raise ValueError(f"Stacked expert tensor has no experts: {name!r}")

    if leaf in {"gate_up_proj", "w13", "w13_weight"}:
        if width % 2:
            raise ValueError(
                f"Stacked expert tensor {name!r} has an odd fused dimension: "
                f"{metadata.shape}"
            )
        projection_names = (
            ("gate_proj", "up_proj")
            if leaf == "gate_up_proj"
            else ("w1", "w3")
        )
        split_width = width // 2
        return tuple(
            _LogicalWeight(
                name=(
                    f"{prefix}.experts.{expert_index}."
                    f"{projection_name}.weight"
                ),
                physical_name=name,
                expert_index=expert_index,
                split_kind=f"{projection_name}:{split_width}",
            )
            for expert_index in range(experts)
            for projection_name in projection_names
        )

    projection_name = "down_proj" if leaf == "down_proj" else "w2"
    return tuple(
        _LogicalWeight(
            name=(
                f"{prefix}.experts.{expert_index}.{projection_name}.weight"
            ),
            physical_name=name,
            expert_index=expert_index,
            split_kind="stacked",
        )
        for expert_index in range(experts)
    )


def _logical_metadata(
    source: CheckpointWeightSource,
) -> tuple[dict[str, TensorMetadata], dict[str, _LogicalWeight]]:
    metadata_by_name: dict[str, TensorMetadata] = {}
    records: dict[str, _LogicalWeight] = {}
    for physical_name in source.tensor_names():
        metadata = source.metadata(physical_name)
        for record in _expanded_logical_weights(physical_name, metadata):
            if record.name in records:
                raise ValueError(
                    "Multiple checkpoint tensors map to logical HIGGS weight "
                    f"{record.name!r}"
                )
            records[record.name] = record
            if record.name == physical_name:
                metadata_by_name[record.name] = metadata
                continue

            shape = metadata.shape
            if record.split_kind == "stacked":
                logical_shape = shape[1:]
            else:
                split_width = int(record.split_kind.rsplit(":", 1)[1])
                logical_shape = (split_width, shape[2])
            metadata_by_name[record.name] = TensorMetadata(
                name=record.name,
                shape=tuple(logical_shape),
                dtype=metadata.dtype,
                shard=metadata.shard,
                storage_index=metadata.storage_index,
            )
    return metadata_by_name, records


class StreamingHiggsConfigProvider:
    """Collect HIGGS MSE summaries with bounded checkpoint residency.

    The provider intentionally returns a normal ``QuantizationConfig``.  It is
    a configuration-selection layer, so the returned object can be consumed by
    model-free conversion, ordinary oneshot, or the streaming quantization
    stages without changing their modifier lifecycle.
    """

    def __init__(
        self,
        candidate_schemes: Sequence[str | QuantizationScheme],
        *,
        targets: str | Sequence[str] = "Linear",
        ignore: Sequence[str] | None = None,
        device: torch.device | str | None = None,
        target_avg_bitwidth: float | None = None,
        target_avg_act_bitwidth: float | None = None,
        allow_unquantized: bool = False,
        materializer: WeightMaterializer | None = None,
        target_dtype: torch.dtype = torch.bfloat16,
    ):
        self.candidate_schemes = tuple(candidate_schemes)
        self.targets = (
            [targets] if isinstance(targets, str) else list(targets)
        )
        self.ignore = list(ignore) if ignore is not None else ["lm_head"]
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.target_avg_bitwidth = target_avg_bitwidth
        self.target_avg_act_bitwidth = target_avg_act_bitwidth
        self.allow_unquantized = allow_unquantized
        self.materializer = materializer or CastWeightMaterializer()
        self.target_dtype = target_dtype

    @property
    def config_fingerprint(self) -> str:
        """Return a stable fingerprint for the global HIGGS decision inputs."""

        return fingerprint_json(
            {
                "candidate_schemes": [
                    scheme.model_dump(mode="json")
                    if isinstance(scheme, QuantizationScheme)
                    else scheme
                    for scheme in self.candidate_schemes
                ],
                "targets": self.targets,
                "ignore": self.ignore,
                "device": str(self.device),
                "target_dtype": str(self.target_dtype),
                "target_avg_bitwidth": self.target_avg_bitwidth,
                "target_avg_act_bitwidth": self.target_avg_act_bitwidth,
                "allow_unquantized": self.allow_unquantized,
                "materializer": self.materializer.identifier,
            }
        )

    def _collector(self) -> HiggsMSECollectorConverter:
        ignore = list(self.ignore)
        for value in self.ignore:
            if not value.startswith("re:") and not value.startswith("model."):
                ignore.append(f"model.{value}")
        return HiggsMSECollectorConverter(
            candidate_schemes=list(self.candidate_schemes),
            targets=self.targets,
            ignore=list(dict.fromkeys(ignore)),
            device=self.device,
            target_avg_bitwidth=self.target_avg_bitwidth,
            target_avg_act_bitwidth=self.target_avg_act_bitwidth,
            allow_unquantized=self.allow_unquantized,
        )

    @staticmethod
    def _group_records(
        matched_names: Iterable[str],
    ) -> tuple[tuple[str, ...], ...]:
        matched = set(matched_names)
        fused = get_fused_names(matched)
        grouped = [tuple(group.values()) for group in fused]
        grouped_names = {name for group in grouped for name in group}
        grouped.extend((name,) for name in sorted(matched - grouped_names))
        return tuple(grouped)

    @staticmethod
    def _slice_logical_weight(
        record: _LogicalWeight,
        physical: torch.Tensor,
    ) -> torch.Tensor:
        if record.expert_index is None:
            if physical.ndim != 2:
                raise ValueError(
                    "HIGGS only supports 2-D logical weights; matched "
                    f"{record.physical_name!r} with shape {tuple(physical.shape)}"
                )
            return physical

        if physical.ndim != 3:
            raise ValueError(
                "Stacked expert metadata expected a 3-D tensor for "
                f"{record.physical_name!r}, got {tuple(physical.shape)}"
            )
        expert = physical[record.expert_index]
        if record.split_kind in {"stacked", None}:
            return expert
        _, split_width_text = record.split_kind.rsplit(":", 1)
        split_width = int(split_width_text)
        if record.name.rsplit(".", 2)[-2] == "gate_proj":
            return expert[:split_width]
        return expert[split_width:]

    def _materialize_group(
        self,
        source: CheckpointWeightSource,
        records: Mapping[str, _LogicalWeight],
        names: Sequence[str],
        raw_values: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        physical_names = tuple(
            dict.fromkeys(records[name].physical_name for name in names)
        )
        decoded: dict[str, torch.Tensor] = {}
        try:
            for physical_name in physical_names:
                metadata = source.metadata(physical_name)
                decoded_physical = self.materializer.materialize_cpu(
                    physical_name,
                    {
                        name: raw_values[name]
                        for name in (
                            physical_name,
                            *self.materializer.dependencies(
                                physical_name,
                                metadata,
                            ),
                        )
                    },
                    target_dtype=self.target_dtype,
                )
                for name in names:
                    record = records[name]
                    if record.physical_name == physical_name:
                        decoded[name] = self._slice_logical_weight(
                            record, decoded_physical
                        )
        finally:
            decoded_physical = None
        return decoded

    def _group_requests(
        self,
        source: CheckpointWeightSource,
        records: Mapping[str, _LogicalWeight],
        names: Sequence[str],
    ) -> tuple[str, ...]:
        physical_names = tuple(
            dict.fromkeys(records[name].physical_name for name in names)
        )
        requests = []
        for physical_name in physical_names:
            metadata = source.metadata(physical_name)
            requests.extend(
                (
                    physical_name,
                    *self.materializer.dependencies(physical_name, metadata),
                )
            )
        return tuple(dict.fromkeys(requests))

    def collect(
        self,
        checkpoint: str | Path | None = None,
        *,
        source: CheckpointWeightSource | None = None,
    ) -> HiggsMSECollectorConverter:
        """Collect bounded MSE summaries and return the reusable collector."""

        if source is None:
            if checkpoint is None:
                raise ValueError("checkpoint is required when source is not supplied")
            source = self.materializer.create_source(str(checkpoint))

        metadata, records = _logical_metadata(source)
        meta_tensors = {
            name: _metadata_tensor(value) for name, value in metadata.items()
        }
        collector = self._collector()
        matched = list(
            match_quantizable_tensors(meta_tensors, collector.ignore, self.targets)
        )
        if not matched:
            raise ValueError(
                f"No HIGGS quantizable weights matched targets={self.targets!r} "
                f"and ignore={self.ignore!r}"
            )
        matched_names = [tensor_name for _, tensor_name in matched]
        collector.set_weight_names(set(matched_names))

        groups = self._group_records(matched_names)
        requests = (
            self._group_requests(source, records, group) for group in groups
        )
        for group, raw_values in zip(
            groups, source.iter_tensor_groups_cpu(requests)
        ):
            values = self._materialize_group(source, records, group, raw_values)
            try:
                collector.process(values)
            finally:
                values.clear()
        logger.info(
            "Streaming HIGGS collected bounded MSE summaries for "
            f"{len(collector.mse_matrix)} logical weights"
        )
        return collector

    def create_config(
        self,
        checkpoint: str | Path | None = None,
        *,
        source: CheckpointWeightSource | None = None,
    ) -> QuantizationConfig:
        """Collect summaries and solve the global HIGGS ILP."""

        return self.collect(checkpoint, source=source).create_config()

    __call__ = create_config


def get_higgs_config_streaming(
    checkpoint: str | Path,
    candidate_schemes: Sequence[str | QuantizationScheme],
    **kwargs,
) -> QuantizationConfig:
    """Convenience wrapper for bounded HIGGS config generation."""

    return StreamingHiggsConfigProvider(
        candidate_schemes,
        **kwargs,
    ).create_config(checkpoint)
