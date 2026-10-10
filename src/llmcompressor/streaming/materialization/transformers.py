"""Lazy application of Transformers checkpoint conversion mappings."""

from __future__ import annotations

from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from llmcompressor.streaming.checkpoint import (
    CheckpointWeightSource,
    TensorMetadata,
)

from .base import WeightMaterializer

try:
    from transformers.core_model_loading import (
        WeightConverter,
        WeightRenaming,
        dot_natural_key,
        rename_source_key,
    )
    from transformers.modeling_utils import get_model_conversion_mapping
except ImportError:  # pragma: no cover - Transformers < 5
    WeightConverter = None
    WeightRenaming = None
    dot_natural_key = None
    rename_source_key = None
    get_model_conversion_mapping = None

__all__ = [
    "TransformersWeightMaterializer",
    "apply_transformers_weight_conversions",
]


@dataclass(frozen=True)
class _SourceTensor:
    name: str
    source_pattern: str


@dataclass(frozen=True)
class _ConversionGroup:
    target_name: str
    sources: tuple[_SourceTensor, ...]
    converter: Any | None
    transformed: bool


class _TransformersConversionSource(CheckpointWeightSource):
    def __init__(
        self,
        source: CheckpointWeightSource,
        groups: Mapping[str, _ConversionGroup],
        primary_names: Sequence[str],
    ):
        self._source = source
        self._groups = dict(groups)
        self._primary_names = tuple(primary_names)

    def tensor_names(self):
        return self._primary_names

    def _raw_name(self, name: str) -> str:
        group = self._groups.get(name)
        return group.sources[0].name if group is not None else name

    def metadata(self, name: str) -> TensorMetadata:
        raw_name = self._raw_name(name)
        metadata = self._source.metadata(raw_name)
        if raw_name == name:
            return metadata
        return TensorMetadata(
            name=name,
            shape=metadata.shape,
            dtype=metadata.dtype,
            shard=metadata.shard,
            storage_index=metadata.storage_index,
        )

    def load_tensors_cpu(self, names) -> dict[str, torch.Tensor]:
        requested = list(dict.fromkeys(names))
        raw_names = [self._raw_name(name) for name in requested]
        raw_values = self._source.load_tensors_cpu(raw_names)
        return {
            name: raw_values[raw_name]
            for name, raw_name in zip(requested, raw_names)
        }

    def iter_tensor_groups_cpu(self, groups):
        logical_groups = [tuple(dict.fromkeys(group)) for group in groups]
        raw_groups = [
            tuple(self._raw_name(name) for name in group)
            for group in logical_groups
        ]
        for logical, raw, raw_values in zip(
            logical_groups,
            raw_groups,
            self._source.iter_tensor_groups_cpu(raw_groups),
        ):
            yield {
                logical_name: raw_values[raw_name]
                for logical_name, raw_name in zip(logical, raw)
            }


class TransformersWeightMaterializer(WeightMaterializer):
    """Compose a materializer with a model's registered weight conversions."""

    def __init__(
        self,
        materializer: WeightMaterializer,
        model: nn.Module,
        conversions: Sequence[Any],
    ):
        self.materializer = materializer
        self.model = model
        self.conversions = tuple(conversions)
        self._groups: dict[str, _ConversionGroup] = {}
        self._source: CheckpointWeightSource | None = None
        self._target_shapes = {
            name: tuple(tensor.shape)
            for name, tensor in model.state_dict().items()
        }

    def configuration(self) -> Mapping[str, Any]:
        return {
            "materializer": {
                "identifier": self.materializer.identifier,
                "configuration": self.materializer.configuration(),
            },
            "transformers_weight_conversions": [
                repr(conversion) for conversion in self.conversions
            ],
        }

    def create_source(self, checkpoint: str) -> CheckpointWeightSource:
        source = self.materializer.create_source(checkpoint)
        groups, primary_names = self._build_groups(source)
        self._groups = groups
        self._source = source
        return _TransformersConversionSource(source, groups, primary_names)

    def _build_groups(
        self,
        source: CheckpointWeightSource,
    ) -> tuple[dict[str, _ConversionGroup], tuple[str, ...]]:
        if WeightRenaming is None or WeightConverter is None:
            raise RuntimeError(
                "Transformers checkpoint conversions require Transformers 5"
            )
        renamings = [
            conversion
            for conversion in self.conversions
            if isinstance(conversion, WeightRenaming)
        ]
        converters = [
            conversion
            for conversion in self.conversions
            if isinstance(conversion, WeightConverter)
        ]
        pattern_to_converter = {
            pattern: converter
            for converter in converters
            for pattern in converter.source_patterns
        }
        meta_state = self.model.state_dict()
        base_model_prefix = getattr(self.model, "base_model_prefix", None)
        pending: dict[str, dict[str, Any]] = {}
        primary_names = []

        source_names = list(source.tensor_names())
        if dot_natural_key is not None:
            source_names.sort(key=dot_natural_key)
        else:  # pragma: no cover - Transformers < 5 fallback
            source_names.sort()

        for original_name in source_names:
            renamed_name, source_pattern = rename_source_key(
                original_name,
                renamings,
                converters,
                base_model_prefix,
                meta_state,
            )
            if renamed_name not in meta_state and original_name in meta_state:
                renamed_name, source_pattern = rename_source_key(
                    original_name,
                    [],
                    [],
                    base_model_prefix,
                    meta_state,
                )
            if renamed_name not in meta_state:
                renamed_name = original_name
                source_pattern = None

            converter = (
                pattern_to_converter[source_pattern]
                if source_pattern is not None
                else None
            )
            if converter is not None and len(converter.target_patterns) != 1:
                raise NotImplementedError(
                    "Streaming Transformers conversion does not yet support "
                    f"one-to-many transforms: {converter!r}"
                )
            entry = pending.get(renamed_name)
            if entry is None:
                entry = {
                    "sources": [],
                    "converter": converter,
                    "transformed": (
                        original_name != renamed_name or converter is not None
                    ),
                }
                pending[renamed_name] = entry
                primary_names.append(renamed_name)
            elif entry["converter"] is not converter:
                raise ValueError(
                    "Checkpoint conversion maps incompatible sources to "
                    f"{renamed_name!r}"
                )
            entry["sources"].append(
                _SourceTensor(
                    name=original_name,
                    source_pattern=source_pattern or original_name,
                )
            )

        groups = {
            target_name: _ConversionGroup(
                target_name=target_name,
                sources=tuple(entry["sources"]),
                converter=entry["converter"],
                transformed=entry["transformed"],
            )
            for target_name, entry in pending.items()
        }
        return groups, tuple(primary_names)

    def _group(self, tensor_name: str) -> _ConversionGroup:
        try:
            return self._groups[tensor_name]
        except KeyError as error:
            raise KeyError(
                f"Unknown converted checkpoint tensor {tensor_name!r}"
            ) from error

    def _source_metadata(self, name: str) -> TensorMetadata:
        if self._source is None:
            raise RuntimeError("create_source() must be called before materialization")
        return self._source.metadata(name)

    def dependencies(
        self, tensor_name: str, metadata: TensorMetadata
    ) -> list[str]:
        group = self._group(tensor_name)
        dependencies = []
        for index, source_tensor in enumerate(group.sources):
            if index:
                dependencies.append(source_tensor.name)
            source_metadata = self._source_metadata(source_tensor.name)
            dependencies.extend(
                self.materializer.dependencies(
                    source_tensor.name,
                    source_metadata,
                )
            )
        return list(dict.fromkeys(dependencies))

    def logical_shape(
        self, tensor_name: str, metadata: TensorMetadata
    ) -> tuple[int, ...]:
        target_shape = self._target_shapes.get(tensor_name)
        if target_shape is not None:
            return target_shape
        return self.materializer.logical_shape(tensor_name, metadata)

    def estimate_workspace_bytes(
        self,
        tensor_name: str,
        metadata: TensorMetadata,
        target_dtype: torch.dtype,
    ) -> int:
        group = self._group(tensor_name)
        base_workspace = sum(
            self.materializer.estimate_workspace_bytes(
                source_tensor.name,
                self._source_metadata(source_tensor.name),
                target_dtype,
            )
            for source_tensor in group.sources
        )
        if group.converter is None:
            return base_workspace

        decoded_source_bytes = 0
        for source_tensor in group.sources:
            source_metadata = self._source_metadata(source_tensor.name)
            elements = 1
            for dimension in self.materializer.logical_shape(
                source_tensor.name,
                source_metadata,
            ):
                elements *= dimension
            decoded_source_bytes += elements * target_dtype.itemsize

        target_elements = 1
        for dimension in self._target_shapes[group.target_name]:
            target_elements *= dimension
        intermediate_outputs = max(0, len(group.converter.operations) - 1)
        return (
            base_workspace
            + decoded_source_bytes
            + intermediate_outputs * target_elements * target_dtype.itemsize
        )

    def _materialize_source(
        self,
        group: _ConversionGroup,
        index: int,
        tensors: Mapping[str, torch.Tensor],
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        source_tensor = group.sources[index]
        request_name = group.target_name if index == 0 else source_tensor.name
        values = {source_tensor.name: tensors[request_name]}
        source_metadata = self._source_metadata(source_tensor.name)
        for dependency in self.materializer.dependencies(
            source_tensor.name,
            source_metadata,
        ):
            values[dependency] = tensors[dependency]
        return self.materializer.materialize_cpu(
            source_tensor.name,
            values,
            target_dtype=target_dtype,
        )

    def materialize_cpu(
        self,
        tensor_name: str,
        tensors: Mapping[str, torch.Tensor],
        *,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        group = self._group(tensor_name)
        values = [
            self._materialize_source(
                group,
                index,
                tensors,
                target_dtype,
            )
            for index in range(len(group.sources))
        ]
        if group.converter is None:
            if len(values) != 1:
                raise ValueError(
                    f"Multiple checkpoint tensors map to {tensor_name!r} "
                    "without a WeightConverter"
                )
            return values[0]

        converter = deepcopy(group.converter)
        for source_tensor, value in zip(group.sources, values):
            future = Future()
            future.set_result(value)
            converter.add_tensor(
                tensor_name,
                source_tensor.name,
                source_tensor.source_pattern,
                future,
            )
        converted = converter.convert(
            tensor_name,
            model=self.model,
            config=getattr(self.model, "config", None),
        )
        try:
            result = converted[tensor_name]
        except KeyError as error:
            raise KeyError(
                f"Transformers converter did not produce {tensor_name!r}; "
                f"produced={sorted(converted)}"
            ) from error
        if isinstance(result, list):
            if len(result) != 1:
                raise ValueError(
                    f"Transformers converter returned {len(result)} tensors "
                    f"for {tensor_name!r}"
                )
            result = result[0]
        return result

    def output_config_updates(self) -> Mapping[str, Any]:
        return self.materializer.output_config_updates()

    def output_tensor_name(self, tensor_name: str) -> str:
        group = self._groups.get(tensor_name)
        if group is not None and group.transformed:
            return tensor_name
        return self.materializer.output_tensor_name(tensor_name)

    def output_weight_scale_name(self, output_module_name: str) -> str:
        weight_name = f"{output_module_name}.weight"
        group = self._groups.get(weight_name)
        if group is not None and group.transformed:
            return f"{output_module_name}.weight_scale"
        return self.materializer.output_weight_scale_name(output_module_name)

    def transform_output_config(
        self, config: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return self.materializer.transform_output_config(config)


def apply_transformers_weight_conversions(
    materializer: WeightMaterializer,
    model: nn.Module,
) -> WeightMaterializer:
    """Wrap a materializer with the conversions registered for ``model``."""
    if isinstance(materializer, TransformersWeightMaterializer):
        return materializer
    if get_model_conversion_mapping is None:
        return materializer
    conversions = get_model_conversion_mapping(model, None, None)
    if not conversions:
        return materializer
    return TransformersWeightMaterializer(materializer, model, conversions)
