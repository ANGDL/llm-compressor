"""A header-planned, non-caching CPU weights Mapping for Accelerate offload."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass

import torch
from compressed_tensors.compressors import BaseCompressor
from compressed_tensors.quantization import QuantizationScheme
from torch import nn

from llmcompressor.streaming.checkpoint import CheckpointWeightSource
from llmcompressor.streaming.loading import HostMemoryBudget
from llmcompressor.streaming.materialization import WeightMaterializer


@dataclass(frozen=True)
class InferenceWeight:
    """Logical execution tensor and its checkpoint dependencies."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    sources: tuple[str, ...]
    preparation_bytes: int
    format: str = "dense"
    packing: str = "none"
    weight_bits: int | None = None
    activation_bits: int | None = None
    backend: str = "native_float"
    fallback_reason: str | None = None
    source_dtype: str | None = None
    source_shape: tuple[int, ...] | None = None

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize

    def to_dict(self) -> dict:
        result = asdict(self)
        result["dtype"] = str(self.dtype)
        result["nbytes"] = self.nbytes
        result["weight_type"] = "int" if self.weight_bits is not None else None
        result["activation_type"] = "int" if self.activation_bits is not None else None
        return result


class StreamingWeights(Mapping[str, torch.Tensor]):
    """Decode only the tensor requested by an Accelerate module hook.

    Keys, membership and length are metadata-only. No decoded tensor is retained
    by this Mapping, including after a failed read or forward.
    """

    def __init__(
        self,
        model: nn.Module,
        source: CheckpointWeightSource,
        materializer: WeightMaterializer,
        schemes: Mapping[str, QuantizationScheme],
        *,
        format_name: str | None = None,
        host_memory_budget: HostMemoryBudget | None = None,
        max_preparation_bytes: int | None = None,
    ):
        if max_preparation_bytes is not None and max_preparation_bytes <= 0:
            raise ValueError("max_preparation_bytes must be positive")
        self.source = source
        self.materializer = materializer
        self.schemes = dict(schemes)
        self.host_memory_budget = host_memory_budget or HostMemoryBudget()
        self.max_preparation_bytes = max_preparation_bytes
        self.tensor_reads = 0
        self.materializations = 0
        self.loads: Counter[str] = Counter()
        self.peak_preparation_bytes = 0
        self._weights: dict[str, InferenceWeight] = {}
        self._local_sources: dict[str, dict[str, str]] = {}
        self._aliases = self._tied_sources(model)
        for name, tensor in model.state_dict().items():
            module_name, _, local_name = name.rpartition(".")
            scheme = schemes.get(module_name) if local_name == "weight" else None
            if scheme is not None:
                weight = self._quantized_weight(name, tensor, scheme, format_name)
            else:
                weight = self._ordinary_weight(name, tensor)
            if (
                max_preparation_bytes is not None
                and weight.preparation_bytes > max_preparation_bytes
            ):
                raise MemoryError(
                    f"CPU preparation for {name!r} needs {weight.preparation_bytes} "
                    f"bytes, limit is {max_preparation_bytes}"
                )
            self._weights[name] = weight

    @staticmethod
    def _tied_sources(model: nn.Module) -> dict[str, str]:
        expanded = getattr(model, "get_expanded_tied_weights_keys", None)
        declared = (
            expanded()
            if callable(expanded)
            else getattr(model, "_tied_weights_keys", {})
        )
        aliases = dict(declared) if isinstance(declared, dict) else {}
        groups: dict[int, list[str]] = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            groups.setdefault(id(parameter), []).append(name)
        for names in groups.values():
            for name in names[1:]:
                aliases.setdefault(name, names[0])
        return aliases

    def _metadata(self, name):
        try:
            return self.source.metadata(name)
        except KeyError:
            return None

    def _source_name(self, name: str) -> str:
        current, seen = name, set()
        while self._metadata(current) is None:
            if current in seen or current not in self._aliases:
                raise KeyError(f"Checkpoint is missing model tensor {name!r}")
            seen.add(current)
            current = self._aliases[current]
        return current

    def _raw_bytes(self, names) -> int:
        return sum(
            math.prod(self.source.metadata(name).shape)
            * self.source.metadata(name).dtype.itemsize
            for name in dict.fromkeys(names)
        )

    def _ordinary_weight(self, name, tensor):
        source_name = self._source_name(name)
        metadata = self.source.metadata(source_name)
        sources = (
            source_name,
            *self.materializer.dependencies(source_name, metadata),
        )
        for dependency in sources:
            self.source.metadata(dependency)
        shape = (
            self.materializer.logical_shape(source_name, metadata)
            if tensor.dtype.is_floating_point
            else metadata.shape
        )
        if tuple(shape) != tuple(tensor.shape):
            raise ValueError(
                f"Checkpoint shape mismatch for {name!r}: "
                f"{tuple(shape)} vs model {tuple(tensor.shape)}"
            )
        output_bytes = tensor.numel() * tensor.element_size()
        workspace = (
            self.materializer.estimate_workspace_bytes(
                source_name, metadata, tensor.dtype
            )
            if tensor.dtype.is_floating_point
            else 0
        )
        return InferenceWeight(
            name,
            tuple(tensor.shape),
            tensor.dtype,
            tuple(sources),
            self._raw_bytes(sources) + output_bytes + workspace,
            source_dtype=str(metadata.dtype),
            source_shape=metadata.shape,
        )

    def _quantized_weight(self, name, tensor, scheme, format_name):
        module_name = name.removesuffix(".weight")
        selected_format = scheme.format or format_name
        if selected_format not in ("int-quantized", "pack-quantized"):
            raise ValueError(
                f"Unsupported compressed format {selected_format!r} at {module_name!r}"
            )
        bits = scheme.weights.num_bits
        locals_to_sources = {}
        if selected_format == "pack-quantized":
            locals_to_sources["weight_packed"] = f"{module_name}.weight_packed"
            locals_to_sources["weight_shape"] = f"{module_name}.weight_shape"
            expected = (*tensor.shape[:-1], math.ceil(tensor.shape[-1] * bits / 32))
            packing = "int32"
            weight_metadata = self.source.metadata(locals_to_sources["weight_packed"])
            if weight_metadata.dtype != torch.int32:
                raise TypeError(f"Expected int32 packed weights for {name!r}")
            shape_metadata = self.source.metadata(locals_to_sources["weight_shape"])
            if shape_metadata.shape != (tensor.ndim,):
                raise ValueError(f"Invalid packed weight_shape header for {name!r}")
        else:
            locals_to_sources["weight"] = name
            weight_metadata = self.source.metadata(name)
            if weight_metadata.dtype != torch.int8:
                raise TypeError(f"Expected signed int8 weight storage for {name!r}")
            if (
                bits == 4
                and tensor.shape[-1] % 2 == 0
                and weight_metadata.shape == (*tensor.shape[:-1], tensor.shape[-1] // 2)
            ):
                expected = weight_metadata.shape
                packing = "signed_nibbles"
            else:
                expected, packing = tuple(tensor.shape), "none"
        if tuple(weight_metadata.shape) != tuple(expected):
            raise ValueError(f"Invalid packed weight shape for {name!r}")
        scale_name = f"{module_name}.weight_scale"
        if self._metadata(scale_name) is None:
            scale_name = f"{module_name}.scale"
        scale_metadata = self.source.metadata(scale_name)
        scale_shape = (
            (tensor.shape[0], 1) if scheme.weights.strategy == "channel" else (1,)
        )
        if (
            scale_metadata.shape != scale_shape
            or not scale_metadata.dtype.is_floating_point
        ):
            raise ValueError(f"Invalid weight scale header for {name!r}")
        locals_to_sources["weight_scale"] = scale_name
        self._local_sources[name] = locals_to_sources
        sources = tuple(locals_to_sources.values())
        # CT's int32 unpacker pads each row to 32 elements and uses int32
        # temporaries. Include these and simultaneous floating decode outputs.
        workspace = tensor.numel() * (12 + tensor.element_size())
        if packing == "int32":
            padded_elements = tensor.shape[0] * math.ceil(tensor.shape[1] / 32) * 32
            workspace += padded_elements * 12
        preparation = self._raw_bytes(sources) + workspace
        return InferenceWeight(
            name,
            tuple(tensor.shape),
            tensor.dtype,
            sources,
            preparation,
            format=selected_format,
            packing=packing,
            weight_bits=bits,
            activation_bits=(
                scheme.input_activations.num_bits
                if scheme.input_activations is not None
                else None
            ),
            backend="dequantized_float",
            fallback_reason="No validated integer kernel with streaming offload",
            source_dtype=str(weight_metadata.dtype),
            source_shape=weight_metadata.shape,
        )

    def __getitem__(self, name: str) -> torch.Tensor:
        weight = self._weights[name]
        with self.host_memory_budget.reserve(
            f"inference weight {name!r}", weight.preparation_bytes
        ):
            raw = self.source.load_tensors_cpu(weight.sources)
            self.tensor_reads += len(weight.sources)
            if weight.weight_bits is not None:
                local = {
                    key: raw[value] for key, value in self._local_sources[name].items()
                }
                scale = local["weight_scale"]
                if not torch.isfinite(scale).all() or not (scale > 0).all():
                    raise ValueError(
                        f"Weight scales must be positive and finite: {name}"
                    )
                if weight.packing == "signed_nibbles":
                    packed = local["weight"]
                    unpacked = torch.empty(weight.shape, dtype=torch.int8)
                    unpacked[..., 0::2] = ((packed & 15) ^ 8) - 8
                    unpacked[..., 1::2] = packed >> 4
                    local["weight"] = unpacked
                if "weight_shape" in local:
                    if tuple(local["weight_shape"].tolist()) != weight.shape:
                        raise ValueError(
                            f"Packed weight_shape disagrees with model: {name}"
                        )
                elif weight.weight_bits < 8:
                    limit = 2 ** (weight.weight_bits - 1)
                    if (local["weight"] < -limit).any() or (
                        local["weight"] >= limit
                    ).any():
                        raise ValueError(
                            f"Integer weight exceeds its bit range: {name}"
                        )
                scheme = self.schemes[name.removesuffix(".weight")]
                compressor = BaseCompressor.get_value_from_registry(weight.format)
                value = compressor.decompress(local, scheme)["weight"].to(weight.dtype)
            elif weight.dtype.is_floating_point:
                value = self.materializer.materialize_cpu(
                    weight.sources[0], raw, target_dtype=weight.dtype
                )
            else:
                value = raw[weight.sources[0]].to(weight.dtype)
            if value.device.type != "cpu" or tuple(value.shape) != weight.shape:
                raise ValueError(f"Decoded tensor device/shape mismatch for {name!r}")
            if value.dtype != weight.dtype:
                raise ValueError(f"Decoded tensor dtype mismatch for {name!r}")
            self.materializations += 1
            self.loads[name] += 1
            self.peak_preparation_bytes = max(
                self.peak_preparation_bytes, weight.preparation_bytes
            )
            return value

    def __iter__(self) -> Iterator[str]:
        return iter(self._weights)

    def __len__(self) -> int:
        return len(self._weights)

    def __contains__(self, key: object) -> bool:
        return key in self._weights

    def describe(self) -> list[dict]:
        return [
            {**weight.to_dict(), "materializations": self.loads[weight.name]}
            for weight in self._weights.values()
        ]
