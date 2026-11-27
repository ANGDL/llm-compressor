"""Execution topology metadata for multimodal and auxiliary streaming paths."""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.fx import Node

__all__ = [
    "StreamingExecutionDescriptor",
    "describe_streaming_execution",
    "fingerprint_streaming_value",
    "fingerprint_streaming_values",
]


def _canonical(value: Any, digest: "hashlib._Hash") -> None:
    def frame(payload: bytes):
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    if isinstance(value, torch.Tensor):
        tensor = value.detach().to("cpu").contiguous()
        frame(b"tensor")
        frame(str(tensor.dtype).encode())
        frame(json.dumps(tuple(tensor.shape)).encode())
        digest.update(tensor.nbytes.to_bytes(8, "big"))
        data = memoryview(tensor.reshape(-1).view(torch.uint8).numpy())
        for offset in range(0, len(data), 1024 * 1024):
            digest.update(data[offset : offset + 1024 * 1024])
        return
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        frame(type(value).__qualname__.encode())
        frame(repr(value).encode())
        return
    if is_dataclass(value) and not isinstance(value, type):
        frame(f"{type(value).__module__}.{type(value).__qualname__}".encode())
        _canonical(
            {field.name: getattr(value, field.name) for field in fields(value)}, digest
        )
        return
    if isinstance(value, Mapping):
        frame(b"mapping")
        digest.update(len(value).to_bytes(8, "big"))
        for key in sorted(value, key=lambda item: repr(item)):
            _canonical(key, digest)
            _canonical(value[key], digest)
        return
    if isinstance(value, (list, tuple)):
        frame(type(value).__qualname__.encode())
        digest.update(len(value).to_bytes(8, "big"))
        for item in value:
            _canonical(item, digest)
        return
    raise TypeError(f"Unsupported streaming execution value {type(value).__qualname__}")


def fingerprint_streaming_value(value: Any) -> str:
    digest = hashlib.sha256()
    _canonical(value, digest)
    return digest.hexdigest()


def fingerprint_streaming_values(values: Any) -> str:
    digest = hashlib.sha256()
    count = 0
    for value in values:
        digest.update(b"item")
        _canonical(value, digest)
        count += 1
    digest.update(b"end")
    digest.update(count.to_bytes(8, "big"))
    return digest.hexdigest()


def streaming_configuration(value: Any) -> Any:
    """Normalize explicit constructor/processor configuration for cache identity."""
    if isinstance(value, Mapping):
        return {key: streaming_configuration(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(streaming_configuration(item) for item in value)
    if isinstance(value, (torch.dtype, torch.device, Path)):
        return str(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = dict(to_dict())
        result.pop("_name_or_path", None)
        result.pop("_streaming_meta_init", None)
        return streaming_configuration(result)
    return value


def fingerprint_execution_graph(subgraphs) -> str:
    """Bind graph semantics without depending on placeholder insertion order."""

    def encode(value):
        if isinstance(value, Node):
            return ("node", value.name)
        if isinstance(value, Mapping):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return (type(value).__name__, tuple(encode(item) for item in value))
        if isinstance(value, slice):
            return (
                "slice",
                encode(value.start),
                encode(value.stop),
                encode(value.step),
            )
        if isinstance(value, functools.partial):
            return (
                "partial",
                encode(value.func),
                encode(value.args),
                encode(value.keywords),
            )
        if callable(value):
            return (
                "callable",
                getattr(value, "__module__", ""),
                getattr(value, "__qualname__", str(value)),
            )
        if isinstance(
            value, (torch.dtype, torch.device, torch.layout, torch.memory_format)
        ):
            return (type(value).__name__, str(value))
        if value is Ellipsis:
            return ("ellipsis",)
        return value

    graphs = []
    for subgraph in subgraphs:
        nodes = {}
        order = []
        for node in subgraph.graph.nodes:
            nodes[node.name] = {
                "op": node.op,
                "target": encode(node.target),
                "args": encode(node.args),
                "kwargs": encode(node.kwargs),
            }
            if node.op != "placeholder":
                order.append(node.name)
        graphs.append({"nodes": nodes, "execution_order": order})
    return fingerprint_streaming_value(graphs)


def streaming_input_contract(value: Any, *, specialize_numeric: bool = False) -> Any:
    """Permit tensor lengths and numeric metadata to vary, preserving branch flags."""
    if isinstance(value, torch.Tensor):
        return ("tensor", str(value.dtype), value.ndim)
    if isinstance(value, Mapping):
        return {
            key: streaming_input_contract(item, specialize_numeric=specialize_numeric)
            for key, item in value.items()
        }
    if is_dataclass(value) and not isinstance(value, type):
        return (
            f"{type(value).__module__}.{type(value).__qualname__}",
            {
                field.name: streaming_input_contract(
                    getattr(value, field.name), specialize_numeric=specialize_numeric
                )
                for field in fields(value)
            },
        )
    if isinstance(value, (tuple, list)):
        return (
            type(value).__name__,
            tuple(
                streaming_input_contract(item, specialize_numeric=specialize_numeric)
                for item in value
            ),
        )
    if type(value) in (int, float) and not specialize_numeric:
        return (type(value).__name__,)
    if value is None or isinstance(value, (bool, int, float, str)):
        return (type(value).__name__, value)
    raise TypeError(f"Unsupported traced input type {type(value).__name__}")


class FrozenCalibrationInputs(Sequence):
    """Replay the exact CPU batches included in the execution identity."""

    def __init__(self, store, count, fingerprint):
        self.store = store
        self.count = count
        self.fingerprint = fingerprint

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not isinstance(index, int) or index < 0 or index >= self.count:
            raise IndexError(index)
        return self.store.get(0, index)


def freeze_calibration_inputs(batches, root: str | Path) -> FrozenCalibrationInputs:
    from .activations import DiskBoundaryActivationStore
    from .admission import DiskSpaceBudget

    store = DiskBoundaryActivationStore(root)
    store.delete(0)
    count = 0
    contract = None

    def snapshot():
        nonlocal count, contract
        for index, batch in enumerate(batches):
            current = streaming_input_contract(batch)
            if contract is None:
                contract = current
            elif current != contract:
                raise ValueError(
                    "Calibration input structure or non-tensor branch changed; "
                    "use a separate traced streaming plan for this input combination"
                )
            DiskSpaceBudget(root).check(
                "calibration input snapshot", _input_bytes(batch) * 2 + 4096
            )
            store.put(0, index, batch)
            count += 1
            yield (index, batch)

    fingerprint = fingerprint_streaming_values(snapshot())
    if not count:
        raise ValueError("Streaming calibration inputs cannot be empty")
    return FrozenCalibrationInputs(store, count, fingerprint)


def _input_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.nbytes
    if isinstance(value, Mapping):
        return sum(_input_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_input_bytes(item) for item in value)
    return 0


@dataclass(frozen=True)
class StreamingExecutionDescriptor:
    """Serializable dependencies for one streaming execution graph."""

    modalities: tuple[str, ...] = ()
    branches: tuple[tuple[str, tuple[str, ...]], ...] = ()
    shared_tensors: tuple[tuple[str, str], ...] = ()
    runtime_dependencies: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = ()
    input_fingerprint: str = ""
    topology_fingerprint: str = ""
    processor_fingerprint: str = ""
    graph_fingerprint: str = ""
    model_configuration_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "modalities": list(self.modalities),
            "branches": [
                {"name": name, "stages": list(stages)} for name, stages in self.branches
            ],
            "shared_tensors": [
                {"alias": alias, "canonical": canonical}
                for alias, canonical in self.shared_tensors
            ],
            "runtime_dependencies": [
                {
                    "stage": stage,
                    "modules": list(modules),
                    "parameters": list(parameters),
                }
                for stage, modules, parameters in self.runtime_dependencies
            ],
            "input_fingerprint": self.input_fingerprint,
            "topology_fingerprint": self.topology_fingerprint,
            "processor_fingerprint": self.processor_fingerprint,
            "graph_fingerprint": self.graph_fingerprint,
            "model_configuration_fingerprint": self.model_configuration_fingerprint,
        }

    def with_inputs(self, input_fingerprint: str, processor_identity: Any = None):
        return replace(
            self,
            input_fingerprint=input_fingerprint,
            processor_fingerprint=fingerprint_streaming_value(processor_identity),
        )

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        targets: Sequence[str],
        sample_batch: Mapping[str, Any] | None = None,
    ) -> "StreamingExecutionDescriptor":
        hook = getattr(model, "streaming_execution_metadata", None)
        raw = hook() if callable(hook) else {}
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise TypeError("streaming_execution_metadata() must return a mapping")

        input_names = set(sample_batch or {})
        inferred = []
        for modality, names in (
            ("text", {"input_ids", "inputs_embeds"}),
            ("image", {"pixel_values", "images", "image_grid_thw"}),
            ("audio", {"input_features", "input_values", "audio_values"}),
            ("video", {"pixel_values_videos", "video_grid_thw"}),
        ):
            if input_names & names:
                inferred.append(modality)
        modalities = tuple(str(item) for item in raw.get("modalities", inferred))
        branches = []
        for name, stages in raw.get("branches", {"main": tuple(targets)}).items():
            branches.append((str(name), tuple(str(stage) for stage in stages)))

        shared = raw.get("shared_tensors", ())
        if isinstance(shared, Mapping):
            shared = shared.items()
        shared_tensors = tuple(
            (str(alias), str(canonical)) for alias, canonical in shared
        )
        canonical_names = {}
        inferred_shared = []
        for name, parameter in model.named_parameters(remove_duplicate=False):
            canonical = canonical_names.setdefault(id(parameter), name)
            if canonical != name:
                inferred_shared.append((name, canonical))
        shared_tensors = tuple(dict.fromkeys((*shared_tensors, *inferred_shared)))

        dependencies = []
        for item in raw.get("runtime_dependencies", ()):
            if not isinstance(item, Mapping):
                raise TypeError("runtime_dependencies entries must be mappings")
            dependencies.append(
                (
                    str(item["stage"]),
                    tuple(str(name) for name in item.get("modules", ())),
                    tuple(str(name) for name in item.get("parameters", ())),
                )
            )

        descriptor = cls(
            modalities=modalities,
            branches=tuple(branches),
            shared_tensors=shared_tensors,
            runtime_dependencies=tuple(dependencies),
            input_fingerprint=(
                fingerprint_streaming_value(sample_batch)
                if sample_batch is not None
                else ""
            ),
            model_configuration_fingerprint=fingerprint_streaming_value(
                {
                    "config": streaming_configuration(getattr(model, "config", None)),
                    "classes": [
                        (name, type(module).__module__, type(module).__qualname__)
                        for name, module in model.named_modules()
                    ],
                }
            ),
        )
        topology = descriptor.to_dict()
        topology.pop("input_fingerprint", None)
        return cls(
            **{
                **descriptor.__dict__,
                "topology_fingerprint": fingerprint_streaming_value(topology),
            }
        )

    def validate(self, model: nn.Module, targets: Sequence[str]) -> None:
        target_set = set(targets)
        if len(target_set) != len(tuple(targets)):
            raise ValueError("Streaming execution targets must be unique")
        for name, stages in self.branches:
            if not name:
                raise ValueError("Streaming execution branch names cannot be empty")
            unknown = set(stages) - target_set - {"input", "prefix"}
            if unknown:
                raise ValueError(
                    f"Execution branch {name!r} references unknown stages "
                    f"{sorted(unknown)}"
                )
        parameter_names = {
            name for name, _parameter in model.named_parameters(remove_duplicate=False)
        }
        for stage, modules, parameters in self.runtime_dependencies:
            if stage not in target_set and stage != "prefix":
                raise ValueError(f"Unknown runtime dependency stage {stage!r}")
            for name in modules:
                if not name:
                    raise ValueError(
                        "Runtime dependencies must not load the whole model"
                    )
                model.get_submodule(name)
            missing = set(parameters) - parameter_names
            if missing:
                raise ValueError(
                    f"Runtime dependency parameters are missing: {sorted(missing)}"
                )
        state = model.state_dict(keep_vars=True)
        state_names = set(state)
        for alias, canonical in self.shared_tensors:
            if alias not in state_names or canonical not in state_names:
                raise ValueError(
                    f"Shared tensor pair must exist in model state: {alias!r}, "
                    f"{canonical!r}"
                )
            if state[alias] is not state[canonical]:
                raise ValueError(
                    f"Declared shared tensors must alias the same object: "
                    f"{alias!r}, {canonical!r}"
                )


def describe_streaming_execution(
    model: nn.Module,
    targets: Sequence[str],
    sample_batch: Mapping[str, Any] | None = None,
) -> StreamingExecutionDescriptor:
    descriptor = StreamingExecutionDescriptor.from_model(model, targets, sample_batch)
    descriptor.validate(model, targets)
    return descriptor
