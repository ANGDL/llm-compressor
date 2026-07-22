"""Streaming execution backed by the oneshot sequential trace plan."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import nn
from torch.fx import Graph

from llmcompressor.pipelines.sequential.helpers import Subgraph
from llmcompressor.pipelines.sequential.plan import (
    SequentialExecutionPlan,
    trace_sequential_plan,
)

from .checkpoint import CheckpointWeightSource
from .loading import TargetWeightLoader
from .materialization import WeightMaterializer

__all__ = ["TracedBoundaryAdapter", "trace_streaming_boundaries"]


def _project_through_target(subgraph: Subgraph, target_name: str) -> Subgraph:
    """Return the original partition through its target, excluding its suffix."""

    graph = Graph()
    node_map = {}
    target_node = None
    for node in subgraph.graph.nodes:
        if node.op == "output":
            break
        copied = graph.node_copy(node, lambda dependency: node_map[dependency])
        node_map[node] = copied
        if node.op == "call_module" and str(node.target) == target_name:
            target_node = copied
            break
    if target_node is None:
        raise ValueError(f"Target {target_name!r} is absent from traced partition")
    graph.output({target_node.name: target_node})
    for node in reversed(tuple(graph.nodes)):
        if node.op in {"placeholder", "get_attr"} and not node.users:
            graph.erase_node(node)
    graph.lint()
    return Subgraph(
        graph=graph,
        input_names={
            str(node.target) for node in graph.nodes if node.op == "placeholder"
        },
        consumed_names=set(),
    )


def _move_tensors(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    return value


@contextmanager
def _loaded_modules(
    loader: TargetWeightLoader,
    module_names: Sequence[str],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Iterator[None]:
    contexts = []
    try:
        for name in module_names:
            context = loader.loaded(name, device=device, dtype=dtype)
            context.__enter__()
            contexts.append(context)
        yield
    finally:
        for context in reversed(contexts):
            context.__exit__(None, None, None)


@dataclass
class TracedBoundaryAdapter:
    """Execute original sequential subgraphs with checkpoint-backed weights."""

    model: nn.Module
    plan: SequentialExecutionPlan
    loader: TargetWeightLoader
    device: torch.device
    dtype: torch.dtype
    _prefix_modules: tuple[str, ...]
    _target_modules: tuple[tuple[str, ...], ...]
    _target_subgraphs: tuple[Subgraph, ...]

    @property
    def targets(self) -> tuple[str, ...]:
        return self.plan.target_names

    @property
    def prefix(self):
        return self.plan.subgraphs[0]

    @property
    def target_subgraphs(self):
        return self._target_subgraphs

    def calibration_boundaries(
        self, batches: Sequence[Mapping[str, Any]] | Any
    ) -> Iterator[dict[str, Any]]:
        """Execute the traced prefix without rewriting the model input contract."""

        prefix = self.plan.subgraphs[0]
        with _loaded_modules(
            self.loader,
            self._prefix_modules,
            device=self.device,
            dtype=self.dtype,
        ):
            with torch.no_grad():
                for batch in batches:
                    if not isinstance(batch, Mapping):
                        raise TypeError(
                            "Traced streaming calibration batches must be mappings"
                        )
                    values = _move_tensors(dict(batch), self.device)
                    inputs = {name: values[name] for name in prefix.input_names}
                    yield prefix.forward(self.model, **inputs)

    def forward_target(self, target: nn.Module, value: Mapping[str, Any]):
        """Execute the original target partition from the shared trace plan."""

        target_name = getattr(target, "_streaming_target_name", None)
        if target_name not in self.plan.target_names:
            raise ValueError(f"Unknown streaming target {target_name!r}")
        target_index = self.plan.target_names.index(target_name)
        subgraph = self._target_subgraphs[target_index]
        inputs = {name: value[name] for name in subgraph.input_names}
        original = self.model.get_submodule(target_name)
        self.model.set_submodule(target_name, target)
        try:
            with _loaded_modules(
                self.loader,
                self._target_modules[target_index],
                device=self.device,
                dtype=self.dtype,
            ):
                output = subgraph.forward(self.model, **inputs)
        finally:
            self.model.set_submodule(target_name, original)
        result = {**value, **output}
        subgraph_index = self.plan.target_subgraph_indices[target_index]
        for name in self.plan.subgraphs[subgraph_index].consumed_names:
            result.pop(name, None)
        return result


def trace_streaming_boundaries(
    *,
    model: nn.Module,
    source: CheckpointWeightSource,
    sample_batch: Mapping[str, Any],
    sequential_targets: Sequence[str] | str | None = None,
    target_names: Sequence[str] | None = None,
    materializer: WeightMaterializer | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    tracing_ignore: Sequence[str] = (),
) -> TracedBoundaryAdapter:
    """Trace model boundaries with the same implementation used by oneshot."""

    plan = trace_sequential_plan(
        model,
        dict(sample_batch),
        sequential_targets,
        tracing_ignore,
        targets_per_subgraph=1,
    )
    if target_names is not None and tuple(target_names) != plan.target_names:
        raise ValueError(
            "Explicit target names differ from the shared sequential trace: "
            f"expected {plan.target_names}, got {tuple(target_names)}"
        )

    prefix = plan.subgraphs[0]
    prefix_modules = []
    available = set(source.tensor_names())
    input_embedding = (
        model.get_input_embeddings()
        if hasattr(model, "get_input_embeddings")
        else None
    )
    embedding_name = next(
        (name for name, module in model.named_modules() if module is input_embedding),
        None,
    )
    if "input_ids" in prefix.input_names and embedding_name is not None:
        prefix_modules.append(embedding_name)
    for node in prefix.graph.find_nodes(op="call_module"):
        name = str(node.target)
        if any(
            tensor == name or tensor.startswith(f"{name}.")
            for tensor in available
        ):
            prefix_modules.append(name)

    target_modules = []
    target_subgraphs = []
    for target_index, subgraph_index in enumerate(plan.target_subgraph_indices):
        target_name = plan.target_names[target_index]
        target_subgraph = _project_through_target(
            plan.subgraphs[subgraph_index], target_name
        )
        target_subgraphs.append(target_subgraph)
        dependencies = []
        for node in target_subgraph.graph.find_nodes(op="call_module"):
            name = str(node.target)
            if name == target_name or name.startswith(f"{target_name}."):
                continue
            if any(
                tensor == name or tensor.startswith(f"{name}.")
                for tensor in available
            ):
                dependencies.append(name)
        target_modules.append(tuple(dict.fromkeys(dependencies)))

    return TracedBoundaryAdapter(
        model=model,
        plan=plan,
        loader=TargetWeightLoader(model, source, materializer),
        device=torch.device(device),
        dtype=dtype,
        _prefix_modules=tuple(dict.fromkeys(prefix_modules)),
        _target_modules=tuple(target_modules),
        _target_subgraphs=tuple(target_subgraphs),
    )
