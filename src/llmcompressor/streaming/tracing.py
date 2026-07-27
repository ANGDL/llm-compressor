"""Streaming execution backed by the oneshot sequential trace plan."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from llmcompressor.utils.helpers import (
    disable_cache,
    disable_hf_kernels,
    eval_context,
)

from .checkpoint import CheckpointWeightSource
from .loading import SubgraphWeightSession, TargetWeightLoader
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


@dataclass
class TracedBoundaryAdapter:
    """Execute original sequential subgraphs with checkpoint-backed weights."""

    model: nn.Module
    plan: SequentialExecutionPlan
    loader: TargetWeightLoader
    weight_session: SubgraphWeightSession
    device: torch.device
    dtype: torch.dtype
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
        with self.weight_session.loaded(
            prefix,
            device=self.device,
            dtype=self.dtype,
            include_modules=self._prefix_fallback_modules(),
        ):
            with (
                torch.no_grad(),
                disable_cache(self.model),
                eval_context(self.model),
                disable_hf_kernels(self.model),
            ):
                for batch in batches:
                    if not isinstance(batch, Mapping):
                        raise TypeError(
                            "Traced streaming calibration batches must be mappings"
                        )
                    values = _move_tensors(dict(batch), self.device)
                    inputs = {name: values[name] for name in prefix.input_names}
                    yield prefix.forward(self.model, **inputs)

    def _prefix_fallback_modules(self) -> tuple[str, ...]:
        """Cover embeddings hidden behind traced call_function nodes."""

        if "input_ids" not in self.plan.subgraphs[0].input_names:
            return ()
        get_embeddings = getattr(self.model, "get_input_embeddings", None)
        if not callable(get_embeddings):
            return ()
        embedding = get_embeddings()
        return tuple(
            name for name, module in self.model.named_modules() if module is embedding
        )

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
            with self.weight_session.loaded(
                subgraph,
                device=self.device,
                dtype=self.dtype,
                exclude_modules=(target_name,),
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

    target_subgraphs = []
    for target_index, subgraph_index in enumerate(plan.target_subgraph_indices):
        target_name = plan.target_names[target_index]
        target_subgraph = _project_through_target(
            plan.subgraphs[subgraph_index], target_name
        )
        target_subgraphs.append(target_subgraph)

    loader = TargetWeightLoader(model, source, materializer)
    weight_session = SubgraphWeightSession(model, source, materializer)

    return TracedBoundaryAdapter(
        model=model,
        plan=plan,
        loader=loader,
        weight_session=weight_session,
        device=torch.device(device),
        dtype=dtype,
        _target_subgraphs=tuple(target_subgraphs),
    )
