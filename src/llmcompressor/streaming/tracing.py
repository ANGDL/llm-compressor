"""Streaming execution backed by the oneshot sequential trace plan."""

from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from operator import getitem
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
from .materialization import StreamingDTypePolicy, WeightMaterializer

__all__ = ["TracedBoundaryAdapter", "trace_streaming_boundaries"]


class _PrefixDependencyDetected(RuntimeError):
    def __init__(self, module_name: str):
        super().__init__(module_name)
        self.module_name = module_name


def _project_through_target(subgraph: Subgraph, target_name: str) -> Subgraph:
    """Return the original partition through its target, excluding its suffix."""

    graph = Graph()
    node_map = {}
    target_node = None
    output_nodes = []
    for node in subgraph.graph.nodes:
        if node.op == "output":
            break
        if target_node is not None:
            if node.op != "call_function" or node.target is not getitem:
                break
            if any(dependency not in node_map for dependency in node.all_input_nodes):
                break
        copied = graph.node_copy(node, lambda dependency: node_map[dependency])
        node_map[node] = copied
        if node.op == "call_module" and str(node.target) == target_name:
            target_node = copied
            output_nodes = [(node, copied)]
        elif target_node is not None:
            output_nodes = [
                pair for pair in output_nodes if pair[0] not in node.all_input_nodes
            ]
            output_nodes.append((node, copied))
    if target_node is None:
        raise ValueError(f"Target {target_name!r} is absent from traced partition")
    graph.output({node.name: copied for node, copied in output_nodes})
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
    _prefix_runtime_modules: tuple[str, ...] | None = None

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
        batch_iterator = iter(batches)
        if self._prefix_runtime_modules is None:
            try:
                sample_batch = next(batch_iterator)
            except StopIteration as error:
                raise ValueError("Streaming calibration batches are empty") from error
            self._prefix_runtime_modules = self._discover_prefix_runtime_modules(
                prefix, sample_batch
            )
            batch_iterator = chain((sample_batch,), batch_iterator)
        with self.weight_session.loaded(
            prefix,
            device=self.device,
            dtype=self.dtype,
            include_modules=self._prefix_runtime_modules,
        ):
            with (
                torch.no_grad(),
                disable_cache(self.model),
                eval_context(self.model),
                disable_hf_kernels(self.model),
            ):
                for batch in batch_iterator:
                    if not isinstance(batch, Mapping):
                        raise TypeError(
                            "Traced streaming calibration batches must be mappings"
                        )
                    values = _move_tensors(dict(batch), self.device)
                    inputs = {name: values[name] for name in prefix.input_names}
                    yield prefix.forward(self.model, **inputs)

    def _discover_prefix_runtime_modules(
        self, prefix: Subgraph, sample_batch: Mapping[str, Any]
    ) -> tuple[str, ...]:
        """Discover checkpoint branches hidden inside autowrapped prefix calls."""

        if not isinstance(sample_batch, Mapping):
            raise TypeError("Traced streaming calibration batches must be mappings")
        candidates = self._prefix_dependency_candidates()
        discovered = []
        values = _move_tensors(dict(sample_batch), self.device)
        inputs = {name: values[name] for name in prefix.input_names}

        with (
            torch.no_grad(),
            disable_cache(self.model),
            eval_context(self.model),
            disable_hf_kernels(self.model),
        ):
            while True:
                pending = tuple(name for name in candidates if name not in discovered)
                with self.weight_session.loaded(
                    prefix,
                    device=self.device,
                    dtype=self.dtype,
                    include_modules=discovered,
                ):
                    try:
                        with self._detect_prefix_dependency_calls(pending):
                            prefix.forward(self.model, **inputs)
                    except Exception as error:
                        dependency = self._dependency_from_exception(error)
                        if dependency is None:
                            raise
                    else:
                        return tuple(discovered)
                discovered.append(dependency)

    def _prefix_dependency_candidates(self) -> tuple[str, ...]:
        target_ancestors = {""}
        for target_name in self.plan.target_names:
            current = target_name
            while current:
                target_ancestors.add(current)
                current, _, _ = current.rpartition(".")

        candidates = []
        for name, _module in self.model.named_modules():
            if not name or name in target_ancestors:
                continue
            parent, _, _ = name.rpartition(".")
            if parent in target_ancestors and self.weight_session.has_checkpoint_state(
                name
            ):
                candidates.append(name)
        return tuple(candidates)

    @contextmanager
    def _detect_prefix_dependency_calls(
        self, candidates: Sequence[str]
    ) -> Generator[None, None, None]:
        handles = []

        def hook(module_name: str, module: nn.Module, _args):
            if any(parameter.is_meta for parameter in module.parameters()):
                raise _PrefixDependencyDetected(module_name)
            if any(buffer.is_meta for buffer in module.buffers()):
                raise _PrefixDependencyDetected(module_name)

        try:
            for module_name in candidates:
                root = self.model.get_submodule(module_name)
                for module in root.modules():
                    handles.append(
                        module.register_forward_pre_hook(
                            lambda module, args, name=module_name: hook(
                                name, module, args
                            )
                        )
                    )
            yield
        finally:
            for handle in handles:
                handle.remove()

    @staticmethod
    def _dependency_from_exception(error: Exception) -> str | None:
        current: BaseException | None = error
        seen = set()
        while current is not None and id(current) not in seen:
            if isinstance(current, _PrefixDependencyDetected):
                return current.module_name
            seen.add(id(current))
            current = current.__cause__ or current.__context__
        return None

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
    dtype_policy: StreamingDTypePolicy | None = None,
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

    loader = TargetWeightLoader(
        model, source, materializer, dtype_policy=dtype_policy
    )
    weight_session = SubgraphWeightSession(
        model, source, materializer, dtype_policy=dtype_policy
    )

    return TracedBoundaryAdapter(
        model=model,
        plan=plan,
        loader=loader,
        weight_session=weight_session,
        device=torch.device(device),
        dtype=dtype,
        _target_subgraphs=tuple(target_subgraphs),
    )
