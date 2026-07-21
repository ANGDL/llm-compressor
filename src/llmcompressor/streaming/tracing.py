"""Generic discovery of streaming boundaries from a model execution graph."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import nn
from torch.fx import Graph

from llmcompressor.pipelines.sequential.helpers import Subgraph, trace_subgraphs

from .checkpoint import CheckpointWeightSource
from .loading import TargetWeightLoader
from .materialization import WeightMaterializer

__all__ = ["TracedBoundaryAdapter", "trace_streaming_boundaries"]


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


def _move_module_state(module: nn.Module, device: torch.device) -> None:
    for child in module.modules():
        for name, buffer in child._buffers.items():
            if buffer is not None and not buffer.is_meta:
                child._buffers[name] = buffer.to(device)


def _truncate_after_target(subgraph: Subgraph, target_name: str) -> Subgraph:
    """Drop suffix operations so a target partition returns target output only."""

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
    graph.lint()
    return Subgraph(
        graph=graph,
        input_names={
            str(node.target) for node in graph.nodes if node.op == "placeholder"
        },
        consumed_names=set(),
    )


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
    """Execute the traced prefix once, then one sequential target at a time."""

    model: nn.Module
    targets: tuple[str, ...]
    prefix: Subgraph
    target_subgraphs: tuple[Subgraph, ...]
    loader: TargetWeightLoader
    device: torch.device
    dtype: torch.dtype
    _prefix_modules: tuple[str, ...]
    _embedding_name: str | None

    def calibration_boundaries(
        self, batches: Sequence[Mapping[str, Any]] | Any
    ) -> Iterator[dict[str, Any]]:
        """Yield first-target inputs produced by the traced model prefix."""

        _move_module_state(self.model, self.device)
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
                    inputs = _move_tensors(dict(batch), self.device)
                    if "input_ids" in inputs and "inputs_embeds" not in inputs:
                        if self._embedding_name is None:
                            raise ValueError(
                                "Traced prefix requires input embeddings, but no "
                                "checkpoint-backed input embedding was found"
                            )
                        inputs["inputs_embeds"] = self.model.get_submodule(
                            self._embedding_name
                        )(inputs["input_ids"])
                    prefix_inputs = {
                        name: inputs[name] for name in self.prefix.input_names
                    }
                    yield self.prefix.forward(self.model, **prefix_inputs)

    def forward_target(self, target: nn.Module, value: Mapping[str, Any]):
        """Execute the traced graph partition corresponding to ``target``."""

        index = int(getattr(target, "_streaming_target_index"))
        inputs = {
            name: value[name]
            for name in self.target_subgraphs[index].input_names
        }
        target_name = self.targets[index]
        original = self.model.get_submodule(target_name)
        self.model.set_submodule(target_name, target)
        try:
            output = self.target_subgraphs[index].forward(self.model, **inputs)
        finally:
            self.model.set_submodule(target_name, original)
        return {**value, **output}


def trace_streaming_boundaries(
    *,
    model: nn.Module,
    source: CheckpointWeightSource,
    sample_batch: Mapping[str, Any],
    sequential_targets: Sequence[str],
    target_names: Sequence[str],
    materializer: WeightMaterializer | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    tracing_ignore: Sequence[str] = (),
) -> TracedBoundaryAdapter:
    """Trace model-agnostic prefix and target boundary functions.

    The existing sequential pipeline tracer remains the source of truth for model
    call conventions, mask construction, rotary inputs, and other auxiliary values.
    Only checkpoint-backed modules in the prefix are temporarily materialized.
    """

    target_names = tuple(target_names)
    trace_batch = dict(sample_batch)
    embedding_name = None
    if "input_ids" in trace_batch and hasattr(model, "get_input_embeddings"):
        embedding = model.get_input_embeddings()
        embedding_name = next(
            (name for name, module in model.named_modules() if module is embedding),
            None,
        )
        input_ids = trace_batch.pop("input_ids")
        trace_batch["inputs_embeds"] = torch.empty(
            (*input_ids.shape, embedding.embedding_dim),
            dtype=dtype,
            device="meta",
        )
    subgraphs = trace_subgraphs(
        model,
        trace_batch,
        list(sequential_targets),
        list(tracing_ignore),
        targets_per_subgraph=1,
    )
    target_set = set(target_names)
    target_subgraphs = []
    prefix_subgraphs = []
    for subgraph in subgraphs:
        called = {
            str(node.target)
            for node in subgraph.graph.find_nodes(op="call_module")
        }
        matched = called & target_set
        if matched:
            if len(matched) != 1:
                raise ValueError(
                    "A traced streaming partition must contain exactly one "
                    f"sequential target; got {sorted(matched)}"
                )
            target_name = next(iter(matched))
            target_subgraphs.append(
                (target_name, _truncate_after_target(subgraph, target_name))
            )
        elif not target_subgraphs:
            prefix_subgraphs.append(subgraph)

    if len(prefix_subgraphs) != 1:
        raise ValueError(
            f"Expected one model prefix partition, got {len(prefix_subgraphs)}"
        )
    by_name = dict(target_subgraphs)
    missing = [name for name in target_names if name not in by_name]
    if missing:
        raise ValueError(f"Tracing did not produce target partitions for {missing}")

    prefix = prefix_subgraphs[0]
    prefix_modules = []
    for node in prefix.graph.find_nodes(op="call_module"):
        name = str(node.target)
        module = model.get_submodule(name)
        has_checkpoint_state = any(
            tensor_name == name or tensor_name.startswith(f"{name}.")
            for tensor_name in source.tensor_names()
        )
        if has_checkpoint_state and any(True for _ in module.parameters()):
            prefix_modules.append(name)
    if embedding_name is not None:
        prefix_modules.insert(0, embedding_name)

    return TracedBoundaryAdapter(
        model=model,
        targets=target_names,
        prefix=prefix,
        target_subgraphs=tuple(by_name[name] for name in target_names),
        loader=TargetWeightLoader(model, source, materializer),
        device=torch.device(device),
        dtype=dtype,
        _prefix_modules=tuple(dict.fromkeys(prefix_modules)),
        _embedding_name=embedding_name,
    )
