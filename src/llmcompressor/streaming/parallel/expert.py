"""Single-process expert ownership and concurrent execution of traced FX graphs."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import nn
from torch.fx import Graph, Node
from torch.fx.node import map_arg
from torch.utils._pytree import tree_map

from llmcompressor.pipelines.sequential.helpers import Subgraph

from ..admission import ModifierMemoryEstimate, estimate_modifier_memory
from ..artifacts import fingerprint_json
from ..loading.session import SubgraphLoadPlan
from .placement import PlacementEntry, PlacementPlan


def resolve_expert_parallel_devices(
    devices: Sequence[torch.device | str] | None,
    primary: torch.device,
) -> tuple[torch.device, ...]:
    """Validate an explicit single-process CUDA topology before loading weights."""
    if devices is None:
        return ()
    resolved = tuple(torch.device(device) for device in devices)
    if len(resolved) < 2:
        raise ValueError("expert_parallel_devices requires at least two devices")
    if any(device.type != "cuda" or device.index is None for device in resolved):
        raise ValueError(
            "Expert parallel devices must be explicit cuda:<index> devices"
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError("Expert parallel devices must be unique")
    if primary.index is None and primary.type == "cuda":
        primary = torch.device("cuda", torch.cuda.current_device())
    if resolved[0] != primary:
        raise ValueError(
            "The first expert parallel device must be the execution device"
        )
    if not torch.cuda.is_available() or any(
        device.index >= torch.cuda.device_count() for device in resolved
    ):
        raise ValueError("An expert parallel device is unavailable")
    return resolved


@dataclass
class ExpertParallelPlan:
    """Assign individually traced experts to devices without replicating weights."""

    devices: tuple[torch.device, ...]
    placement: PlacementPlan
    module_devices: dict[nn.Module, torch.device]
    _owners: dict[str, int] = field(init=False, repr=False)
    backend = "single-process-expert-parallel-v1"

    def __post_init__(self):
        self._owners = {
            entry.logical_name: entry.owner_rank for entry in self.placement.entries
        }

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        targets: Sequence[str],
        devices: Sequence[torch.device | str],
    ) -> ExpertParallelPlan:
        resolved = tuple(torch.device(device) for device in devices)
        if len(resolved) < 2:
            raise ValueError("Expert parallel placement requires at least two slots")
        target_names = set(targets)
        target_ancestors = set()
        target_positions = {}
        for target_index, target in enumerate(targets):
            target_positions.setdefault(target, target_index)
            ancestor = target.rpartition(".")[0]
            while ancestor:
                target_ancestors.add(ancestor)
                target_positions.setdefault(ancestor, target_index)
                ancestor = ancestor.rpartition(".")[0]
        entries = []
        for collection_name, collection in model.named_modules():
            if not (
                getattr(collection, "_streaming_expert_collection", False)
                or "expert" in collection_name.rsplit(".", 1)[-1].lower()
            ):
                continue
            children = tuple(
                (name, child)
                for name, child in collection._modules.items()
                if child is not None
            )
            if not children or not (
                isinstance(collection, (nn.ModuleList, nn.ModuleDict))
                or isinstance(getattr(collection, "num_experts", None), int)
            ):
                continue
            for expert_index, (child_name, _child) in enumerate(children):
                name = f"{collection_name}.{child_name}"
                if name in target_names or name in target_ancestors:
                    entries.append(
                        PlacementEntry(
                            name, expert_index % len(resolved), expert_id=expert_index
                        )
                    )
                elif any(ancestor in target_names for ancestor in _ancestors(name)):
                    raise ValueError(
                        "Expert parallelism requires independently traced expert "
                        f"targets; {name!r} is hidden inside a larger target"
                    )
        if not entries:
            raise ValueError("No independently traced expert collection was found")
        entries.sort(key=lambda entry: target_positions[entry.logical_name])
        entries = [
            replace(entry, owner_rank=index % len(resolved))
            for index, entry in enumerate(entries)
        ]
        plan = cls(
            resolved,
            PlacementPlan(tuple(entries), ep_size=len(resolved)),
            {},
        )
        identities: dict[int, torch.device] = {}
        for name, module in model.named_modules(remove_duplicate=False):
            owner = plan.device_for(name)
            previous = plan.module_devices.setdefault(module, owner)
            if previous != owner:
                raise ValueError(f"Module {name!r} is shared across expert owners")
            for value in (
                *module.parameters(recurse=False),
                *module.buffers(recurse=False),
            ):
                previous = identities.setdefault(id(value), owner)
                if previous != owner:
                    raise ValueError(
                        f"State in {name!r} is shared across expert owners"
                    )
        return plan

    def device_for(self, name: str) -> torch.device:
        current = name
        while current:
            if current in self._owners:
                return self.devices[self._owners[current]]
            current = current.rpartition(".")[0]
        return self.devices[0]

    def is_expert(self, name: str) -> bool:
        current = name
        while current:
            if current in self._owners:
                return True
            current = current.rpartition(".")[0]
        return False

    def split_load_plan(
        self, plan: SubgraphLoadPlan
    ) -> tuple[tuple[torch.device, SubgraphLoadPlan], ...]:
        """Preserve exact parameter plans while grouping them by owner device."""
        result = []
        for device in self.devices:
            targets = tuple(
                target
                for target in plan.targets
                if self.device_for(target.name) == device
            )
            if targets or device == self.devices[0]:
                result.append(
                    (
                        device,
                        SubgraphLoadPlan(
                            tuple(dict.fromkeys(target.name for target in targets)),
                            targets,
                        ),
                    )
                )
        return tuple(result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "devices": [str(device) for device in self.devices],
            "placement": self.placement.to_dict(),
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())

    def estimate_memory(
        self, modifiers, modules, *, device, num_tokens, max_batch_tokens
    ) -> ModifierMemoryEstimate:
        from llmcompressor.modifiers.gptq import GPTQModifier
        from llmcompressor.modifiers.gptq.helpers import (
            batch_key,
            quantization_peak_bytes,
        )

        owned = tuple(
            module for module in modules if self.module_devices[module] == device
        )
        estimate = estimate_modifier_memory(
            modifiers, owned, num_tokens=num_tokens, max_batch_tokens=max_batch_tokens
        )
        scratch = estimate.scratch_bytes
        for modifier in modifiers:
            if not isinstance(modifier, GPTQModifier):
                continue
            batches = {}
            for module in owned:
                if not modifier._is_calibrated(module):
                    continue
                key = batch_key(module)
                if key is None:
                    key = id(module)
                batches.setdefault(key, []).append(module)
            for members in batches.values():
                batch_size = (
                    len(members)
                    if modifier.batched_quantization == "auto"
                    else min(len(members), modifier.batched_quantization or 1)
                )
                scratch = max(
                    scratch,
                    batch_size
                    * quantization_peak_bytes(members[0], modifier.block_size),
                )
        return ModifierMemoryEstimate(
            estimate.held_bytes, scratch, estimate.details, estimate.host_bytes
        )


def _join_subgraphs(subgraphs: Sequence[Subgraph]) -> tuple[Graph, set[str]]:
    graph = Graph()
    values: dict[str, Node] = {}
    external: dict[str, Node] = {}
    consumed = set()
    for subgraph in subgraphs:
        node_map = {}
        for node in subgraph.graph.nodes:
            if node.op == "placeholder":
                name = str(node.target)
                if name not in values:
                    if name not in external:
                        external[name] = graph.placeholder(name)
                    values[name] = external[name]
                node_map[node] = values[name]
            elif node.op == "output":
                values.update(map_arg(node.args[0], node_map.__getitem__))
            else:
                node_map[node] = graph.node_copy(node, node_map.__getitem__)
        for name in subgraph.consumed_names:
            values.pop(name, None)
            consumed.add(name)
    graph.output(values)
    graph.lint()
    return graph, consumed


def _ancestors(name: str):
    current = name.rpartition(".")[0]
    while current:
        yield current
        current = current.rpartition(".")[0]


@dataclass(frozen=True)
class _ExecutionGraph:
    graph: Graph
    consumed: set[str]
    dependencies: dict[Node, tuple[Node, ...]]
    users: dict[Node, tuple[Node, ...]]
    ready: tuple[Node, ...]
    unmet: dict[Node, int]
    remaining_users: dict[Node, int]

    @classmethod
    def build(cls, subgraphs: Sequence[Subgraph]) -> _ExecutionGraph:
        graph, consumed = _join_subgraphs(subgraphs)
        nodes = tuple(graph.nodes)
        dependencies = {}
        users: dict[Node, list[Node]] = {node: [] for node in nodes}
        aliases: dict[Node, Node] = {}
        last_writes: dict[Node, Node] = {}
        last_effect = None
        for node in nodes:
            required = set(node.all_input_nodes)
            for source in tuple(required):
                writer = last_writes.get(aliases.get(source, source))
                if writer is not None:
                    required.add(writer)
            mutates = (
                node.op == "call_method"
                and str(node.target).endswith("_")
                and not str(node.target).startswith("__")
            )
            if mutates or (node.op == "call_function" and node.is_impure()):
                if last_effect is not None:
                    required.add(last_effect)
                last_effect = node
            if mutates:
                receiver = node.args[0]
                if not isinstance(receiver, Node):
                    raise ValueError("FX mutation receiver must be a graph node")
                root = aliases.get(receiver, receiver)
                aliases[node] = root
                last_writes[root] = node
            if node.op == "output" and last_effect is not None:
                required.add(last_effect)
            dependencies[node] = tuple(required)
            for source in required:
                users[source].append(node)
        return cls(
            graph,
            consumed,
            dependencies,
            {node: tuple(members) for node, members in users.items()},
            tuple(node for node in nodes if not dependencies[node]),
            {node: len(required) for node, required in dependencies.items()},
            {node: len(members) for node, members in users.items()},
        )


class ExpertParallelExecutor:
    """Schedule independent expert calls concurrently and preserve FX dataflow."""

    def __init__(self, model: nn.Module, plan: ExpertParallelPlan):
        self.model = model
        self.plan = plan
        self._workers = {
            device: ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"ep-{index}")
            for index, device in enumerate(plan.devices)
        }
        self._graphs: dict[tuple[int, ...], _ExecutionGraph] = {}

    def close(self) -> None:
        for worker in self._workers.values():
            worker.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @torch.no_grad()
    def _expert_forward(self, name, args, kwargs, input_event):
        device = self.plan.device_for(name)
        primary = self.plan.devices[0]
        if input_event is not None:
            input_event.synchronize()
        context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        with context:
            moved_args, moved_kwargs = tree_map(
                lambda value: (
                    value.to(device) if isinstance(value, torch.Tensor) else value
                ),
                (args, kwargs),
            )
            output = self.model.get_submodule(name)(*moved_args, **moved_kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            output = tree_map(
                lambda value: (
                    value.to(primary) if isinstance(value, torch.Tensor) else value
                ),
                output,
            )
            if primary.type == "cuda":
                torch.cuda.synchronize(primary)
            return output

    @torch.no_grad()
    def forward(
        self, subgraphs: Sequence[Subgraph], value: Mapping[str, Any]
    ) -> dict[str, Any]:
        key = tuple(id(subgraph) for subgraph in subgraphs)
        if key not in self._graphs:
            self._graphs.clear()
            self._graphs[key] = _ExecutionGraph.build(subgraphs)
        schedule = self._graphs[key]
        dependencies, users = schedule.dependencies, schedule.users
        ready = deque(schedule.ready)
        values: dict[Node, Any] = {}
        pending: dict[Future, Node] = {}
        remaining_users = schedule.remaining_users.copy()
        output = None

        def finish(node, result):
            nonlocal output
            values[node] = result
            if node.op == "output":
                output = result
            for source in dependencies[node]:
                remaining_users[source] -= 1
                if remaining_users[source] == 0:
                    values.pop(source, None)
            for user in users[node]:
                unmet[user] -= 1
                if unmet[user] == 0:
                    ready.append(user)

        unmet = schedule.unmet.copy()
        try:
            while ready or pending:
                while ready:
                    node = ready.popleft()
                    args, kwargs = map_arg((node.args, node.kwargs), values.__getitem__)
                    if node.op == "placeholder":
                        finish(node, value[str(node.target)])
                    elif node.op == "get_attr":
                        attribute = self.model
                        for component in str(node.target).split("."):
                            attribute = getattr(attribute, component)
                        finish(node, attribute)
                    elif node.op == "call_module":
                        name = str(node.target)
                        if self.plan.is_expert(name):
                            input_event = (
                                torch.cuda.current_stream(
                                    self.plan.devices[0]
                                ).record_event()
                                if self.plan.devices[0].type == "cuda"
                                else None
                            )
                            future = self._workers[self.plan.device_for(name)].submit(
                                self._expert_forward, name, args, kwargs, input_event
                            )
                            pending[future] = node
                        else:
                            finish(
                                node, self.model.get_submodule(name)(*args, **kwargs)
                            )
                    elif node.op == "call_function":
                        finish(node, node.target(*args, **kwargs))
                    elif node.op == "call_method":
                        finish(
                            node,
                            getattr(args[0], str(node.target))(*args[1:], **kwargs),
                        )
                    elif node.op == "output":
                        finish(node, args[0])
                    else:
                        raise ValueError(
                            f"Unsupported expert parallel FX operation {node.op}"
                        )
                if pending:
                    completed, _remaining = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        node = pending.pop(future)
                        finish(node, future.result())
        except BaseException:
            for future in pending:
                future.cancel()
            wait(pending)
            raise
        if output is None:
            raise RuntimeError("Expert parallel graph did not produce an output")
        return {
            **{
                name: item for name, item in value.items()
                if name not in schedule.consumed
            },
            **output,
        }
