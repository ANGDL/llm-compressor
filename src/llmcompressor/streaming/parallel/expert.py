"""Single-process expert ownership and concurrent execution of traced FX graphs."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import nn
from torch.fx import Graph, Node
from torch.fx.node import map_arg
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from llmcompressor.pipelines.sequential.helpers import Subgraph

from ..admission import ModifierMemoryEstimate, estimate_modifier_memory
from ..artifacts import fingerprint_json
from ..loading.session import SubgraphLoadPlan
from ..moe.proxy import install_local_expert_proxy
from .placement import PlacementEntry, PlacementPlan
from .topology import EPTopology


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
    expert_collections: tuple[str, ...] = ()
    expert_bytes: tuple[tuple[str, int], ...] = ()
    adapters: tuple[Any, ...] = ()
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
        *,
        adapters: Sequence[Any] = (),
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
        collections = _discover_expert_collections(model, targets, adapters)
        entries = []
        expert_sizes: dict[str, int] = {}
        for collection_name, collection in collections:
            children = tuple(
                (name, child)
                for name, child in collection._modules.items()
                if child is not None
            )
            for expert_index, (child_name, child) in enumerate(children):
                name = f"{collection_name}.{child_name}"
                if name in target_names or name in target_ancestors:
                    entries.append(PlacementEntry(name, 0, expert_id=expert_index))
                    expert_sizes[name] = _module_storage_bytes(child)
                elif any(ancestor in target_names for ancestor in _ancestors(name)):
                    raise ValueError(
                        "Expert parallelism requires independently traced expert "
                        f"targets; {name!r} is hidden inside a larger target"
                    )
        if not entries:
            raise ValueError("No independently traced expert collection was found")
        entries.sort(key=lambda entry: target_positions[entry.logical_name])
        owners = _balanced_expert_owners(
            tuple(entry.logical_name for entry in entries),
            expert_sizes,
            len(resolved),
        )
        entries = [
            replace(entry, owner_rank=owners[entry.logical_name])
            for entry in entries
        ]
        plan = cls(
            resolved,
            PlacementPlan(tuple(entries), ep_size=len(resolved)),
            {},
            tuple(collection_name for collection_name, _collection in collections),
            tuple(sorted(expert_sizes.items())),
            tuple(adapters),
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

    @classmethod
    def from_model_with_topology(
        cls,
        model: nn.Module,
        targets: Sequence[str],
        topology: EPTopology,
        devices: Sequence[torch.device | str],
        *,
        adapters: Sequence[Any] = (),
    ) -> "ExpertParallelPlan":
        """Build a local-device plan after validating an EP topology."""

        if topology.config.ep_size != len(devices):
            raise ValueError(
                "topology ep_size must match the number of expert owner devices"
            )
        if topology.config.dp_size != 1:
            raise NotImplementedError(
                "single-process ExpertParallelExecutor does not support dp_size > 1"
            )
        return cls.from_model(model, targets, devices, adapters=adapters)

    @property
    def owner_by_name(self) -> dict[str, int]:
        return dict(self._owners)

    def local_expert_ids(
        self, collection_name: str, owner_rank: int
    ) -> tuple[int, ...]:
        """Return stable local expert ids for an external adapter."""

        if owner_rank < 0 or owner_rank >= len(self.devices):
            raise ValueError("owner_rank is outside the EP device list")
        return tuple(
            entry.expert_id
            for entry in self.placement.entries
            if entry.logical_name.startswith(f"{collection_name}.")
            and entry.owner_rank == owner_rank
            and entry.expert_id is not None
        )

    @contextmanager
    def install_standard_proxies(
        self,
        model: nn.Module,
        *,
        topology: EPTopology | None = None,
    ):
        """Install adapter-owned local proxies for standard MoE forwards.

        Native traced paths that call individual expert leaves continue to use
        :class:`ExpertParallelExecutor`. This context is for model adapters
        whose parent MoE module calls an expert container directly.
        """

        if not self.adapters:
            yield
            return
        process_group = None if topology is None else topology.ep_group
        with ExitStack() as stack:
            for adapter in self.adapters:
                module_name = adapter.expert_module_name()
                local_ids = self.local_expert_ids(
                    module_name,
                    0 if topology is None else topology.ep_rank,
                )
                stack.enter_context(
                    install_local_expert_proxy(
                        model,
                        module_name,
                        local_ids,
                        process_group=process_group,
                    )
                )
            yield

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
            "expert_collections": list(self.expert_collections),
            "expert_bytes": [list(item) for item in self.expert_bytes],
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


def _module_storage_bytes(module: nn.Module) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in module.parameters()
    ) + sum(
        tensor.numel() * tensor.element_size()
        for tensor in module.buffers()
        if tensor is not None
    )


def _balanced_expert_owners(
    names: Sequence[str],
    sizes: Mapping[str, int],
    owner_count: int,
) -> dict[str, int]:
    """Assign experts by descending bytes with stable tie breaking."""

    totals = [0] * owner_count
    owners = {}
    ordered = sorted(
        names,
        key=lambda name: (-sizes.get(name, 0), names.index(name)),
    )
    for name in ordered:
        owner = min(range(owner_count), key=lambda rank: (totals[rank], rank))
        owners[name] = owner
        totals[owner] += sizes.get(name, 0)
    return owners


def _discover_expert_collections(
    model: nn.Module,
    targets: Sequence[str],
    adapters: Sequence[Any],
) -> tuple[tuple[str, nn.Module], ...]:
    names = []
    seen = set()
    for adapter in adapters:
        validate = getattr(adapter, "validate", None)
        if callable(validate):
            validate(model)
        module_name = adapter.expert_module_name()
        collection = model.get_submodule(module_name)
        if module_name not in seen:
            names.append((module_name, collection))
            seen.add(module_name)
    for collection_name, collection in model.named_modules():
        last_component = collection_name.rsplit(".", 1)[-1].lower()
        if not (
            getattr(collection, "_streaming_expert_collection", False)
            or "expert" in last_component
        ):
            continue
        children = tuple(
            child for child in collection._modules.values() if child is not None
        )
        if not children or not (
            isinstance(collection, (nn.ModuleList, nn.ModuleDict))
            or isinstance(getattr(collection, "num_experts", None), int)
        ):
            continue
        if collection_name not in seen:
            names.append((collection_name, collection))
            seen.add(collection_name)
    return tuple(names)


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


@dataclass(frozen=True)
class _PreparedExpertBatch:
    names: tuple[str, ...]
    args: tuple[Any, ...]
    kwargs: tuple[Any, ...]
    lengths: tuple[int, ...]
    output_lengths: tuple[tuple[int, ...], ...]
    output_indices: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class _ExpertBatchResult:
    values: tuple[Any, ...]
    event: Any | None = None


_DEFAULT_MAX_BATCH_INPUT_BYTES = 128 * 1024 * 1024
_DEFAULT_FORWARD_BATCH_SIZE = 4


class ExpertParallelExecutor:
    """Schedule independent expert calls concurrently and preserve FX dataflow."""

    def __init__(
        self,
        model: nn.Module,
        plan: ExpertParallelPlan,
        *,
        max_batch_input_bytes: int = _DEFAULT_MAX_BATCH_INPUT_BYTES,
        forward_batch_size: int = _DEFAULT_FORWARD_BATCH_SIZE,
    ):
        if (
            isinstance(max_batch_input_bytes, bool)
            or not isinstance(max_batch_input_bytes, int)
            or max_batch_input_bytes <= 0
        ):
            raise ValueError("max_batch_input_bytes must be a positive integer")
        if (
            isinstance(forward_batch_size, bool)
            or not isinstance(forward_batch_size, int)
            or forward_batch_size <= 0
        ):
            raise ValueError("forward_batch_size must be a positive integer")
        self.model = model
        self.plan = plan
        self.max_batch_input_bytes = max_batch_input_bytes
        self.forward_batch_size = forward_batch_size
        self._workers = {
            device: ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"ep-{index}")
            for index, device in enumerate(plan.devices)
        }
        self._streams = {
            device: (
                torch.cuda.Stream(device=device)
                if device.type == "cuda"
                else None
            )
            for device in plan.devices
        }
        self._graphs: dict[tuple[int, ...], _ExecutionGraph] = {}

    def close(self) -> None:
        self.synchronize()
        for worker in self._workers.values():
            worker.shutdown(wait=True, cancel_futures=True)

    def synchronize(self) -> None:
        """Wait for queued owner streams before weights are unloaded or changed."""

        for stream in self._streams.values():
            if stream is not None:
                stream.synchronize()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @staticmethod
    def _copy_inputs(value, device, stream):
        def copy(item):
            if not isinstance(item, torch.Tensor):
                return item
            return item.to(device, non_blocking=stream is not None)

        return tree_map(copy, value)

    @staticmethod
    def _copy_outputs(value, primary, stream, primary_stream):
        def copy(item):
            if not isinstance(item, torch.Tensor):
                return item
            output = item.to(primary, non_blocking=stream is not None)
            if (
                stream is not None
                and primary_stream is not None
                and output.device.type == "cuda"
            ):
                output.record_stream(primary_stream)
            return output

        return tree_map(copy, value)

    @staticmethod
    def _same_static_value(first, current):
        if type(first) is not type(current):
            return False
        if isinstance(first, (bool, int, float, str, bytes, type(None))):
            return first == current
        return first is current

    @staticmethod
    def _call_leaves(args, kwargs):
        return tree_flatten((args, kwargs))

    @staticmethod
    def _same_device(actual, expected):
        if actual.type != expected.type:
            return False
        return actual.type != "cuda" or actual == expected

    @classmethod
    def _inputs_compatible(cls, entries, primary):
        if not entries:
            return False
        first_leaves, first_spec = cls._call_leaves(
            entries[0][2], entries[0][3]
        )
        tensor_indices = []
        for leaf_index, leaf in enumerate(first_leaves):
            if isinstance(leaf, torch.Tensor):
                if leaf.ndim < 1 or not cls._same_device(leaf.device, primary):
                    return False
                tensor_indices.append(leaf_index)
            else:
                if leaf is not None and not isinstance(
                    leaf, (bool, int, float, str, bytes)
                ):
                    return False
        if not tensor_indices:
            return False

        for _node, _name, args, kwargs in entries:
            leaves, spec = cls._call_leaves(args, kwargs)
            if spec != first_spec or len(leaves) != len(first_leaves):
                return False
            task_length = None
            tensor_ids = set()
            for leaf_index, (first, current) in enumerate(
                zip(first_leaves, leaves)
            ):
                if isinstance(first, torch.Tensor):
                    if (
                        not isinstance(current, torch.Tensor)
                        or current.ndim != first.ndim
                        or not cls._same_device(current.device, primary)
                        or current.dtype != first.dtype
                        or current.shape[1:] != first.shape[1:]
                        or id(current) in tensor_ids
                    ):
                        return False
                    tensor_ids.add(id(current))
                    if task_length is None:
                        task_length = current.shape[0]
                    elif task_length != current.shape[0]:
                        return False
                elif not cls._same_static_value(first, current):
                    return False
            if task_length is None:
                return False
        return True

    @classmethod
    def _prepare_batch(cls, entries, primary):
        if not cls._inputs_compatible(entries, primary):
            return None
        first_leaves, spec = cls._call_leaves(entries[0][2], entries[0][3])
        grouped_entries: dict[str, list[tuple[int, Any]]] = {}
        for entry_index, (_node, name, args, kwargs) in enumerate(entries):
            grouped_entries.setdefault(name, []).append(
                (entry_index, (args, kwargs))
            )

        names = []
        lengths = []
        output_lengths = []
        output_indices = []
        grouped_args = []
        grouped_kwargs = []
        for name, group in grouped_entries.items():
            names.append(name)
            group_leaves = [
                cls._call_leaves(args, kwargs)[0]
                for _entry_index, (args, kwargs) in group
            ]
            group_lengths = tuple(
                next(
                    leaves[leaf_index].shape[0]
                    for leaf_index in range(len(first_leaves))
                    if isinstance(leaves[leaf_index], torch.Tensor)
                )
                for leaves in group_leaves
            )
            lengths.append(sum(group_lengths))
            output_lengths.append(group_lengths)
            output_indices.append(tuple(entry_index for entry_index, _ in group))
            combined_leaves = [None] * len(first_leaves)
            for leaf_index, first in enumerate(first_leaves):
                if isinstance(first, torch.Tensor):
                    combined_leaves[leaf_index] = torch.cat(
                        [
                            leaves[leaf_index]
                            for leaves in group_leaves
                        ],
                        dim=0,
                    )
                else:
                    combined_leaves[leaf_index] = first
            combined_args, combined_kwargs = tree_unflatten(
                combined_leaves, spec
            )
            grouped_args.append(combined_args)
            grouped_kwargs.append(combined_kwargs)
        return _PreparedExpertBatch(
            tuple(names),
            tuple(grouped_args),
            tuple(grouped_kwargs),
            tuple(lengths),
            tuple(output_lengths),
            tuple(output_indices),
        )

    def _group_ready_experts(self, entries, primary):
        groups = []
        current = []
        current_bytes = 0
        for entry in entries:
            input_bytes = sum(
                item.numel() * item.element_size()
                for item in self._call_leaves(entry[2], entry[3])[0]
                if isinstance(item, torch.Tensor)
            )
            exceeds_budget = (
                current
                and current_bytes + input_bytes > self.max_batch_input_bytes
            )
            incompatible = (
                current
                and not self._inputs_compatible(current + [entry], primary)
            )
            if exceeds_budget or incompatible:
                groups.append(current)
                current = []
                current_bytes = 0
            current.append(entry)
            current_bytes += input_bytes
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _copy_batch_outputs(outputs, lengths, primary, stream, primary_stream):
        flattened = [tree_flatten(output) for output in outputs]
        first_leaves, first_spec = flattened[0]
        for leaves, spec in flattened[1:]:
            if spec != first_spec or len(leaves) != len(first_leaves):
                return tuple(
                    ExpertParallelExecutor._copy_outputs(
                        output, primary, stream, primary_stream
                    )
                    for output in outputs
                )
        for leaf_index, first in enumerate(first_leaves):
            if isinstance(first, torch.Tensor):
                if (
                    first.ndim < 1
                    or first.shape[0] != lengths[0]
                    or any(
                        not isinstance(leaves[leaf_index], torch.Tensor)
                        or leaves[leaf_index].ndim != first.ndim
                        or leaves[leaf_index].shape[0] != lengths[index]
                        or leaves[leaf_index].shape[1:] != first.shape[1:]
                        or leaves[leaf_index].dtype != first.dtype
                        for index, (leaves, _spec) in enumerate(flattened)
                    )
                ):
                    return tuple(
                        ExpertParallelExecutor._copy_outputs(
                            output, primary, stream, primary_stream
                        )
                        for output in outputs
                    )
            elif any(
                not ExpertParallelExecutor._same_static_value(
                    first, leaves[leaf_index]
                )
                for leaves, _spec in flattened
            ):
                return tuple(
                    ExpertParallelExecutor._copy_outputs(
                        output, primary, stream, primary_stream
                    )
                    for output in outputs
                )

        split_leaves = [[] for _ in outputs]
        for leaf_index, first in enumerate(first_leaves):
            if isinstance(first, torch.Tensor):
                combined = torch.cat(
                    [leaves[leaf_index] for leaves, _spec in flattened],
                    dim=0,
                )
                copied = combined.to(
                    primary, non_blocking=stream is not None
                )
                if (
                    stream is not None
                    and primary_stream is not None
                    and copied.device.type == "cuda"
                ):
                    copied.record_stream(primary_stream)
                pieces = copied.split(lengths, dim=0)
                for index, piece in enumerate(pieces):
                    split_leaves[index].append(piece)
            else:
                for index in range(len(outputs)):
                    split_leaves[index].append(first)
        return tuple(
            tree_unflatten(leaves, first_spec) for leaves in split_leaves
        )

    @staticmethod
    def _split_output(output, lengths):
        leaves, spec = tree_flatten(output)
        split_leaves = [[] for _length in lengths]
        for leaf in leaves:
            if isinstance(leaf, torch.Tensor):
                pieces = leaf.split(lengths, dim=0)
                for index, piece in enumerate(pieces):
                    split_leaves[index].append(piece)
            else:
                for split in split_leaves:
                    split.append(leaf)
        return tuple(tree_unflatten(item, spec) for item in split_leaves)

    def _expert_forward(self, name, args, kwargs, input_event, primary_stream):
        device = self.plan.device_for(name)
        primary = self.plan.devices[0]
        context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        owner_stream = self._streams[device]
        with context:
            stream_context = (
                torch.cuda.stream(owner_stream)
                if owner_stream is not None
                else nullcontext()
            )
            with stream_context:
                if input_event is not None:
                    owner_stream.wait_event(input_event)
                moved_args, moved_kwargs = self._copy_inputs(
                    (args, kwargs), device, owner_stream
                )
                output = self.model.get_submodule(name)(
                    *moved_args, **moved_kwargs
                )
                output = self._copy_outputs(
                    output, primary, owner_stream, primary_stream
                )
                output_event = (
                    torch.cuda.Event() if owner_stream is not None else None
                )
                if output_event is not None:
                    output_event.record(owner_stream)
                return _ExpertBatchResult((output,), output_event)

    def _expert_forward_batch(
        self, prepared, input_event, primary_stream
    ):
        device = self.plan.device_for(prepared.names[0])
        if any(self.plan.device_for(name) != device for name in prepared.names):
            raise ValueError("An expert batch must have one owner device")
        primary = self.plan.devices[0]
        context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        owner_stream = self._streams[device]
        with context:
            stream_context = (
                torch.cuda.stream(owner_stream)
                if owner_stream is not None
                else nullcontext()
            )
            with stream_context:
                if input_event is not None:
                    owner_stream.wait_event(input_event)
                moved_args, moved_kwargs = self._copy_inputs(
                    (prepared.args, prepared.kwargs), device, owner_stream
                )
                grouped_outputs = []
                for name, task_args, task_kwargs, output_lengths in zip(
                    prepared.names,
                    moved_args,
                    moved_kwargs,
                    prepared.output_lengths,
                ):
                    grouped_outputs.append(
                        self._split_output(
                            self.model.get_submodule(name)(
                                *task_args, **task_kwargs
                            ),
                            output_lengths,
                        )
                    )
                outputs = [None] * sum(
                    len(indices) for indices in prepared.output_indices
                )
                for group_outputs, indices in zip(
                    grouped_outputs, prepared.output_indices
                ):
                    for output, index in zip(group_outputs, indices):
                        outputs[index] = output
                outputs = self._copy_batch_outputs(
                    outputs,
                    tuple(
                        length
                        for output_lengths in prepared.output_lengths
                        for length in output_lengths
                    ),
                    primary,
                    owner_stream,
                    primary_stream,
                )
                output_event = (
                    torch.cuda.Event() if owner_stream is not None else None
                )
                if output_event is not None:
                    output_event.record(owner_stream)
                return _ExpertBatchResult(outputs, output_event)

    @torch.no_grad()
    def forward(
        self, subgraphs: Sequence[Subgraph], value: Mapping[str, Any]
    ) -> dict[str, Any]:
        return self.forward_many(subgraphs, (value,))[0]

    @torch.no_grad()
    def forward_many(
        self,
        subgraphs: Sequence[Subgraph],
        values: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        """Execute independent inputs while batching repeated expert calls."""
        if not values:
            return ()
        key = tuple(id(subgraph) for subgraph in subgraphs)
        if key not in self._graphs:
            self._graphs.clear()
            self._graphs[key] = _ExecutionGraph.build(subgraphs)
        schedule = self._graphs[key]
        dependencies, users = schedule.dependencies, schedule.users
        ready = deque(
            (index, node)
            for index in range(len(values))
            for node in schedule.ready
        )
        node_values = [dict(value) for value in values]
        pending: dict[Future, tuple[tuple[int, Node], ...]] = {}
        remaining_users = [
            schedule.remaining_users.copy() for _value in values
        ]
        outputs = [None] * len(values)
        primary = self.plan.devices[0]
        primary_stream = (
            torch.cuda.current_stream(primary)
            if primary.type == "cuda"
            else None
        )

        def finish(index, node, result, event=None):
            if event is not None:
                primary_stream.wait_event(event)
            node_values[index][node] = result
            if node.op == "output":
                outputs[index] = result
            for source in dependencies[node]:
                remaining_users[index][source] -= 1
                if remaining_users[index][source] == 0:
                    node_values[index].pop(source, None)
            for user in users[node]:
                unmet[index][user] -= 1
                if unmet[index][user] == 0:
                    ready.append((index, user))

        unmet = [schedule.unmet.copy() for _value in values]
        try:
            while ready or pending:
                ready_experts = {}
                while ready:
                    index, node = ready.popleft()
                    args, kwargs = map_arg(
                        (node.args, node.kwargs),
                        node_values[index].__getitem__,
                    )
                    if node.op == "placeholder":
                        finish(index, node, values[index][str(node.target)])
                    elif node.op == "get_attr":
                        attribute = self.model
                        for component in str(node.target).split("."):
                            attribute = getattr(attribute, component)
                        finish(index, node, attribute)
                    elif node.op == "call_module":
                        name = str(node.target)
                        if self.plan.is_expert(name):
                            ready_experts.setdefault(
                                self.plan.device_for(name), []
                            ).append(((index, node), name, args, kwargs))
                        else:
                            finish(
                                index,
                                node,
                                self.model.get_submodule(name)(*args, **kwargs),
                            )
                    elif node.op == "call_function":
                        finish(index, node, node.target(*args, **kwargs))
                    elif node.op == "call_method":
                        finish(
                            index,
                            node,
                            getattr(args[0], str(node.target))(*args[1:], **kwargs),
                        )
                    elif node.op == "output":
                        finish(index, node, args[0])
                    else:
                        raise ValueError(
                            f"Unsupported expert parallel FX operation {node.op}"
                        )
                dispatches = []
                primary_context = (
                    torch.cuda.device(primary)
                    if primary.type == "cuda"
                    else nullcontext()
                )
                primary_stream_context = (
                    torch.cuda.stream(primary_stream)
                    if primary_stream is not None
                    else nullcontext()
                )
                with primary_context, primary_stream_context:
                    for owner, entries in ready_experts.items():
                        for group in self._group_ready_experts(entries, primary):
                            prepared = (
                                self._prepare_batch(group, primary)
                                if len(group) > 1
                                else None
                            )
                            dispatches.append((owner, group, prepared))
                    input_event = (
                        primary_stream.record_event()
                        if primary_stream is not None and dispatches
                        else None
                    )
                for owner, entries, prepared in dispatches:
                    if prepared is not None:
                        future = self._workers[owner].submit(
                            self._expert_forward_batch,
                            prepared,
                            input_event,
                            primary_stream,
                        )
                        pending[future] = tuple(entry[0] for entry in entries)
                        continue
                    for _node, name, args, kwargs in entries:
                        future = self._workers[owner].submit(
                            self._expert_forward,
                            name,
                            args,
                            kwargs,
                            input_event,
                            primary_stream,
                        )
                        pending[future] = (_node,)
                if pending:
                    completed, _remaining = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        nodes = pending.pop(future)
                        result = future.result()
                        if isinstance(result, _ExpertBatchResult):
                            if result.event is not None:
                                primary_stream.wait_event(result.event)
                            if len(nodes) != len(result.values):
                                raise RuntimeError(
                                    "Expert batch returned an unexpected result count"
                                )
                            for (index, node), result_value in zip(
                                nodes, result.values
                            ):
                                finish(index, node, result_value)
                        else:
                            if len(nodes) != 1:
                                raise RuntimeError(
                                    "Expert batch fallback returned multiple nodes"
                                )
                            index, node = nodes[0]
                            finish(index, node, result)
        except BaseException:
            for future in pending:
                future.cancel()
            wait(pending)
            raise
        if any(output is None for output in outputs):
            raise RuntimeError("Expert parallel graph did not produce an output")
        return tuple(
            {
                **{
                    name: item
                    for name, item in value.items()
                    if name not in schedule.consumed
                },
                **output,
            }
            for value, output in zip(values, outputs)
        )
