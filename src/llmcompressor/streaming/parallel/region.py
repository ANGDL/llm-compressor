"""Route-profiled expert regions with independent resident-window lifetimes."""

from __future__ import annotations

from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, replace
from math import ceil
from threading import Lock
from time import perf_counter

import torch
from torch import nn
from torch.fx import Graph, Node
from torch.fx.node import map_arg
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.sequential.helpers import Subgraph
from llmcompressor.utils.helpers import DisableQuantization

from .._logging import streaming_logger
from ..activations.store import _tensor_bytes
from ..admission import estimate_modifier_dispatch_memory
from ..calibration import calibration_batch_context
from .expert import _join_subgraphs
from .load_balance import (
    AsyncResidentWindowScheduler,
    ExpertCostProfile,
    OwnerMemoryProfile,
    build_load_balanced_placement,
    plan_resident_windows,
)
from .placement import PlacementEntry, PlacementPlan
from .token_budget import select_token_capacity


class UnsupportedExpertRegion(ValueError):
    """A graph cannot expose independent expert inputs without model execution."""


def _mutates(node):
    target_name = getattr(node.target, "__name__", "")
    schema = getattr(node.target, "_schema", None)
    return (
        node.op == "call_method"
        and str(node.target).endswith("_")
        and not str(node.target).startswith("__")
    ) or (
        node.op == "call_function"
        and (
            node.is_impure()
            or target_name == "setitem"
            or (target_name.endswith("_") and not target_name.startswith("__"))
            or getattr(schema, "is_mutable", False)
            or node.kwargs.get("out") is not None
        )
    )


def _rows(args, kwargs):
    tensors = [
        value
        for value in tree_flatten((args, kwargs))[0]
        if isinstance(value, torch.Tensor)
    ]
    if not tensors or tensors[0].ndim != 2:
        raise UnsupportedExpertRegion(
            "expert inputs must have a two-dimensional token axis"
        )
    count = tensors[0].shape[0]
    if any(value.ndim == 0 or value.shape[0] != count for value in tensors):
        raise UnsupportedExpertRegion(
            "expert tensor arguments must share their token axis"
        )
    return int(count)


def _move(value, device):
    memo = {}

    def move(item):
        if isinstance(item, torch.Tensor):
            if id(item) not in memo:
                memo[id(item)] = item.to(device=device)
            return memo[id(item)]
        return item

    return tree_map(move, value)


@dataclass
class ExpertRegionGraph:
    """An original FX graph and the pure dependency slices of its expert calls."""

    model: nn.Module
    graph: Graph
    consumed: set[str]
    calls: tuple[Node, ...]
    required: dict[str, set[Node]]

    @classmethod
    def build(cls, model, plan, subgraphs):
        graph, consumed = _join_subgraphs(subgraphs)
        calls = tuple(node for node in graph.nodes if node.op == "call_module")
        if not calls or any(not plan.is_expert(str(node.target)) for node in calls):
            raise UnsupportedExpertRegion("region contains non-expert module execution")
        if len({str(node.target) for node in calls}) != len(calls):
            raise UnsupportedExpertRegion(
                "an expert is called more than once per boundary"
            )
        required = {}
        for call in calls:
            dependencies = set(call.all_input_nodes)
            pending = list(dependencies)
            while pending:
                node = pending.pop()
                if node.op == "call_module" or _mutates(node):
                    raise UnsupportedExpertRegion(
                        "expert input depends on an effect or another expert"
                    )
                if node.op == "get_attr":
                    attribute = model
                    for component in str(node.target).split("."):
                        attribute = getattr(attribute, component)
                    if isinstance(attribute, torch.Tensor) and (
                        attribute.is_meta or attribute.device.type != "cpu"
                    ):
                        raise UnsupportedExpertRegion(
                            "routing requires unmaterialized model state"
                        )
                for source in node.all_input_nodes:
                    if source not in dependencies:
                        dependencies.add(source)
                        pending.append(source)
            required[str(call.target)] = dependencies
        return cls(model, graph, consumed, calls, required)

    @property
    def input_names(self):
        return {
            str(node.target) for node in self.graph.nodes if node.op == "placeholder"
        }

    @property
    def expert_names(self):
        return tuple(str(node.target) for node in self.calls)

    def _evaluate(self, value, required, *, metadata=False):
        memo = {}

        def prepare(item):
            if not isinstance(item, torch.Tensor):
                return item
            if metadata and item.is_floating_point():
                if id(item) not in memo:
                    memo[id(item)] = item.to(device="meta")
                return memo[id(item)]
            return item

        environment = {}
        for node in self.graph.nodes:
            if node not in required:
                continue
            args, kwargs = map_arg((node.args, node.kwargs), environment.__getitem__)
            if node.op == "placeholder":
                result = tree_map(prepare, value[str(node.target)])
            elif node.op == "get_attr":
                result = self.model
                for component in str(node.target).split("."):
                    result = getattr(result, component)
                result = tree_map(prepare, result)
            elif node.op == "call_function":
                result = node.target(*args, **kwargs)
            elif node.op == "call_method":
                result = getattr(args[0], str(node.target))(*args[1:], **kwargs)
            else:
                raise UnsupportedExpertRegion(
                    f"unsupported routing operation {node.op}"
                )
            environment[node] = result
        return environment

    def inputs(self, value, names=None, *, metadata=False):
        names = set(self.expert_names if names is None else names)
        required = set().union(*(self.required[name] for name in names))
        environment = self._evaluate(value, required, metadata=metadata)
        result = {}
        for call in self.calls:
            if str(call.target) in names:
                args, kwargs = map_arg(
                    (call.args, call.kwargs), environment.__getitem__
                )
                _rows(args, kwargs)
                result[str(call.target)] = (args, kwargs)
        return result

    def combine_subgraph(self):
        output = next(node for node in self.graph.nodes if node.op == "output")
        required = {output, *(node for node in self.graph.nodes if _mutates(node))}
        pending = list(required)
        calls = set(self.calls)
        while pending:
            node = pending.pop()
            if node in calls:
                continue
            for source in node.all_input_nodes:
                if source not in required:
                    required.add(source)
                    pending.append(source)
        graph = Graph()
        mapping = {}
        for node in self.graph.nodes:
            if node not in required:
                continue
            if node in calls:
                mapping[node] = graph.placeholder(f"ep_result_{node.name}")
            else:
                mapping[node] = graph.node_copy(node, mapping.__getitem__)
        graph.lint()
        return Subgraph(
            graph,
            input_names={
                str(node.target) for node in graph.nodes if node.op == "placeholder"
            },
            consumed_names=self.consumed,
        )


def coalesce_expert_regions(adapter, plan):
    """Merge adjacent independent experts within one collection across trace groups."""
    groups = []
    pending = []
    collection = None

    def flush():
        nonlocal pending
        if pending:
            try:
                ExpertRegionGraph.build(
                    adapter.model,
                    plan,
                    tuple(adapter.target_subgraphs[index] for index in pending),
                )
            except UnsupportedExpertRegion:
                groups.extend((index,) for index in pending)
            else:
                groups.append(tuple(pending))
            pending = []

    for original in adapter.target_groups:
        ordinary = []
        for index in original:
            name = adapter.targets[index]
            owner_collection = next(
                (
                    root
                    for root in plan.expert_collections
                    if name.startswith(root + ".")
                ),
                None,
            )
            eligible = owner_collection is not None
            if eligible:
                try:
                    ExpertRegionGraph.build(
                        adapter.model, plan, (adapter.target_subgraphs[index],)
                    )
                except UnsupportedExpertRegion:
                    eligible = False
            if not eligible:
                flush()
                ordinary.append(index)
                continue
            if ordinary:
                groups.append(tuple(ordinary))
                ordinary = []
            if owner_collection != collection:
                flush()
                collection = owner_collection
            pending.append(index)
        if ordinary:
            groups.append(tuple(ordinary))
    flush()
    return tuple(groups)


@dataclass(frozen=True)
class ExpertRegionProfile:
    token_counts: dict[str, int]
    batch_counts: dict[str, tuple[int, ...]]


def profile_expert_region(graph, boundaries, boundary_index, batches):
    counts = {name: [] for name in graph.expert_names}
    with torch.no_grad(), HooksMixin.disable_hooks_local():
        for batch in batches:
            value = boundaries.read_for_routing(
                boundary_index, batch, graph.input_names
            )
            try:
                inputs = graph.inputs(value, metadata=True)
            except (RuntimeError, NotImplementedError) as error:
                raise UnsupportedExpertRegion(
                    "route metadata requires floating-point model execution"
                ) from error
            for name, (args, kwargs) in inputs.items():
                counts[name].append(_rows(args, kwargs))
    return ExpertRegionProfile(
        {name: sum(values) for name, values in counts.items()},
        {name: tuple(values) for name, values in counts.items()},
    )


def _concatenate(entries):
    flattened = [tree_flatten(entry) for entry in entries]
    specification = flattened[0][1]
    if any(current != specification for _leaves, current in flattened):
        raise UnsupportedExpertRegion("expert call structures differ across boundaries")
    combined = []
    tensors = {}
    for position, first in enumerate(flattened[0][0]):
        leaves = [values[position] for values, _specification in flattened]
        if isinstance(first, torch.Tensor):
            identities = tuple(map(id, leaves))
            if identities not in tensors:
                tensors[identities] = torch.cat(leaves, dim=0)
            combined.append(tensors[identities])
        elif any(type(value) is not type(first) or value != first for value in leaves):
            raise UnsupportedExpertRegion(
                "expert static arguments differ across boundaries"
            )
        else:
            combined.append(first)
    return tree_unflatten(combined, specification)


def _split_result(result, lengths):
    leaves, specification = tree_flatten(result)
    outputs = [[] for _length in lengths]
    for leaf in leaves:
        if isinstance(leaf, torch.Tensor):
            if leaf.ndim == 0 or leaf.shape[0] != sum(lengths):
                raise UnsupportedExpertRegion(
                    "expert output does not preserve its token axis"
                )
            values = leaf.split(lengths, dim=0)
        else:
            values = [leaf] * len(lengths)
        for output, value in zip(outputs, values):
            output.append(value)
    return tuple(tree_unflatten(output, specification) for output in outputs)


def execute_expert_region(
    *,
    graph,
    profile,
    adapter,
    parallel,
    modifier_adapter,
    modifiers,
    boundaries,
    boundary_index,
    destination_boundary,
    batches,
    device_budget,
    host_budget,
    target_dtype,
    token_budget,
    label,
    save_window,
    propagate=True,
):
    """Calibrate, update, propagate, and release each owner's admitted windows."""
    costs = {}
    runtime = {}
    estimates = {}
    expert_modules = {}
    for name in graph.expert_names:
        modules = tuple(adapter.model.get_submodule(name).modules())
        expert_modules[name] = modules
        estimate = estimate_modifier_dispatch_memory(
            modifiers,
            modules,
            total_tokens=profile.token_counts[name],
            dispatch_tokens=0,
            phase="calibration",
        )
        estimates[name] = estimate
        plan = adapter.weight_session.plan_modules((name,), allow_missing_state=True)
        weights = adapter.weight_session.estimate_materialized_bytes(
            plan, dtype=target_dtype
        )
        costs[name] = ExpertCostProfile(
            storage_bytes=weights,
            route_tokens=profile.token_counts[name],
            persistent_state_bytes=estimate.peak_bytes,
        )
        linears = [module for module in modules if isinstance(module, nn.Linear)]
        if not linears:
            raise UnsupportedExpertRegion("expert has no bounded Linear workspace")
        runtime[name] = sum(
            4 * (2 * module.in_features + 2 * module.out_features) for module in linears
        ) + max(8 * module.in_features for module in linears)
    placement = build_load_balanced_placement(costs, len(parallel.devices))
    memory = []
    for owner, device in enumerate(parallel.devices):
        available = device_budget.usable_bytes(device)
        if available is None:
            raise UnsupportedExpertRegion(
                "resident windows require a known device budget"
            )
        names = [name for name, rank in placement.owners if rank == owner]
        maximum = sum(profile.token_counts[name] for name in names)
        if token_budget != "auto":
            maximum = min(maximum, token_budget)
        memory.append(
            OwnerMemoryProfile(
                usable_bytes=available,
                dispatch_bytes_per_token=max(
                    (runtime[name] for name in names), default=1
                ),
                max_dispatch_tokens=max(1, maximum),
            )
        )
    windows = plan_resident_windows(placement, costs, memory)

    def required_bytes(window, tokens):
        extra = max(
            max(
                0,
                estimate_modifier_dispatch_memory(
                    modifiers,
                    expert_modules[name],
                    total_tokens=profile.token_counts[name],
                    dispatch_tokens=tokens,
                    phase="calibration",
                ).peak_bytes
                - estimates[name].peak_bytes,
            )
            for name in window.expert_names
        )
        return (
            window.resident_bytes
            + extra
            + memory[window.owner_rank].dispatch_bytes_per_token * tokens
        )

    admitted_windows = []
    for window in windows:
        capacity = (
            select_token_capacity(
                owner=(
                    f"{label} resident window {window.owner_rank}:{window.window_index}"
                ),
                usable_bytes=memory[window.owner_rank].usable_bytes,
                required_bytes=lambda tokens: required_bytes(window, tokens),
                maximum_tokens=window.max_dispatch_tokens,
            )
            if window.route_tokens
            else 0
        )
        admitted_windows.append(
            replace(
                window,
                max_dispatch_tokens=capacity,
                activation_batches=sum(
                    max(1, ceil(profile.token_counts[name] / capacity))
                    if capacity
                    else 1
                    for name in window.expert_names
                ),
            )
        )
    windows = tuple(admitted_windows)
    replacements = placement.owner_by_name
    parallel.placement = PlacementPlan(
        tuple(
            PlacementEntry(
                entry.logical_name,
                replacements.get(entry.logical_name, entry.owner_rank),
                entry.replicated,
                entry.replica_group,
                entry.expert_id,
            )
            for entry in parallel.placement.entries
        ),
        ep_size=len(parallel.devices),
    )
    parallel._owners.update(replacements)
    for name, owner in replacements.items():
        for module in adapter.model.get_submodule(name).modules():
            parallel.module_devices[module] = parallel.devices[owner]
    parallel.placement_policy = placement.policy
    streaming_logger.info(
        f"{label} | EP routed windows | policy={placement.policy} | "
        f"experts={len(costs)} | windows={len(windows)} | "
        f"routed_tokens={sum(profile.token_counts.values()):,}"
    )
    for window in windows:
        streaming_logger.info(
            f"{label} | EP window | rank={window.owner_rank} | "
            f"window={window.window_index} | experts={len(window.expert_names)} | "
            f"tokens={window.route_tokens:,} | "
            f"capacity={window.max_dispatch_tokens:,} | "
            f"dispatches={window.activation_batches} | "
            f"resident_gib={window.resident_bytes / 2**30:.2f}"
        )
    result_store = {}
    result_reservations = []
    result_lock = Lock()
    save_lock = Lock()
    all_modules = tuple(
        dict.fromkeys(
            module
            for name in graph.expert_names
            for module in adapter.model.get_submodule(name).modules()
        )
    )
    begin = getattr(modifier_adapter, "begin_owner_update", None)
    finalize = getattr(modifier_adapter, "finalize_owner_update", None)
    if callable(begin):
        begin(all_modules)

    def prefetch(window):
        plan = adapter.weight_session.plan_modules(
            window.expert_names, allow_missing_state=True
        )
        required = adapter.weight_session.estimate_preparation(plan, dtype=target_dtype)
        reservation = host_budget.reserve(
            f"{label} owner {window.owner_rank} expert weights", required
        )
        try:
            return adapter.weight_session.prepare(
                plan, dtype=target_dtype, reservation=reservation
            )
        except BaseException:
            reservation.close()
            raise

    def forward_expert(name, capacity, *, capture):
        pieces = []
        entries = []
        lengths = []
        segments = []
        samples = 0
        tokens = 0
        device = parallel.device_for(name)

        def submit():
            nonlocal entries, lengths, segments, samples, tokens
            if not entries:
                return
            with host_budget.reserve(
                f"{label} expert dispatch {name}", _tensor_bytes(entries)
            ) as gathered:
                combined = _concatenate(entries)
                gathered.commit(_tensor_bytes(combined))
                args, kwargs = _move(combined, device)
                del combined
                context = HooksMixin.disable_hooks_local() if capture else nullcontext()
                with context, calibration_batch_context(lengths, sample_count=samples):
                    result = adapter.model.get_submodule(name)(*args, **kwargs)
                if capture:
                    size = _tensor_bytes(result)
                    reservation = host_budget.reserve(
                        f"{label} expert output {name}", size
                    )
                    with result_lock:
                        result_reservations.append(reservation)
                    copied = _move(result, "cpu")
                    reservation.commit(size)
                    outputs = _split_result(copied, lengths)
                    pieces.extend(zip(segments, outputs))
                del args, kwargs, result
            entries, lengths, segments = [], [], []
            samples = tokens = 0

        with ExitStack() as inputs_stack:
            for batch in batches:
                metadata = boundaries.execution_metadata(
                    boundary_index, batch, graph.input_names
                )
                read_reservation = inputs_stack.enter_context(
                    host_budget.reserve(
                        f"{label} routing boundary {name}",
                        metadata.tensor_bytes,
                    )
                )
                value = boundaries.read_for_routing(
                    boundary_index, batch, graph.input_names
                )
                read_reservation.commit(read_reservation.reserved_bytes)
                routed = graph.inputs(value, (name,), metadata=True)[name]
                reservation = inputs_stack.enter_context(
                    host_budget.reserve(
                        f"{label} expert inputs {name}",
                        _tensor_bytes(routed),
                    )
                )
                args, kwargs = graph.inputs(value, (name,))[name]
                reservation.commit(reservation.reserved_bytes)
                count = _rows(args, kwargs)
                offset = 0
                while offset < count or count == 0:
                    if entries and tokens == capacity:
                        submit()
                    length = min(capacity - tokens, count - offset)
                    entry = tree_map(
                        lambda item: (
                            item[offset : offset + length]
                            if isinstance(item, torch.Tensor)
                            else item
                        ),
                        (args, kwargs),
                    )
                    entries.append(entry)
                    lengths.append(length)
                    segments.append((batch, offset))
                    samples += int(offset == 0)
                    tokens += length
                    if not count:
                        break
                    offset += length
            submit()
        if capture:
            by_batch = {}
            for (batch, offset), output in pieces:
                by_batch.setdefault(batch, []).append((offset, output))
            for batch, outputs in by_batch.items():
                ordered = [output for _offset, output in sorted(outputs)]
                if len(ordered) > 1:
                    reservation = host_budget.reserve(
                        f"{label} concatenate expert output {name}",
                        _tensor_bytes(ordered),
                    )
                    with result_lock:
                        result_reservations.append(reservation)
                    output = _concatenate(ordered)
                    reservation.commit(_tensor_bytes(output))
                else:
                    output = ordered[0]
                with result_lock:
                    result_store[batch, name] = output

    def compute(window, prepared):
        device = parallel.devices[window.owner_rank]
        device_context = (
            torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        )
        with device_context, torch.no_grad(), ExitStack() as stack:
            stack.callback(prepared.close)
            stack.enter_context(
                device_budget.reserve(
                    f"{label} owner {window.owner_rank} window {window.window_index}",
                    required_bytes(window, window.max_dispatch_tokens),
                    device=device,
                )
            )
            prepared.move_to(device)
            loaded = stack.enter_context(
                adapter.weight_session.installed(prepared, reinitialize=False)
            )
            modules = tuple(
                dict.fromkeys(
                    module
                    for name in window.expert_names
                    for module in adapter.model.get_submodule(name).modules()
                )
            )
            stack.enter_context(modifier_adapter.expert_window(adapter.model, modules))
            for phase in ("calibration", "update", "propagation"):
                if phase == "propagation" and not propagate:
                    continue
                started = perf_counter()
                streaming_logger.info(
                    f"{label} | EP window {phase} | rank={window.owner_rank} | "
                    f"window={window.window_index} | start"
                )
                if phase == "update":
                    modifier_adapter.update_owner(device, modules)
                else:
                    reported = started
                    for index, name in enumerate(window.expert_names):
                        with DisableQuantization(adapter.model.get_submodule(name)):
                            forward_expert(
                                name,
                                window.max_dispatch_tokens,
                                capture=phase == "propagation",
                            )
                        if perf_counter() - reported >= 30:
                            streaming_logger.info(
                                f"{label} | EP window {phase} | "
                                f"rank={window.owner_rank} | "
                                f"window={window.window_index} | "
                                f"experts={index + 1}/{len(window.expert_names)}"
                            )
                            reported = perf_counter()
                streaming_logger.info(
                    f"{label} | EP window {phase} | rank={window.owner_rank} | "
                    f"window={window.window_index} | complete | "
                    f"time={perf_counter() - started:.2f}s"
                )
            with save_lock:
                save_window(window, loaded, modules)
            streaming_logger.info(
                f"{label} | EP window complete | rank={window.owner_rank} | "
                f"window={window.window_index} | tokens={window.route_tokens:,}"
            )
        return None

    try:
        AsyncResidentWindowScheduler(max_prefetch=1).run(
            windows,
            prefetch=prefetch,
            compute=compute,
            discard=lambda prepared: prepared.close(),
        )
        if callable(finalize):
            finalize()
        if not propagate:
            return {
                "placement": placement.to_dict(),
                "windows": [window.to_dict() for window in windows],
            }
        combine = graph.combine_subgraph()
        primary = parallel.devices[0]
        call_names = {
            f"ep_result_{node.name}": str(node.target) for node in graph.calls
        }
        input_names = combine.input_names - call_names.keys()
        for batch in batches:
            metadata = boundaries.execution_metadata(boundary_index, batch, input_names)
            outputs_size = sum(
                _tensor_bytes(result_store[batch, name])
                for placeholder, name in call_names.items()
                if placeholder in combine.input_names
            )
            with device_budget.reserve(
                f"{label} combine batch {batch}",
                2 * (metadata.tensor_bytes + outputs_size),
                device=primary,
            ):
                value = boundaries.get_for_execution(
                    boundary_index, batch, input_names, device=primary
                )
                for placeholder, name in call_names.items():
                    if placeholder in combine.input_names:
                        value[placeholder] = _move(
                            result_store.pop((batch, name)), primary
                        )
                with torch.no_grad(), HooksMixin.disable_hooks_local():
                    output = combine.forward(
                        adapter.model,
                        **{name: value[name] for name in combine.input_names},
                    )
                boundaries.commit_propagated_batch(
                    boundary_index,
                    destination_boundary,
                    batch,
                    output,
                    consumed_names=graph.consumed,
                )
                del value, output
    finally:
        result_store.clear()
        for reservation in result_reservations:
            reservation.close()
    return {
        "placement": placement.to_dict(),
        "windows": [window.to_dict() for window in windows],
    }
