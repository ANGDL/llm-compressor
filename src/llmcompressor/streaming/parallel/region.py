"""Route-profiled expert regions with independent resident-window lifetimes."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack, contextmanager, nullcontext
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
from ..admission import DeviceMemoryError, estimate_modifier_dispatch_memory
from ..calibration import calibration_batch_context
from ..loading.host_memory import HostMemoryError
from .expert import _join_subgraphs
from .load_balance import (
    AsyncResidentWindowScheduler,
    ExpertCostProfile,
    OwnerMemoryProfile,
    ResidentWindow,
    build_fixed_owner_placement,
    build_load_balanced_placement,
    plan_resident_windows,
)
from .placement import PlacementEntry, PlacementPlan
from .route_cache import is_cacheable_route
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
    required: dict[str, tuple[Node, ...]]

    def __post_init__(self):
        self._calls_by_name = {str(call.target): call for call in self.calls}
        required = set().union(*self.required.values())
        self._ordered_dependencies = tuple(
            node for node in self.graph.nodes if node in required
        )
        self._input_names = frozenset(
            str(node.target) for node in self.graph.nodes if node.op == "placeholder"
        )
        self.route_nodes = tuple(
            node for node in self._ordered_dependencies if is_cacheable_route(node)
        )

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
        positions = {node: index for index, node in enumerate(graph.nodes)}
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
            required[str(call.target)] = tuple(
                sorted(dependencies, key=positions.__getitem__)
            )
        return cls(model, graph, consumed, calls, required)

    @property
    def input_names(self):
        return self._input_names

    @property
    def expert_names(self):
        return tuple(self._calls_by_name)

    def _evaluate(self, value, required, *, metadata=False, routes=None):
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
        for node in required:
            if routes is not None and node in routes:
                result = tree_map(prepare, routes[node]) if metadata else routes[node]
            elif node.op == "placeholder":
                result = tree_map(prepare, value[str(node.target)])
            elif node.op == "get_attr":
                result = self.model
                for component in str(node.target).split("."):
                    result = getattr(result, component)
                result = tree_map(prepare, result)
            elif node.op in ("call_function", "call_method"):
                args, kwargs = map_arg(
                    (node.args, node.kwargs), environment.__getitem__
                )
                if node.op == "call_function":
                    result = node.target(*args, **kwargs)
                else:
                    result = getattr(args[0], str(node.target))(*args[1:], **kwargs)
            else:
                raise UnsupportedExpertRegion(
                    f"unsupported routing operation {node.op}"
                )
            environment[node] = result
        return environment

    def inputs(self, value, names=None, *, metadata=False, routes=None):
        if names is None:
            calls, required = self.calls, self._ordered_dependencies
        else:
            calls = tuple(dict.fromkeys(self._calls_by_name[name] for name in names))
            if len(calls) == 1:
                required = self.required[str(calls[0].target)]
            else:
                selected = set().union(
                    *(self.required[str(call.target)] for call in calls)
                )
                required = tuple(
                    node for node in self._ordered_dependencies if node in selected
                )
        environment = self._evaluate(value, required, metadata=metadata, routes=routes)
        result = {}
        for call in calls:
            result[str(call.target)] = map_arg(
                (call.args, call.kwargs), environment.__getitem__
            )
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
    boundary_bytes: dict[int, int]
    input_bytes: dict[str, tuple[int, ...]]


def profile_expert_region(
    graph, boundaries, boundary_index, batches, *, host_budget=None, route_cache=None
):
    counts = {name: [] for name in graph.expert_names}
    input_bytes = {name: [] for name in graph.expert_names}
    boundary_bytes = {}
    borrowed = boundaries.routing_inputs_are_borrowed
    with torch.no_grad(), HooksMixin.disable_hooks_local():
        for batch in batches:
            if not borrowed and host_budget is not None:
                boundary_bytes[batch] = boundaries.execution_metadata(
                    boundary_index, batch, graph.input_names
                ).tensor_bytes
                read_context = host_budget.reserve(
                    f"expert route profile batch {batch}", boundary_bytes[batch]
                )
            else:
                read_context = nullcontext()
            with read_context as reservation:
                value = boundaries.read_for_routing(
                    boundary_index, batch, graph.input_names
                )
                if reservation is not None:
                    reservation.commit(reservation.reserved_bytes)
                else:
                    boundary_bytes[batch] = _tensor_bytes(value)
                routes = None
                if route_cache is not None:
                    route_cache.prepare(batch, value, graph.route_nodes)
                    routes = route_cache.batch(batch)
                try:
                    inputs = graph.inputs(value, metadata=True, routes=routes)
                except (RuntimeError, NotImplementedError) as error:
                    raise UnsupportedExpertRegion(
                        "route metadata requires floating-point model execution"
                    ) from error
                for name, (args, kwargs) in inputs.items():
                    counts[name].append(_rows(args, kwargs))
                    input_bytes[name].append(_tensor_bytes((args, kwargs)))
                del value, inputs, args, kwargs
    return ExpertRegionProfile(
        {name: sum(values) for name, values in counts.items()},
        {name: tuple(values) for name, values in counts.items()},
        boundary_bytes,
        {name: tuple(values) for name, values in input_bytes.items()},
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


def _window_activation_batches(
    window,
    capacity: int,
    route_tokens: Mapping[str, int],
    *,
    expert_names=None,
) -> int:
    names = window.expert_names if expert_names is None else expert_names
    return sum(
        max(1, ceil(route_tokens[name] / capacity)) if capacity else 1
        for name in names
    )


def _admit_runtime_window(
    window,
    *,
    required_bytes,
    device_budget,
    device,
    label,
    route_tokens,
):
    """Admit a planned window, adapting its dispatch capacity to live headroom."""
    owner = f"{label} owner {window.owner_rank} window {window.window_index}"
    planned_capacity = window.max_dispatch_tokens
    planned_required = required_bytes(planned_capacity)
    try:
        reservation = device_budget.reserve(
            owner,
            planned_required,
            device=device,
        )
    except DeviceMemoryError as initial_error:
        if planned_capacity < 1 or not window.route_tokens:
            raise
        usable = device_budget.usable_bytes(device)
        if usable is None:
            raise
        try:
            capacity = select_token_capacity(
                owner=f"{owner} runtime fallback",
                usable_bytes=usable,
                required_bytes=required_bytes,
                maximum_tokens=planned_capacity,
            )
        except DeviceMemoryError as fallback_error:
            raise DeviceMemoryError(
                f"{owner} could not fit after runtime capacity fallback: "
                f"planned_capacity={planned_capacity}, "
                f"planned_required={planned_required}, "
                f"usable={usable}; {fallback_error}"
            ) from initial_error
        effective = replace(
            window,
            max_dispatch_tokens=capacity,
            activation_batches=_window_activation_batches(
                window, capacity, route_tokens
            ),
        )
        try:
            reservation = device_budget.reserve(
                owner,
                required_bytes(effective.max_dispatch_tokens),
                device=device,
            )
        except DeviceMemoryError as retry_error:
            raise DeviceMemoryError(
                f"{owner} failed after runtime capacity fallback: "
                f"planned_capacity={planned_capacity}, "
                f"runtime_capacity={capacity}, usable={usable}; {retry_error}"
            ) from initial_error
        if capacity != planned_capacity:
            streaming_logger.warning(
                f"{label} | EP window admission fallback | "
                f"rank={window.owner_rank} | window={window.window_index} | "
                f"planned_capacity={planned_capacity:,} | "
                f"runtime_capacity={capacity:,} | "
                f"planned_required={planned_required:,} | "
                f"runtime_usable={usable:,}"
            )
        return effective, reservation
    return window, reservation


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
    route_cache=None,
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
    memory = []
    total_tokens = sum(profile.token_counts.values())
    for device in parallel.devices:
        available = device_budget.usable_bytes(device)
        if available is None:
            raise UnsupportedExpertRegion(
                "resident windows require a known device budget"
            )
        memory.append(
            OwnerMemoryProfile(
                usable_bytes=available,
                dispatch_bytes_per_token=max(runtime.values(), default=1),
                max_dispatch_tokens=max(
                    1,
                    total_tokens
                    if token_budget == "auto"
                    else min(total_tokens, token_budget),
                ),
            )
        )
    try:
        placement = build_fixed_owner_placement(
            costs,
            len(parallel.devices),
            owner_memory=memory,
        )
    except MemoryError:
        placement = build_load_balanced_placement(costs, len(parallel.devices))
        streaming_logger.warning(
            f"{label} | fixed-owner residency does not fit | "
            "using fixed owner-local resident windows"
        )
    memory = tuple(
        replace(
            memory[owner],
            dispatch_bytes_per_token=max(
                (runtime[name] for name, rank in placement.owners if rank == owner),
                default=1,
            ),
            max_dispatch_tokens=max(
                1,
                (
                    sum(
                        profile.token_counts[name]
                        for name, rank in placement.owners
                        if rank == owner
                    )
                    if token_budget == "auto"
                    else min(
                        token_budget,
                        sum(
                            profile.token_counts[name]
                            for name, rank in placement.owners
                            if rank == owner
                        ),
                    )
                ),
            ),
        )
        for owner in range(len(parallel.devices))
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
                activation_batches=_window_activation_batches(
                    window, capacity, profile.token_counts
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

    def forward_expert(name, capacity, *, capture, prepared_inputs=None):
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
            for position, batch in enumerate(batches):
                if prepared_inputs is None:
                    read_reservation = None
                    if not boundaries.routing_inputs_are_borrowed:
                        read_reservation = inputs_stack.enter_context(
                            host_budget.reserve(
                                f"{label} routing boundary {name}",
                                profile.boundary_bytes[batch],
                            )
                        )
                    value = boundaries.read_for_routing(
                        boundary_index, batch, graph.input_names
                    )
                    if read_reservation is not None:
                        read_reservation.commit(read_reservation.reserved_bytes)
                    reservation = inputs_stack.enter_context(
                        host_budget.reserve(
                            f"{label} expert inputs {name}",
                            profile.input_bytes[name][position],
                        )
                    )
                    routes = route_cache.batch(batch) if route_cache else None
                    args, kwargs = graph.inputs(value, (name,), routes=routes)[name]
                    reservation.commit(reservation.reserved_bytes)
                else:
                    args, kwargs = prepared_inputs[batch][name]
                count = profile.batch_counts[name][position]
                if _rows(args, kwargs) != count:
                    raise UnsupportedExpertRegion(
                        f"expert {name} input rows changed after route profiling"
                    )
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

    def input_groups(names):
        if not boundaries.routing_inputs_are_borrowed or not graph.route_nodes:
            for name in names:
                yield (name,), 0
            return
        limit = min(512 * 2**20, host_budget.effective_available_bytes() // 32)
        pending = []
        size = 0
        for name in names:
            expert_bytes = sum(profile.input_bytes[name])
            if pending and size + expert_bytes > limit:
                yield tuple(pending), size
                pending, size = [], 0
            if expert_bytes > limit:
                yield (name,), 0
            else:
                pending.append(name)
                size += expert_bytes
        if pending:
            yield tuple(pending), size

    @contextmanager
    def prepare_group_inputs(names, size, device):
        if len(names) == 1:
            yield None
            return
        try:
            reservation = host_budget.reserve(f"{label} {device} routed inputs", size)
        except HostMemoryError:
            yield None
            return
        prepared_inputs = {}
        with reservation:
            try:
                for position, batch in enumerate(batches):
                    value = boundaries.read_for_routing(
                        boundary_index, batch, graph.input_names
                    )
                    routes = route_cache.batch(batch) if route_cache else None
                    prepared_inputs[batch] = graph.inputs(value, names, routes=routes)
                    reservation.commit(
                        sum(profile.input_bytes[name][position] for name in names)
                    )
                yield prepared_inputs
            finally:
                prepared_inputs.clear()

    window_stride = len(costs) + 1

    def split_runtime_window(window, usable):
        if len(window.expert_names) < 2:
            return ()
        parts = []
        current = []
        current_bytes = 0
        for name in window.expert_names:
            resident_bytes = costs[name].total_resident_bytes
            if resident_bytes > usable:
                return ()
            if current and current_bytes + resident_bytes > usable:
                parts.append(tuple(current))
                current = []
                current_bytes = 0
            current.append(name)
            current_bytes += resident_bytes
        if current:
            parts.append(tuple(current))
        if len(parts) < 2:
            return ()
        return tuple(
            ResidentWindow(
                owner_rank=window.owner_rank,
                window_index=window.window_index * window_stride + index,
                expert_names=names,
                resident_bytes=sum(
                    costs[name].total_resident_bytes for name in names
                ),
                route_tokens=sum(profile.token_counts[name] for name in names),
                work_units=sum(
                    costs[name].routed_work_units for name in names
                ),
                max_dispatch_tokens=min(
                    window.max_dispatch_tokens,
                    sum(profile.token_counts[name] for name in names),
                ),
                activation_batches=_window_activation_batches(
                    window,
                    min(
                        window.max_dispatch_tokens,
                        sum(profile.token_counts[name] for name in names),
                    ),
                    profile.token_counts,
                    expert_names=names,
                ),
            )
            for index, names in enumerate(parts)
        )

    def compute_window(window, prepared):
        device = parallel.devices[window.owner_rank]
        device_context = (
            torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        )
        with device_context, torch.no_grad(), ExitStack() as stack:
            stack.callback(prepared.close)
            planned_window = window
            window, reservation = _admit_runtime_window(
                planned_window,
                required_bytes=lambda tokens: required_bytes(planned_window, tokens),
                device_budget=device_budget,
                device=device,
                label=label,
                route_tokens=profile.token_counts,
            )
            stack.enter_context(reservation)
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
                    after_update = getattr(modifier_adapter, "after_update", None)
                    if callable(after_update):
                        after_update(modules)
                else:
                    reported = started
                    done = 0
                    for names, size in input_groups(window.expert_names):
                        with prepare_group_inputs(names, size, device) as inputs:
                            for name in names:
                                with DisableQuantization(
                                    adapter.model.get_submodule(name)
                                ):
                                    forward_expert(
                                        name,
                                        window.max_dispatch_tokens,
                                        capture=phase == "propagation",
                                        prepared_inputs=inputs,
                                    )
                                done += 1
                        if perf_counter() - reported >= 30:
                            streaming_logger.info(
                                f"{label} | EP window {phase} | "
                                f"rank={window.owner_rank} | "
                                f"window={window.window_index} | "
                                f"experts={done}/{len(window.expert_names)}"
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
        return window

    def compute(window, prepared):
        try:
            return compute_window(window, prepared)
        except DeviceMemoryError:
            device = parallel.devices[window.owner_rank]
            usable = device_budget.usable_bytes(device)
            if usable is None or window.resident_bytes <= usable:
                raise
            split = split_runtime_window(window, usable)
            if not split:
                raise
            streaming_logger.warning(
                f"{label} | EP runtime resident split | "
                f"rank={window.owner_rank} | window={window.window_index} | "
                f"original_experts={len(window.expert_names)} | "
                f"split_windows={len(split)} | usable={usable:,}"
            )
            for subwindow in split:
                subprepared = prefetch(subwindow)
                compute_window(subwindow, subprepared)
            return window

    try:
        completed_windows = AsyncResidentWindowScheduler(max_prefetch=1).run(
            windows,
            prefetch=prefetch,
            compute=compute,
            discard=lambda prepared: prepared.close(),
        )
        windows = tuple(window for _planned, window in completed_windows)
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
