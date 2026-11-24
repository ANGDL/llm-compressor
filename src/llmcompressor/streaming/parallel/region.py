"""Route-profiled expert regions with independent resident-window lifetimes."""

from __future__ import annotations

from collections import deque
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
from ..activations.store import _tensor_bytes, _tensors
from ..activations.transfer import transfer_tree, transfer_tree_to_cpu
from ..admission import DeviceMemoryError, estimate_modifier_dispatch_memory
from ..calibration import calibration_batch_context
from ..loading.host_memory import HostMemoryError
from ..loading.session import SubgraphLoadPlan
from .expert import _join_subgraphs
from .load_balance import (
    AsyncResidentWindowScheduler,
    ExpertCostProfile,
    OwnerMemoryProfile,
    ResidentWindow,
    build_fixed_owner_placement,
    build_load_balanced_placement,
    plan_resident_windows,
    plan_staging_bytes,
)
from .placement import PlacementEntry, PlacementPlan
from .route_cache import is_cacheable_route
from .token_budget import select_token_capacity


class UnsupportedExpertRegion(ValueError):
    """A graph cannot expose independent expert inputs without model execution."""


def _select_destination_output(output, destination_input_names):
    """Drop values no downstream target consumes from a combined boundary."""

    if destination_input_names is None or not isinstance(output, dict):
        return output
    return {
        name: value for name, value in output.items() if name in destination_input_names
    }


def _output_storage_bytes(value):
    storages = {}
    for tensor in _tensors(value):
        storage = tensor.untyped_storage()
        storages[storage.device, storage.data_ptr()] = storage.nbytes()
    return sum(storages.values())


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
        self.direct_input_experts = frozenset(
            name
            for name, dependencies in self.required.items()
            if all(node.op == "placeholder" for node in dependencies)
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
        max(1, ceil(route_tokens[name] / capacity)) if capacity else 1 for name in names
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
    destination_input_names=None,
    allow_device_prefetch=True,
):
    """Calibrate, update, propagate, and release each owner's admitted windows."""
    planning_started = perf_counter()
    costs = {}
    runtime = {}
    workspaces = {}
    expert_modules = {}
    weight_session = adapter.weight_session
    region_plan = weight_session.plan_modules(
        graph.expert_names, allow_missing_state=True
    )
    target_plans = {target.name: target for target in region_plan.targets}
    if target_plans.keys() != set(graph.expert_names):
        raise UnsupportedExpertRegion("resident window expert roots overlap")

    def weight_plan(names):
        return SubgraphLoadPlan(names, tuple(target_plans[name] for name in names))

    for name in graph.expert_names:
        modules = tuple(adapter.model.get_submodule(name).modules())
        expert_modules[name] = modules
        estimate_update_work = getattr(modifier_adapter, "estimate_update_work", None)
        update_work = (
            float(estimate_update_work(modules))
            if callable(estimate_update_work)
            else 0.0
        )
        estimate = estimate_modifier_dispatch_memory(
            modifiers,
            modules,
            total_tokens=profile.token_counts[name],
            dispatch_tokens=int(bool(profile.token_counts[name])),
            phase="calibration",
        )
        workspaces[name] = estimate.scratch_bytes + estimate.batch_bytes
        weights = weight_session.estimate_materialized_bytes(
            weight_plan((name,)), dtype=target_dtype
        )
        costs[name] = ExpertCostProfile(
            storage_bytes=weights,
            route_tokens=profile.token_counts[name],
            persistent_state_bytes=estimate.held_bytes,
            update_work_units=update_work,
        )
        linears = [module for module in modules if isinstance(module, nn.Linear)]
        if not linears:
            raise UnsupportedExpertRegion("expert has no bounded Linear workspace")
        runtime[name] = sum(
            4 * (2 * module.in_features + 2 * module.out_features) for module in linears
        ) + max(8 * module.in_features for module in linears)
    memory = []
    workspace = max(workspaces.values(), default=0)
    total_tokens = sum(profile.token_counts.values())
    for device in parallel.devices:
        available = device_budget.usable_bytes(device)
        if available is None:
            raise UnsupportedExpertRegion(
                "resident windows require a known device budget"
            )
        if workspace > available:
            raise DeviceMemoryError(
                f"{label} owner {device} has {available} usable bytes for "
                f"{workspace} bytes of expert workspace"
            )
        memory.append(
            OwnerMemoryProfile(
                usable_bytes=available,
                fixed_bytes=workspace,
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
    staging_bytes = plan_staging_bytes(
        windows,
        costs,
        memory,
        {
            rank
            for rank, device in enumerate(parallel.devices)
            if allow_device_prefetch and device.type == "cuda"
        },
    )
    if staging_bytes:
        windows = plan_resident_windows(
            placement, costs, memory, staging_bytes=staging_bytes
        )

    def required_bytes(window, tokens):
        workspace = max(workspaces[name] for name in window.expert_names)
        if tokens:
            for name in window.expert_names:
                estimate = estimate_modifier_dispatch_memory(
                    modifiers,
                    expert_modules[name],
                    total_tokens=profile.token_counts[name],
                    dispatch_tokens=min(tokens, profile.token_counts[name]),
                    phase="calibration",
                )
                workspace = max(
                    workspace, estimate.scratch_bytes + estimate.batch_bytes
                )
        return (
            window.resident_bytes
            + workspace
            + memory[window.owner_rank].dispatch_bytes_per_token * tokens
        )

    def with_capacity(window, capacity):
        return replace(
            window,
            max_dispatch_tokens=capacity,
            activation_batches=_window_activation_batches(
                window, capacity, profile.token_counts
            ),
        )

    admitted_windows = []
    for window in windows:
        capacity = (
            select_token_capacity(
                owner=(
                    f"{label} resident window {window.owner_rank}:{window.window_index}"
                ),
                usable_bytes=(
                    memory[window.owner_rank].usable_bytes
                    - staging_bytes.get(window.owner_rank, 0)
                ),
                required_bytes=lambda tokens: required_bytes(window, tokens),
                maximum_tokens=min(
                    window.max_dispatch_tokens,
                    max(profile.token_counts[name] for name in window.expert_names),
                ),
            )
            if window.route_tokens
            else 0
        )
        admitted_windows.append(with_capacity(window, capacity))
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
    owner_expert_counts = tuple(
        sum(owner == rank for _name, owner in placement.owners)
        for rank in range(placement.ep_size)
    )
    owner_work_units = tuple(round(value, 2) for value in placement.owner_work_units)
    streaming_logger.info(
        f"{label} | EP routed windows | policy={placement.policy} | "
        f"experts={len(costs)} | windows={len(windows)} | "
        f"routed_tokens={sum(profile.token_counts.values()):,} | "
        f"owner_experts={owner_expert_counts} | "
        f"owner_route_tokens={placement.owner_route_tokens} | "
        f"owner_work_units={owner_work_units} | "
        f"planning_s={perf_counter() - planning_started:.2f}"
    )
    for window in windows:
        streaming_logger.info(
            f"{label} | EP window | rank={window.owner_rank} | "
            f"window={window.window_index} | experts={len(window.expert_names)} | "
            f"tokens={window.route_tokens:,} | "
            f"capacity={window.max_dispatch_tokens:,} | "
            f"dispatches={window.activation_batches} | "
            f"resident_gib={window.resident_bytes / 2**30:.2f} | "
            f"staging_gib={staging_bytes.get(window.owner_rank, 0) / 2**30:.2f}"
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

    buffered_owners = set()
    prepared_windows = {}
    prepared_windows_lock = Lock()
    window_plans = tuple(
        (window, weight_plan(window.expert_names)) for window in windows
    )
    if window_plans:
        aggregate_plan = SubgraphLoadPlan(
            tuple(
                dict.fromkeys(
                    name for _window, plan in window_plans for name in plan.module_names
                )
            ),
            tuple(target for _window, plan in window_plans for target in plan.targets),
        )
        required = adapter.weight_session.estimate_preparation(
            aggregate_plan, dtype=target_dtype
        )
        layer_prepare_started = perf_counter()
        layer_reservation = host_budget.reserve(
            f"{label} routed MoE layer expert weights", required
        )
        try:
            prepared = adapter.weight_session.prepare_many(
                tuple(plan for _window, plan in window_plans),
                dtype=target_dtype,
                reservation=layer_reservation,
            )
        except BaseException:
            layer_reservation.close()
            raise
        with prepared_windows_lock:
            prepared_windows.update(
                {
                    (window.owner_rank, window.window_index): value
                    for (window, _plan), value in zip(window_plans, prepared)
                }
            )
        streaming_logger.info(
            f"{label} | EP layer CPU prepare | "
            f"windows={len(window_plans)} | "
            f"targets={len(aggregate_plan.targets)} | "
            f"resident_gib={sum(value.nbytes for value in prepared) / 2**30:.2f} | "
            f"time={perf_counter() - layer_prepare_started:.2f}s"
        )

    def prefetch(window):
        started = perf_counter()
        key = (window.owner_rank, window.window_index)
        with prepared_windows_lock:
            prepared = prepared_windows.pop(key, None)
        if prepared is None:
            raise RuntimeError(
                f"{label} has no CPU-prepared weights for "
                f"rank={window.owner_rank}, window={window.window_index}"
            )
        staged = False
        if window.owner_rank in buffered_owners and window.window_index != 0:
            device = parallel.devices[window.owner_rank]
            try:
                size = prepared.nbytes
                if size > staging_bytes[window.owner_rank]:
                    raise DeviceMemoryError(
                        f"{label} {device} staged weights require {size} bytes, "
                        f"planned {staging_bytes[window.owner_rank]}"
                    )
                transfer = host_budget.reserve(
                    f"{label} {device} staging pinned weights", size
                )
            except HostMemoryError:
                streaming_logger.warning(
                    f"{label} | EP pinned prefetch unavailable | device={device} | "
                    f"window={window.window_index} | using host-only prefetch"
                )
            except BaseException:
                prepared.close()
                raise
            else:
                try:
                    prepared.move_to_async(device, transfer_reservation=transfer)
                    staged = True
                except BaseException:
                    transfer.close()
                    prepared.close()
                    raise
        streaming_logger.info(
            f"{label} | EP window prefetch | rank={window.owner_rank} | "
            f"window={window.window_index} | device_staged={staged} | "
            f"time={perf_counter() - started:.2f}s"
        )
        return prepared

    borrowed_experts = (
        graph.direct_input_experts
        if boundaries.routing_inputs_are_borrowed
        else frozenset()
    )
    borrowed_boundaries = {}

    def forward_expert(
        name,
        capacity,
        *,
        capture,
        prepared_inputs=None,
        phase_stats=None,
    ):
        pieces = []
        entries = []
        lengths = []
        segments = []
        samples = 0
        tokens = 0
        device = parallel.device_for(name)
        borrow_inputs = name in borrowed_experts

        def submit():
            nonlocal entries, lengths, segments, samples, tokens
            if not entries:
                return
            submit_started = perf_counter()
            if phase_stats is not None:
                phase_stats["dispatches"] += 1
                phase_stats["tokens"] += tokens
            pack_started = perf_counter()
            with host_budget.reserve(
                f"{label} expert dispatch {name}", _tensor_bytes(entries)
            ) as gathered:
                combined = _concatenate(entries)
                gathered.commit(_tensor_bytes(combined))
                if phase_stats is not None:
                    phase_stats["pack_s"] += perf_counter() - pack_started
                input_move_started = perf_counter()
                args, kwargs = _move(combined, device)
                if phase_stats is not None:
                    phase_stats["input_move_s"] += perf_counter() - input_move_started
                del combined
                context = HooksMixin.disable_hooks_local() if capture else nullcontext()
                with context, calibration_batch_context(lengths, sample_count=samples):
                    forward_started = perf_counter()
                    result = adapter.model.get_submodule(name)(*args, **kwargs)
                if phase_stats is not None:
                    phase_stats["forward_s"] += perf_counter() - forward_started
                if capture:
                    reserve_started = perf_counter()
                    size = _tensor_bytes(result)
                    reservation = host_budget.reserve(
                        f"{label} expert output {name}", size
                    )
                    if phase_stats is not None:
                        phase_stats["output_reserve_s"] += (
                            perf_counter() - reserve_started
                        )
                    with result_lock:
                        result_reservations.append(reservation)
                    output_move_started = perf_counter()
                    copied = _move(result, "cpu")
                    if phase_stats is not None:
                        phase_stats["output_move_s"] += (
                            perf_counter() - output_move_started
                        )
                    reservation.commit(size)
                    split_started = perf_counter()
                    outputs = _split_result(copied, lengths)
                    if phase_stats is not None:
                        phase_stats["split_s"] += perf_counter() - split_started
                    pieces.extend(zip(segments, outputs))
                del args, kwargs, result
            entries, lengths, segments = [], [], []
            samples = tokens = 0
            if phase_stats is not None:
                phase_stats["submit_s"] += perf_counter() - submit_started

        with ExitStack() as inputs_stack:
            loop_started = perf_counter()
            for position, batch in enumerate(batches):
                if borrow_inputs:
                    fx_started = perf_counter()
                    args, kwargs = graph.inputs(
                        borrowed_boundaries[batch], (name,)
                    )[name]
                    if phase_stats is not None:
                        phase_stats["fx_s"] += perf_counter() - fx_started
                elif prepared_inputs is None:
                    read_reservation = None
                    if not boundaries.routing_inputs_are_borrowed:
                        reserve_started = perf_counter()
                        read_reservation = inputs_stack.enter_context(
                            host_budget.reserve(
                                f"{label} routing boundary {name}",
                                profile.boundary_bytes[batch],
                            )
                        )
                        if phase_stats is not None:
                            phase_stats["reserve_s"] += (
                                perf_counter() - reserve_started
                            )
                    read_started = perf_counter()
                    value = boundaries.read_for_routing(
                        boundary_index, batch, graph.input_names
                    )
                    if phase_stats is not None:
                        phase_stats["read_s"] += perf_counter() - read_started
                        phase_stats["reads"] += 1
                    if read_reservation is not None:
                        read_reservation.commit(read_reservation.reserved_bytes)
                    reserve_started = perf_counter()
                    reservation = inputs_stack.enter_context(
                        host_budget.reserve(
                            f"{label} expert inputs {name}",
                            profile.input_bytes[name][position],
                        )
                    )
                    if phase_stats is not None:
                        phase_stats["reserve_s"] += (
                            perf_counter() - reserve_started
                        )
                    routes = route_cache.batch(batch) if route_cache else None
                    fx_started = perf_counter()
                    args, kwargs = graph.inputs(value, (name,), routes=routes)[name]
                    if phase_stats is not None:
                        phase_stats["fx_s"] += perf_counter() - fx_started
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
                    slice_started = perf_counter()
                    entry = tree_map(
                        lambda item: (
                            item[offset : offset + length]
                            if isinstance(item, torch.Tensor)
                            else item
                        ),
                        (args, kwargs),
                    )
                    if phase_stats is not None:
                        phase_stats["slice_s"] += perf_counter() - slice_started
                    entries.append(entry)
                    lengths.append(length)
                    segments.append((batch, offset))
                    samples += int(offset == 0)
                    tokens += length
                    if not count:
                        break
                    offset += length
            if phase_stats is not None:
                phase_stats["loop_s"] += perf_counter() - loop_started
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
            expert_bytes = (
                0 if name in borrowed_experts else sum(profile.input_bytes[name])
            )
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
        names = tuple(name for name in names if name not in borrowed_experts)
        if len(names) < 2:
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
        minimum_tokens = int(bool(window.route_tokens))
        usable -= required_bytes(window, minimum_tokens) - window.resident_bytes
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
                resident_bytes=sum(costs[name].total_resident_bytes for name in names),
                route_tokens=sum(profile.token_counts[name] for name in names),
                work_units=sum(costs[name].routed_work_units for name in names),
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

    def compute_window(window, prepared, *, buffered=False):
        device = parallel.devices[window.owner_rank]
        device_context = (
            torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        )
        with device_context, torch.no_grad(), ExitStack() as stack:
            if not buffered:
                planned_window = window
                try:
                    window, reservation = _admit_runtime_window(
                        planned_window,
                        required_bytes=lambda tokens: required_bytes(
                            planned_window, tokens
                        ),
                        device_budget=device_budget,
                        device=device,
                        label=label,
                        route_tokens=profile.token_counts,
                    )
                except DeviceMemoryError:
                    raise
                except BaseException:
                    prepared.close()
                    raise
            stack.callback(prepared.close)
            if not buffered:
                stack.enter_context(reservation)
            prefetched_to_device = buffered and prepared.device == device
            transfer_started = perf_counter()
            prepared.move_to(device)
            if prefetched_to_device:
                stream = torch.cuda.current_stream(device)
                for target in prepared.targets:
                    for values in (target.parameter_values, target.buffer_values):
                        for value in values.values():
                            value.record_stream(stream)
            streaming_logger.info(
                f"{label} | EP window activate | rank={window.owner_rank} | "
                f"window={window.window_index} | "
                f"device_staged={prefetched_to_device} | "
                f"transfer_wait_s={perf_counter() - transfer_started:.2f}"
            )
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
                    phase_stats = {
                        "dispatches": 0,
                        "tokens": 0,
                        "reads": 0,
                        "expert_s": 0.0,
                        "loop_s": 0.0,
                        "submit_s": 0.0,
                        "reserve_s": 0.0,
                        "slice_s": 0.0,
                        "read_s": 0.0,
                        "fx_s": 0.0,
                        "pack_s": 0.0,
                        "input_move_s": 0.0,
                        "forward_s": 0.0,
                        "output_move_s": 0.0,
                        "output_reserve_s": 0.0,
                        "split_s": 0.0,
                    }
                    reported = started
                    done = 0
                    for names, size in input_groups(window.expert_names):
                        with prepare_group_inputs(names, size, device) as inputs:
                            for name in names:
                                with DisableQuantization(
                                    adapter.model.get_submodule(name)
                                ):
                                    expert_started = perf_counter()
                                    forward_expert(
                                        name,
                                        window.max_dispatch_tokens,
                                        capture=phase == "propagation",
                                        prepared_inputs=inputs,
                                        phase_stats=phase_stats,
                                    )
                                    phase_stats["expert_s"] += (
                                        perf_counter() - expert_started
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
                        f"{label} | EP window {phase} dispatch timing | "
                        f"rank={window.owner_rank} | window={window.window_index} | "
                        f"dispatches={phase_stats['dispatches']} | "
                        f"tokens={phase_stats['tokens']:,} | "
                        f"reads={phase_stats['reads']} | "
                        f"read_s={phase_stats['read_s']:.2f} | "
                        f"fx_s={phase_stats['fx_s']:.2f} | "
                        f"expert_s={phase_stats['expert_s']:.2f} | "
                        f"loop_s={phase_stats['loop_s']:.2f} | "
                        f"submit_s={phase_stats['submit_s']:.2f} | "
                        f"reserve_s={phase_stats['reserve_s']:.2f} | "
                        f"slice_s={phase_stats['slice_s']:.2f} | "
                        f"pack_s={phase_stats['pack_s']:.2f} | "
                        f"input_move_s={phase_stats['input_move_s']:.2f} | "
                        f"forward_s={phase_stats['forward_s']:.2f} | "
                        f"output_move_s={phase_stats['output_move_s']:.2f} | "
                        f"output_reserve_s={phase_stats['output_reserve_s']:.2f} | "
                        f"split_s={phase_stats['split_s']:.2f}"
                    )
                streaming_logger.info(
                    f"{label} | EP window {phase} | rank={window.owner_rank} | "
                    f"window={window.window_index} | complete | "
                    f"time={perf_counter() - started:.2f}s"
                )
            save_wait_started = perf_counter()
            with save_lock:
                save_started = perf_counter()
                save_window(window, loaded, modules)
                save_elapsed = perf_counter() - save_started
            streaming_logger.info(
                f"{label} | EP window complete | rank={window.owner_rank} | "
                f"window={window.window_index} | tokens={window.route_tokens:,} | "
                f"save_wait_s={save_started - save_wait_started:.2f} | "
                f"save_s={save_elapsed:.2f}"
            )
        return window

    def compute(window, prepared):
        if window.owner_rank in buffered_owners:
            return (compute_window(window, prepared, buffered=True),)
        try:
            return (compute_window(window, prepared),)
        except DeviceMemoryError:
            device = parallel.devices[window.owner_rank]
            usable = device_budget.usable_bytes(device)
            minimum_tokens = int(bool(window.route_tokens))
            if usable is None or required_bytes(window, minimum_tokens) <= usable:
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
            subplans = tuple(weight_plan(subwindow.expert_names) for subwindow in split)
            subprepared = prepared.partition(subplans)
            actual = []
            for subwindow, state in zip(split, subprepared):
                actual.append(compute_window(subwindow, state))
            return tuple(actual)

    try:
        if borrowed_experts:
            read_started = perf_counter()
            for batch in batches:
                borrowed_boundaries[batch] = boundaries.read_for_routing(
                    boundary_index, batch, graph.input_names
                )
            streaming_logger.info(
                f"{label} | EP borrowed expert inputs | "
                f"experts={len(borrowed_experts)} | "
                f"batches={len(borrowed_boundaries)} | "
                f"read_s={perf_counter() - read_started:.2f}"
            )
        with ExitStack() as buffers:
            for owner, staging in staging_bytes.items():
                current = max(
                    required_bytes(window, window.max_dispatch_tokens)
                    for window in windows
                    if window.owner_rank == owner
                )
                device = parallel.devices[owner]
                try:
                    reservation = device_budget.reserve(
                        f"{label} owner {owner} resident and staged weights",
                        current + staging,
                        device=device,
                    )
                except DeviceMemoryError:
                    streaming_logger.warning(
                        f"{label} | EP double-buffer admission fallback | "
                        f"rank={owner} | using host-only prefetch"
                    )
                else:
                    buffers.enter_context(reservation)
                    buffered_owners.add(owner)
            try:
                completed_windows = AsyncResidentWindowScheduler(max_prefetch=1).run(
                    windows,
                    prefetch=prefetch,
                    compute=compute,
                    discard=lambda prepared: prepared.close(),
                )
            finally:
                with prepared_windows_lock:
                    leftovers = tuple(prepared_windows.values())
                    prepared_windows.clear()
                for prepared in leftovers:
                    prepared.close()
        borrowed_boundaries.clear()
        windows = tuple(
            actual for _planned, results in completed_windows for actual in results
        )
        if callable(finalize):
            finalize()
        if not propagate:
            return {
                "placement": placement.to_dict(),
                "windows": [window.to_dict() for window in windows],
            }
        combine_started_at = perf_counter()
        combine = graph.combine_subgraph()
        call_names = {
            f"ep_result_{node.name}": str(node.target) for node in graph.calls
        }
        input_names = combine.input_names - call_names.keys()
        result_names = {
            placeholder: name
            for placeholder, name in call_names.items()
            if placeholder in combine.input_names
        }

        if all(node.op in {"placeholder", "output"} for node in combine.graph.nodes):
            output_node = combine.graph.find_nodes(op="output")[0]
            live_outputs = _select_destination_output(
                output_node.args[0], destination_input_names
            )
            live_inputs = set()
            map_arg(live_outputs, live_inputs.add)
            source_inputs = (
                {str(node.target) for node in live_inputs} - call_names.keys()
            )
            host_bytes = 0
            commit_seconds = 0.0
            for batch in batches:
                source = (
                    boundaries.read_for_routing(boundary_index, batch, source_inputs)
                    if source_inputs
                    else {}
                )
                projected = {
                    node: (
                        result_store[batch, call_names[node.target]]
                        if node.target in call_names
                        else source[node.target]
                    )
                    for node in live_inputs
                }
                output = map_arg(live_outputs, projected.__getitem__)
                host_bytes += _tensor_bytes(output)
                commit_started = perf_counter()
                boundaries.commit_propagated_batch(
                    boundary_index,
                    destination_boundary,
                    batch,
                    output,
                    consumed_names=graph.consumed,
                )
                commit_seconds += perf_counter() - commit_started
                for name in result_names.values():
                    del result_store[batch, name]
                del source, projected, output
            streaming_logger.info(
                f"{label} | EP combine coalesced transfer | "
                f"batches={len(batches)} | mode=cpu_passthrough | "
                f"tensors=0 | bytes=0 | output_tensors=0 | output_bytes=0 | "
                f"output_flushes=0 | prefetch_fallbacks=0 | host_bytes={host_bytes} | "
                f"commit_s={commit_seconds:.2f} | "
                f"wall_s={perf_counter() - combine_started_at:.2f}"
            )
            return {
                "placement": placement.to_dict(),
                "windows": [window.to_dict() for window in windows],
            }

        primary = parallel.devices[0]
        copy_stream = (
            torch.cuda.Stream(device=primary) if primary.type == "cuda" else None
        )
        output_stream = (
            torch.cuda.Stream(device=primary) if primary.type == "cuda" else None
        )

        def prepare_batch(batch):
            prepare_started = perf_counter()
            payload = dict(
                boundaries.read_for_routing(boundary_index, batch, input_names)
            )
            for placeholder, name in result_names.items():
                payload[placeholder] = result_store[batch, name]
            transfer = transfer_tree(
                payload,
                primary,
                stream=copy_stream,
                host_budget=host_budget,
                device_budget=device_budget,
                owner=f"{label} combine batch {batch}",
            )
            for name in result_names.values():
                del result_store[batch, name]
            return transfer, perf_counter() - prepare_started

        output_cap = 512 * 2**20
        if primary.type == "cuda":
            usable = device_budget.usable_bytes(primary)
            if usable is not None:
                output_cap = max(1, min(output_cap, usable // 8))
        max_output_batches = 8
        max_output_transfers = 2
        admission_errors = (DeviceMemoryError, HostMemoryError, torch.OutOfMemoryError)

        combined_tensors = 0
        combined_bytes = 0
        allocate_seconds = 0.0
        pack_seconds = 0.0
        submit_seconds = 0.0
        prepare_seconds = 0.0
        wait_seconds = 0.0
        combine_seconds = 0.0
        commit_seconds = 0.0
        close_seconds = 0.0
        output_pack_seconds = 0.0
        output_wait_seconds = 0.0
        output_tensors = 0
        output_bytes = 0
        output_flushes = 0
        prefetch_fallbacks = 0
        pending_outputs = {}
        pending_output_bytes = 0
        inflight_outputs = deque()
        inflight_output_bytes = 0
        pending = None
        current = None

        def commit_outputs(cpu_outputs):
            nonlocal commit_seconds
            for output_batch, output_value in cpu_outputs.items():
                commit_started = perf_counter()
                boundaries.commit_propagated_batch(
                    boundary_index,
                    destination_boundary,
                    output_batch,
                    output_value,
                    consumed_names=graph.consumed,
                )
                commit_seconds += perf_counter() - commit_started

        def drain_outputs(*, keep):
            nonlocal output_wait_seconds, output_tensors, output_bytes
            nonlocal inflight_output_bytes
            while len(inflight_outputs) > keep:
                transfer, storage_bytes = inflight_outputs.popleft()
                try:
                    wait_started = perf_counter()
                    cpu_outputs = transfer.wait()
                    output_wait_seconds += perf_counter() - wait_started
                    output_tensors += transfer.tensor_count
                    output_bytes += transfer.total_bytes
                    commit_outputs(cpu_outputs)
                    del cpu_outputs
                finally:
                    transfer.close()
                    inflight_output_bytes -= storage_bytes

        def flush_outputs(*, keep):
            nonlocal pending_output_bytes, inflight_output_bytes
            nonlocal output_pack_seconds, output_flushes
            if pending_outputs:
                flushed = pending_outputs.copy()
                storage_bytes = pending_output_bytes
                pending_outputs.clear()
                pending_output_bytes = 0
                pack_started = perf_counter()
                output_flushes += 1
                try:
                    transfer = transfer_tree_to_cpu(
                        flushed,
                        stream=output_stream,
                        wait_stream=(
                            torch.cuda.current_stream(primary)
                            if output_stream is not None
                            else None
                        ),
                        host_budget=host_budget,
                        owner=f"{label} combine output staging",
                    )
                except (HostMemoryError, torch.OutOfMemoryError):
                    output_pack_seconds += perf_counter() - pack_started
                    drain_outputs(keep=0)
                    commit_outputs(flushed)
                else:
                    output_pack_seconds += perf_counter() - pack_started
                    inflight_outputs.append((transfer, storage_bytes))
                    inflight_output_bytes += storage_bytes
                del flushed
            drain_outputs(keep=keep)

        try:
            for index, batch in enumerate(batches):
                current, pending = pending, None
                if current is None:
                    flush_outputs(keep=0)
                    current = prepare_batch(batch)
                if index + 1 < len(batches):
                    try:
                        pending = prepare_batch(batches[index + 1])
                    except admission_errors:
                        prefetch_fallbacks += 1
                transfer, prepare_elapsed = current
                prepare_seconds += prepare_elapsed
                allocate_seconds += transfer.allocate_seconds
                pack_seconds += transfer.pack_seconds
                submit_seconds += transfer.submit_seconds
                with transfer:
                    wait_started = perf_counter()
                    value = transfer.wait()
                    wait_seconds += perf_counter() - wait_started
                    combine_started = perf_counter()
                    with torch.no_grad(), HooksMixin.disable_hooks_local():
                        output = combine.forward(
                            adapter.model,
                            **{name: value[name] for name in combine.input_names},
                        )
                    combine_seconds += perf_counter() - combine_started
                    output = _select_destination_output(output, destination_input_names)
                    combined_tensors += transfer.tensor_count
                    combined_bytes += transfer.total_bytes
                    del value
                    close_started = perf_counter()
                close_seconds += perf_counter() - close_started
                current = None
                output_size = _output_storage_bytes(output)
                if pending_output_bytes + output_size > output_cap:
                    flush_outputs(keep=max_output_transfers - 1)
                while (
                    inflight_outputs
                    and inflight_output_bytes + pending_output_bytes + output_size
                    > output_cap
                ):
                    drain_outputs(keep=len(inflight_outputs) - 1)
                pending_outputs[batch] = output
                del output
                pending_output_bytes += output_size
                if (
                    pending_output_bytes >= output_cap
                    or len(pending_outputs) >= max_output_batches
                ):
                    flush_outputs(
                        keep=0 if output_size > output_cap else max_output_transfers - 1
                    )
            flush_outputs(keep=0)
            if batches:
                streaming_logger.info(
                    f"{label} | EP combine coalesced transfer | "
                    f"batches={len(batches)} | tensors={combined_tensors} | "
                    f"bytes={combined_bytes} | copy_stream={copy_stream is not None} | "
                    f"allocate_s={allocate_seconds:.2f} | "
                    f"pack_s={pack_seconds:.2f} | "
                    f"submit_s={submit_seconds:.2f} | "
                    f"prepare_s={prepare_seconds:.2f} | "
                    f"wait_s={wait_seconds:.2f} | "
                    f"prefetch_fallbacks={prefetch_fallbacks} | "
                    f"combine_s={combine_seconds:.2f} | "
                    f"output_flushes={output_flushes} | "
                    f"output_pack_s={output_pack_seconds:.2f} | "
                    f"output_wait_s={output_wait_seconds:.2f} | "
                    f"output_tensors={output_tensors} | "
                    f"output_bytes={output_bytes} | "
                    f"commit_s={commit_seconds:.2f} | "
                    f"close_s={close_seconds:.2f} | "
                    f"wall_s={perf_counter() - combine_started_at:.2f}"
                )
        finally:
            for transfer, _storage_bytes in inflight_outputs:
                transfer.close()
            inflight_outputs.clear()
            pending_outputs.clear()
            for item in (current, pending):
                if item is None:
                    continue
                transfer, _elapsed = item
                transfer.close()
    finally:
        borrowed_boundaries.clear()
        with prepared_windows_lock:
            leftovers = tuple(prepared_windows.values())
            prepared_windows.clear()
        for prepared in leftovers:
            prepared.close()
        result_store.clear()
        for reservation in result_reservations:
            reservation.close()
    return {
        "placement": placement.to_dict(),
        "windows": [window.to_dict() for window in windows],
    }
