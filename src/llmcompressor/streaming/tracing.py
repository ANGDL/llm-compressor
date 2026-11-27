"""Streaming execution backed by the oneshot sequential trace plan."""

from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from itertools import chain, groupby
from time import perf_counter
from typing import Any, Iterator

import torch
from torch import nn
from torch.fx import Graph

from llmcompressor.pipelines.sequential.helpers import (
    SPECIALIZED_INPUT_META_KEY,
    Subgraph,
    partition_graph,
    trace_consumed_names,
)
from llmcompressor.pipelines.sequential.plan import (
    SequentialExecutionPlan,
    trace_sequential_plan,
)
from llmcompressor.utils.helpers import (
    disable_cache,
    disable_hf_kernels,
    eval_context,
)

from ._logging import streaming_logger
from .activations.store import _tensors
from .activations.transfer import PreparedTreeTransfer, prepare_tree_transfer
from .admission import DeviceMemoryBudget, DeviceMemoryError
from .checkpoint import CheckpointWeightSource
from .data_worker import CalibrationDataWorker, initial_boundary_async_enabled
from .execution import (
    FrozenCalibrationInputs,
    StreamingExecutionDescriptor,
    describe_streaming_execution,
    fingerprint_execution_graph,
    streaming_input_contract,
)
from .loading import HostMemoryBudget, SubgraphWeightSession, TargetWeightLoader
from .materialization import StreamingDTypePolicy, WeightMaterializer

__all__ = ["TracedBoundaryAdapter", "trace_streaming_boundaries"]


class _PrefixDependencyDetected(RuntimeError):
    def __init__(self, module_name: str):
        super().__init__(module_name)
        self.module_name = module_name


def _project_through_target(
    subgraph: Subgraph,
    target_name: str,
    required_output_names: set[str] | None = None,
    *,
    additional_targets: Sequence[str] = (),
) -> Subgraph:
    """Keep the target and dependencies needed by later streaming partitions."""

    nodes_by_name = {}
    target_nodes = {}
    target_names = {target_name, *additional_targets}
    for node in subgraph.graph.nodes:
        if node.op == "output":
            if required_output_names is None:
                outputs = node.args[0]
                required_output_names = (
                    set(outputs) if isinstance(outputs, dict) else set()
                )
            break
        nodes_by_name[node.name] = node
        if node.op == "call_module" and str(node.target) in target_names:
            target_nodes[str(node.target)] = node
    missing = target_names - target_nodes.keys()
    if missing:
        raise ValueError(
            f"Targets {sorted(missing)!r} are absent from traced partition"
        )

    output_nodes = {
        name: node
        for name, node in nodes_by_name.items()
        if name in (required_output_names or ())
    }
    if not output_nodes:
        target_node = target_nodes[target_name]
        output_nodes[target_node.name] = target_node
    required_nodes = {*target_nodes.values(), *output_nodes.values()}
    required_nodes.update(
        node
        for node in nodes_by_name.values()
        if (
            node.op == "call_method"
            and str(node.target).endswith("_")
            and not str(node.target).startswith("__")
        )
        or (node.op == "call_function" and node.is_impure())
    )
    pending = list(required_nodes)
    while pending:
        for dependency in pending.pop().all_input_nodes:
            if dependency not in required_nodes:
                required_nodes.add(dependency)
                pending.append(dependency)

    graph = Graph()
    node_map = {}
    for node in subgraph.graph.nodes:
        if node in required_nodes:
            node_map[node] = graph.node_copy(
                node, lambda dependency: node_map[dependency]
            )
    graph.output({name: node_map[node] for name, node in output_nodes.items()})
    graph.lint()
    return Subgraph(
        graph=graph,
        input_names={
            str(node.target) for node in graph.nodes if node.op == "placeholder"
        },
        consumed_names=set(),
    )


def _split_target_subgraph(
    model: nn.Module, subgraph: Subgraph, target_names: Sequence[str]
) -> list[Subgraph]:
    """Assign each operation to exactly one target slice in execution order."""
    targets = set(target_names)
    partitions = [[]]
    seen_targets = []
    for node in subgraph.graph.nodes:
        if node.op == "call_module" and str(node.target) in targets:
            if seen_targets:
                partitions.append([])
            seen_targets.append(str(node.target))
        partitions[-1].append(node)
    if tuple(seen_targets) != tuple(target_names):
        raise ValueError("Projected target order differs from the sequential plan")
    return partition_graph(model, partitions)


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
    _prefix_runtime_parameters: tuple[str, ...] | None = None
    execution: StreamingExecutionDescriptor | None = None
    input_contract: Any = None
    specialized_input_names: tuple[str, ...] = ()
    _initial_host_budget: HostMemoryBudget | None = None
    _initial_data_worker: CalibrationDataWorker | None = None

    def _input_contract(self, batch):
        return {
            name: streaming_input_contract(
                value, specialize_numeric=name in self.specialized_input_names
            )
            for name, value in batch.items()
        }

    def _validate_input_contract(self, batch):
        if not isinstance(batch, Mapping):
            raise TypeError("Traced streaming calibration batches must be mappings")
        if self.input_contract is not None and (
            self._input_contract(batch) != self.input_contract
        ):
            raise ValueError("Calibration inputs differ from the traced input contract")

    @property
    def targets(self) -> tuple[str, ...]:
        return self.plan.target_names

    @property
    def prefix(self):
        return self.plan.subgraphs[0]

    @property
    def target_subgraphs(self):
        return self._target_subgraphs

    @property
    def target_groups(self) -> tuple[tuple[int, ...], ...]:
        """Return consecutive target indices owned by one traced subgraph."""

        groups: list[list[int]] = []
        previous_subgraph_index = None
        for target_index, subgraph_index in enumerate(
            self.plan.target_subgraph_indices
        ):
            if subgraph_index != previous_subgraph_index:
                groups.append([])
                previous_subgraph_index = subgraph_index
            groups[-1].append(target_index)
        return tuple(tuple(group) for group in groups)

    def calibration_boundaries(
        self, batches: Sequence[Mapping[str, Any]] | Any
    ) -> Iterator[dict[str, Any]]:
        """Execute the traced prefix without rewriting the model input contract."""

        with ExitStack() as stack:
            batches = stack.enter_context(self._prepared_calibration_batches(batches))
            prefix = self.plan.subgraphs[0]
            batch_iterator = iter(batches)
            if self._prefix_runtime_modules is None:
                try:
                    sample_batch = next(batch_iterator)
                except StopIteration as error:
                    raise ValueError(
                        "Streaming calibration batches are empty"
                    ) from error
                sample_value = (
                    sample_batch.value
                    if isinstance(sample_batch, PreparedTreeTransfer)
                    else sample_batch
                )
                self._validate_input_contract(sample_value)
                self._prefix_runtime_parameters = self._declared_runtime_parameters(
                    "prefix"
                )
                self._prefix_runtime_modules = self._discover_prefix_runtime_modules(
                    prefix, sample_value
                )
                batch_iterator = chain((sample_batch,), batch_iterator)
            stack.enter_context(
                self.weight_session.loaded(
                    prefix,
                    device=self.device,
                    dtype=self.dtype,
                    include_modules=self._prefix_runtime_modules,
                    include_parameters=self._prefix_runtime_parameters or (),
                )
            )
            stream = (
                torch.cuda.Stream(device=self.device)
                if self.device.type == "cuda" and initial_boundary_async_enabled()
                else None
            )
            device_budget = DeviceMemoryBudget()
            if stream is not None:
                stack.callback(torch.cuda.current_stream(self.device).synchronize)
            with (
                torch.no_grad(),
                disable_cache(self.model),
                eval_context(self.model),
                disable_hf_kernels(self.model),
            ):
                for batch_index, batch in enumerate(batch_iterator):
                    with ExitStack() as transfers:
                        started = perf_counter()
                        if isinstance(batch, PreparedTreeTransfer):
                            try:
                                transfer = batch.transfer(
                                    self.device,
                                    stream=stream,
                                    device_budget=device_budget,
                                )
                            except DeviceMemoryError:
                                values = _move_tensors(batch.value, self.device)
                            else:
                                transfers.enter_context(transfer)
                                values = transfer.wait()
                        else:
                            self._validate_input_contract(batch)
                            values = _move_tensors(dict(batch), self.device)
                        inputs = {name: values[name] for name in prefix.input_names}
                        moved = perf_counter()
                        result = prefix.forward(self.model, **inputs)
                        streaming_logger.info(
                            f"initial boundary | prefix batch={batch_index} | "
                            f"input_submit_s={moved - started:.2f} | "
                            f"forward_submit_s={perf_counter() - moved:.2f}"
                        )
                        yield result
                        del result, inputs, values

    @contextmanager
    def _prepared_calibration_batches(self, batches):
        """Prepare at most one upcoming frozen batch on a CPU-only worker."""
        if (
            not initial_boundary_async_enabled()
            or not isinstance(batches, FrozenCalibrationInputs)
            or batches.batch_bytes is None
        ):
            yield iter(batches)
            return
        packed = self.device.type == "cuda" and batches.batch_transfer_bytes is not None

        def prepare(value):
            self._validate_input_contract(value)
            return (
                prepare_tree_transfer(value)
                if packed and all(tensor.is_contiguous() for tensor in _tensors(value))
                else value
            )

        with CalibrationDataWorker(
            len(batches),
            batches.read_owned,
            batches.estimate_transfer_bytes if packed else batches.estimate_read_bytes,
            host_budget=self._initial_host_budget or HostMemoryBudget(),
            prepare=prepare,
        ) as worker:
            self._initial_data_worker = worker
            worker.prefetch()

            def iterate():
                for _index in range(len(batches)):
                    with worker.take() as batch:
                        worker.prefetch()
                        yield batch.value

            iterator = iterate()
            try:
                yield iterator
            finally:
                iterator.close()
                self._initial_data_worker = None

    def discard_initial_prefetch(self) -> None:
        """Yield speculative input capacity to the current boundary snapshot."""
        if self._initial_data_worker is not None:
            self._initial_data_worker.discard_prefetch()

    def _discover_prefix_runtime_modules(
        self, prefix: Subgraph, sample_batch: Mapping[str, Any]
    ) -> tuple[str, ...]:
        """Discover checkpoint branches hidden inside autowrapped prefix calls."""

        if not isinstance(sample_batch, Mapping):
            raise TypeError("Traced streaming calibration batches must be mappings")
        candidates = self._prefix_dependency_candidates()
        discovered = list(self._execution_dependencies("prefix")[0])
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
                with ExitStack() as stack:
                    stack.enter_context(
                        self.weight_session.loaded(
                            prefix,
                            device=self.device,
                            dtype=self.dtype,
                            include_modules=discovered,
                            include_parameters=self._prefix_runtime_parameters or (),
                        )
                    )
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

    def _declared_runtime_parameters(self, stage: str) -> tuple[str, ...]:
        """Read model-declared direct parameter dependencies for one stage."""

        hook = getattr(
            self.model,
            "streaming_runtime_parameter_dependencies",
            None,
        )
        dependencies = hook(stage) if callable(hook) else ()
        if dependencies is None:
            dependencies = ()
        if isinstance(dependencies, str):
            dependencies = (dependencies,)
        declared = self._execution_dependencies(stage)[1]
        result = tuple(dict.fromkeys((*map(str, dependencies), *declared)))
        for name in result:
            owner_name, separator, local_name = name.rpartition(".")
            if not separator:
                owner_name = ""
            owner = self.model.get_submodule(owner_name)
            if local_name not in owner._parameters:
                raise ValueError(
                    f"Model-declared streaming parameter dependency {name!r} "
                    "does not name a direct parameter"
                )
        return result

    def _execution_dependencies(self, stage):
        modules, parameters = [], []
        if self.execution is not None:
            for (
                name,
                module_names,
                parameter_names,
            ) in self.execution.runtime_dependencies:
                if name == stage:
                    modules.extend(module_names)
                    parameters.extend(parameter_names)
        return tuple(dict.fromkeys(modules)), tuple(dict.fromkeys(parameters))

    def _target_load_dependencies(self, index):
        name = self.targets[index]
        modules, _parameters = self._execution_dependencies(name)
        parameters = self._declared_runtime_parameters(name)
        for dependency in (*modules, *parameters):
            if any(
                target != name
                and (dependency == target or dependency.startswith(target + "."))
                for target in self.targets
            ):
                raise ValueError(
                    "Runtime dependencies on another quantized target require "
                    "an explicit propagated boundary"
                )
        for dependency in modules:
            if any(
                target != name and target.startswith(dependency + ".")
                for target in self.targets
            ):
                raise ValueError(
                    "Runtime dependency modules must not contain other targets"
                )
        return modules, parameters

    def plan_target(self, index):
        modules, parameters = self._target_load_dependencies(index)
        return self.weight_session.plan(
            self.target_subgraphs[index],
            include_modules=modules,
            include_parameters=parameters,
        )

    def plan_group(self, indices: Sequence[int]):
        """Plan the union working set for one multi-target execution unit."""

        if not indices:
            raise ValueError("A streaming target group cannot be empty")
        if len(indices) == 1:
            return self.plan_target(indices[0])
        dependencies = [self._target_load_dependencies(index) for index in indices]
        return self.weight_session.plan_subgraphs(
            tuple(self.target_subgraphs[index] for index in indices),
            include_modules=tuple(
                name for modules, _parameters in dependencies for name in modules
            ),
            include_parameters=tuple(
                name for _modules, parameters in dependencies for name in parameters
            ),
        )

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
            modules, _parameters = self._execution_dependencies(target_name)
            parameters = tuple(
                name
                for name in self._declared_runtime_parameters(target_name)
                if not name.startswith(target_name + ".")
            )
            helper_plan = self.weight_session.plan(
                subgraph,
                include_modules=modules,
                include_parameters=parameters,
                exclude_modules=(target_name,),
            )
            with self.weight_session.loaded_plan(
                helper_plan,
                device=self.device,
                dtype=self.dtype,
            ):
                output = subgraph.forward(self.model, **inputs)
        finally:
            self.model.set_submodule(target_name, original)
        result = {**value, **output}
        for name in subgraph.consumed_names:
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
    targets_per_subgraph: int = 1,
) -> TracedBoundaryAdapter:
    """Trace model boundaries with the same implementation used by oneshot."""

    plan = trace_sequential_plan(
        model,
        dict(sample_batch),
        sequential_targets,
        tracing_ignore,
        targets_per_subgraph=targets_per_subgraph,
    )
    requested_target_names = tuple(target_names) if target_names is not None else None
    if requested_target_names is not None and (
        len(requested_target_names) != len(plan.target_names)
        or set(requested_target_names) != set(plan.target_names)
    ):
        raise ValueError(
            "Explicit target names differ from the shared sequential trace: "
            f"expected {plan.target_names}, got {requested_target_names}"
        )
    projected_groups = []
    required_names: set[str] = set()
    groups = [
        (subgraph_index, tuple(indices))
        for subgraph_index, indices in groupby(
            range(len(plan.target_names)),
            key=lambda index: plan.target_subgraph_indices[index],
        )
    ]
    for subgraph_index, target_indices in reversed(groups):
        target_names_in_group = tuple(
            plan.target_names[index] for index in target_indices
        )
        target_subgraph = _project_through_target(
            plan.subgraphs[subgraph_index],
            target_names_in_group[-1],
            required_names,
            additional_targets=target_names_in_group[:-1],
        )
        projected_groups.append(
            _split_target_subgraph(model, target_subgraph, target_names_in_group)
        )
        output_node = next(
            node for node in target_subgraph.graph.nodes if node.op == "output"
        )
        required_names.difference_update(output_node.args[0])
        required_names.update(target_subgraph.input_names)
    target_subgraphs = [
        subgraph for group in reversed(projected_groups) for subgraph in group
    ]
    trace_consumed_names(target_subgraphs)
    execution = replace(
        describe_streaming_execution(model, plan.target_names, sample_batch),
        graph_fingerprint=fingerprint_execution_graph(
            (plan.subgraphs[0], *target_subgraphs)
        ),
    )

    loader = TargetWeightLoader(model, source, materializer, dtype_policy=dtype_policy)
    weight_session = SubgraphWeightSession(
        model, source, materializer, dtype_policy=dtype_policy
    )

    specialized_input_names = tuple(
        dict.fromkeys(
            str(node.target)
            for subgraph in plan.subgraphs
            for node in subgraph.graph.nodes
            if node.op == "placeholder" and node.meta.get(SPECIALIZED_INPUT_META_KEY)
        )
    )
    adapter = TracedBoundaryAdapter(
        model=model,
        plan=plan,
        loader=loader,
        weight_session=weight_session,
        device=torch.device(device),
        dtype=dtype,
        _target_subgraphs=tuple(target_subgraphs),
        execution=execution,
        specialized_input_names=specialized_input_names,
    )
    adapter.input_contract = adapter._input_contract(sample_batch)
    return adapter
