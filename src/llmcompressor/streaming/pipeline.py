"""Per-subgraph closed-loop execution for streaming PTQ."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from functools import lru_cache
from math import prod
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

import torch
from compressed_tensors.compressors import compress_module
from compressed_tensors.quantization.utils import is_module_quantized

from llmcompressor.core import LifecycleCallbacks, active_session, create_session
from llmcompressor.modifiers.quantization.calibration import (
    set_quantization_scale_dtype,
)
from llmcompressor.modifiers.quantization.quantization.mixin import QuantizationMixin
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache, IntermediateValue
from llmcompressor.pipelines.sequential.helpers import Subgraph
from llmcompressor.recipe import Recipe
from llmcompressor.utils.hardware_ops import available_hardware_ops
from llmcompressor.utils.helpers import DisableQuantization
from llmcompressor.utils.int4_packing import DEFAULT_INT4_PACKING_WORKSPACE_BYTES
from llmcompressor.utils.metric_logging import CompressionLogger

from ._logging import streaming_logger
from .activations import (
    BoundaryActivationStore,
    DiskBoundaryActivationStore,
    InMemoryBoundaryActivationStore,
)
from .admission import (
    DeviceMemoryBudget,
    DeviceMemoryError,
    DiskSpaceBudget,
    estimate_modifier_memory,
)
from .artifacts import (
    ArtifactStore,
    CalibrationInfo,
    ExecutionInfo,
    ParallelInfo,
    RecipeInfo,
    SequentialInfo,
    SoftwareInfo,
    StreamingRunManifest,
    fingerprint_checkpoint,
    fingerprint_json,
)
from .checkpoint import (
    AsyncDirectSafetensorsWriter,
    DirectSafetensorsWriter,
    SingleParticipantCoordinator,
    StreamingCheckpointWriter,
)
from .compaction import BoundaryCompactor
from .execution import (
    FrozenCalibrationInputs,
    StreamingExecutionDescriptor,
    describe_streaming_execution,
    freeze_calibration_inputs,
)
from .loading import HostMemoryBudget, HostMemoryError, SubgraphPrefetcher
from .materialization import (
    StreamingDTypePolicy,
    WeightMaterializer,
    materialize_weights,
)
from .memory_adapters import (
    get_modifier_memory_adapter,
    modifier_calibration_context,
)
from .modifier_adapters import (
    StreamingModifierAdapter,
    get_modifier_adapter,
    modifier_adapters,
    target_rng_scope,
)
from .modifier_contract import (
    ModifierContractError,
    build_modifier_contracts,
    calibration_outputs_reusable,
)
from .output import (
    prepare_quantized_tensor_for_save,
    quantized_module_formats,
    quantized_tensor_bytes_for_save,
)
from .parallel import (
    OwnerModuleGroup,
    OwnerUpdateScheduler,
    ParallelContext,
    PlacementPlan,
    partition_modules_by_owner,
)
from .parallel.expert import ExpertParallelExecutor, ExpertParallelPlan
from .parallel.region import (
    ExpertRegionGraph,
    UnsupportedExpertRegion,
    coalesce_expert_regions,
    execute_expert_region,
    profile_expert_region,
)
from .parallel.route_cache import ExpertRouteCache
from .parallel.telemetry import EPPhaseLogger
from .parallel.token_budget import (
    TOKEN_BUDGET_POLICY,
    select_token_capacity,
    validate_token_budget,
)
from .recovery import build_resume_plan
from .tied_weights import infer_transformers_tied_weights
from .tracing import TracedBoundaryAdapter

__all__ = ["run_subgraph_streaming_pipeline"]


def _empty_device_cache(device: torch.device) -> None:
    backend = getattr(torch, device.type, None)
    empty_cache = getattr(backend, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.get_device_module(device).reset_peak_memory_stats(device)


_ASYNC_H2D_ENV = "LLMCOMPRESSOR_STREAMING_ASYNC_H2D"


def _async_h2d_enabled() -> bool:
    """Return whether prefetch may enqueue a background CPU-to-CUDA copy."""

    value = os.environ.get(_ASYNC_H2D_ENV, "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _stage_logger(
    model: torch.nn.Module, name: str, device: torch.device
) -> CompressionLogger:
    device_ids = (
        (
            device.index
            if device.index is not None
            else torch.get_device_module(device).current_device(),
        )
        if device.type == "cuda"
        else ()
    )
    metrics = CompressionLogger(model, device_ids=device_ids)
    metrics.set_results(
        name="streaming",
        summary=f"{name} | device={device}",
    )
    return metrics


def _target_label(index: int, total: int, target: str) -> str:
    width = max(2, len(str(total)))
    targets = target.split(" + ")
    if len(targets) > 2:
        target = f"{targets[0]} ... {targets[-1]} ({len(targets)} targets)"
    return f"[{index + 1:0{width}d}/{total:0{width}d}] {target}"


def _ep_phase_logger(executor, label, phase, **kwargs):
    return (
        executor.log_phase(label, phase, **kwargs)
        if executor is not None
        else nullcontext()
    )


_TOKEN_METADATA_NAMES = frozenset(
    {
        "attention_mask",
        "image_mask",
        "image_token_types",
        "input_ids",
        "position_ids",
        "token_type_ids",
    }
)


def _tensor_token_count(tensor: torch.Tensor, name: str | None = None) -> int:
    """Estimate token count without charging routing or feature dimensions."""

    if tensor.ndim == 0:
        return 0
    if tensor.ndim == 1:
        return int(tensor.shape[0])
    if name in _TOKEN_METADATA_NAMES or not tensor.is_floating_point():
        return int(prod(tensor.shape[:2]))
    if tensor.ndim == 2:
        return int(tensor.shape[0])
    return int(prod(tensor.shape[:2]))


def _boundary_token_count(value: Any, name: str | None = None) -> int:
    if isinstance(value, torch.Tensor):
        return _tensor_token_count(value, name)
    if isinstance(value, Mapping):
        named = []
        unnamed = []
        for item_name, item in value.items():
            count = _boundary_token_count(item, str(item_name))
            if str(item_name) in _TOKEN_METADATA_NAMES:
                named.append(count)
            else:
                unnamed.append(count)
        return max(named or unnamed, default=0)
    if isinstance(value, (list, tuple)):
        return max(
            (_boundary_token_count(item, name) for item in value),
            default=0,
        )
    return 0


def _boundary_nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.nbytes
    if isinstance(value, Mapping):
        return sum(_boundary_nbytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_boundary_nbytes(item) for item in value)
    return 0


def _debug_tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    numel = tensor.numel()
    summary: dict[str, Any] = {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": numel,
    }
    if tensor.is_meta:
        summary["meta"] = True
        return summary
    if numel == 0:
        summary["empty"] = True
        return summary
    detached = tensor.detach()
    summary["all_zero"] = bool(torch.count_nonzero(detached).item() == 0)
    if detached.dtype.is_floating_point:
        summary["finite"] = bool(torch.isfinite(detached).all().item())
        summary["abs_max"] = float(detached.float().abs().amax().item())
    return summary


def _debug_value_summary(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _debug_tensor_summary(value)
    if isinstance(value, Mapping):
        return {str(name): _debug_value_summary(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_debug_value_summary(item) for item in value]
    return type(value).__name__


def _debug_boundary(
    stage: str,
    *,
    boundary: int,
    batch: int,
    producer: str,
    consumer: str,
    value: Any,
) -> None:
    streaming_logger.opt(lazy=True).debug(
        "boundary | {stage} | index={boundary} | batch={batch} | "
        "producer={producer} | consumer={consumer} | values={values}",
        stage=lambda: stage,
        boundary=lambda: boundary,
        batch=lambda: batch,
        producer=lambda: producer,
        consumer=lambda: consumer,
        values=lambda: _debug_value_summary(value),
    )


def _output_shard(source, tensor_name: str, owner_name: str) -> str:
    available = set(source.tensor_names())
    if tensor_name in available:
        return source.metadata(tensor_name).shard.name
    candidate = owner_name
    while candidate:
        weight_name = f"{candidate}.weight"
        if weight_name in available:
            return source.metadata(weight_name).shard.name
        candidate = candidate.rpartition(".")[0]
    shards = {source.metadata(name).shard.name for name in available}
    if len(shards) == 1:
        return next(iter(shards))
    raise KeyError(f"Cannot assign output shard for tensor {tensor_name!r}")


def _serialized_tensor_name(
    materializer: WeightMaterializer | None, tensor_name: str
) -> str:
    return (
        tensor_name
        if materializer is None
        else materializer.output_tensor_name(tensor_name)
    )


def _serialized_module_name(
    materializer: WeightMaterializer | None, module_name: str
) -> str:
    return (
        module_name
        if materializer is None
        else materializer.output_module_name(module_name)
    )


def _prepared_output_tensors(
    materializer: WeightMaterializer | None,
    tensor_name: str,
    tensor: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    return (
        {tensor_name: tensor}
        if materializer is None
        else materializer.prepare_output_tensors(tensor_name, tensor)
    )


def _normalized_recipe_payload(recipe: Recipe) -> dict[str, Any]:
    payload = recipe.model_dump(mode="json", exclude={"modifiers"})
    payload["modifiers"] = [
        {
            "type": type(modifier).__name__,
            "arguments": modifier.model_dump(mode="json"),
        }
        for modifier in recipe.modifiers
    ]
    return payload


def _initialize_run(
    *,
    checkpoint: str | Path,
    artifact_dir: Path,
    recipe: Recipe,
    dataset_fingerprint: str,
    targets: tuple[str, ...],
    targets_per_subgraph: int = 1,
    materializer: WeightMaterializer,
    target_dtype: torch.dtype,
    num_samples: int,
    max_seq_length: int | None,
    seed: int | None,
    pack_to_int8: bool,
    dtype_policy: StreamingDTypePolicy | None = None,
    parallel_context: ParallelContext | None = None,
    placement: PlacementPlan | None = None,
    replace_existing: bool = False,
    execution: StreamingExecutionDescriptor | None = None,
    expert_parallel: ExpertParallelPlan | None = None,
    expert_parallel_token_budget: str | int = "auto",
    target_groups: tuple[tuple[int, ...], ...] = (),
) -> str:
    normalized_recipe = _normalized_recipe_payload(recipe)
    source_info = fingerprint_checkpoint(checkpoint)
    modifier_contracts = build_modifier_contracts(recipe.modifiers)
    modifier_contract_sha256 = fingerprint_json(
        [contract.to_dict() for contract in modifier_contracts]
    )
    parallel_context = parallel_context or ParallelContext.single_device()
    placement = placement or PlacementPlan(ep_size=parallel_context.ep_size)
    if expert_parallel is not None:
        parallel_context = ParallelContext(ep_size=len(expert_parallel.devices))
        placement = expert_parallel.placement
    run_fingerprint = fingerprint_json(
        {
            "source": source_info.content_fingerprint,
            "recipe": normalized_recipe,
            "dataset": dataset_fingerprint,
            "targets": targets,
            "targets_per_subgraph": targets_per_subgraph,
            "materializer": materializer.manifest_info(
                target_dtype=target_dtype, dtype_policy=dtype_policy
            ).config_sha256,
            "output": {"pack_to_int8": pack_to_int8},
            "execution": {
                "propagation_mode": "sequential-compatible",
                "boundary_schema": 1,
                "modifier_contract_sha256": modifier_contract_sha256,
                "execution_graph_sha256": fingerprint_json(
                    execution.to_dict() if execution is not None else {}
                ),
            },
            "parallel": {
                "context": parallel_context.to_dict(),
                "placement": placement.fingerprint,
                **(
                    {
                        "expert_parallel": expert_parallel.to_dict(),
                        "token_budget_policy": TOKEN_BUDGET_POLICY,
                        "token_budget": expert_parallel_token_budget,
                        "target_groups": target_groups,
                    }
                    if expert_parallel is not None
                    else {}
                ),
            },
        }
    )
    manifest = StreamingRunManifest(
        source=source_info,
        recipe=RecipeInfo(fingerprint_json(normalized_recipe)),
        calibration=CalibrationInfo(
            dataset_fingerprint=dataset_fingerprint,
            num_samples=num_samples,
            max_seq_length=max_seq_length,
            seed=seed,
        ),
        sequential=SequentialInfo(targets, targets_per_subgraph=targets_per_subgraph),
        materializer=materializer.manifest_info(
            target_dtype=target_dtype, dtype_policy=dtype_policy
        ),
        software=SoftwareInfo.from_versions({"torch": torch.__version__}),
        execution=ExecutionInfo(
            modifier_contract_sha256=modifier_contract_sha256,
            execution_graph_sha256=fingerprint_json(
                execution.to_dict() if execution is not None else {}
            ),
        ),
        parallel=ParallelInfo.from_context(
            parallel_context.to_dict(),
            placement_fingerprint=placement.fingerprint,
            backend=expert_parallel.backend if expert_parallel else "single-device",
            devices=tuple(map(str, expert_parallel.devices)) if expert_parallel else (),
            target_groups=target_groups if expert_parallel else (),
            expert_owners=(
                tuple(
                    (entry.logical_name, entry.owner_rank)
                    for entry in expert_parallel.placement.entries
                )
                if expert_parallel
                else ()
            ),
            token_budget_policy=TOKEN_BUDGET_POLICY if expert_parallel else "",
            token_budget=expert_parallel_token_budget,
        ),
    )
    ArtifactStore(artifact_dir).initialize(
        manifest,
        normalized_recipe=normalized_recipe,
        targets=targets,
        replace_existing=replace_existing,
    )
    return run_fingerprint


def _prepare_loaded_tensor_for_save(
    loaded,
    name: str,
    tensor: torch.Tensor,
    formats: Mapping[str, str],
    *,
    pack_to_int8: bool,
) -> torch.Tensor:
    owner_name, separator, local_name = name.rpartition(".")
    if not separator or local_name != "weight" or owner_name not in formats:
        return tensor
    module = loaded.model.get_submodule(owner_name)
    return prepare_quantized_tensor_for_save(
        name,
        tensor,
        module.quantization_scheme,
        format_name=formats[owner_name],
        pack_to_int8=pack_to_int8,
    )


def _is_under_any(module_name: str, roots: Sequence[str]) -> bool:
    return any(
        module_name == root or module_name.startswith(f"{root}.") for root in roots
    )


def _target_roots(target_name: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(target_name, str):
        return (target_name,)
    roots = tuple(target_name)
    if not roots:
        raise ValueError("At least one target root is required")
    return roots


def _loaded_target_formats(model, roots, exclude_modules):
    formats = {}
    for root in roots:
        try:
            module = model.get_submodule(root)
        except AttributeError:
            continue
        formats.update(
            quantized_module_formats(module.named_modules(prefix=root), prefix=root)
        )
    return {
        name: value
        for name, value in formats.items()
        if not _is_under_any(name, exclude_modules)
    }


def _write_loaded_target(
    transaction,
    loaded,
    target_name: str | Sequence[str],
    source,
    materializer: WeightMaterializer,
    *,
    pack_to_int8: bool,
    exclude_modules: Sequence[str] = (),
) -> set[str]:
    target_roots = _target_roots(target_name)
    written = set()
    formats = _loaded_target_formats(loaded.model, target_roots, exclude_modules)
    for name, tensor in loaded.state_tensors_under(target_roots):
        owner_name = name.rpartition(".")[0]
        if _is_under_any(owner_name, exclude_modules):
            continue
        tensor = _prepare_loaded_tensor_for_save(
            loaded,
            name,
            tensor,
            formats,
            pack_to_int8=pack_to_int8,
        )
        for output_name, output_tensor in materializer.prepare_output_tensors(
            name, tensor
        ).items():
            transaction.write_tensor(
                output_name,
                output_tensor,
                output_shard=_output_shard(source, name, owner_name),
            )
        written.add(name)

    for module_name, format_name in formats.items():
        transaction.mark_quantized(
            materializer.output_module_name(module_name), format_name
        )
        # Compression formats such as pack-quantized replace the source weight
        # with weight_packed/weight_shape. Mark the original source tensor as
        # consumed so the fallback copier cannot reintroduce a second raw weight.
        written.add(f"{module_name}.weight")
    return written


def _loaded_target_delta(
    loaded,
    target_name: str | Sequence[str],
    materializer: WeightMaterializer | None = None,
    *,
    pack_to_int8: bool,
    exclude_modules: Sequence[str] = (),
):
    target_roots = _target_roots(target_name)
    formats = _loaded_target_formats(loaded.model, target_roots, exclude_modules)
    tensors = {}
    written = set()
    owned_cpu_tensors = set()
    for name, tensor in loaded.state_tensors_under(target_roots):
        if _is_under_any(name.rpartition(".")[0], exclude_modules):
            continue
        prepared = _prepare_loaded_tensor_for_save(
            loaded,
            name,
            tensor,
            formats,
            pack_to_int8=pack_to_int8,
        )
        output_tensors = _prepared_output_tensors(materializer, name, prepared)
        for output_name, output_tensor in output_tensors.items():
            if output_name in tensors:
                raise ValueError(f"Duplicate serialized tensor name {output_name!r}")
            tensors[output_name] = output_tensor
        written.add(name)
        if prepared is not tensor and prepared.device.type == "cpu":
            owned_cpu_tensors.update(
                name
                for name, value in output_tensors.items()
                if value.device.type == "cpu"
            )
    output_formats = {
        _serialized_module_name(materializer, name): format_name
        for name, format_name in formats.items()
    }
    for module_name in formats:
        written.add(f"{module_name}.weight")
    return tensors, output_formats, written, owned_cpu_tensors


def _estimate_loaded_target_snapshot_bytes(
    loaded,
    target_name: str | Sequence[str],
    *,
    pack_to_int8: bool,
    exclude_modules: Sequence[str] = (),
) -> int:
    target_roots = _target_roots(target_name)
    formats = _loaded_target_formats(loaded.model, target_roots, exclude_modules)
    total = 0
    workspace = 0
    for name, tensor in loaded.state_tensors_under(target_roots):
        owner_name, _, local_name = name.rpartition(".")
        if _is_under_any(owner_name, exclude_modules):
            continue
        format_name = formats.get(owner_name)
        if local_name == "weight" and format_name is not None:
            module = loaded.model.get_submodule(owner_name)
            output_bytes = quantized_tensor_bytes_for_save(
                name,
                tensor,
                module.quantization_scheme,
                format_name=format_name,
                pack_to_int8=pack_to_int8,
            )
            total += output_bytes
            if tensor.device.type == "cpu" and output_bytes < tensor.nbytes:
                workspace = max(
                    workspace,
                    min(DEFAULT_INT4_PACKING_WORKSPACE_BYTES, 2 * output_bytes),
                )
        else:
            total += tensor.numel() * tensor.element_size()
    return total + workspace


def _write_loaded_target_direct(
    writer: DirectSafetensorsWriter,
    loaded,
    target_name: str | Sequence[str],
    shard_id: str,
    materializer: WeightMaterializer | None = None,
    *,
    pack_to_int8: bool = True,
    exclude_modules: Sequence[str] = (),
    host_budget: HostMemoryBudget | None = None,
) -> set[str]:
    """Write one target and release all device tensor references before returning."""

    tensors = {}
    reservation = None
    try:
        estimate = getattr(writer, "estimate_snapshot_bytes", None)
        if host_budget is not None and callable(estimate):
            required = _estimate_loaded_target_snapshot_bytes(
                loaded,
                target_name,
                pack_to_int8=pack_to_int8,
                exclude_modules=exclude_modules,
            )
            reservation = host_budget.reserve(f"checkpoint shard {shard_id}", required)
        tensors, formats, written, owned_cpu_tensors = _loaded_target_delta(
            loaded,
            target_name,
            materializer,
            pack_to_int8=pack_to_int8,
            exclude_modules=exclude_modules,
        )
        if tensors:
            writer_kwargs = {
                "quantized_modules": formats,
                "owned_cpu_tensors": owned_cpu_tensors,
            }
            if reservation is not None:
                writer_kwargs["reservation"] = reservation
            writer.write_shard(shard_id, tensors, **writer_kwargs)
        elif reservation is not None:
            reservation.close()
        return written
    except Exception:
        if reservation is not None:
            reservation.close()
        raise
    finally:
        tensors.clear()


def _modifier_update(
    modules: Sequence[torch.nn.Module],
    quantizers: Sequence[QuantizationMixin],
    adapters: Sequence[StreamingModifierAdapter] = (),
    owner_groups: Sequence[OwnerModuleGroup] | None = None,
    telemetry: EPPhaseLogger | None = None,
) -> str:
    with ExitStack() as stack:
        for adapter in adapters:
            stack.enter_context(adapter.update_context(active_session().state.model))
        owner_update_adapter = _owner_update_adapter(quantizers, adapters)
        if owner_groups is not None and owner_update_adapter is not None:
            validate = getattr(owner_update_adapter, "validate_owner_groups", None)
            if callable(validate):
                validate(owner_groups)
            begin = getattr(owner_update_adapter, "begin_owner_update", None)
            if callable(begin):
                begin(modules)
            OwnerUpdateScheduler().run(
                owner_groups,
                update=owner_update_adapter.update_owner,
                telemetry=telemetry,
            )
            finalize = getattr(owner_update_adapter, "finalize_owner_update", None)
            if callable(finalize):
                finalize()
            update_parallelism = "owner-local"
        else:
            context = (
                telemetry.owner_work(telemetry.devices[0], modules=len(modules))
                if telemetry is not None
                else nullcontext()
            )
            with context:
                LifecycleCallbacks.sequential_epoch_end(list(modules))
            update_parallelism = "serial-contract"
    for adapter in adapters:
        adapter.after_update(modules)
    for quantizer in quantizers:
        quantizer.end_layerwise_calibration(list(modules))
    return update_parallelism


def _owner_update_adapter(
    quantizers: Sequence[QuantizationMixin],
    adapters: Sequence[StreamingModifierAdapter],
):
    if len(quantizers) != 1 or adapters:
        return None
    candidate = get_modifier_memory_adapter(quantizers[0])
    return candidate if callable(getattr(candidate, "update_owner", None)) else None


def _target_group_input_names(subgraphs: Sequence[Subgraph]) -> set[str]:
    """Find external inputs without including values produced inside the group."""
    available = set()
    required = set()
    for subgraph in subgraphs:
        required.update(subgraph.input_names - available)
        output = next(node for node in subgraph.graph.nodes if node.op == "output")
        available.update(output.args[0])
        available.difference_update(subgraph.consumed_names)
    return required


def _forward_target_group(
    adapter: TracedBoundaryAdapter,
    target_indices: Sequence[int],
    value: Mapping[str, Any],
    executor: ExpertParallelExecutor | None = None,
) -> dict[str, Any]:
    """Execute consecutive target subgraphs without publishing internal boundaries."""

    if executor is not None:
        return executor.forward(
            tuple(adapter.target_subgraphs[index] for index in target_indices), value
        )
    current = dict(value)
    for target_index in target_indices:
        subgraph = adapter.target_subgraphs[target_index]
        inputs = {name: current[name] for name in subgraph.input_names}
        output = subgraph.forward(adapter.model, **inputs)
        next_value = {**current, **output}
        for consumed in subgraph.consumed_names:
            next_value.pop(consumed, None)
        current = next_value
    return current


def _forward_target_group_many(
    adapter: TracedBoundaryAdapter,
    target_indices: Sequence[int],
    values: Sequence[Mapping[str, Any]],
    executor: ExpertParallelExecutor | None = None,
) -> tuple[dict[str, Any], ...]:
    """Forward a bounded set of batches, coalescing EP expert calls."""

    if executor is not None:
        return executor.forward_many(
            tuple(adapter.target_subgraphs[index] for index in target_indices),
            values,
        )
    return tuple(
        _forward_target_group(adapter, target_indices, value) for value in values
    )


def _iter_execution_batches(
    boundaries: BoundaryActivationStore,
    boundary_index: int,
    batch_indices: Sequence[int],
    input_names: set[str],
    *,
    device: torch.device,
    token_capacity: int | None,
    input_bytes_capacity: int | None = None,
    telemetry: EPPhaseLogger | None = None,
):
    """Load independent boundary inputs within token and byte limits."""

    def load(batch_index):
        context = (
            telemetry.primary_work("load", "loading boundary inputs")
            if telemetry is not None
            else nullcontext()
        )
        with context:
            return boundaries.get_for_execution(
                boundary_index, batch_index, input_names, device=device
            )

    if token_capacity is None and input_bytes_capacity is None:
        for batch_index in batch_indices:
            yield (batch_index,), (load(batch_index),)
        return

    current_indices = []
    current_values = []
    current_tokens = 0
    current_bytes = 0
    for batch_index in batch_indices:
        context = (
            telemetry.primary_work("load", "reading boundary metadata")
            if telemetry is not None
            else nullcontext()
        )
        with context:
            describe = getattr(boundaries, "execution_metadata", None)
            if callable(describe):
                metadata = describe(boundary_index, batch_index, input_names)
                value = metadata.values
                input_bytes = metadata.tensor_bytes
                del metadata
            else:
                value = boundaries.get(boundary_index, batch_index, device="meta")
                input_bytes = _boundary_nbytes(value)
            tokens = _boundary_token_count({name: value[name] for name in input_names})
            del value
        if token_capacity is not None and tokens > token_capacity:
            raise DeviceMemoryError(
                f"Boundary {boundary_index} batch {batch_index} has {tokens} "
                f"tokens, exceeding the input token capacity {token_capacity}"
            )
        if input_bytes_capacity is not None and input_bytes > input_bytes_capacity:
            raise DeviceMemoryError(
                f"Boundary {boundary_index} batch {batch_index} has {input_bytes} "
                f"input bytes, exceeding the input byte capacity {input_bytes_capacity}"
            )
        exceeds_capacity = current_values and (
            (token_capacity is not None and current_tokens + tokens > token_capacity)
            or (
                input_bytes_capacity is not None
                and current_bytes + input_bytes > input_bytes_capacity
            )
        )
        if exceeds_capacity:
            yield tuple(current_indices), tuple(current_values)
            current_indices = []
            current_values = []
            current_tokens = 0
            current_bytes = 0
        value = load(batch_index)
        current_indices.append(batch_index)
        current_values.append(value)
        current_tokens += tokens
        current_bytes += input_bytes
    if current_values:
        yield tuple(current_indices), tuple(current_values)


def _reference_tensor(value: Any) -> torch.Tensor | None:
    """Find a floating decoder output in a boundary value."""

    if isinstance(value, torch.Tensor):
        return value if value.is_floating_point() and value.ndim >= 2 else None
    if isinstance(value, Mapping):
        for name in ("hidden_states", "inputs_embeds"):
            if name in value:
                reference = _reference_tensor(value[name])
                if reference is not None:
                    return reference
        for item in value.values():
            reference = _reference_tensor(item)
            if reference is not None:
                return reference
    elif isinstance(value, (tuple, list)):
        for item in value:
            reference = _reference_tensor(item)
            if reference is not None:
                return reference
    return None


def _streaming_reference_cache(
    boundaries: BoundaryActivationStore,
    boundary_index: int,
    batches: Sequence[int],
) -> IntermediatesCache | None:
    """Expose bounded next-boundary outputs using the oneshot cache protocol."""

    intermediates = []
    for batch_index in batches:
        value = boundaries.get(boundary_index, batch_index, device="cpu")
        reference = _reference_tensor(value)
        if reference is None:
            return None
        intermediates.append(
            {"hidden_states": IntermediateValue(value=reference, device=None)}
        )
    return IntermediatesCache(intermediates, offload_device=None)


def _can_reuse_calibration_outputs(recipe: Recipe) -> bool:
    """Require an explicit promise from every modifier contract."""
    return calibration_outputs_reusable(recipe.modifiers)


def _compress_modules(modules: Sequence[torch.nn.Module]) -> None:
    for module in modules:
        if is_module_quantized(module):
            set_quantization_scale_dtype(module)
            compress_module(module)


def _write_remaining_direct_shards(
    *,
    writer: DirectSafetensorsWriter,
    source,
    materializer: WeightMaterializer,
    dtype_policy: StreamingDTypePolicy | None,
    target_dtype: torch.dtype,
    written: set[str],
    omitted: Mapping[str, str],
    host_budget: HostMemoryBudget | None = None,
    start_shard_index: int = 0,
) -> None:
    """Copy non-subgraph tensors into bounded final static shards."""

    omitted_metadata = {
        materializer.output_tensor_name(alias): materializer.output_tensor_name(
            canonical
        )
        for alias, canonical in omitted.items()
    }
    tensors: dict[str, torch.Tensor] = {}
    tensor_bytes = 0
    shard_index = start_shard_index
    max_shard_bytes = 256 * 1024 * 1024

    def flush() -> None:
        nonlocal omitted_metadata, shard_index, tensor_bytes
        if not tensors:
            return
        shard_id = f"static-{shard_index:05d}"
        reservation = None
        estimate = getattr(writer, "estimate_snapshot_bytes", None)
        if host_budget is not None and callable(estimate):
            reservation = host_budget.reserve(
                f"checkpoint shard {shard_id}", estimate(tensors)
            )
        writer_kwargs = {"omitted_tied_weights": omitted_metadata}
        if reservation is not None:
            writer_kwargs["reservation"] = reservation
        try:
            writer.write_shard(shard_id, tensors, **writer_kwargs)
        except Exception:
            if reservation is not None:
                reservation.close()
            raise
        tensors.clear()
        omitted_metadata = {}
        shard_index += 1
        tensor_bytes = 0

    for name in source.tensor_names():
        if name in written or name in omitted:
            continue
        metadata = source.metadata(name)
        if metadata.dtype.is_floating_point or materializer.dependencies(
            name, metadata
        ):
            value = materialize_weights(
                source,
                (name,),
                materializer,
                target_dtype=target_dtype,
                dtype_policy=dtype_policy,
            )[name]
        else:
            value = source.load_tensors_cpu((name,))[name]
        output_tensors = materializer.prepare_output_tensors(name, value)
        if (
            tensors
            and tensor_bytes
            + sum(
                output_tensor.numel() * output_tensor.element_size()
                for output_tensor in output_tensors.values()
            )
            > max_shard_bytes
        ):
            flush()
        for output_name, output_tensor in output_tensors.items():
            if output_name in tensors:
                raise ValueError(f"Duplicate serialized tensor name {output_name!r}")
            tensors[output_name] = output_tensor
        tensor_bytes += sum(
            output_tensor.numel() * output_tensor.element_size()
            for output_tensor in output_tensors.values()
        )
        written.add(name)
    flush()


def _complete_direct_writer_staging(
    *,
    adapter: TracedBoundaryAdapter,
    checkpoint: str | Path,
    publish_dir: Path,
    materializer: WeightMaterializer,
    dtype_policy: StreamingDTypePolicy | None,
    target_dtype: torch.dtype,
    run_fingerprint: str,
) -> None:
    """Validate completed targets and append only missing static tensors."""

    writer = DirectSafetensorsWriter(publish_dir, run_fingerprint=run_fingerprint)
    states = writer.committed_metadata()
    required_target_shards = {
        f"model-subgraph-{index:05d}.safetensors"
        for index in range(len(adapter.target_groups))
    }
    available_target_shards = {
        state["output_shard"]
        for state in states
        if state["output_shard"].startswith("model-subgraph-")
    }
    if available_target_shards != required_target_shards:
        raise RuntimeError(
            "Direct-writer staging does not contain every completed target: "
            f"expected={len(required_target_shards)}, "
            f"actual={len(available_target_shards)}"
        )

    written = {name for state in states for name in state.get("tensor_names", ())}
    static_indices = []
    for state in states:
        shard_name = state["output_shard"]
        if shard_name.startswith("model-static-"):
            index = shard_name.removeprefix("model-static-").removesuffix(
                ".safetensors"
            )
            try:
                static_indices.append(int(index))
            except ValueError as error:
                raise RuntimeError(
                    f"Invalid static shard name: {shard_name}"
                ) from error

    source = adapter.weight_session.source
    source_names = set(source.tensor_names())
    omitted = {
        alias: canonical
        for alias, canonical in infer_transformers_tied_weights(checkpoint).items()
        if alias in source_names and canonical in source_names
    }
    _write_remaining_direct_shards(
        writer=writer,
        source=source,
        materializer=materializer,
        target_dtype=target_dtype,
        dtype_policy=dtype_policy,
        written=written,
        omitted=omitted,
        host_budget=HostMemoryBudget(),
        start_shard_index=max(static_indices, default=-1) + 1,
    )


def _copy_remaining_tensors(
    *,
    writer: StreamingCheckpointWriter,
    source,
    materializer: WeightMaterializer,
    dtype_policy: StreamingDTypePolicy | None,
    target_dtype: torch.dtype,
    written: set[str],
    omitted: Mapping[str, str],
) -> None:
    for index, name in enumerate(source.tensor_names()):
        if name in written or name in omitted:
            continue
        transaction_id = f"source-tensor-{index:08d}"
        if writer.is_transaction_complete(transaction_id):
            continue
        metadata = source.metadata(name)
        if metadata.dtype.is_floating_point or materializer.dependencies(
            name, metadata
        ):
            value = materialize_weights(
                source,
                (name,),
                materializer,
                target_dtype=target_dtype,
                dtype_policy=dtype_policy,
            )[name]
        else:
            value = source.load_tensors_cpu((name,))[name]
        with writer.transaction(transaction_id) as transaction:
            for output_name, output_tensor in materializer.prepare_output_tensors(
                name, value
            ).items():
                transaction.write_tensor(
                    output_name,
                    output_tensor,
                    output_shard=metadata.shard.name,
                )
            for alias, canonical in omitted.items():
                transaction.omit_tied_weight(
                    materializer.output_tensor_name(alias),
                    materializer.output_tensor_name(canonical),
                )
            transaction.commit()
        del value


def run_subgraph_streaming_pipeline(
    *,
    adapter: TracedBoundaryAdapter,
    checkpoint: str | Path,
    work_dir: str | Path,
    calibration_batches: Iterable[Mapping[str, Any]],
    recipe: Recipe,
    dataset_fingerprint: str,
    materializer: WeightMaterializer,
    device: torch.device,
    target_dtype: torch.dtype,
    num_samples: int,
    max_seq_length: int | None,
    seed: int | None,
    checkpoint_progress: bool = False,
    async_save: bool | None = None,
    pack_to_int8: bool = True,
    dtype_policy: StreamingDTypePolicy | None = None,
    execution: StreamingExecutionDescriptor | None = None,
    targets_per_subgraph: int = 1,
    expert_parallel: ExpertParallelPlan | None = None,
    expert_parallel_token_budget: str | int = "auto",
) -> tuple[Path, Path, str]:
    """Calibrate, modify, propagate, and persist one subgraph at a time."""

    device = torch.device(device)
    validate_token_budget(expert_parallel_token_budget)
    # Quantization observes/modifies weights; no parameter needs autograd.
    adapter.model.requires_grad_(False)
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    ):
        raise ModifierContractError(
            "Streaming admission currently requires a single participant; "
            "multi-rank statistics and communication require an EP-B adapter"
        )
    if async_save is None:
        async_save = expert_parallel is not None and not checkpoint_progress
    if checkpoint_progress and async_save:
        raise ValueError(
            "Asynchronous saving is not supported with checkpoint_progress=True"
        )
    work = Path(work_dir)
    dtype_policy = dtype_policy or adapter.weight_session.dtype_policy
    artifact_dir = work / "artifacts"
    staging_dir = work / "staging"
    publish_dir = work / "publish"
    if not isinstance(calibration_batches, FrozenCalibrationInputs):
        calibration_batches = freeze_calibration_inputs(
            calibration_batches, work / "inputs"
        )
    execution = (
        execution
        or adapter.execution
        or describe_streaming_execution(adapter.model, adapter.targets)
    )
    execution = replace(execution, input_fingerprint=calibration_batches.fingerprint)
    adapter.execution = execution
    execution.validate(adapter.model, adapter.targets)
    target_groups = adapter.target_groups
    if not target_groups:
        raise ValueError("Streaming execution plan contains no target groups")
    if any(len(group) > targets_per_subgraph for group in target_groups):
        raise ValueError(
            "Streaming trace produced a target group larger than the requested "
            f"targets_per_subgraph={targets_per_subgraph}"
        )
    unsupported_grouped_adapters = [
        type(adapter).__name__
        for adapter in modifier_adapters(recipe.modifiers)
        if not adapter.supports_target_groups
        and any(len(group) > 1 for group in target_groups)
    ]
    if unsupported_grouped_adapters:
        raise ModifierContractError(
            "Streaming target grouping is incompatible with "
            f"{', '.join(unsupported_grouped_adapters)}; use "
            "sequential_targets_per_subgraph=1"
        )
    for group in target_groups:
        adapter.plan_group(group)
    window_adapter = (
        get_modifier_memory_adapter(recipe.modifiers[0])
        if expert_parallel is not None
        and not checkpoint_progress
        and len(recipe.modifiers) == 1
        else None
    )
    if getattr(window_adapter, "supports_independent_expert_windows", False):
        target_groups = coalesce_expert_regions(adapter, expert_parallel)
    run_fingerprint = _initialize_run(
        checkpoint=checkpoint,
        artifact_dir=artifact_dir,
        recipe=recipe,
        dataset_fingerprint=dataset_fingerprint,
        targets=adapter.targets,
        targets_per_subgraph=targets_per_subgraph,
        materializer=materializer,
        dtype_policy=dtype_policy,
        target_dtype=target_dtype,
        num_samples=num_samples,
        max_seq_length=max_seq_length,
        seed=seed,
        pack_to_int8=pack_to_int8,
        replace_existing=not checkpoint_progress,
        execution=execution or adapter.execution,
        expert_parallel=expert_parallel,
        expert_parallel_token_budget=expert_parallel_token_budget,
        target_groups=target_groups,
    )
    transaction_writer = (
        StreamingCheckpointWriter(
            staging_dir,
            run_fingerprint=run_fingerprint,
            coordinator=SingleParticipantCoordinator(
                staging_dir,
                run_fingerprint=run_fingerprint,
            ),
        )
        if checkpoint_progress
        else None
    )
    direct_writer_base = (
        None
        if checkpoint_progress
        else DirectSafetensorsWriter(publish_dir, run_fingerprint=run_fingerprint)
    )
    direct_writer = direct_writer_base
    async_direct_writer = None
    host_budget = HostMemoryBudget()
    disk_budget = DiskSpaceBudget(work)
    boundaries: BoundaryActivationStore
    memory_frontier = checkpoint_progress and expert_parallel is not None
    if checkpoint_progress and not memory_frontier:
        boundaries = DiskBoundaryActivationStore(
            work / "boundaries", host_budget=host_budget
        )
    else:
        boundaries = InMemoryBoundaryActivationStore(
            storage_device="cpu",
            deduplicate_tensors=True,
            host_budget=host_budget,
        )
    streaming_logger.info(
        "setup | boundary storage | "
        + (
            "in-memory live frontier with durable checkpoint transactions"
            if memory_frontier
            else "disk-backed checkpoint frontier"
            if checkpoint_progress
            else "in-memory"
        )
    )
    source = adapter.weight_session.source
    written: set[str] = set()
    resume_plan = (
        build_resume_plan(transaction_writer)
        if transaction_writer is not None
        else None
    )
    resume_metadata = resume_plan.metadata_by_id if resume_plan is not None else {}
    committed_transaction_ids = list(
        resume_plan.transaction_ids if resume_plan is not None else ()
    )
    if resume_plan is not None and resume_plan.prefix_metadata is not None:
        committed_transaction_ids.insert(0, "prefix-00000")
    if transaction_writer is None or not resume_plan.transaction_ids:
        streaming_logger.info("setup | collecting initial calibration boundary")
        with _stage_logger(adapter.model, "[setup] initial boundary", device):
            boundaries.delete(0)
            prefix_metadata = resume_plan.prefix_metadata if resume_plan else None
            initial = (
                transaction_writer.iter_committed_boundary(
                    "prefix-00000", metadata=prefix_metadata
                )
                if prefix_metadata is not None
                else enumerate(adapter.calibration_boundaries(calibration_batches))
            )
            for batch_index, boundary in initial:
                boundaries.put(0, batch_index, boundary)
                if batch_index == 0:
                    _debug_boundary(
                        "initial",
                        boundary=0,
                        batch=batch_index,
                        producer="<dataset>",
                        consumer=adapter.targets[0],
                        value=boundary,
                    )
                del boundary
        streaming_logger.info(
            "setup | initial boundary ready | "
            f"batches={len(boundaries.batch_indices(0))}"
        )
        if transaction_writer is not None and prefix_metadata is None:
            batches = boundaries.batch_indices(0)
            with transaction_writer.transaction("prefix-00000") as transaction:
                for batch_index in batches:
                    value = boundaries.get(0, batch_index)
                    disk_budget.check(
                        "prefix boundary transaction",
                        _boundary_nbytes(value) * 2 + 4096,
                    )
                    transaction.write_boundary(
                        batch_index,
                        value,
                        boundary=0,
                        expected_batch_count=len(batches),
                    )
                    del value
                transaction.commit()
            committed_transaction_ids.insert(0, "prefix-00000")

    streaming_logger.info("setup | initializing modifier session")
    with create_session() as session, ExitStack() as calibration_stack:
        session.state.layerwise_calibration = True
        with _stage_logger(adapter.model, "[setup] initialize modifiers", device):
            session.initialize(
                model=adapter.model,
                recipe=recipe,
                start=-1,
                calib_data=calibration_batches,
                sequential_targets=adapter.targets,
                copy_data=False,
            )
            runtime_adapters = modifier_adapters(session.lifecycle.recipe.modifiers)
            try:
                calibration_stack.enter_context(
                    modifier_calibration_context(session.lifecycle.recipe.modifiers)
                )
                for modifier_adapter in runtime_adapters:
                    modifier_adapter.before_calibration(adapter.model, adapter.targets)
                    configure = getattr(modifier_adapter, "configure_algorithm", None)
                    if callable(configure):
                        session.state.autoround_kwargs_adapter = configure
                LifecycleCallbacks.calibration_start()
                for modifier_adapter in runtime_adapters:
                    modifier_adapter.validate_targets(adapter.model, adapter.targets)
            except BaseException:
                for modifier_adapter in runtime_adapters:
                    modifier_adapter.abort()
                session.finalize()
                raise
        streaming_logger.info("setup | modifier session ready")
        quantizers = [
            modifier
            for modifier in session.lifecycle.recipe.modifiers
            if isinstance(modifier, QuantizationMixin)
        ]
        modifiers = tuple(session.lifecycle.recipe.modifiers)
        if window_adapter is not None:
            window_adapter = get_modifier_memory_adapter(modifiers[0])
        device_budget = DeviceMemoryBudget()
        reuse_calibration_outputs = _can_reuse_calibration_outputs(recipe)
        streaming_logger.info(
            "setup | activation propagation | "
            + (
                "reuse calibration outputs"
                if reuse_calibration_outputs
                else "recompute after modifier update"
            )
        )
        if async_save:
            if direct_writer_base is None:
                raise RuntimeError("Asynchronous saving requires a direct writer")
            async_direct_writer = AsyncDirectSafetensorsWriter(
                direct_writer_base,
                max_pending=len(expert_parallel.devices) if expert_parallel else 1,
            )
            direct_writer = async_direct_writer
        ep_prefetch = expert_parallel is not None and not checkpoint_progress
        prefetcher = (
            SubgraphPrefetcher(adapter.weight_session)
            if expert_parallel is None or ep_prefetch
            else None
        )
        streaming_logger.info(
            "setup | checkpoint write | "
            + ("asynchronous" if async_save else "synchronous")
        )
        streaming_logger.info(
            "setup | weight prefetch | "
            + (
                "prepare only (expert parallel)"
                if ep_prefetch
                else "disabled for expert parallel progress checkpoints"
                if expert_parallel is not None
                else "prepare + pinned async H2D"
                if _async_h2d_enabled() and device.type == "cuda"
                else "prepare only"
            )
        )
        pipeline_failed = True
        ep_executor = None
        try:
            available_operators = available_hardware_ops()
            operator_names = {operator["name"] for operator in available_operators}
            operator_summary = (
                ", ".join(
                    f"{operator['name']}:"
                    f"{operator['backend']}/{operator['implementation']}"
                    for operator in available_operators
                )
                or "none"
            )
            streaming_logger.info(
                "setup | hardware operators | "
                f"available={operator_summary} | "
                "gptq_factorize_hessian="
                + (
                    "available"
                    if "gptq_factorize_hessian" in operator_names
                    else "missing"
                )
            )
            if expert_parallel is not None:
                ep_executor = ExpertParallelExecutor(adapter.model, expert_parallel)
                streaming_logger.info(
                    f"setup | expert parallel | backend={expert_parallel.backend} | "
                    f"devices={tuple(map(str, expert_parallel.devices))} | "
                    f"experts={len(expert_parallel.placement.entries)} | "
                    f"fingerprint={expert_parallel.fingerprint}"
                )
            prefetch_pending = False
            target_count = len(target_groups)
            group_input_names = tuple(
                _target_group_input_names(
                    tuple(adapter.target_subgraphs[index] for index in target_indices)
                )
                for target_indices in target_groups
            )
            last_input_use = {
                name: group_index
                for group_index, input_names in enumerate(group_input_names)
                for name in input_names
            }

            @lru_cache(maxsize=2)
            def region_for_group(index):
                if not getattr(
                    window_adapter, "supports_independent_expert_windows", False
                ):
                    return None
                try:
                    return ExpertRegionGraph.build(
                        adapter.model,
                        expert_parallel,
                        tuple(
                            adapter.target_subgraphs[target]
                            for target in target_groups[index]
                        ),
                    )
                except UnsupportedExpertRegion:
                    return None

            def submit_ep_prefetch(index, plan, headroom_bytes):
                nonlocal prefetch_pending
                indices = target_groups[index]
                names = tuple(adapter.targets[target] for target in indices)
                label = _target_label(index, target_count, " + ".join(names))
                required = adapter.weight_session.estimate_preparation(
                    plan, dtype=target_dtype
                )
                try:
                    with host_budget.reserve(
                        f"{label} current group headroom", headroom_bytes
                    ):
                        reservation = host_budget.reserve(f"{label} weights", required)
                except HostMemoryError:
                    streaming_logger.info(
                        f"{label} | EP prefetch deferred | "
                        f"headroom_bytes={headroom_bytes} | prefetch_bytes={required}"
                    )
                    return
                prefetcher.submit(
                    plan,
                    dtype=target_dtype,
                    reservation=reservation,
                    label=f"{label} | prepare weights",
                )
                prefetch_pending = True
                streaming_logger.info(f"{label} | EP prefetch submitted | prepare")

            streaming_logger.info(
                f"setup | progress counts target groups | groups={target_count} | "
                "calibration batches and routed tokens are reported separately"
            )
            for group_index, target_indices in enumerate(target_groups):
                group_start = target_indices[0]
                group_end = target_indices[-1]
                next_boundary = group_end + 1
                has_next = group_index + 1 < target_count
                target_names = tuple(adapter.targets[index] for index in target_indices)
                target_name = " + ".join(target_names)
                subgraphs = tuple(
                    adapter.target_subgraphs[index] for index in target_indices
                )
                group_inputs = group_input_names[group_index]
                group_consumed = set().union(
                    *(subgraph.consumed_names for subgraph in subgraphs)
                )
                transaction_id = f"subgraph-{group_index:05d}"
                if (
                    transaction_writer is not None
                    and resume_plan is not None
                    and transaction_id in resume_metadata
                ):
                    label = _target_label(group_index, target_count, target_name)
                    streaming_logger.info(f"{label} | restoring checkpoint")
                    metadata = resume_metadata[transaction_id]
                    written.update(record["name"] for record in metadata["records"])
                    written.update(
                        f"{name}.weight"
                        for subgraph in subgraphs
                        for name in subgraph.submodule_dict(adapter.model)
                        if materializer.output_module_name(name)
                        in metadata.get("quantized_modules", {})
                    )
                    boundary = metadata.get("boundary")
                    is_frontier = transaction_id == resume_plan.transaction_ids[-1]
                    if boundary is not None and is_frontier:
                        boundary_index = int(boundary["index"])
                        boundaries.delete(boundary_index)
                        for (
                            batch_index,
                            value,
                        ) in transaction_writer.iter_committed_boundary(
                            transaction_id,
                            metadata=metadata,
                        ):
                            boundaries.put(
                                boundary_index,
                                batch_index,
                                value,
                            )
                    boundaries.delete(group_start)
                    for name in target_names:
                        for modifier_adapter in runtime_adapters:
                            modifier_adapter.restore_target(
                                adapter.model.get_submodule(name)
                            )
                    continue
                if (
                    transaction_writer is not None
                    and transaction_writer.has_transaction(transaction_id)
                ):
                    raise RuntimeError(
                        f"Committed transaction {transaction_id!r} is corrupt; "
                        "refusing to replace durable output"
                    )
                batches = boundaries.batch_indices(group_start)
                if not batches:
                    raise RuntimeError(
                        f"Missing boundary {group_start} for {target_name!r}"
                    )
                if any(
                    len(batches) < modifier_adapter.minimum_batches
                    for modifier_adapter in runtime_adapters
                ):
                    raise ModifierContractError(
                        "Streaming QAD validation requires at least two batches"
                    )
                target_label = _target_label(group_index, target_count, target_name)
                for owner_device in (
                    expert_parallel.devices if expert_parallel else (device,)
                ):
                    _reset_peak_memory(owner_device)
                streaming_logger.info(
                    f"{target_label} | start | batches={len(batches)} | device={device}"
                )
                target_started_at = perf_counter()
                region_graph = region_for_group(group_index)
                next_ep_plan = None
                if ep_prefetch and has_next:
                    next_region = region_for_group(group_index + 1)
                    next_ep_plan = (
                        adapter.weight_session.plan_modules(
                            next_region.expert_names, allow_missing_state=True
                        )
                        if next_region is not None
                        else adapter.plan_group(target_groups[group_index + 1])
                    )
                if region_graph is not None:
                    region_output = next(
                        node for node in region_graph.graph.nodes if node.op == "output"
                    )
                    destination_input_names = {
                        name
                        for name in region_output.args[0]
                        if last_input_use.get(name, -1) > group_index
                    }

                    written_lock = Lock()

                    def save_expert_window(window, loaded, modules):
                        _compress_modules(modules)
                        saved = _write_loaded_target_direct(
                            direct_writer,
                            loaded,
                            window.expert_names,
                            f"{transaction_id}-r{window.owner_rank}-w{window.window_index}",
                            materializer,
                            pack_to_int8=pack_to_int8,
                            host_budget=host_budget,
                        )
                        with written_lock:
                            written.update(saved)

                    def prefetch_during_combine(expert_output_bytes):
                        boundary_bytes = sum(
                            _boundary_nbytes(
                                boundaries.get(group_start, batch, device="meta")
                            )
                            for batch in batches
                        )
                        submit_ep_prefetch(
                            group_index + 1,
                            next_ep_plan,
                            boundary_bytes + expert_output_bytes,
                        )

                    with (
                        ExitStack() as region_stack,
                        ExpertRouteCache(host_budget, target_label) as route_cache,
                    ):
                        with _stage_logger(
                            adapter.model, f"{target_label} | profile routes", device
                        ):
                            route_profile = profile_expert_region(
                                region_graph,
                                boundaries,
                                group_start,
                                batches,
                                host_budget=host_budget,
                                route_cache=route_cache,
                            )
                        streaming_logger.info(
                            f"{target_label} | EP route cache | "
                            f"route_nodes={len(region_graph.route_nodes)} | "
                            f"batches={len(batches)} | {route_cache.report()}"
                        )
                        prepared_weights = None
                        if prefetch_pending:
                            prepared_weights = prefetcher.take()
                            prefetch_pending = False
                            region_stack.callback(prepared_weights.close)
                        window_plan = execute_expert_region(
                            graph=region_graph,
                            profile=route_profile,
                            adapter=adapter,
                            parallel=expert_parallel,
                            modifier_adapter=window_adapter,
                            modifiers=modifiers,
                            boundaries=boundaries,
                            boundary_index=group_start,
                            destination_boundary=next_boundary,
                            batches=batches,
                            device_budget=device_budget,
                            host_budget=host_budget,
                            target_dtype=target_dtype,
                            token_budget=expert_parallel_token_budget,
                            label=target_label,
                            save_window=save_expert_window,
                            async_save=async_save,
                            propagate=has_next,
                            route_cache=route_cache,
                            destination_input_names=destination_input_names,
                            allow_device_prefetch=_async_h2d_enabled(),
                            prepared_weights=prepared_weights,
                            on_combine_start=(
                                prefetch_during_combine
                                if next_ep_plan is not None
                                else None
                            ),
                        )
                    ArtifactStore(artifact_dir).commit_ep_token_plan(
                        group_index,
                        {"policy": "routed-resident-windows-v1", **window_plan},
                    )
                    boundaries.delete(group_start)
                    streaming_logger.info(
                        f"{target_label} | complete | "
                        f"time={perf_counter() - target_started_at:.2f}s | "
                        "execution=routed-resident-windows"
                    )
                    continue
                baseline_allocated = (
                    torch.get_device_module(device).memory_allocated(device)
                    if device.type == "cuda"
                    else None
                )
                # A previous attempt can fail after publishing only part of the
                # next external boundary but before committing its transaction.
                # Rebuild it from scratch so stale batches are never consumed.
                if has_next:
                    boundaries.delete(next_boundary)
                weight_stack = ExitStack()
                completion_reservation = None
                prefetched_prepared = None
                try:
                    if prefetch_pending:
                        prefetched_prepared = prefetcher.take()
                        prefetch_pending = False
                        weight_stack.callback(prefetched_prepared.close)
                    resident_modules = tuple(
                        dict.fromkeys(
                            module
                            for subgraph in subgraphs
                            for module in subgraph.submodules(adapter.model)
                        )
                    )
                    if runtime_adapters:
                        weight_stack.enter_context(
                            target_rng_scope(seed, group_index, device)
                        )
                    for modifier_adapter in runtime_adapters:
                        modifier_adapter.before_target(resident_modules)
                    num_tokens = 0
                    max_batch_tokens = 0
                    batch_token_counts = []
                    boundary_bytes = 0
                    max_boundary_batch_bytes = 0
                    max_boundary_bytes_per_token = 0
                    for batch_index in batches:
                        value = boundaries.get(group_start, batch_index, device="meta")
                        execution_metadata = boundaries.execution_metadata(
                            group_start, batch_index, group_inputs
                        )
                        batch_tokens = _boundary_token_count(
                            {
                                name: execution_metadata.values[name]
                                for name in group_inputs
                            }
                        )
                        num_tokens += batch_tokens
                        batch_token_counts.append(batch_tokens)
                        max_batch_tokens = max(max_batch_tokens, batch_tokens)
                        boundary_bytes += _boundary_nbytes(value)
                        current_boundary_bytes = execution_metadata.tensor_bytes
                        max_boundary_batch_bytes = max(
                            max_boundary_batch_bytes, current_boundary_bytes
                        )
                        if batch_tokens:
                            max_boundary_bytes_per_token = max(
                                max_boundary_bytes_per_token,
                                (current_boundary_bytes + batch_tokens - 1)
                                // batch_tokens,
                            )
                        del value, execution_metadata
                    num_tokens = max(num_tokens, 1)
                    max_batch_tokens = max(max_batch_tokens, 1)
                    execution_tokens = (
                        max_batch_tokens if expert_parallel is not None else None
                    )
                    execution_bytes = (
                        max_boundary_batch_bytes
                        if expert_parallel is not None
                        else None
                    )
                    with _stage_logger(
                        adapter.model, f"{target_label} | plan weights", device
                    ):
                        plan = (
                            prefetched_prepared.plan
                            if prefetched_prepared is not None
                            else adapter.plan_group(target_indices)
                        )
                        modifier_memory = estimate_modifier_memory(
                            modifiers,
                            resident_modules,
                            num_tokens=num_tokens,
                            max_batch_tokens=max_batch_tokens,
                        )
                        materialized_bytes = (
                            adapter.weight_session.estimate_materialized_bytes(
                                plan, dtype=target_dtype
                            )
                        )
                        device_required = (
                            materialized_bytes + modifier_memory.peak_bytes
                        )
                        boundary_disk_multiplier = (
                            2
                            if checkpoint_progress and memory_frontier
                            else 4
                            if checkpoint_progress
                            else 0
                        )
                        disk_required = (
                            (4 if async_save else 2) * materialized_bytes
                            + boundary_disk_multiplier * boundary_bytes
                            + 1024 * 1024
                        )
                        disk_budget.check(target_label, disk_required)
                        if expert_parallel is not None:
                            owner_plans = expert_parallel.split_load_plan(plan)
                            owner_capacities = {}
                            phases = ["calibration", "update"]
                            if not reuse_calibration_outputs and has_next:
                                phases.append("propagation")
                            token_store = ArtifactStore(artifact_dir)
                            stored_plan = (
                                token_store.load_ep_token_plan(group_index)
                                if checkpoint_progress
                                else None
                            )
                            token_identity = {
                                "run": run_fingerprint,
                                "policy": TOKEN_BUDGET_POLICY,
                                "phases": phases,
                                "input_tokens": batch_token_counts,
                            }
                            if stored_plan is not None and (
                                stored_plan["identity"] != token_identity
                            ):
                                raise ValueError(
                                    f"Incompatible EP token plan for {target_label}"
                                )
                            boundary_bytes_per_token = max(
                                1,
                                max_boundary_bytes_per_token,
                            )
                            for owner_device, owner_plan in owner_plans:
                                owner_weights = (
                                    adapter.weight_session.estimate_materialized_bytes(
                                        owner_plan, dtype=target_dtype
                                    )
                                )
                                runtime_bytes_per_token = (
                                    expert_parallel.runtime_bytes_per_token(
                                        adapter.model,
                                        owner_device,
                                        modules=resident_modules,
                                        boundary_bytes_per_token=(
                                            boundary_bytes_per_token
                                            if owner_device == device
                                            else 0
                                        ),
                                    )
                                )
                                if owner_device == expert_parallel.devices[0]:
                                    transfer_fn = (
                                        expert_parallel.primary_transfer_bytes_per_token
                                    )
                                    owner_transfer = transfer_fn(
                                        adapter.model,
                                        resident_modules,
                                    )
                                    runtime_bytes_per_token += owner_transfer

                                def memory_at(tokens):
                                    return max(
                                        expert_parallel.estimate_memory(
                                            modifiers,
                                            resident_modules,
                                            device=owner_device,
                                            num_tokens=num_tokens,
                                            max_batch_tokens=tokens,
                                            phase=phase,
                                        ).peak_bytes
                                        for phase in phases
                                    )

                                def required_at(tokens):
                                    return (
                                        owner_weights
                                        + memory_at(tokens)
                                        + runtime_bytes_per_token * tokens
                                    )

                                usable = device_budget.usable_bytes(owner_device)
                                if stored_plan is not None:
                                    capacity = stored_plan["capacities"][
                                        str(owner_device)
                                    ]
                                    if (
                                        usable is not None
                                        and required_at(capacity) > usable
                                    ):
                                        raise DeviceMemoryError(
                                            f"Recorded EP token capacity for "
                                            f"{owner_device} no longer fits; "
                                            "start a new run to replan"
                                        )
                                else:
                                    maximum = num_tokens
                                    if expert_parallel_token_budget != "auto":
                                        maximum = min(
                                            maximum, expert_parallel_token_budget
                                        )
                                    capacity = select_token_capacity(
                                        owner=(
                                            f"{target_label} expert owner "
                                            f"{owner_device}"
                                        ),
                                        usable_bytes=usable,
                                        required_bytes=required_at,
                                        maximum_tokens=maximum,
                                        minimum_tokens=max_batch_tokens,
                                    )
                                owner_capacities[owner_device] = capacity
                                owner_required = required_at(capacity)
                                weight_stack.enter_context(
                                    device_budget.reserve(
                                        f"{target_label} expert owner {owner_device}",
                                        owner_required,
                                        device=owner_device,
                                    )
                                )
                                usable_text = (
                                    f"{usable / 2**30:.2f}"
                                    if usable is not None
                                    else "unknown"
                                )
                                streaming_logger.info(
                                    f"{target_label} | EP admission | "
                                    "rank="
                                    f"{expert_parallel.devices.index(owner_device)} | "
                                    f"device={owner_device} | "
                                    f"weights_gib={owner_weights / 2**30:.2f} | "
                                    "modifier_peak_gib="
                                    f"{memory_at(capacity) / 2**30:.2f} | "
                                    "runtime_bytes/token="
                                    f"{runtime_bytes_per_token:,} | "
                                    f"token_capacity={capacity:,} | "
                                    f"required_gib={owner_required / 2**30:.2f} | "
                                    f"usable_gib={usable_text}"
                                )
                            token_store.commit_ep_token_plan(
                                group_index,
                                {
                                    "identity": token_identity,
                                    "capacities": {
                                        str(owner): capacity
                                        for owner, capacity in owner_capacities.items()
                                    },
                                },
                            )
                            ep_executor.set_token_capacity(owner_capacities)
                            streaming_logger.info(
                                f"{target_label} | EP FX input window | "
                                f"token_capacity={max_batch_tokens:,} | "
                                f"byte_capacity={max_boundary_batch_bytes:,}"
                            )
                        elif prefetched_prepared is None:
                            weight_stack.enter_context(
                                device_budget.reserve(
                                    f"{target_label} device working set",
                                    device_required,
                                    device=device,
                                )
                            )
                        elif prefetched_prepared.device_reservation is None:
                            weight_stack.enter_context(
                                device_budget.reserve(
                                    f"{target_label} device working set",
                                    device_required,
                                    device=device,
                                )
                            )
                        elif (
                            prefetched_prepared.device_reservation.reserved_bytes
                            < device_required
                        ):
                            additional_bytes = (
                                device_required
                                - prefetched_prepared.device_reservation.reserved_bytes
                            )
                            weight_stack.enter_context(
                                device_budget.reserve(
                                    f"{target_label} additional device working set",
                                    additional_bytes,
                                    device=device,
                                )
                            )
                        modifier_host_bytes = modifier_memory.host_bytes + (
                            modifier_memory.peak_bytes if device.type == "cpu" else 0
                        )
                        if modifier_host_bytes:
                            weight_stack.enter_context(
                                host_budget.reserve(
                                    f"{target_label} modifier working set",
                                    modifier_host_bytes,
                                )
                            )
                        streaming_logger.info(
                            f"{target_label} | admission | "
                            f"materialized_bytes={materialized_bytes} | "
                            f"modifier_held_bytes={modifier_memory.held_bytes} | "
                            f"modifier_scratch_bytes={modifier_memory.scratch_bytes} | "
                            f"modifier_host_bytes={modifier_host_bytes} | "
                            f"device_required_bytes={device_required} | "
                            f"disk_required_bytes={disk_required} | "
                            f"num_tokens={num_tokens}"
                            f" | max_batch_tokens={max_batch_tokens}"
                        )
                    if prefetched_prepared is None:
                        required = adapter.weight_session.estimate_preparation(
                            plan, dtype=target_dtype
                        )
                        reservation = host_budget.reserve(
                            f"{target_label} weights", required
                        )
                        with _stage_logger(
                            adapter.model,
                            f"{target_label} | prepare weights",
                            device,
                        ):
                            prepared = adapter.weight_session.prepare(
                                plan,
                                dtype=target_dtype,
                                reservation=reservation,
                            )
                    else:
                        prepared = prefetched_prepared
                    weight_stack.callback(prepared.close)
                    with _stage_logger(
                        adapter.model,
                        f"{target_label} | load weights",
                        device,
                    ):
                        if expert_parallel is None:
                            prepared.move_to(device)
                        else:
                            prepared.move_to(
                                device,
                                target_devices={
                                    target.name: expert_parallel.device_for(target.name)
                                    for target in plan.targets
                                },
                            )
                        loaded = weight_stack.enter_context(
                            adapter.weight_session.installed(prepared)
                        )

                    if next_ep_plan is not None:
                        submit_ep_prefetch(
                            group_index + 1,
                            next_ep_plan,
                            boundary_bytes
                            + max_boundary_batch_bytes
                            + _estimate_loaded_target_snapshot_bytes(
                                loaded, target_names, pack_to_int8=pack_to_int8
                            ),
                        )
                    elif prefetcher is not None and has_next:
                        next_indices = target_groups[group_index + 1]
                        next_names = tuple(
                            adapter.targets[index] for index in next_indices
                        )
                        next_name = " + ".join(next_names)
                        next_label = _target_label(
                            group_index + 1, target_count, next_name
                        )
                        with _stage_logger(
                            adapter.model,
                            f"{next_label} | plan weights",
                            device,
                        ):
                            next_plan = adapter.plan_group(next_indices)
                            next_required = adapter.weight_session.estimate_preparation(
                                next_plan, dtype=target_dtype
                            )
                            next_modules = tuple(
                                dict.fromkeys(
                                    module
                                    for index in next_indices
                                    for module in adapter.target_subgraphs[
                                        index
                                    ].submodules(adapter.model)
                                )
                            )
                            next_modifier_memory = estimate_modifier_memory(
                                modifiers,
                                next_modules,
                                num_tokens=num_tokens,
                                max_batch_tokens=max_batch_tokens,
                            )
                            completion_bytes = _estimate_loaded_target_snapshot_bytes(
                                loaded, target_names, pack_to_int8=pack_to_int8
                            )
                            next_reservation = None
                            next_device_reservation = None
                            next_transfer_reservation = None
                            try:
                                completion_reservation = host_budget.reserve(
                                    f"{target_label} completion headroom",
                                    completion_bytes,
                                )
                                next_reservation = host_budget.reserve(
                                    f"{next_label} weights", next_required
                                )
                                next_materialized_bytes = (
                                    adapter.weight_session.estimate_materialized_bytes(
                                        next_plan, dtype=target_dtype
                                    )
                                )
                                try:
                                    next_device_reservation = device_budget.reserve(
                                        f"{next_label} prefetch working set",
                                        next_materialized_bytes
                                        + next_modifier_memory.peak_bytes,
                                        device=device,
                                    )
                                except DeviceMemoryError:
                                    streaming_logger.info(
                                        f"{next_label} | async H2D deferred | "
                                        "device working set does not fit"
                                    )
                                if (
                                    next_device_reservation is not None
                                    and device.type == "cuda"
                                    and _async_h2d_enabled()
                                ):
                                    try:
                                        next_transfer_reservation = host_budget.reserve(
                                            f"{next_label} pinned transfer",
                                            next_materialized_bytes,
                                        )
                                    except HostMemoryError:
                                        next_device_reservation.close()
                                        next_device_reservation = None
                                        streaming_logger.info(
                                            f"{next_label} | async H2D deferred | "
                                            "pinned host staging does not fit"
                                        )
                            except HostMemoryError:
                                if next_device_reservation is not None:
                                    next_device_reservation.close()
                                if next_transfer_reservation is not None:
                                    next_transfer_reservation.close()
                                if completion_reservation is not None:
                                    completion_reservation.close()
                                    completion_reservation = None
                                streaming_logger.info(
                                    f"{next_label} | prefetch deferred | "
                                    f"completion_bytes={completion_bytes} | "
                                    f"prefetch_bytes={next_required}"
                                )
                        if next_reservation is not None:
                            weight_stack.callback(completion_reservation.close)
                            prefetcher.submit(
                                next_plan,
                                dtype=target_dtype,
                                reservation=next_reservation,
                                label=f"{next_label} | prepare weights",
                                device=device,
                                device_reservation=next_device_reservation,
                                transfer_reservation=next_transfer_reservation,
                            )
                            streaming_logger.info(
                                f"{next_label} | prefetch submitted | "
                                + (
                                    "prepare + pinned async H2D"
                                    if next_transfer_reservation is not None
                                    else "prepare"
                                )
                            )
                            prefetch_pending = True

                    with _stage_logger(
                        adapter.model,
                        f"{target_label} | initialize quantization",
                        device,
                    ):
                        for quantizer in quantizers:
                            weight_stack.callback(
                                quantizer.end_layerwise_calibration,
                                list(resident_modules),
                            )
                            quantizer_adapter = get_modifier_adapter(quantizer)
                            start = getattr(quantizer_adapter, "start_quantizer", None)
                            if callable(start):
                                start(adapter.model, resident_modules)
                            else:
                                quantizer.start_layerwise_calibration(
                                    adapter.model, list(resident_modules)
                                )
                    # Match the ordinary sequential calibration pipeline: model
                    # execution is inference-only. Besides avoiding autograd
                    # storage, this is required by models such as DeepSeek-V4
                    # whose attention updates runtime KV buffers in place.
                    with (
                        _stage_logger(
                            adapter.model,
                            f"{target_label} | calibration",
                            device,
                        ),
                        _ep_phase_logger(
                            ep_executor,
                            target_label,
                            "calibration",
                            total_batches=len(batches),
                        ) as ep_metrics,
                    ):
                        with DisableQuantization(adapter.model), torch.no_grad():
                            for batch_indices, values in _iter_execution_batches(
                                boundaries,
                                group_start,
                                batches,
                                group_inputs,
                                device=device,
                                token_capacity=execution_tokens,
                                input_bytes_capacity=execution_bytes,
                                telemetry=ep_metrics,
                            ):
                                for batch_index in batch_indices:
                                    session.state.current_batch_idx = batch_index
                                next_values = _forward_target_group_many(
                                    adapter,
                                    target_indices,
                                    values,
                                    ep_executor,
                                )
                                publish_started_at = perf_counter()
                                if ep_metrics is not None:
                                    ep_metrics.activity(
                                        "publishing calibration outputs"
                                    )
                                for batch_index, value, next_value in zip(
                                    batch_indices, values, next_values
                                ):
                                    if batch_index == batches[0]:
                                        _debug_boundary(
                                            "calibration input",
                                            boundary=group_start,
                                            batch=batch_index,
                                            producer=(
                                                adapter.targets[group_start - 1]
                                                if group_start
                                                else "<dataset>"
                                            ),
                                            consumer=target_name,
                                            value={
                                                name: value[name]
                                                for name in subgraphs[0].input_names
                                            },
                                        )
                                    if reuse_calibration_outputs and has_next:
                                        # These modifiers only collect statistics
                                        # and compute qparams. Snapshot the output
                                        # now, before the next batch can mutate it,
                                        # and consume each input batch as usual.
                                        boundaries.commit_propagated_batch(
                                            group_start,
                                            next_boundary,
                                            batch_index,
                                            next_value,
                                            consumed_names=group_consumed,
                                        )
                                    del next_value, value
                                del next_values, values
                                if ep_metrics is not None:
                                    ep_metrics.host_time(
                                        "boundary", perf_counter() - publish_started_at
                                    )
                                    ep_metrics.advance(len(batch_indices))
                    owner_update_groups = None
                    if expert_parallel is not None:
                        ep_executor.synchronize()
                        owner_update_groups = partition_modules_by_owner(
                            resident_modules,
                            expert_parallel.devices,
                            expert_parallel.module_devices,
                        )
                    owner_local_update = (
                        owner_update_groups is not None
                        and _owner_update_adapter(quantizers, runtime_adapters)
                        is not None
                    )
                    update_modules = (
                        {
                            group.owner: len(group.modules)
                            for group in owner_update_groups
                        }
                        if owner_local_update
                        else {device: len(resident_modules)}
                    )
                    with (
                        _stage_logger(
                            adapter.model,
                            f"{target_label} | modifier update",
                            device,
                        ),
                        _ep_phase_logger(
                            ep_executor,
                            target_label,
                            "modifier update",
                            update_modules=update_modules,
                        ) as ep_metrics,
                    ):
                        if ep_metrics is not None:
                            ep_metrics.activity(
                                "owner-local"
                                if owner_local_update
                                else "serial-contract"
                            )
                        reference_cache = None
                        previous_activations = session.state.sequential_activations
                        previous_propagate_error = (
                            session.state.sequential_propagate_error
                        )
                        if reuse_calibration_outputs and has_next:
                            reference_cache = _streaming_reference_cache(
                                boundaries, next_boundary, batches
                            )
                            if reference_cache is not None:
                                session.state.sequential_activations = reference_cache
                                session.state.sequential_propagate_error = False
                        try:
                            update_parallelism = _modifier_update(
                                resident_modules,
                                quantizers,
                                runtime_adapters,
                                owner_groups=owner_update_groups,
                                telemetry=ep_metrics,
                            )
                            streaming_logger.info(
                                f"{target_label} | modifier update | "
                                f"parallelism={update_parallelism}"
                            )
                        finally:
                            session.state.sequential_activations = previous_activations
                            session.state.sequential_propagate_error = (
                                previous_propagate_error
                            )
                            del reference_cache
                    # Algorithms that update weights need a fresh forward for
                    # the next target. Statistics-only recipes have already
                    # committed that boundary during calibration. The final
                    # target has no successor and never needs propagation.
                    if not reuse_calibration_outputs and has_next:
                        with (
                            _stage_logger(
                                adapter.model,
                                f"{target_label} | activation propagation",
                                device,
                            ),
                            _ep_phase_logger(
                                ep_executor,
                                target_label,
                                "activation propagation",
                                total_batches=len(batches),
                            ) as ep_metrics,
                        ):
                            # Match SequentialPipeline: propagation captures the
                            # modified full-precision output (or an algorithm's
                            # in-place weight update), without adding fake-quant
                            # QDQ to every quantized module.
                            with (
                                HooksMixin.disable_hooks(),
                                (
                                    nullcontext()
                                    if any(
                                        modifier_adapter.propagation_mode
                                        == "post-update-quantized"
                                        for modifier_adapter in runtime_adapters
                                    )
                                    else DisableQuantization(adapter.model)
                                ),
                                torch.no_grad(),
                            ):
                                for batch_indices, values in _iter_execution_batches(
                                    boundaries,
                                    group_start,
                                    batches,
                                    group_inputs,
                                    device=device,
                                    token_capacity=execution_tokens,
                                    input_bytes_capacity=execution_bytes,
                                    telemetry=ep_metrics,
                                ):
                                    next_values = _forward_target_group_many(
                                        adapter,
                                        target_indices,
                                        values,
                                        ep_executor,
                                    )
                                    publish_started_at = perf_counter()
                                    if ep_metrics is not None:
                                        ep_metrics.activity("publishing next boundary")
                                    for batch_index, value, next_value in zip(
                                        batch_indices, values, next_values
                                    ):
                                        boundaries.commit_propagated_batch(
                                            group_start,
                                            next_boundary,
                                            batch_index,
                                            next_value,
                                            consumed_names=group_consumed,
                                        )
                                        if batch_index == batches[0]:
                                            _debug_boundary(
                                                "propagated",
                                                boundary=next_boundary,
                                                batch=batch_index,
                                                producer=target_name,
                                                consumer=(
                                                    " + ".join(
                                                        adapter.targets[index]
                                                        for index in target_groups[
                                                            group_index + 1
                                                        ]
                                                    )
                                                ),
                                                value=next_value,
                                            )
                                        del next_value, value
                                    del next_values, values
                                    if ep_metrics is not None:
                                        ep_metrics.host_time(
                                            "boundary",
                                            perf_counter() - publish_started_at,
                                        )
                                        ep_metrics.advance(len(batch_indices))

                    with _stage_logger(
                        adapter.model,
                        f"{target_label} | compress weights",
                        device,
                    ):
                        _compress_modules(resident_modules)
                    with _stage_logger(
                        adapter.model,
                        f"{target_label} | queue checkpoint",
                        device,
                    ):
                        if completion_reservation is not None:
                            completion_reservation.close()
                        if transaction_writer is not None:
                            with transaction_writer.transaction(
                                transaction_id
                            ) as transaction:
                                if has_next:
                                    for batch_index in batches:
                                        transaction.write_boundary(
                                            batch_index,
                                            boundaries.get(next_boundary, batch_index),
                                            boundary=next_boundary,
                                            expected_batch_count=len(batches),
                                        )
                                written.update(
                                    _write_loaded_target(
                                        transaction,
                                        loaded,
                                        target_names,
                                        source,
                                        materializer,
                                        pack_to_int8=pack_to_int8,
                                    )
                                )
                                transaction.commit()
                            committed_transaction_ids.append(transaction_id)
                            BoundaryCompactor(staging_dir).compact(
                                tuple(committed_transaction_ids)
                            )
                        else:
                            written.update(
                                _write_loaded_target_direct(
                                    direct_writer,
                                    loaded,
                                    target_names,
                                    transaction_id,
                                    materializer,
                                    pack_to_int8=pack_to_int8,
                                    host_budget=host_budget,
                                )
                            )
                finally:
                    with _stage_logger(
                        adapter.model,
                        f"{target_label} | unload weights",
                        device,
                    ):
                        if ep_executor is not None:
                            ep_executor.synchronize()
                        weight_stack.close()
                        boundaries.delete(group_start)
                        _empty_device_cache(device)
                        if expert_parallel is not None:
                            for owner_device in expert_parallel.devices[1:]:
                                _empty_device_cache(owner_device)
                peak_allocated = (
                    torch.get_device_module(device).max_memory_allocated(device)
                    if device.type == "cuda"
                    else None
                )
                streaming_logger.info(
                    f"{target_label} | complete | "
                    f"time={perf_counter() - target_started_at:.2f}s | "
                    f"baseline_allocated_bytes={baseline_allocated} | "
                    f"peak_allocated_bytes={peak_allocated}"
                )
                if expert_parallel is not None:
                    for owner_rank, owner_device in enumerate(expert_parallel.devices):
                        owner_peak = (
                            torch.cuda.max_memory_allocated(owner_device)
                            if owner_device.type == "cuda"
                            else None
                        )
                        peak_text = (
                            f"{owner_peak / 2**30:.2f}"
                            if owner_peak is not None
                            else "unavailable"
                        )
                        streaming_logger.info(
                            f"{target_label} | EP memory | rank={owner_rank} | "
                            f"device={owner_device} | "
                            f"peak_allocated_gib={peak_text}"
                        )
            pipeline_failed = False
        finally:
            try:
                if ep_executor is not None:
                    ep_executor.close()
                if prefetcher is not None:
                    prefetcher.close(suppress_errors=pipeline_failed)
                with _stage_logger(
                    adapter.model, "[finalize] modifier session", device
                ):
                    if pipeline_failed and runtime_adapters:
                        for modifier_adapter in runtime_adapters:
                            modifier_adapter.abort()
                    else:
                        LifecycleCallbacks.calibration_end()
                    session.finalize()
                streaming_logger.info("finalize | modifier session complete")
            except BaseException:
                if async_direct_writer is not None:
                    try:
                        async_direct_writer.close()
                    except Exception:
                        streaming_logger.exception(
                            "Asynchronous checkpoint writer failed while unwinding"
                        )
                raise
            if pipeline_failed and async_direct_writer is not None:
                try:
                    async_direct_writer.close()
                except Exception:
                    streaming_logger.exception(
                        "Asynchronous checkpoint writer failed while unwinding"
                    )

    source_names = set(source.tensor_names())
    omitted = {
        alias: canonical
        for alias, canonical in infer_transformers_tied_weights(checkpoint).items()
        if alias in source_names and canonical in source_names
    }
    streaming_logger.info("finalize | writing static model tensors")
    remaining_failed = True
    try:
        with _stage_logger(adapter.model, "[finalize] write static tensors", device):
            if direct_writer is not None:
                _write_remaining_direct_shards(
                    writer=direct_writer,
                    source=source,
                    materializer=materializer,
                    target_dtype=target_dtype,
                    dtype_policy=dtype_policy,
                    written=written,
                    omitted=omitted,
                    host_budget=host_budget,
                )
            else:
                _copy_remaining_tensors(
                    writer=transaction_writer,
                    source=source,
                    materializer=materializer,
                    target_dtype=target_dtype,
                    dtype_policy=dtype_policy,
                    written=written,
                    omitted=omitted,
                )
                transaction_writer.assemble_shards()
        remaining_failed = False
    finally:
        if async_direct_writer is not None:
            try:
                drain_started = perf_counter()
                async_direct_writer.close()
                streaming_logger.info(
                    f"finalize | checkpoint drain | "
                    f"time={perf_counter() - drain_started:.2f}s"
                )
            except Exception:
                if not remaining_failed:
                    raise
                streaming_logger.exception(
                    "Asynchronous checkpoint writer failed while unwinding"
                )
    streaming_logger.info("complete")
    output_staging = staging_dir if checkpoint_progress else publish_dir
    return artifact_dir, output_staging, run_fingerprint
