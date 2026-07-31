"""Per-subgraph closed-loop execution for streaming PTQ."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from compressed_tensors.compressors import compress_module
from compressed_tensors.quantization.utils import is_module_quantized

from llmcompressor.core import LifecycleCallbacks, create_session
from llmcompressor.modifiers.quantization.calibration import (
    freeze_module_quantization,
    set_quantization_scale_dtype,
)
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.recipe import Recipe
from llmcompressor.utils.helpers import DisableQuantization
from llmcompressor.utils.int4_packing import DEFAULT_INT4_PACKING_WORKSPACE_BYTES
from llmcompressor.utils.metric_logging import CompressionLogger

from ._logging import streaming_logger
from .activations import (
    BoundaryActivationStore,
    DiskBoundaryActivationStore,
    InMemoryBoundaryActivationStore,
)
from .artifacts import (
    ArtifactStore,
    CalibrationInfo,
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
    StreamingCheckpointWriter,
)
from .loading import HostMemoryBudget, SubgraphPrefetcher
from .materialization import WeightMaterializer, materialize_weights
from .output import (
    prepare_quantized_tensor_for_save,
    quantized_module_formats,
    quantized_tensor_bytes_for_save,
)
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
        torch.cuda.reset_peak_memory_stats(device)


def _stage_logger(
    model: torch.nn.Module, name: str, device: torch.device
) -> CompressionLogger:
    device_ids = (
        (device.index if device.index is not None else torch.cuda.current_device(),)
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
    return f"[{index + 1:0{width}d}/{total:0{width}d}] {target}"


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


def _initialize_run(
    *,
    checkpoint: str | Path,
    artifact_dir: Path,
    recipe: Recipe,
    dataset_fingerprint: str,
    targets: tuple[str, ...],
    materializer: WeightMaterializer,
    target_dtype: torch.dtype,
    num_samples: int,
    max_seq_length: int | None,
    seed: int | None,
    pack_to_int8: bool,
    replace_existing: bool = False,
) -> str:
    normalized_recipe = recipe.model_dump(mode="json")
    source_info = fingerprint_checkpoint(checkpoint)
    run_fingerprint = fingerprint_json(
        {
            "source": source_info.content_fingerprint,
            "recipe": normalized_recipe,
            "dataset": dataset_fingerprint,
            "targets": targets,
            "materializer": materializer.manifest_info(
                target_dtype=target_dtype
            ).config_sha256,
            "output": {"pack_to_int8": pack_to_int8},
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
        sequential=SequentialInfo(targets),
        materializer=materializer.manifest_info(target_dtype=target_dtype),
        software=SoftwareInfo.from_versions({"torch": torch.__version__}),
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
        module_name == root or module_name.startswith(f"{root}.")
        for root in roots
    )


def _write_loaded_target(
    transaction,
    loaded,
    target_name: str,
    source,
    *,
    pack_to_int8: bool,
    exclude_modules: Sequence[str] = (),
) -> set[str]:
    written = set()
    formats = quantized_module_formats(
        loaded.model.named_modules(), prefix=target_name
    )
    formats = {
        name: value
        for name, value in formats.items()
        if not _is_under_any(name, exclude_modules)
    }
    for name, tensor in loaded.state_tensors_under((target_name,)):
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
        transaction.write_tensor(
            name,
            tensor,
            output_shard=_output_shard(source, name, owner_name),
        )
        written.add(name)

    for module_name, format_name in formats.items():
        transaction.mark_quantized(module_name, format_name)
        # Compression formats such as pack-quantized replace the source weight
        # with weight_packed/weight_shape. Mark the original source tensor as
        # consumed so the fallback copier cannot reintroduce a second raw weight.
        written.add(f"{module_name}.weight")
    return written


def _loaded_target_delta(
    loaded,
    target_name: str,
    *,
    pack_to_int8: bool,
    exclude_modules: Sequence[str] = (),
):
    formats = quantized_module_formats(
        loaded.model.named_modules(), prefix=target_name
    )
    formats = {
        name: value
        for name, value in formats.items()
        if not _is_under_any(name, exclude_modules)
    }
    tensors = {}
    owned_cpu_tensors = set()
    for name, tensor in loaded.state_tensors_under((target_name,)):
        if _is_under_any(name.rpartition(".")[0], exclude_modules):
            continue
        prepared = _prepare_loaded_tensor_for_save(
            loaded,
            name,
            tensor,
            formats,
            pack_to_int8=pack_to_int8,
        )
        tensors[name] = prepared
        if prepared is not tensor and prepared.device.type == "cpu":
            owned_cpu_tensors.add(name)
    written = set(tensors)
    for module_name in formats:
        written.add(f"{module_name}.weight")
    return tensors, formats, written, owned_cpu_tensors


def _estimate_loaded_target_snapshot_bytes(
    loaded,
    target_name: str,
    *,
    pack_to_int8: bool,
    exclude_modules: Sequence[str] = (),
) -> int:
    formats = quantized_module_formats(
        loaded.model.named_modules(), prefix=target_name
    )
    total = 0
    workspace = 0
    for name, tensor in loaded.state_tensors_under((target_name,)):
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
    target_name: str,
    shard_id: str,
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
            reservation = host_budget.reserve(
                f"checkpoint shard {shard_id}", required
            )
        tensors, formats, written, owned_cpu_tensors = _loaded_target_delta(
            loaded,
            target_name,
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


def _modifier_update(modules: Sequence[torch.nn.Module]) -> None:
    LifecycleCallbacks.sequential_epoch_end(list(modules))
    for module in modules:
        if is_module_quantized(module):
            freeze_module_quantization(module)


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
    target_dtype: torch.dtype,
    written: set[str],
    omitted: Mapping[str, str],
    host_budget: HostMemoryBudget | None = None,
    start_shard_index: int = 0,
) -> None:
    """Copy non-subgraph tensors into bounded final static shards."""

    omitted_metadata = dict(omitted)
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
            )[name]
        else:
            value = source.load_tensors_cpu((name,))[name]
        value_bytes = value.numel() * value.element_size()
        if tensors and tensor_bytes + value_bytes > max_shard_bytes:
            flush()
        tensors[name] = value
        tensor_bytes += value_bytes
        written.add(name)
    flush()


def _complete_direct_writer_staging(
    *,
    adapter: TracedBoundaryAdapter,
    checkpoint: str | Path,
    publish_dir: Path,
    materializer: WeightMaterializer,
    target_dtype: torch.dtype,
    run_fingerprint: str,
) -> None:
    """Validate completed targets and append only missing static tensors."""

    writer = DirectSafetensorsWriter(
        publish_dir, run_fingerprint=run_fingerprint
    )
    states = writer.committed_metadata()
    required_target_shards = {
        f"model-subgraph-{index:05d}.safetensors"
        for index in range(len(adapter.targets))
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

    written = {
        name for state in states for name in state.get("tensor_names", ())
    }
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
        for alias, canonical in infer_transformers_tied_weights(
            checkpoint
        ).items()
        if alias in source_names and canonical in source_names
    }
    _write_remaining_direct_shards(
        writer=writer,
        source=source,
        materializer=materializer,
        target_dtype=target_dtype,
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
            )[name]
        else:
            value = source.load_tensors_cpu((name,))[name]
        with writer.transaction(transaction_id) as transaction:
            transaction.write_tensor(
                name, value, output_shard=metadata.shard.name
            )
            for alias, canonical in omitted.items():
                transaction.omit_tied_weight(alias, canonical)
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
    async_save: bool = False,
    pack_to_int8: bool = True,
) -> tuple[Path, Path, str]:
    """Calibrate, modify, propagate, and persist one subgraph at a time."""

    device = torch.device(device)
    if checkpoint_progress and async_save:
        raise ValueError(
            "Asynchronous saving is not supported with checkpoint_progress=True"
        )
    work = Path(work_dir)
    artifact_dir = work / "artifacts"
    staging_dir = work / "staging"
    publish_dir = work / "publish"
    run_fingerprint = _initialize_run(
        checkpoint=checkpoint,
        artifact_dir=artifact_dir,
        recipe=recipe,
        dataset_fingerprint=dataset_fingerprint,
        targets=adapter.targets,
        materializer=materializer,
        target_dtype=target_dtype,
        num_samples=num_samples,
        max_seq_length=max_seq_length,
        seed=seed,
        pack_to_int8=pack_to_int8,
        replace_existing=not checkpoint_progress,
    )
    transaction_writer = (
        StreamingCheckpointWriter(staging_dir, run_fingerprint=run_fingerprint)
        if checkpoint_progress
        else None
    )
    direct_writer_base = (
        None
        if checkpoint_progress
        else DirectSafetensorsWriter(
            publish_dir, run_fingerprint=run_fingerprint
        )
    )
    direct_writer = direct_writer_base
    async_direct_writer = None
    host_budget = HostMemoryBudget()
    boundaries: BoundaryActivationStore
    if checkpoint_progress:
        boundaries = DiskBoundaryActivationStore(
            work / "boundaries", host_budget=host_budget
        )
    else:
        boundaries = InMemoryBoundaryActivationStore(
            storage_device="cpu",
            deduplicate_tensors=True,
            host_budget=host_budget,
        )
    source = adapter.weight_session.source
    written: set[str] = set()
    first_transaction = "subgraph-00000"
    if transaction_writer is None or not transaction_writer.is_transaction_complete(
        first_transaction
    ):
        if transaction_writer is not None and transaction_writer.has_transaction(
            first_transaction
        ):
            raise RuntimeError(
                f"Committed transaction {first_transaction!r} is corrupt; "
                "refusing to replace durable output"
            )
        # Boundary zero is not covered by a preceding subgraph transaction.
        # Always recreate it when the first subgraph is incomplete so a crash
        # while snapshotting the dataloader cannot leave a partial input set.
        streaming_logger.info("setup | collecting initial calibration boundary")
        with _stage_logger(
            adapter.model, "[setup] initial boundary", device
        ):
            boundaries.delete(0)
            for batch_index, boundary in enumerate(
                adapter.calibration_boundaries(calibration_batches)
            ):
                boundaries.put(0, batch_index, boundary)
        streaming_logger.info(
            "setup | initial boundary ready | "
            f"batches={len(boundaries.batch_indices(0))}"
        )

    streaming_logger.info("setup | initializing modifier session")
    with create_session() as session:
        with _stage_logger(
            adapter.model, "[setup] initialize modifiers", device
        ):
            session.initialize(
                model=adapter.model,
                recipe=recipe,
                start=-1,
                calib_data=calibration_batches,
                sequential_targets=adapter.targets,
                copy_data=False,
            )
            LifecycleCallbacks.calibration_start()
        streaming_logger.info("setup | modifier session ready")
        if async_save:
            if direct_writer_base is None:
                raise RuntimeError("Asynchronous saving requires a direct writer")
            async_direct_writer = AsyncDirectSafetensorsWriter(
                direct_writer_base, max_pending=1
            )
            direct_writer = async_direct_writer
        prefetcher = (
            SubgraphPrefetcher(adapter.weight_session)
            if adapter.targets
            else None
        )
        pipeline_failed = True
        try:
            prefetch_pending = False
            for target_index, (target_name, subgraph) in enumerate(
                zip(adapter.targets, adapter.target_subgraphs)
            ):
                transaction_id = f"subgraph-{target_index:05d}"
                if (
                    transaction_writer is not None
                    and transaction_writer.is_transaction_complete(transaction_id)
                ):
                    label = _target_label(
                        target_index, len(adapter.targets), target_name
                    )
                    streaming_logger.info(f"{label} | restoring checkpoint")
                    metadata = next(
                        value
                        for value in transaction_writer.committed_metadata()
                        if value["transaction_id"] == transaction_id
                    )
                    written.update(
                        record["name"] for record in metadata["records"]
                    )
                    committed_boundary = transaction_writer.committed_boundary(
                        transaction_id
                    )
                    if committed_boundary is not None:
                        boundary_index, values = committed_boundary
                        boundaries.delete(boundary_index)
                        for batch_index, value in enumerate(values):
                            boundaries.put(boundary_index, batch_index, value)
                    boundaries.delete(target_index)
                    continue
                if (
                    transaction_writer is not None
                    and transaction_writer.has_transaction(transaction_id)
                ):
                    raise RuntimeError(
                        f"Committed transaction {transaction_id!r} is corrupt; "
                        "refusing to replace durable output"
                    )
                batches = boundaries.batch_indices(target_index)
                if not batches:
                    raise RuntimeError(
                        f"Missing boundary {target_index} for {target_name!r}"
                    )
                target_count = len(adapter.targets)
                target_label = _target_label(
                    target_index, target_count, target_name
                )
                _reset_peak_memory(device)
                streaming_logger.info(
                    f"{target_label} | start | "
                    f"batches={len(batches)} | device={device}"
                )
                target_started_at = perf_counter()
                # A previous attempt can fail after publishing only part of the
                # next external boundary but before committing its transaction.
                # Rebuild it from scratch so stale batches are never consumed.
                if target_index + 1 < len(adapter.targets):
                    boundaries.delete(target_index + 1)
                weight_stack = ExitStack()
                try:
                    if not prefetch_pending:
                        with _stage_logger(
                            adapter.model,
                            f"{target_label} | plan weights",
                            device,
                        ):
                            plan = adapter.weight_session.plan(subgraph)
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
                        prepared = prefetcher.take()
                        prefetch_pending = False
                    with _stage_logger(
                        adapter.model,
                        f"{target_label} | load weights",
                        device,
                    ):
                        prepared.move_to(device)
                        loaded = weight_stack.enter_context(
                            adapter.weight_session.installed(prepared)
                        )

                    next_index = target_index + 1
                    if prefetcher is not None and next_index < target_count:
                        next_name = adapter.targets[next_index]
                        next_label = _target_label(
                            next_index, target_count, next_name
                        )
                        with _stage_logger(
                            adapter.model,
                            f"{next_label} | plan weights",
                            device,
                        ):
                            next_plan = adapter.weight_session.plan(
                                adapter.target_subgraphs[next_index]
                            )
                            next_required = (
                                adapter.weight_session.estimate_preparation(
                                    next_plan, dtype=target_dtype
                                )
                            )
                            next_reservation = host_budget.reserve(
                                f"{next_label} weights", next_required
                            )
                        prefetcher.submit(
                            next_plan,
                            dtype=target_dtype,
                            reservation=next_reservation,
                            label=f"{next_label} | prepare weights",
                        )
                        prefetch_pending = True

                    resident_modules = tuple(
                        subgraph.submodules(adapter.model)
                    )
                    # Match the ordinary sequential calibration pipeline: model
                    # execution is inference-only. Besides avoiding autograd
                    # storage, this is required by models such as DeepSeek-V4
                    # whose attention updates runtime KV buffers in place.
                    with _stage_logger(
                        adapter.model,
                        f"{target_label} | calibration",
                        device,
                    ):
                        with DisableQuantization(adapter.model), torch.no_grad():
                            for batch_index in batches:
                                session.state.current_batch_idx = batch_index
                                value = boundaries.get(
                                    target_index,
                                    batch_index,
                                    device=device,
                                )
                                inputs = {
                                    name: value[name]
                                    for name in subgraph.input_names
                                }
                                subgraph.forward(adapter.model, **inputs)
                                del inputs, value
                    with _stage_logger(
                        adapter.model,
                        f"{target_label} | modifier update",
                        device,
                    ):
                        _modifier_update(resident_modules)
                    with _stage_logger(
                        adapter.model,
                        f"{target_label} | activation propagation",
                        device,
                    ):
                        # Match SequentialPipeline: propagation captures the
                        # modified full-precision output (or an algorithm's
                        # in-place weight update), without adding fake-quant
                        # QDQ to every quantized module.
                        with (
                            HooksMixin.disable_hooks(),
                            DisableQuantization(adapter.model),
                            torch.no_grad(),
                        ):
                            for batch_index in batches:
                                value = boundaries.get(
                                    target_index,
                                    batch_index,
                                    device=device,
                                )
                                inputs = {
                                    name: value[name]
                                    for name in subgraph.input_names
                                }
                                output = subgraph.forward(adapter.model, **inputs)
                                if target_index + 1 < len(adapter.targets):
                                    next_value = {**value, **output}
                                    for consumed in adapter.plan.subgraphs[
                                        adapter.plan.target_subgraph_indices[
                                            target_index
                                        ]
                                    ].consumed_names:
                                        next_value.pop(consumed, None)
                                    boundaries.commit_propagated_batch(
                                        target_index,
                                        target_index + 1,
                                        batch_index,
                                        next_value,
                                    )
                                    del next_value
                                del inputs, output, value

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
                        if transaction_writer is not None:
                            with transaction_writer.transaction(
                                transaction_id
                            ) as transaction:
                                if target_index + 1 < len(adapter.targets):
                                    for batch_index in batches:
                                        transaction.write_boundary(
                                            batch_index,
                                            boundaries.get(
                                                target_index + 1, batch_index
                                            ),
                                            boundary=target_index + 1,
                                        )
                                written.update(
                                    _write_loaded_target(
                                        transaction,
                                        loaded,
                                        target_name,
                                        source,
                                        pack_to_int8=pack_to_int8,
                                    )
                                )
                                transaction.commit()
                        else:
                            written.update(
                                _write_loaded_target_direct(
                                    direct_writer,
                                    loaded,
                                    target_name,
                                    transaction_id,
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
                        weight_stack.close()
                        boundaries.delete(target_index)
                        _empty_device_cache(device)
                streaming_logger.info(
                    f"{target_label} | complete | "
                    f"time={perf_counter() - target_started_at:.2f}s"
                )
            pipeline_failed = False
        finally:
            try:
                if prefetcher is not None:
                    prefetcher.close(suppress_errors=pipeline_failed)
                with _stage_logger(
                    adapter.model, "[finalize] modifier session", device
                ):
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
        for alias, canonical in infer_transformers_tied_weights(
            checkpoint
        ).items()
        if alias in source_names and canonical in source_names
    }
    streaming_logger.info("finalize | writing static model tensors")
    remaining_failed = True
    try:
        with _stage_logger(
            adapter.model, "[finalize] write static tensors", device
        ):
            if direct_writer is not None:
                _write_remaining_direct_shards(
                    writer=direct_writer,
                    source=source,
                    materializer=materializer,
                    target_dtype=target_dtype,
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
                    written=written,
                    omitted=omitted,
                )
                transaction_writer.assemble_shards()
        remaining_failed = False
    finally:
        if async_direct_writer is not None:
            try:
                async_direct_writer.close()
            except Exception:
                if not remaining_failed:
                    raise
                streaming_logger.exception(
                    "Asynchronous checkpoint writer failed while unwinding"
                )
    streaming_logger.info("complete")
    output_staging = staging_dir if checkpoint_progress else publish_dir
    return artifact_dir, output_staging, run_fingerprint
