"""Per-subgraph closed-loop execution for streaming PTQ."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from compressed_tensors.compressors import compress_module
from compressed_tensors.quantization.utils import is_module_quantized
from loguru import logger

from llmcompressor.core import LifecycleCallbacks, create_session
from llmcompressor.modifiers.quantization.calibration import (
    freeze_module_quantization,
)
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.recipe import Recipe
from llmcompressor.utils.helpers import DisableQuantization

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
from .checkpoint import DirectSafetensorsWriter, StreamingCheckpointWriter
from .materialization import WeightMaterializer, materialize_weights
from .output import quantized_module_formats
from .tied_weights import infer_transformers_tied_weights
from .tracing import TracedBoundaryAdapter

__all__ = ["run_subgraph_streaming_pipeline"]


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
    persist: bool = True,
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
    if persist:
        ArtifactStore(artifact_dir).initialize(
            manifest, normalized_recipe=normalized_recipe, targets=targets
        )
    return run_fingerprint


def _write_loaded_target(
    transaction, loaded, target_name: str, source
) -> set[str]:
    written = set()
    formats = quantized_module_formats(
        loaded.model.named_modules(), prefix=target_name
    )
    for name, tensor in loaded.state_tensors_under((target_name,)):
        owner_name = name.rpartition(".")[0]
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


def _loaded_target_delta(loaded, target_name: str):
    tensors = dict(loaded.state_tensors_under((target_name,)))
    formats = quantized_module_formats(
        loaded.model.named_modules(), prefix=target_name
    )
    written = set(tensors)
    for module_name in formats:
        written.add(f"{module_name}.weight")
    return tensors, formats, written


def _write_remaining_direct_shards(
    *,
    writer: DirectSafetensorsWriter,
    source,
    materializer: WeightMaterializer,
    target_dtype: torch.dtype,
    written: set[str],
    omitted: Mapping[str, str],
) -> None:
    """Copy non-subgraph tensors into bounded final static shards."""

    omitted_metadata = dict(omitted)
    tensors: dict[str, torch.Tensor] = {}
    tensor_bytes = 0
    shard_index = 0
    max_shard_bytes = 256 * 1024 * 1024

    def flush() -> None:
        nonlocal omitted_metadata, shard_index, tensor_bytes
        if not tensors:
            return
        writer.write_shard(
            f"static-{shard_index:05d}",
            tensors,
            omitted_tied_weights=omitted_metadata,
        )
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
                device=torch.device("cpu"),
            )[name]
        else:
            value = source.load_tensors(
                (name,), device=torch.device("cpu")
            )[name]
        value_bytes = value.numel() * value.element_size()
        if tensors and tensor_bytes + value_bytes > max_shard_bytes:
            flush()
        tensors[name] = value
        tensor_bytes += value_bytes
        written.add(name)
    flush()


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
                device=torch.device("cpu"),
            )[name]
        else:
            value = source.load_tensors(
                (name,), device=torch.device("cpu")
            )[name]
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
) -> tuple[Path, Path]:
    """Calibrate, modify, propagate, and persist one subgraph at a time."""

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
        persist=checkpoint_progress,
    )
    transaction_writer = (
        StreamingCheckpointWriter(staging_dir, run_fingerprint=run_fingerprint)
        if checkpoint_progress
        else None
    )
    direct_writer = (
        None
        if checkpoint_progress
        else DirectSafetensorsWriter(
            publish_dir, run_fingerprint=run_fingerprint
        )
    )
    boundaries: BoundaryActivationStore
    if checkpoint_progress:
        boundaries = DiskBoundaryActivationStore(work / "boundaries")
    else:
        boundaries = InMemoryBoundaryActivationStore(
            storage_device="cpu", deduplicate_tensors=True
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
        boundaries.delete(0)
        for batch_index, boundary in enumerate(
            adapter.calibration_boundaries(calibration_batches)
        ):
            boundaries.put(0, batch_index, boundary)

    with create_session() as session:
        session.initialize(
            model=adapter.model,
            recipe=recipe,
            start=-1,
            calib_data=calibration_batches,
            sequential_targets=adapter.targets,
            copy_data=False,
        )
        LifecycleCallbacks.calibration_start()
        try:
            for target_index, (target_name, subgraph) in enumerate(
                zip(adapter.targets, adapter.target_subgraphs)
            ):
                transaction_id = f"subgraph-{target_index:05d}"
                if (
                    transaction_writer is not None
                    and transaction_writer.is_transaction_complete(transaction_id)
                ):
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
                logger.info(
                    "streaming pipeline: processing subgraph "
                    f"{target_index + 1}/{len(adapter.targets)} {target_name}"
                )
                # A previous attempt can fail after publishing only part of the
                # next external boundary but before committing its transaction.
                # Rebuild it from scratch so stale batches are never consumed.
                if target_index + 1 < len(adapter.targets):
                    boundaries.delete(target_index + 1)
                with adapter.weight_session.loaded(
                    subgraph, device=device, dtype=target_dtype
                ) as loaded:
                    # Match the ordinary sequential calibration pipeline: model
                    # execution is inference-only. Besides avoiding autograd
                    # storage, this is required by models such as DeepSeek-V4
                    # whose attention updates runtime KV buffers in place.
                    with DisableQuantization(adapter.model), torch.no_grad():
                        for batch_index in batches:
                            session.state.current_batch_idx = batch_index
                            value = boundaries.get(
                                target_index, batch_index, device=device
                            )
                            inputs = {
                                name: value[name] for name in subgraph.input_names
                            }
                            subgraph.forward(adapter.model, **inputs)
                    LifecycleCallbacks.sequential_epoch_end(
                        subgraph.submodules(adapter.model)
                    )
                    for module in subgraph.submodules(adapter.model):
                        if is_module_quantized(module):
                            freeze_module_quantization(module)
                    with HooksMixin.disable_hooks(), torch.no_grad():
                        for batch_index in batches:
                            value = boundaries.get(
                                target_index, batch_index, device=device
                            )
                            inputs = {
                                name: value[name] for name in subgraph.input_names
                            }
                            output = subgraph.forward(adapter.model, **inputs)
                            if target_index + 1 < len(adapter.targets):
                                next_value = {**value, **output}
                                for consumed in adapter.plan.subgraphs[
                                    adapter.plan.target_subgraph_indices[target_index]
                                ].consumed_names:
                                    next_value.pop(consumed, None)
                                boundaries.put(
                                    target_index + 1, batch_index, next_value
                                )

                    for module in subgraph.submodules(adapter.model):
                        if is_module_quantized(module):
                            compress_module(module)
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
                                    transaction, loaded, target_name, source
                                )
                            )
                            transaction.commit()
                    else:
                        tensors, formats, target_written = _loaded_target_delta(
                            loaded, target_name
                        )
                        if tensors:
                            direct_writer.write_shard(
                                transaction_id,
                                tensors,
                                quantized_modules=formats,
                            )
                        written.update(target_written)
                boundaries.delete(target_index)
        finally:
            LifecycleCallbacks.calibration_end()
            session.finalize()

    if direct_writer is not None:
        _initialize_run(
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
        )

    source_names = set(source.tensor_names())
    omitted = {
        alias: canonical
        for alias, canonical in infer_transformers_tied_weights(
            checkpoint
        ).items()
        if alias in source_names and canonical in source_names
    }
    if direct_writer is not None:
        _write_remaining_direct_shards(
            writer=direct_writer,
            source=source,
            materializer=materializer,
            target_dtype=target_dtype,
            written=written,
            omitted=omitted,
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
    return artifact_dir, staging_dir if checkpoint_progress else publish_dir
