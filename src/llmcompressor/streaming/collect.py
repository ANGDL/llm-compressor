"""Stage-one orchestration for resumable streaming calibration."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from typing import Any

import torch
from loguru import logger
from torch import nn

from .activations import BoundaryActivationStore, DiskBoundaryActivationStore
from .artifacts import (
    ArtifactStore,
    CalibrationInfo,
    ExecutionInfo,
    RecipeInfo,
    SequentialInfo,
    SoftwareInfo,
    StreamingRunManifest,
    TargetStatisticsMetadata,
    fingerprint_checkpoint,
    fingerprint_json,
)
from .execution import (
    fingerprint_streaming_value,
    fingerprint_streaming_values,
    freeze_calibration_inputs,
    streaming_configuration,
)
from .loading import SubgraphWeightSession, build_meta_model
from .materialization import (
    CastWeightMaterializer,
    StreamingDTypePolicy,
    WeightMaterializer,
)
from .statistics import (
    CandidateStatisticsCache,
    CandidateStatisticsKey,
    GPTQStatisticsCollector,
    IMatrixStatisticsCollector,
    StatisticsCollectorGroup,
)

__all__ = ["collect_calibration_statistics"]

_UNSUPPORTED_MODIFIERS = {
    "AWQModifier": "requires calibration-time scale search",
    "SmoothQuantModifier": "requires repeated calibration forwards",
    "AutoSmoothQuantModifier": "requires repeated calibration forwards",
    "AutoRoundModifier": "optimizes weights with private forwards",
}
_SUPPORTED_ALGORITHMS = {"gptq", "imatrix"}


def _walk_names(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_names(key)
            yield from _walk_names(item)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            yield from _walk_names(item)
    else:
        # Recipe objects are accepted by the validation pass even when callers
        # have not converted them to the normalized JSON form yet.
        yield type(value).__name__


def _validate_recipe(recipe: Any, calibration_mode: str) -> None:
    if calibration_mode != "reference":
        raise ValueError(
            "Streaming calibration only supports calibration_mode='reference'; "
            f"got {calibration_mode!r}"
        )
    names = tuple(_walk_names(recipe))
    for modifier, reason in _UNSUPPORTED_MODIFIERS.items():
        if modifier in names:
            raise ValueError(
                f"Streaming calibration does not support {modifier}: {reason}"
            )
    if any("prun" in name.lower() for name in names):
        raise ValueError(
            "Streaming calibration does not support pruning modifiers because "
            "they dynamically change model structure"
        )


def _default_modules(target: nn.Module, target_name: str) -> dict[str, nn.Module]:
    modules = {}
    for local_name, module in target.named_modules():
        if isinstance(module, nn.Linear):
            name = f"{target_name}.{local_name}" if local_name else target_name
            modules[name] = module
    if not modules and isinstance(target, nn.Linear):
        modules[target_name] = target
    return modules


def _invoke_target(target: nn.Module, value: Any) -> Any:
    if isinstance(value, Mapping):
        return target(**value)
    if isinstance(value, tuple):
        return target(*value)
    return target(value)


def _module_fingerprints(source_fingerprint: str, target: nn.Module, target_name: str):
    names = []
    for local_name, _ in target.named_parameters(recurse=True):
        names.append(f"{target_name}.{local_name}" if local_name else target_name)
    # Per-tensor content hashes are intentionally not required in stage one. The
    # checkpoint identity in the manifest still prevents cross-checkpoint reuse.
    return tuple((name, source_fingerprint) for name in names)


def collect_calibration_statistics(
    *,
    model_factory: Callable[..., nn.Module],
    checkpoint: str,
    artifact_dir: str,
    calibration_batches: Iterable[Any] | Callable[[], Iterable[Any]],
    targets: Sequence[str],
    recipe: Mapping[str, Any] | Sequence[Any],
    dataset_fingerprint: str,
    model_args: Sequence[Any] = (),
    model_kwargs: Mapping[str, Any] | None = None,
    execution_model: nn.Module | None = None,
    materializer: WeightMaterializer | None = None,
    dtype_policy: StreamingDTypePolicy | None = None,
    target_module_selector: Callable[[nn.Module, str], Mapping[str, nn.Module]]
    | None = None,
    forward_target: Callable[[nn.Module, Any], Any] | None = None,
    boundary_store: BoundaryActivationStore | None = None,
    algorithms: Sequence[str] = ("gptq", "imatrix"),
    device: torch.device | str = "cpu",
    target_dtype: torch.dtype = torch.bfloat16,
    calibration_mode: str = "reference",
    num_samples: int | None = None,
    max_seq_length: int | None = None,
    seed: int | None = None,
    statistics_cache: CandidateStatisticsCache | None = None,
    candidate_config: Any | None = None,
    prefix_fingerprint: str = "",
) -> ArtifactStore:
    """Collect GPTQ/iMatrix statistics one target at a time.

    ``model_factory`` must construct the model without loading checkpoint data;
    it is called inside a meta-device context. ``forward_target`` receives the
    currently materialized target and one boundary batch, and returns the next
    boundary value. The default invokes the target with a mapping, tuple, or
    single positional batch. A callable ``calibration_batches`` is replayed only
    when target zero needs to be initialized, which keeps resume memory bounded.
    """
    if not targets:
        raise ValueError("targets must contain at least one sequential target")
    if len(set(targets)) != len(targets):
        raise ValueError("targets must be unique and ordered")
    if not algorithms:
        raise ValueError("algorithms must contain at least one algorithm")
    unknown = set(algorithms) - _SUPPORTED_ALGORITHMS
    if unknown:
        raise ValueError(f"Unsupported streaming algorithms: {sorted(unknown)}")
    _validate_recipe(recipe, calibration_mode)
    if (
        statistics_cache is not None
        and (forward_target is not None or target_module_selector is not None)
        and candidate_config is None
    ):
        raise ValueError(
            "Cached custom forwards/selectors require candidate_config "
            "to identify their execution and transform semantics"
        )
    try:
        dataset_fingerprint = str(dataset_fingerprint)
        if len(dataset_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in dataset_fingerprint
        ):
            raise ValueError
    except (TypeError, ValueError) as error:
        raise ValueError("dataset_fingerprint must be a SHA-256 digest") from error

    materializer = materializer or CastWeightMaterializer()
    source = materializer.create_source(checkpoint)
    source_info = fingerprint_checkpoint(checkpoint)
    input_identity = {}
    if (
        statistics_cache is not None
        or candidate_config is not None
        or prefix_fingerprint
    ):
        frozen = freeze_calibration_inputs(
            calibration_batches()
            if callable(calibration_batches)
            else calibration_batches,
            f"{artifact_dir}/cache-inputs",
        )
        calibration_batches = frozen
        input_identity = {
            "inputs": frozen.fingerprint,
            "transform": fingerprint_streaming_value(candidate_config),
            "prefix": prefix_fingerprint,
            "algorithms": sorted(algorithms),
        }
    manifest = StreamingRunManifest(
        source=source_info,
        recipe=RecipeInfo(fingerprint_json(recipe)),
        calibration=CalibrationInfo(
            dataset_fingerprint=dataset_fingerprint,
            num_samples=num_samples if num_samples is not None else 0,
            max_seq_length=max_seq_length,
            seed=seed,
        ),
        sequential=SequentialInfo(tuple(targets), calibration_mode=calibration_mode),
        materializer=materializer.manifest_info(
            target_dtype=target_dtype, dtype_policy=dtype_policy
        ),
        software=SoftwareInfo.from_versions({"torch": torch.__version__}),
        execution=ExecutionInfo(
            execution_graph_sha256=fingerprint_json(input_identity)
            if input_identity
            else ""
        ),
    )
    store = ArtifactStore(artifact_dir)
    store.initialize(manifest, normalized_recipe=recipe, targets=targets)

    kwargs = dict(model_kwargs or {})
    model = execution_model or build_meta_model(model_factory, *model_args, **kwargs)
    weight_session = SubgraphWeightSession(
        model, source, materializer, dtype_policy=dtype_policy
    )
    boundaries = boundary_store or DiskBoundaryActivationStore(
        store.root / "boundaries"
    )
    make_modules = target_module_selector or _default_modules
    run_forward = forward_target or _invoke_target
    batch_factory = (
        calibration_batches
        if callable(calibration_batches)
        else lambda: calibration_batches
    )

    for index, target_name in enumerate(targets):
        if store.is_target_complete(index):
            logger.info(
                f"streaming collect: skipping complete target "
                f"{index + 1}/{len(targets)} {target_name}"
            )
            continue
        logger.info(
            f"streaming collect: processing target "
            f"{index + 1}/{len(targets)} {target_name}"
        )
        if index == 0 and not boundaries.batch_indices(0):
            for batch_index, batch in enumerate(batch_factory()):
                boundaries.put(0, batch_index, batch)
        batches = boundaries.batch_indices(index)
        if not batches:
            raise RuntimeError(
                f"Missing calibration boundary {index}; cannot resume target "
                f"{target_name!r}"
            )
        with weight_session.loaded_modules(
            (target_name,), device=device, dtype=target_dtype
        ) as loaded:
            target = loaded.modules[0]
            target._streaming_target_name = target_name
            modules = dict(make_modules(target, target_name))
            if not modules:
                raise ValueError(
                    f"No Linear modules found under target {target_name!r}"
                )
            key = None
            if statistics_cache is not None:
                key = CandidateStatisticsKey(
                    source_fingerprint=fingerprint_streaming_values(
                        (name, tensor) for name, tensor in target.state_dict().items()
                    ),
                    dataset_fingerprint=dataset_fingerprint,
                    target_name=target_name,
                    target_index=index,
                    boundary_fingerprint=_boundary_fingerprint(
                        boundaries, index, batches
                    ),
                    prefix_fingerprint=prefix_fingerprint,
                    algorithms=tuple(sorted(algorithms)),
                    observer_fingerprint=fingerprint_json(
                        {
                            "algorithms": sorted(algorithms),
                            "counting": "raw-unmasked-input-v1",
                            "torch": torch.__version__,
                            "modules": sorted(modules),
                        }
                    ),
                    transform_fingerprint=fingerprint_streaming_value(
                        {
                            "declared": candidate_config,
                            "materializer": manifest.materializer.config_sha256,
                            "materializer_type": manifest.materializer.identifier,
                            "model_configuration": streaming_configuration(
                                (model_args, model_kwargs)
                            ),
                            "classes": [
                                (
                                    name,
                                    type(module).__module__,
                                    type(module).__qualname__,
                                )
                                for name, module in target.named_modules()
                            ],
                        }
                    ),
                    dtype=str(target_dtype),
                )
            expected_names = tuple(
                f"{module_name}.{statistic}"
                for module_name in modules
                for statistic in (
                    (
                        ("gptq_hessian", "gptq_num_samples")
                        if "gptq" in algorithms
                        else ()
                    )
                    + (
                        ("imatrix_sum", "imatrix_count")
                        if "imatrix" in algorithms
                        else ()
                    )
                )
            )
            statistics = (
                statistics_cache.get(key, expected_names=tuple(sorted(expected_names)))
                if key is not None
                else None
            )
            collector_group = None
            if statistics is None:
                collectors = []
                if "gptq" in algorithms:
                    collectors.append(GPTQStatisticsCollector(modules))
                if "imatrix" in algorithms:
                    collectors.append(IMatrixStatisticsCollector(modules))
                collector_group = StatisticsCollectorGroup(collectors)
            with torch.no_grad(), collector_group or nullcontext():
                for batch_index in batches:
                    value = boundaries.get(index, batch_index, device=device)
                    output = run_forward(target, value)
                    if index + 1 < len(targets):
                        boundaries.put(index + 1, batch_index, output)
            if collector_group is not None:
                statistics = collector_group.export()
                if key is not None:
                    statistics_cache.put(key, statistics)
        metadata = TargetStatisticsMetadata(
            target_name=target_name,
            target_index=index,
            algorithms=tuple(algorithms),
            tensor_names=tuple(sorted(statistics)),
            source_tensor_fingerprints=_module_fingerprints(
                source_info.content_fingerprint, target, target_name
            ),
            completed=False,
        )
        store.commit_target(metadata, statistics)
        boundaries.delete(index)
        logger.info(
            f"streaming collect: committed target "
            f"{index + 1}/{len(targets)} {target_name}"
        )
    return store


def _boundary_fingerprint(
    boundaries: BoundaryActivationStore,
    boundary: int,
    batches: Sequence[int],
) -> str:
    return fingerprint_streaming_values(
        (batch, boundaries.get(boundary, batch)) for batch in batches
    )
