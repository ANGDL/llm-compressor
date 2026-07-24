"""Stage-two quantization into resumable output staging shards."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import torch
from compressed_tensors.compressors import compress_module
from compressed_tensors.compressors.format import infer_module_format
from compressed_tensors.quantization import QuantizationScheme
from loguru import logger

from llmcompressor.entrypoints.model_free.lifecycle import (
    initialize_quantized_linear,
)
from llmcompressor.modifiers.gptq.gptq_quantize import quantize_weight
from llmcompressor.modifiers.quantization.calibration import (
    apply_calibration_status,
    freeze_module_quantization,
    initialize_observer,
    observe,
    update_qparams,
)

from .artifacts import (
    ArtifactCompatibilityError,
    ArtifactStore,
    fingerprint_checkpoint,
    fingerprint_json,
)
from .checkpoint import (
    SafetensorsWeightSource,  # noqa: F401
    StreamingCheckpointWriter,
)
from .materialization import (
    CastWeightMaterializer,
    WeightMaterializer,
)
from .tied_weights import infer_transformers_tied_weights

__all__ = ["quantize_streaming"]


def _scheme_fingerprint(
    schemes: Mapping[str, QuantizationScheme],
    *,
    use_gptq: bool,
    blocksize: int,
    dampening_frac: float,
) -> str:
    value = {
        "schemes": {
            name: deepcopy(scheme).model_dump(mode="json")
            for name, scheme in sorted(schemes.items())
        },
        "use_gptq": use_gptq,
        "blocksize": blocksize,
        "dampening_frac": dampening_frac,
        "tied_weight_policy": "deduplicate-identical-v1",
    }
    return fingerprint_json(value)


def _statistics_by_module(store: ArtifactStore) -> dict[str, dict[str, torch.Tensor]]:
    manifest = store.load_manifest()
    result: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for index in range(len(manifest.sequential.targets)):
        if not store.is_target_complete(index):
            raise RuntimeError(
                f"Calibration statistics for target {index} are not complete"
            )
        for name, tensor in store.load_target_statistics(index).items():
            module_name, separator, statistic_name = name.rpartition(".")
            if not separator or not module_name:
                raise ValueError(f"Invalid statistic tensor name {name!r}")
            if statistic_name in result[module_name]:
                raise ValueError(f"Duplicate statistic tensor {name!r}")
            result[module_name][statistic_name] = tensor
    return dict(result)


def _validate_statistics(
    schemes: Mapping[str, QuantizationScheme],
    statistics: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    use_gptq: bool,
) -> None:
    for module_name, scheme in schemes.items():
        if scheme.weights is None:
            raise ValueError(f"Scheme for {module_name!r} has no weight arguments")
        available = statistics.get(module_name, {})
        required = set()
        if use_gptq:
            required.update(("gptq_hessian", "gptq_num_samples"))
        if scheme.weights.observer == "imatrix_mse":
            required.update(("imatrix_sum", "imatrix_count"))
        missing = required - set(available)
        if missing:
            raise RuntimeError(
                f"Missing statistics for module {module_name!r}: {sorted(missing)}"
            )
        if use_gptq and available["gptq_num_samples"].item() <= 0:
            raise ValueError(f"GPTQ sample count for {module_name!r} must be positive")
        if use_gptq:
            width = available["gptq_hessian"].shape
            if len(width) != 2 or width[0] != width[1]:
                raise ValueError(
                    f"GPTQ Hessian for {module_name!r} must be square; got {width}"
                )


def _restore_imatrix(module: torch.nn.Module, statistics: Mapping[str, torch.Tensor]):
    observer = module.weight_observer
    observer._imatrix_sum = statistics["imatrix_sum"].to(module.weight.device)
    observer._imatrix_count = statistics["imatrix_count"].to(
        module.weight.device
    )


def _quantize_module(
    weight: torch.Tensor,
    scheme: QuantizationScheme,
    statistics: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    use_gptq: bool,
    blocksize: int,
    dampening_frac: float,
) -> torch.nn.Module:
    module = initialize_quantized_linear(weight, scheme, device)
    initialize_observer(module, "weight")
    apply_calibration_status(module)
    if scheme.weights.observer == "imatrix_mse":
        _restore_imatrix(module, statistics)
    observe(module, "weight")

    if use_gptq:
        count = statistics["gptq_num_samples"].to(device=device)
        hessian = statistics["gptq_hessian"].to(device=device) / count
        _, qparams = quantize_weight(
            module=module,
            quant_args=scheme.weights,
            hessian=hessian,
            blocksize=blocksize,
            percdamp=dampening_frac,
        )
        for name, value in qparams.items():
            setattr(module, name, torch.nn.Parameter(value, requires_grad=False))
    else:
        update_qparams(module, "weight")

    freeze_module_quantization(module)
    compress_module(module)
    return module


def quantize_streaming(
    *,
    checkpoint: str | Path,
    artifact_dir: str | Path,
    staging_dir: str | Path,
    schemes: Mapping[str, QuantizationScheme],
    use_gptq: bool = True,
    materializer: WeightMaterializer | None = None,
    device: torch.device | str = "cpu",
    target_dtype: torch.dtype = torch.bfloat16,
    blocksize: int = 128,
    dampening_frac: float = 0.01,
) -> Path:
    """Quantize exact named Linear modules into resumable staging shards.

    Stage two deliberately does not create a model index or config. Each state
    record is written only after its corresponding shard has been atomically
    published. Stage three consumes these records to build the final checkpoint.
    """
    if not schemes:
        raise ValueError("schemes must contain at least one exact module name")
    if blocksize <= 0:
        raise ValueError("blocksize must be positive")
    if dampening_frac < 0:
        raise ValueError("dampening_frac must be non-negative")

    materializer = materializer or CastWeightMaterializer()
    source = materializer.create_source(str(checkpoint))
    tied_weights = infer_transformers_tied_weights(checkpoint)
    artifact_store = ArtifactStore(artifact_dir)
    manifest = artifact_store.load_manifest()
    current_source = fingerprint_checkpoint(checkpoint)
    if current_source.content_fingerprint != manifest.source.content_fingerprint:
        raise ArtifactCompatibilityError(
            "Calibration artifacts belong to a different source checkpoint"
        )
    if materializer.manifest_info(target_dtype=target_dtype) != manifest.materializer:
        raise ArtifactCompatibilityError(
            "Calibration artifacts were created with a different materializer "
            "or target dtype"
        )
    statistics = _statistics_by_module(artifact_store)
    _validate_statistics(schemes, statistics, use_gptq=use_gptq)
    run_fingerprint = _scheme_fingerprint(
        schemes,
        use_gptq=use_gptq,
        blocksize=blocksize,
        dampening_frac=dampening_frac,
    )

    missing_weights = [
        f"{module_name}.weight"
        for module_name in schemes
        if f"{module_name}.weight" not in source.tensor_names()
    ]
    if missing_weights:
        raise KeyError(
            "Quantized weights are absent from checkpoint: "
            f"{missing_weights}"
        )

    staging = Path(staging_dir)
    writer = StreamingCheckpointWriter(
        staging, run_fingerprint=run_fingerprint
    )
    device = torch.device(device)
    source_names = set(source.tensor_names())
    omitted_tied_weights = {}
    for alias, canonical in tied_weights.items():
        if alias not in source_names or canonical not in source_names:
            continue
        tied_values = source.load_tensors([alias, canonical], device=device)
        if torch.equal(tied_values[alias], tied_values[canonical]):
            omitted_tied_weights[alias] = canonical
        del tied_values
    names_by_shard: dict[Path, list[str]] = defaultdict(list)
    for name in source_names:
        if name in omitted_tied_weights:
            continue
        names_by_shard[source.metadata(name).shard].append(name)

    for source_shard, names in sorted(
        names_by_shard.items(), key=lambda item: item[0].name
    ):
        output_name = source_shard.name
        transaction_id = f"source-shard-{output_name}"
        if writer.is_transaction_complete(transaction_id):
            logger.info(
                f"streaming quantize: skipping complete shard {output_name}"
            )
            continue

        logger.info(f"streaming quantize: processing shard {output_name}")
        quantized_weights = {
            f"{module_name}.weight": (module_name, scheme)
            for module_name, scheme in schemes.items()
        }
        with writer.transaction(transaction_id) as transaction:
            for name in names:
                if name in quantized_weights:
                    module_name, scheme = quantized_weights[name]
                    dependency_names = materializer.dependencies(
                        name, source.metadata(name)
                    )
                    requested = [name, *dependency_names]
                    raw_values = source.load_tensors(requested, device=device)
                    weight = materializer.materialize(
                        name,
                        raw_values,
                        target_dtype=target_dtype,
                        device=device,
                    )
                    metadata = source.metadata(name)
                    expected_shape = materializer.logical_shape(name, metadata)
                    if (
                        not weight.dtype.is_floating_point
                        or weight.dtype != target_dtype
                        or tuple(weight.shape) != expected_shape
                        or weight.device != device
                    ):
                        raise ValueError(
                            f"Materializer returned an invalid tensor for {name!r}"
                        )
                    module = _quantize_module(
                        weight,
                        deepcopy(scheme),
                        statistics[module_name],
                        device=device,
                        use_gptq=use_gptq,
                        blocksize=blocksize,
                        dampening_frac=dampening_frac,
                    )
                    for module_tensor_name, output_value in module.state_dict(
                        prefix=f"{module_name}."
                    ).items():
                        transaction.write_tensor(
                            module_tensor_name,
                            output_value,
                            output_shard=output_name,
                        )
                    transaction.mark_quantized(
                        module_name,
                        scheme.format
                        or infer_module_format(torch.nn.Linear, scheme).value,
                    )
                    del module, weight, raw_values
                    continue

                raw_values = source.load_tensors([name], device=device)
                value = raw_values[name]
                dependency_names = materializer.dependencies(
                    name, source.metadata(name)
                )
                if dependency_names:
                    raw_values.update(
                        source.load_tensors(dependency_names, device=device)
                    )
                    value = materializer.materialize(
                        name,
                        raw_values,
                        target_dtype=target_dtype,
                        device=device,
                    )
                transaction.write_tensor(name, value, output_shard=output_name)
                del raw_values, value

            for alias, canonical in omitted_tied_weights.items():
                transaction.omit_tied_weight(alias, canonical)
            transaction.commit()
        logger.info(f"streaming quantize: committed shard {output_name}")

    writer.assemble_shards()

    return staging
