"""Public orchestration entry point for the three streaming PTQ stages."""

from __future__ import annotations

import resource
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from compressed_tensors.compressors.format import infer_module_format
from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme
from loguru import logger
from torch import nn

from .collect import collect_calibration_statistics
from .finalize import finalize_streaming_checkpoint
from .materialization import WeightMaterializer
from .quantize import quantize_streaming

__all__ = ["streaming_oneshot"]


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _resource_summary(device: torch.device, path: Path) -> str:
    rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gpu_peak = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0
    )
    return (
        f"rss_peak_mib={rss_kib / 1024:.1f}, "
        f"gpu_peak_mib={gpu_peak / (1024**2):.1f}, "
        f"disk_mib={_directory_bytes(path) / (1024**2):.1f}"
    )


def _streaming_oneshot_from_boundaries(
    *,
    model_factory: Callable[..., nn.Module],
    checkpoint: str | Path,
    output_dir: str | Path,
    work_dir: str | Path,
    calibration_batches: Iterable[Any] | Callable[[], Iterable[Any]],
    targets: Sequence[str],
    recipe: Mapping[str, Any] | Sequence[Any],
    dataset_fingerprint: str,
    schemes: Mapping[str, QuantizationScheme],
    model_args: Sequence[Any] = (),
    model_kwargs: Mapping[str, Any] | None = None,
    materializer: WeightMaterializer | None = None,
    target_module_selector: Callable[[nn.Module, str], Mapping[str, nn.Module]]
    | None = None,
    forward_target: Callable[[nn.Module, Any], Any] | None = None,
    algorithms: Sequence[str] = ("gptq", "imatrix"),
    use_gptq: bool = True,
    device: torch.device | str = "cpu",
    target_dtype: torch.dtype = torch.bfloat16,
    blocksize: int = 128,
    dampening_frac: float = 0.01,
    num_samples: int | None = None,
    max_seq_length: int | None = None,
    seed: int | None = None,
    validate_config: bool = True,
) -> Path:
    """Run collect, quantize, and finalize with resumable intermediates."""
    work = Path(work_dir)
    artifact_dir = work / "artifacts"
    staging_dir = work / "staging"
    output = Path(output_dir)
    device = torch.device(device)
    if output.exists() and (output / "FINALIZED").is_file():
        logger.info(
            f"streaming finalize: resume hit, output already complete: {output}"
        )
        return output

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    logger.info(f"streaming collect: starting {len(targets)} sequential targets")
    collect_calibration_statistics(
        model_factory=model_factory,
        checkpoint=str(checkpoint),
        artifact_dir=str(artifact_dir),
        calibration_batches=calibration_batches,
        targets=targets,
        recipe=recipe,
        dataset_fingerprint=dataset_fingerprint,
        model_args=model_args,
        model_kwargs=model_kwargs,
        materializer=materializer,
        target_module_selector=target_module_selector,
        forward_target=forward_target,
        algorithms=algorithms,
        device=device,
        target_dtype=target_dtype,
        num_samples=num_samples,
        max_seq_length=max_seq_length,
        seed=seed,
    )
    logger.info(
        f"streaming collect: completed in {time.monotonic() - started:.2f}s; "
        f"{_resource_summary(device, artifact_dir)}"
    )

    started = time.monotonic()
    logger.info("streaming quantize: starting output staging shards")
    quantize_streaming(
        checkpoint=checkpoint,
        artifact_dir=artifact_dir,
        staging_dir=staging_dir,
        schemes=schemes,
        use_gptq=use_gptq,
        materializer=materializer,
        device=device,
        target_dtype=target_dtype,
        blocksize=blocksize,
        dampening_frac=dampening_frac,
    )
    logger.info(
        f"streaming quantize: completed in {time.monotonic() - started:.2f}s; "
        f"{_resource_summary(device, staging_dir)}"
    )

    formats = {
        scheme.format or infer_module_format(nn.Linear, scheme).value
        for scheme in schemes.values()
    }
    config_groups: dict[str, QuantizationScheme] = {}
    for index, (module_name, scheme) in enumerate(schemes.items()):
        serialized = scheme.model_dump()
        serialized["targets"] = [module_name]
        config_groups[f"group_{index}"] = QuantizationScheme.model_validate(
            serialized
        )
    qconfig = QuantizationConfig(
        config_groups=config_groups,
        format=next(iter(formats)) if len(formats) == 1 else "mixed-precision",
        quantization_status="compressed",
    ).model_dump(mode="json")
    started = time.monotonic()
    logger.info("streaming finalize: publishing checkpoint")
    result = finalize_streaming_checkpoint(
        checkpoint=checkpoint,
        artifact_dir=artifact_dir,
        staging_dir=staging_dir,
        output_dir=output,
        quantization_config=qconfig,
        validate_config=validate_config,
    )
    logger.info(
        f"streaming finalize: completed in {time.monotonic() - started:.2f}s; "
        f"{_resource_summary(device, result)}"
    )
    return result


def streaming_oneshot(
    *,
    model: str | Path | None = None,
    dataset: Any = None,
    recipe: Any = None,
    output_dir: str | Path,
    work_dir: str | Path,
    num_calibration_samples: int = 512,
    max_seq_length: int | None = None,
    batch_size: int = 1,
    shuffle_calibration_samples: bool = True,
    splits: str | None = None,
    preprocessing_func: Callable[[Any], Any] | None = None,
    tokenizer: Any = None,
    dataset_fingerprint: str | None = None,
    device: torch.device | str = "cpu",
    target_dtype: torch.dtype = torch.bfloat16,
    # Advanced boundary-mode arguments. These keep the low-level API available
    # for model adapters while normal callers use ``model`` and ``dataset``.
    model_factory: Callable[..., nn.Module] | None = None,
    checkpoint: str | Path | None = None,
    calibration_batches: Iterable[Any] | Callable[[], Iterable[Any]] | None = None,
    targets: Sequence[str] | None = None,
    schemes: Mapping[str, QuantizationScheme] | None = None,
    model_args: Sequence[Any] = (),
    model_kwargs: Mapping[str, Any] | None = None,
    materializer: WeightMaterializer | None = None,
    target_module_selector: Callable[[nn.Module, str], Mapping[str, nn.Module]]
    | None = None,
    forward_target: Callable[[nn.Module, Any], Any] | None = None,
    algorithms: Sequence[str] = ("gptq", "imatrix"),
    use_gptq: bool = True,
    blocksize: int = 128,
    dampening_frac: float = 0.01,
    seed: int | None = None,
    validate_config: bool = True,
) -> Path:
    """Run resumable streaming PTQ.

    Normal callers provide a local Hugging Face ``model``, a calibration
    ``dataset``, and a standard quantization recipe. The sequential tracer derives
    model-prefix and decoder-target boundaries without loading the full model.
    Advanced callers may instead supply explicit boundary-mode arguments.
    """
    if model is not None:
        if checkpoint is not None or calibration_batches is not None:
            raise ValueError(
                "Pass either model/dataset or checkpoint/calibration_batches, not both"
            )
        from .pretrained import streaming_oneshot_from_pretrained

        return streaming_oneshot_from_pretrained(
            model=model,
            dataset=dataset,
            recipe=recipe,
            output_dir=output_dir,
            work_dir=work_dir,
            num_calibration_samples=num_calibration_samples,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            shuffle_calibration_samples=shuffle_calibration_samples,
            splits=splits,
            preprocessing_func=preprocessing_func,
            tokenizer=tokenizer,
            dataset_fingerprint=dataset_fingerprint,
            device=device,
            target_dtype=target_dtype,
            blocksize=blocksize,
            dampening_frac=dampening_frac,
            seed=seed,
            validate_config=validate_config,
        )

    required = {
        "model_factory": model_factory,
        "checkpoint": checkpoint,
        "calibration_batches": calibration_batches,
        "targets": targets,
        "dataset_fingerprint": dataset_fingerprint,
        "schemes": schemes,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise TypeError(
            "Boundary mode requires explicit arguments: " + ", ".join(missing)
        )
    return _streaming_oneshot_from_boundaries(
        model_factory=model_factory,
        checkpoint=checkpoint,
        output_dir=output_dir,
        work_dir=work_dir,
        calibration_batches=calibration_batches,
        targets=targets,
        recipe=recipe,
        dataset_fingerprint=dataset_fingerprint,
        schemes=schemes,
        model_args=model_args,
        model_kwargs=model_kwargs,
        materializer=materializer,
        target_module_selector=target_module_selector,
        forward_target=forward_target,
        algorithms=algorithms,
        use_gptq=use_gptq,
        device=device,
        target_dtype=target_dtype,
        blocksize=blocksize,
        dampening_frac=dampening_frac,
        num_samples=num_calibration_samples,
        max_seq_length=max_seq_length,
        seed=seed,
        validate_config=validate_config,
    )
