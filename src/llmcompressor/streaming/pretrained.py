"""Convenience frontend for supported pretrained streaming PTQ models."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.utils import match_named_modules
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llmcompressor.args import DatasetArguments
from llmcompressor.datasets import get_calibration_dataloader
from llmcompressor.modeling.moe.context import moe_calibration_context
from llmcompressor.modeling.moe_context import (
    moe_calibration_context as moe_module_replacement_context,
)
from llmcompressor.modeling.offset_norm import norm_calibration_context
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.recipe import Recipe

from .artifacts import fingerprint_json
from .finalize import finalize_streaming_checkpoint
from .loading import build_meta_model
from .materialization import CastWeightMaterializer, WeightMaterializer
from .output import build_quantization_config
from .pipeline import run_subgraph_streaming_pipeline
from .tracing import trace_streaming_boundaries

__all__ = ["streaming_oneshot_from_pretrained"]


def _recipe_quantizer(recipe: Recipe):
    if any(
        type(modifier).__name__ == "AutoRoundModifier"
        for modifier in recipe.modifiers
    ):
        raise ValueError(
            "Pretrained streaming mode does not yet support AutoRound: its "
            "modifier owns quantization and checkpoint compression, which needs "
            "a dedicated streaming output adapter"
        )
    quantizers = [
        modifier
        for modifier in recipe.modifiers
        if isinstance(modifier, (QuantizationModifier, GPTQModifier))
    ]
    if len(quantizers) != 1:
        raise ValueError(
            "Pretrained streaming mode requires exactly one "
            "QuantizationModifier or GPTQModifier"
        )
    supported_transforms = {
        "AWQModifier",
        "IMatrixGatherer",
        "SparseGPTModifier",
        "SmoothQuantModifier",
        "WandaPruningModifier",
    }
    unsupported = [
        type(modifier).__name__
        for modifier in recipe.modifiers
        if modifier is not quantizers[0]
        and type(modifier).__name__ not in supported_transforms
    ]
    if unsupported:
        raise ValueError(
            "Pretrained streaming mode supports AWQ, SmoothQuant, iMatrix, "
            "SparseGPT, or Wanda plus one quantizer; "
            f"got unsupported modifiers {unsupported}"
        )
    return quantizers[0]


def _exact_schemes(
    model: nn.Module, quantizer: QuantizationModifier | GPTQModifier
) -> dict[str, QuantizationScheme]:
    config = quantizer.resolved_config
    if config.kv_cache_scheme is not None:
        raise ValueError("Streaming PTQ does not support KV-cache quantization")
    schemes = {}
    for scheme in config.config_groups.values():
        for name, module in match_named_modules(model, scheme.targets, config.ignore):
            if not isinstance(module, nn.Linear):
                raise ValueError(
                    "Pretrained streaming mode supports dense Linear targets only; "
                    f"got {type(module).__name__} at {name!r}"
                )
            if name in schemes:
                raise ValueError(
                    f"Overlapping quantization schemes match {name!r}"
                )
            exact = deepcopy(scheme)
            exact.targets = [name]
            schemes[name] = exact
    if not schemes:
        raise ValueError("Recipe does not match any Linear modules")
    return schemes


def _dataset_fingerprint(
    dataset: Any,
    *,
    dataset_fingerprint: str | None,
    splits: str | None,
    num_calibration_samples: int,
    max_seq_length: int | None,
    seed: int | None,
    shuffle_calibration_samples: bool,
) -> str:
    if dataset_fingerprint is not None:
        return dataset_fingerprint
    source = dataset
    if isinstance(dataset, DataLoader):
        source = dataset.dataset
    value = getattr(source, "_fingerprint", None)
    if value is None:
        raise ValueError(
            "dataset_fingerprint is required when dataset has no Hugging Face "
            "dataset fingerprint"
        )
    return fingerprint_json(
        {
            "dataset": value,
            "splits": splits,
            "num_calibration_samples": num_calibration_samples,
            "max_seq_length": max_seq_length,
            "seed": seed,
            "shuffle_calibration_samples": shuffle_calibration_samples,
        }
    )


def _prepare_messages_dataset(dataset: Any, tokenizer: Any) -> Any:
    """Apply the standard chat template when callers pass a messages dataset."""
    columns = getattr(dataset, "column_names", ())
    if "messages" not in columns or "input_ids" in columns:
        return dataset
    if tokenizer is None:
        raise ValueError("A tokenizer is required for a messages calibration dataset")
    return dataset.map(
        lambda sample: {
            "text": tokenizer.apply_chat_template(sample["messages"], tokenize=False)
        }
    )


def streaming_oneshot_from_pretrained(
    *,
    model: str | Path,
    model_config: Any,
    dataset: Any,
    recipe: Any,
    output_dir: str | Path,
    work_dir: str | Path,
    num_calibration_samples: int,
    max_seq_length: int | None,
    batch_size: int,
    shuffle_calibration_samples: bool,
    moe_calibrate_all_experts: bool,
    splits: str | None,
    preprocessing_func: Callable[[Any], Any] | None,
    tokenizer: Any,
    dataset_fingerprint: str | None,
    device: torch.device | str,
    target_dtype: torch.dtype,
    materializer: WeightMaterializer | None,
    blocksize: int,
    dampening_frac: float,
    seed: int | None,
    validate_config: bool,
    checkpoint_progress: bool,
    overwrite_output: bool,
) -> Path:
    """Run traced streaming PTQ with an oneshot-like model/dataset interface."""
    checkpoint = Path(model).expanduser()
    output = Path(output_dir).expanduser()
    work = Path(work_dir).expanduser()
    resolved_output = output.resolve()
    resolved_work = work.resolve()
    if resolved_work.is_relative_to(resolved_output):
        raise ValueError(
            "work_dir must not be inside output_dir; use sibling directories so "
            "working artifacts do not make output_dir appear pre-existing"
        )
    if resolved_output.is_relative_to(resolved_work):
        raise ValueError(
            "output_dir must not be inside work_dir; use sibling directories"
        )
    work.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint_progress and work.stat().st_dev != output.parent.stat().st_dev:
        raise OSError(
            "work_dir and output_dir must be on the same filesystem for "
            "copy-free publication"
        )
    if (
        output.exists()
        and (output / "FINALIZED").is_file()
        and not overwrite_output
    ):
        return output
    if output.exists() and any(output.iterdir()) and not overwrite_output:
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output}. Pass "
            "overwrite_output=True to replace it."
        )
    publish = work / "publish"
    if not checkpoint_progress and publish.exists():
        shutil.rmtree(publish)
    if not checkpoint.is_dir():
        raise ValueError("Pretrained streaming mode requires a local model directory")
    config = model_config or AutoConfig.from_pretrained(
        checkpoint, local_files_only=True
    )
    materializer = materializer or CastWeightMaterializer()
    if tokenizer is None and not isinstance(dataset, DataLoader):
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    if preprocessing_func is None and not isinstance(dataset, DataLoader):
        dataset = _prepare_messages_dataset(dataset, tokenizer)
    dataset_args = DatasetArguments(
        dataset=dataset,
        splits=splits,
        preprocessing_func=preprocessing_func,
        num_calibration_samples=num_calibration_samples,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        shuffle_calibration_samples=shuffle_calibration_samples,
    )
    dataloader = get_calibration_dataloader(dataset_args, tokenizer)
    if dataloader is None:
        raise ValueError("Streaming PTQ requires calibration data")

    parsed_recipe = Recipe.create_instance(recipe)
    for modifier in parsed_recipe.modifiers:
        if type(modifier).__name__ == "IMatrixGatherer":
            modifier.attach_by_initialize = False
    quantizer = _recipe_quantizer(parsed_recipe)
    meta_model = build_meta_model(
        AutoModelForCausalLM.from_config,
        config,
        keep_nonpersistent_buffers=True,
    )
    schemes = _exact_schemes(meta_model, quantizer)
    uses_imatrix = any(
        scheme.weights is not None and scheme.weights.observer == "imatrix_mse"
        for scheme in schemes.values()
    )
    uses_calibration_transform = any(
        type(modifier).__name__
        in {
            "AWQModifier",
            "SmoothQuantModifier",
            "SparseGPTModifier",
            "WandaPruningModifier",
        }
        for modifier in parsed_recipe.modifiers
    )
    use_gptq = isinstance(quantizer, GPTQModifier)
    if not use_gptq and not uses_imatrix and not uses_calibration_transform:
        raise ValueError(
            "RTN streaming PTQ requires a calibration modifier or "
            "weights.observer='imatrix_mse'"
        )
    device = torch.device(device)
    fingerprint = _dataset_fingerprint(
        dataset,
        dataset_fingerprint=dataset_fingerprint,
        splits=splits,
        num_calibration_samples=num_calibration_samples,
        max_seq_length=max_seq_length,
        seed=seed,
        shuffle_calibration_samples=shuffle_calibration_samples,
    )

    try:
        sample_batch = next(iter(dataloader))
    except StopIteration as error:
        raise ValueError("Streaming PTQ calibration dataset is empty") from error
    with ExitStack() as stack:
        stack.enter_context(norm_calibration_context(meta_model))
        if moe_calibrate_all_experts:
            stack.enter_context(moe_calibration_context())
        stack.enter_context(
            moe_module_replacement_context(
                meta_model,
                calibrate_all_experts=moe_calibrate_all_experts,
            )
        )
        adapter = trace_streaming_boundaries(
            model=meta_model,
            source=materializer.create_source(str(checkpoint)),
            sample_batch=sample_batch,
            sequential_targets=tuple(
                getattr(meta_model, "_no_split_modules", ())
                or getattr(config, "_no_split_modules", ())
            ),
            materializer=materializer,
            device=device,
            dtype=target_dtype,
        )
        targets = adapter.targets
        covered = tuple(
            name
            for name in schemes
            if any(
                name == target or name.startswith(f"{target}.")
                for target in targets
            )
        )
        missing = sorted(set(schemes) - set(covered))
        if missing:
            raise ValueError(
                "Recipe matches modules outside the inferred sequential targets: "
                f"{missing}"
            )

        artifact_dir, staging_dir = run_subgraph_streaming_pipeline(
            adapter=adapter,
            checkpoint=checkpoint,
            work_dir=work_dir,
            calibration_batches=dataloader,
            recipe=parsed_recipe,
            dataset_fingerprint=fingerprint,
            materializer=materializer,
            device=device,
            target_dtype=target_dtype,
            num_samples=num_calibration_samples,
            max_seq_length=max_seq_length,
            seed=seed,
            checkpoint_progress=checkpoint_progress,
        )
        qconfig = build_quantization_config(quantizer.resolved_config)
        return finalize_streaming_checkpoint(
            checkpoint=checkpoint,
            artifact_dir=artifact_dir,
            staging_dir=staging_dir,
            output_dir=output_dir,
            quantization_config=qconfig,
            materializer=materializer,
            validate_config=validate_config,
            cleanup_staging=False,
            publish_in_place=not checkpoint_progress,
            overwrite_output=overwrite_output,
            recipe_yaml=parsed_recipe.yaml(),
        )
