"""Convenience frontend for supported pretrained streaming PTQ models."""

from __future__ import annotations

from collections.abc import Callable
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
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.recipe import Recipe

from .artifacts import fingerprint_json
from .checkpoint import SafetensorsWeightSource
from .loading import build_meta_model
from .oneshot import _streaming_oneshot_from_boundaries
from .tracing import trace_streaming_boundaries

__all__ = ["streaming_oneshot_from_pretrained"]


def _recipe_quantizer(recipe: Recipe):
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
    unsupported = [
        type(modifier).__name__
        for modifier in recipe.modifiers
        if modifier is not quantizers[0]
        and type(modifier).__name__ != "IMatrixGatherer"
    ]
    if unsupported:
        raise ValueError(
            "Pretrained streaming mode only supports IMatrixGatherer plus one "
            f"quantizer; got {unsupported}"
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
    dataset: Any,
    recipe: Any,
    output_dir: str | Path,
    work_dir: str | Path,
    num_calibration_samples: int,
    max_seq_length: int | None,
    batch_size: int,
    shuffle_calibration_samples: bool,
    splits: str | None,
    preprocessing_func: Callable[[Any], Any] | None,
    tokenizer: Any,
    dataset_fingerprint: str | None,
    device: torch.device | str,
    target_dtype: torch.dtype,
    blocksize: int,
    dampening_frac: float,
    seed: int | None,
    validate_config: bool,
) -> Path:
    """Run traced streaming PTQ with an oneshot-like model/dataset interface."""
    checkpoint = Path(model).expanduser()
    if not checkpoint.is_dir():
        raise ValueError("Pretrained streaming mode requires a local model directory")
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
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
    quantizer = _recipe_quantizer(parsed_recipe)
    meta_model = build_meta_model(
        AutoModelForCausalLM.from_config,
        config,
        attn_implementation="eager",
        keep_nonpersistent_buffers=True,
    )
    schemes = _exact_schemes(meta_model, quantizer)
    targets = tuple(
        f"model.layers.{index}" for index in range(config.num_hidden_layers)
    )
    normalized_recipe = parsed_recipe.model_dump(mode="json")
    uses_imatrix = any(
        scheme.weights is not None and scheme.weights.observer == "imatrix_mse"
        for scheme in schemes.values()
    )
    use_gptq = isinstance(quantizer, GPTQModifier)
    if not use_gptq and not uses_imatrix:
        raise ValueError("RTN streaming PTQ requires weights.observer='imatrix_mse'")
    algorithms = tuple(
        name
        for name, enabled in (("gptq", use_gptq), ("imatrix", uses_imatrix))
        if enabled
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
    adapter = trace_streaming_boundaries(
        model=meta_model,
        source=SafetensorsWeightSource(checkpoint),
        sample_batch=sample_batch,
        sequential_targets=tuple(
            getattr(config, "_no_split_modules", ())
            or getattr(meta_model, "_no_split_modules", ())
        ),
        target_names=targets,
        device=device,
        dtype=target_dtype,
    )

    def boundaries():
        return adapter.calibration_boundaries(dataloader)

    return _streaming_oneshot_from_boundaries(
        model_factory=AutoModelForCausalLM.from_config,
        checkpoint=checkpoint,
        output_dir=output_dir,
        work_dir=work_dir,
        calibration_batches=boundaries,
        targets=targets,
        recipe=normalized_recipe,
        dataset_fingerprint=fingerprint,
        schemes=schemes,
        model_args=(config,),
        model_kwargs={"attn_implementation": "eager"},
        forward_target=adapter.forward_target,
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
