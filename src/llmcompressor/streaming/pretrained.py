"""Convenience frontend for supported pretrained streaming PTQ models."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from compressed_tensors.quantization import QuantizationScheme

try:
    from compressed_tensors.quantization.lifecycle.initialize import (
        _is_attention_module as _compressed_tensors_is_attention_module,
    )
except ImportError:
    _compressed_tensors_is_attention_module = None
from compressed_tensors.utils import match_named_modules
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llmcompressor.args import DatasetArguments
from llmcompressor.datasets import get_calibration_dataloader
from llmcompressor.modeling.moe.context import moe_calibration_context
from llmcompressor.modeling.moe.linearize import (
    get_non_linearized_moes,
    linearize_moe,
)
from llmcompressor.modeling.moe_context import (
    moe_calibration_context as moe_module_replacement_context,
)
from llmcompressor.modeling.offset_norm import norm_calibration_context
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.recipe import Recipe
from llmcompressor.utils.pytorch.module import (
    infer_sequential_targets,
    iter_module_ancestors,
)

from ._cleanup import cleanup_successful_work_dir as _cleanup_successful_work_dir
from .artifacts import fingerprint_json
from .checkpoint import with_work_directory_lock
from .execution import freeze_calibration_inputs
from .finalize import finalize_streaming_checkpoint
from .loading import build_meta_model
from .materialization import (
    CastWeightMaterializer,
    StreamingDTypePolicy,
    WeightMaterializer,
    apply_transformers_weight_conversions,
)
from .modifier_adapters import get_modifier_adapter, validate_streaming_modifiers
from .mtp import StreamingMTPWrapper
from .output import build_quantization_config
from .parallel.expert import ExpertParallelPlan, resolve_expert_parallel_devices
from .parallel.token_budget import validate_token_budget
from .pipeline import (
    _complete_direct_writer_staging,
    _initialize_run,
    run_subgraph_streaming_pipeline,
)
from .propagation import resolve_propagate_error
from .tracing import trace_streaming_boundaries

__all__ = ["streaming_oneshot_from_pretrained"]


def _recipe_quantizer(recipe: Recipe):
    quantizers = [
        modifier
        for modifier in recipe.modifiers
        if isinstance(modifier, (QuantizationModifier, GPTQModifier))
        or (
            (adapter := get_modifier_adapter(modifier)) is not None
            and adapter.owns_quantization
        )
    ]
    validate_streaming_modifiers(
        recipe.modifiers, quantizers[0] if len(quantizers) == 1 else None
    )
    if len(quantizers) != 1:
        raise ValueError(
            "Pretrained streaming mode requires exactly one "
            "QuantizationModifier, GPTQModifier, or adapted AutoRoundModifier"
        )
    supported_transforms = {
        "AWQModifier",
        "SmoothQuantModifier",
    }
    unsupported = [
        type(modifier).__name__
        for modifier in recipe.modifiers
        if modifier is not quantizers[0]
        and type(modifier).__name__ not in supported_transforms
        and get_modifier_adapter(modifier) is None
    ]
    if unsupported:
        raise ValueError(
            "Pretrained streaming mode supports AWQ, SmoothQuant, or iMatrix "
            "plus one quantizer; "
            f"got unsupported modifiers {unsupported}"
        )
    return quantizers[0]


def _exact_schemes(
    model: nn.Module,
    quantizer: QuantizationModifier | GPTQModifier,
    *,
    excluded_prefixes: Sequence[str] = (),
) -> dict[str, QuantizationScheme]:
    config = quantizer.resolved_config
    if config.kv_cache_scheme is not None:
        raise ValueError("Streaming PTQ does not support KV-cache quantization")
    schemes = {}
    for scheme in config.config_groups.values():
        for name, module in match_named_modules(model, scheme.targets, config.ignore):
            if any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in excluded_prefixes
            ):
                continue
            if not isinstance(module, nn.Linear):
                raise ValueError(
                    "Pretrained streaming mode supports dense Linear targets only; "
                    f"got {type(module).__name__} at {name!r}"
                )
            if name in schemes:
                raise ValueError(f"Overlapping quantization schemes match {name!r}")
            exact = deepcopy(scheme)
            exact.targets = [name]
            schemes[name] = exact
    if not schemes:
        raise ValueError("Recipe does not match any Linear modules")
    return schemes


def _uncovered_scheme_names(
    schemes: dict[str, QuantizationScheme], targets: Sequence[str]
) -> list[str]:
    target_names = set(targets)
    return [
        name
        for name in schemes
        if name not in target_names
        and not any(parent in target_names for parent in iter_module_ancestors(name))
    ]


def _attention_ancestor(model: nn.Module, module_name: str) -> str | None:
    for parent in iter_module_ancestors(module_name):
        if _is_streaming_attention_module(model.get_submodule(parent)):
            return parent
    return None


def _is_streaming_attention_module(module: nn.Module) -> bool:
    """Return whether a module should be a tracing/quantization boundary.

    The compressed-tensors helper is the primary source of truth for standard
    Transformers attention implementations. Model adapters can opt in with the
    stable ``_streaming_attention_boundary`` marker, which is needed by native
    attention implementations that do not inherit a Transformers attention
    class. The class-name fallback keeps custom ``*Attention`` implementations
    compatible with older model adapters while the marker remains the preferred
    extension point.
    """
    if getattr(module, "_streaming_attention_boundary", False):
        return True
    if (
        _compressed_tensors_is_attention_module is not None
        and _compressed_tensors_is_attention_module(module)
    ):
        return True
    return module.__class__.__name__.endswith("Attention")


def _is_expert_path(module_name: str) -> bool:
    return any("expert" in part.lower() for part in module_name.split("."))


def _expert_owner(
    model: nn.Module,
    module_name: str,
    expert_descendant_counts: dict[str, int],
) -> str | None:
    current = module_name
    while current:
        if expert_descendant_counts.get(current, 0) > 1:
            module = model.get_submodule(current)
            if not isinstance(module, (nn.ModuleList, nn.ModuleDict)):
                return current
        current, _, _ = current.rpartition(".")
    return None


def _group_expert_targets(
    model: nn.Module,
    candidate_names: set[str],
) -> tuple[set[str], int]:
    expert_descendant_counts: dict[str, int] = {}
    for name in candidate_names:
        current = name
        while current:
            if _is_expert_path(current):
                expert_descendant_counts[current] = (
                    expert_descendant_counts.get(current, 0) + 1
                )
            current, _, _ = current.rpartition(".")
    grouped = set()
    owners = set()
    for name in candidate_names:
        owner = _expert_owner(model, name, expert_descendant_counts)
        if owner is None:
            grouped.add(name)
        else:
            owners.add(owner)
    grouped.update(owners)
    return grouped, len(owners)


def _has_expert_collection(model: nn.Module) -> bool:
    """Return whether the model exposes multiple routed expert modules."""
    for name, module in model.named_modules():
        if "expert" not in name.lower():
            continue
        if isinstance(module, (nn.ModuleList, nn.ModuleDict)) and len(module) > 1:
            return True
        count = getattr(module, "num_experts", None)
        if isinstance(count, int) and count > 1:
            return True
    return False


def _quantized_sequential_targets(
    model: nn.Module,
    schemes: dict[str, QuantizationScheme],
    sequential_targets: Sequence[str] | str | None = None,
    group_moe_experts: bool = False,
) -> tuple[str, ...]:
    """Select trace targets which cover every exact quantization target.

    The fine-grained GPTQ policy uses the same target semantics as the oneshot
    sequential pipeline's ``sequential_targets="Linear"`` mode. This is
    important for MoE layers: keeping a whole decoder block alive until
    ``sequential_epoch_end`` retains one Hessian per expert and can exceed
    device memory before GPTQ gets a chance to compress anything.

    Attention modules are retained as a single target when the sequential tracer
    treats them as leaves. Their nested quantized linears are then compressed at
    the attention boundary, matching oneshot's tracing behavior.
    """
    patterns = infer_sequential_targets(model, sequential_targets)
    if isinstance(patterns, str):
        patterns = [patterns]
    scheme_paths = set(schemes)
    for name in schemes:
        scheme_paths.update(iter_module_ancestors(name))
    candidates = {
        name
        for name, _ in match_named_modules(model, patterns)
        if name in scheme_paths
    }

    promoted = set(candidates)
    for name in candidates:
        ancestor = _attention_ancestor(model, name)
        if ancestor is not None:
            promoted.discard(name)
            promoted.add(ancestor)

    if group_moe_experts:
        promoted, expert_owner_count = _group_expert_targets(model, promoted)
        if expert_owner_count:
            logger.info(
                "Grouped quantized MoE modules into "
                f"{expert_owner_count} expert-owner target(s)"
            )

    selected = []
    selected_names = set()
    for name, _ in model.named_modules():
        if name not in promoted:
            continue
        if any(parent in selected_names for parent in iter_module_ancestors(name)):
            continue
        selected.append(name)
        selected_names.add(name)

    missing = _uncovered_scheme_names(schemes, selected)
    if missing:
        raise ValueError(
            "Recipe matches modules outside the inferred sequential targets: "
            f"{sorted(missing)}"
        )
    return tuple(selected)


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


@with_work_directory_lock
def streaming_oneshot_from_pretrained(
    *,
    model: str | Path,
    model_config: Any,
    model_factory: Callable[..., nn.Module] | None = None,
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
    async_save: bool | None,
    pack_to_int8: bool,
    finalize_only: bool = False,
    dtype_policy: StreamingDTypePolicy | None = None,
    sequential_targets: Sequence[str] | str | None = None,
    sequential_targets_per_subgraph: int = 1,
    expert_parallel_devices: Sequence[torch.device | str] | None = None,
    expert_parallel_token_budget: str | int = "auto",
    processor_identity: Any = None,
    propagate_error: bool | None = None,
) -> Path:
    """Run traced streaming PTQ with an oneshot-like model/dataset interface."""
    validate_token_budget(expert_parallel_token_budget)
    if expert_parallel_devices is None and expert_parallel_token_budget != "auto":
        raise ValueError(
            "expert_parallel_token_budget requires expert_parallel_devices"
        )
    if (
        not isinstance(sequential_targets_per_subgraph, int)
        or isinstance(sequential_targets_per_subgraph, bool)
        or sequential_targets_per_subgraph < 1
    ):
        raise ValueError(
            "sequential_targets_per_subgraph must be a positive integer"
        )
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
    resolved_checkpoint = checkpoint.resolve()
    if resolved_work.is_relative_to(
        resolved_checkpoint
    ) or resolved_checkpoint.is_relative_to(resolved_work):
        raise ValueError("work_dir and checkpoint must use separate directories")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and (output / "FINALIZED").is_file() and not overwrite_output:
        return output
    if output.exists() and any(output.iterdir()) and not overwrite_output:
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output}. Pass "
            "overwrite_output=True to replace it."
        )
    work.mkdir(parents=True, exist_ok=True)
    if not checkpoint_progress and work.stat().st_dev != output.parent.stat().st_dev:
        raise OSError(
            "work_dir and output_dir must be on the same filesystem for "
            "copy-free publication"
        )
    publish = work / "publish"
    if finalize_only and checkpoint_progress:
        raise ValueError(
            "finalize_only is only supported for checkpoint_progress=False"
        )
    if not checkpoint.is_dir():
        raise ValueError("Pretrained streaming mode requires a local model directory")
    parsed_recipe = Recipe.create_instance(recipe)
    resolve_propagate_error(parsed_recipe.modifiers, propagate_error)
    quantizer = _recipe_quantizer(parsed_recipe)
    ep_devices = resolve_expert_parallel_devices(
        expert_parallel_devices, torch.device(device)
    )
    if ep_devices and (
        len(parsed_recipe.modifiers) != 1
        or type(quantizer) not in (QuantizationModifier, GPTQModifier)
    ):
        raise ValueError(
            "Expert parallel streaming currently supports one GPTQModifier or "
            "QuantizationModifier; other modifier combinations require an EP adapter"
        )
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

    factory = model_factory or AutoModelForCausalLM.from_config
    meta_model = build_meta_model(
        factory,
        config,
        keep_nonpersistent_buffers=True,
    )
    meta_model.eval()
    # Streaming PTQ is inference-only. Freeze the meta parameters before any
    # materialization so installed/restored weights never carry autograd state.
    meta_model.requires_grad_(False)
    model_was_linearized = len(get_non_linearized_moes(meta_model)) > 0
    if model_was_linearized:
        linearize_moe(meta_model)

    mtp_wrapper = StreamingMTPWrapper.attach(meta_model)
    schemes = _exact_schemes(
        meta_model,
        quantizer,
        excluded_prefixes=("mtp",) if mtp_wrapper.attached else (),
    )
    dtype_policy = dtype_policy or StreamingDTypePolicy.for_quantized_modules(
        target_dtype, schemes
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

    dataloader = freeze_calibration_inputs(dataloader, work / "inputs")
    try:
        sample_batch = next(iter(dataloader))
    except StopIteration as error:
        raise ValueError("Streaming PTQ calibration dataset is empty") from error

    with ExitStack() as stack:
        stack.enter_context(norm_calibration_context(meta_model))
        if moe_calibrate_all_experts:
            stack.enter_context(moe_calibration_context())
        if not model_was_linearized:
            stack.enter_context(
                moe_module_replacement_context(
                    meta_model,
                    calibrate_all_experts=moe_calibrate_all_experts,
                )
            )
        materializer = apply_transformers_weight_conversions(
            materializer,
            meta_model,
            extra_conversions=mtp_wrapper.weight_conversions,
        )
        target_patterns = sequential_targets
        automatic_expert_targets = (
            target_patterns is None
            and _has_expert_collection(meta_model)
            and (
                isinstance(quantizer, GPTQModifier)
                or (
                    bool(ep_devices)
                    and isinstance(quantizer, QuantizationModifier)
                )
            )
        )
        if automatic_expert_targets:
            target_patterns = ["Linear"]
        target_started = perf_counter()
        sequential_targets = _quantized_sequential_targets(
            meta_model,
            schemes,
            sequential_targets=target_patterns,
            group_moe_experts=automatic_expert_targets,
        )
        if automatic_expert_targets and sequential_targets_per_subgraph == 1:
            sequential_targets_per_subgraph = 8 * (len(ep_devices) or 1)
            grouping = (
                "Automatic GPTQ MoE target grouping"
                if isinstance(quantizer, GPTQModifier)
                else "Automatic RTN MoE target grouping"
            )
            logger.info(
                f"{grouping}: "
                f"using up to {sequential_targets_per_subgraph} consecutive "
                "expert-owner targets per subgraph"
            )
        if automatic_expert_targets:
            target_mode = (
                "automatic GPTQ Linear"
                if isinstance(quantizer, GPTQModifier)
                else "automatic RTN Linear"
            )
        elif target_patterns is not None:
            target_mode = "explicit"
        else:
            target_mode = "inferred"
        logger.info(
            "Streaming sequential targets: "
            f"{len(sequential_targets)} target(s) ({target_mode}) | "
            f"selection_s={perf_counter() - target_started:.2f}"
        )
        trace_started = perf_counter()
        adapter = trace_streaming_boundaries(
            model=meta_model,
            source=materializer.create_source(str(checkpoint)),
            sample_batch=sample_batch,
            sequential_targets=sequential_targets,
            target_names=sequential_targets,
            materializer=materializer,
            dtype_policy=dtype_policy,
            device=device,
            dtype=target_dtype,
            tracing_ignore=dataset_args.tracing_ignore,
            targets_per_subgraph=sequential_targets_per_subgraph,
        )
        logger.info(
            "Streaming sequential trace ready | "
            f"trace_s={perf_counter() - trace_started:.2f}"
        )
        expert_parallel = None
        if ep_devices:
            placement_started = perf_counter()
            expert_parallel = ExpertParallelPlan.from_model(
                meta_model, adapter.targets, ep_devices
            )
            logger.info(
                "Streaming EP placement ready | "
                f"placement_s={perf_counter() - placement_started:.2f}"
            )
        adapter.execution = adapter.execution.with_inputs(
            dataloader.fingerprint,
            {
                "declared": processor_identity,
                "processor_files": {
                    name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
                    for name in (
                        "preprocessor_config.json",
                        "processor_config.json",
                        "tokenizer_config.json",
                        "special_tokens_map.json",
                    )
                    if (checkpoint / name).is_file()
                },
                "tokenizer": (
                    {
                        "class": (
                            f"{type(tokenizer).__module__}."
                            f"{type(tokenizer).__qualname__}"
                        ),
                        "vocab": tokenizer.get_vocab(),
                        "special_tokens": tokenizer.special_tokens_map,
                        "chat_template": getattr(tokenizer, "chat_template", None),
                    }
                    if tokenizer is not None
                    and callable(getattr(tokenizer, "get_vocab", None))
                    else None
                ),
            },
        )
        missing = _uncovered_scheme_names(schemes, adapter.targets)
        if missing:
            raise ValueError(
                "Recipe matches modules outside the inferred sequential targets: "
                f"{sorted(missing)}"
            )

        if finalize_only:
            if not publish.is_dir():
                raise FileNotFoundError(
                    f"No completed direct-writer staging directory: {publish}"
                )
            artifact_dir = work / "artifacts"
            staging_dir = publish
            run_fingerprint = _initialize_run(
                checkpoint=checkpoint,
                artifact_dir=artifact_dir,
                recipe=parsed_recipe,
                dataset_fingerprint=fingerprint,
                targets=adapter.targets,
                targets_per_subgraph=sequential_targets_per_subgraph,
                materializer=materializer,
                dtype_policy=dtype_policy,
                target_dtype=target_dtype,
                num_samples=num_calibration_samples,
                max_seq_length=max_seq_length,
                seed=seed,
                pack_to_int8=pack_to_int8,
                replace_existing=True,
                execution=adapter.execution,
                expert_parallel=expert_parallel,
                expert_parallel_token_budget=expert_parallel_token_budget,
                target_groups=adapter.target_groups,
                propagate_error=propagate_error,
            )
            _complete_direct_writer_staging(
                adapter=adapter,
                checkpoint=checkpoint,
                publish_dir=publish,
                materializer=materializer,
                dtype_policy=dtype_policy,
                target_dtype=target_dtype,
                run_fingerprint=run_fingerprint,
            )
        else:
            if not checkpoint_progress and publish.exists():
                shutil.rmtree(publish)
            artifact_dir, staging_dir, run_fingerprint = (
                run_subgraph_streaming_pipeline(
                    adapter=adapter,
                    checkpoint=checkpoint,
                    work_dir=work_dir,
                    calibration_batches=dataloader,
                    recipe=parsed_recipe,
                    dataset_fingerprint=fingerprint,
                    materializer=materializer,
                    dtype_policy=dtype_policy,
                    device=device,
                    target_dtype=target_dtype,
                    num_samples=num_calibration_samples,
                    max_seq_length=max_seq_length,
                    seed=seed,
                    checkpoint_progress=checkpoint_progress,
                    async_save=async_save,
                    pack_to_int8=pack_to_int8,
                    targets_per_subgraph=sequential_targets_per_subgraph,
                    execution=adapter.execution,
                    expert_parallel=expert_parallel,
                    expert_parallel_token_budget=expert_parallel_token_budget,
                    propagate_error=propagate_error,
                )
            )
        qconfig = build_quantization_config(quantizer.resolved_config)
        if mtp_wrapper.attached:
            ignored = list(qconfig.get("ignore") or [])
            for pattern in mtp_wrapper.ignored_targets:
                if pattern not in ignored:
                    ignored.append(pattern)
            qconfig["ignore"] = ignored
        result = finalize_streaming_checkpoint(
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
            expected_run_fingerprint=run_fingerprint,
        )
        _cleanup_successful_work_dir(work)
        return result
