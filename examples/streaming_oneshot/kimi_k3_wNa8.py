"""Streaming Kimi K3 WNA8 (routed-expert W4, other-linear W8) PTQ.

Kimi K2.5 cannot be used as a naming template here: its experts are
``gate_proj/up_proj/down_proj`` and its MoE block has no K3 latent routed
projections.  K3 instead has ``block_sparse_moe.experts.<id>.w1/w2/w3`` stored
as MXFP4 ``weight_packed`` + ``weight_scale`` and uses SITU activation.  The
expert target set below is generated from the K3 safetensors index, so a layer
is quantized to INT4 only when its native checkpoint contains a scale tensor.

The recipe is deliberately fixed to IMatrixGatherer + QuantizationModifier
(RTN).  Activation and weight quantization are symmetric; weights are static
per-channel and activations are dynamic per-token.  ``pack_to_int8=True`` is
also fixed because K3's downstream compressed-tensors/vLLM loaders consume the
packed INT4 layout.  The model-level ``output_attn_res_proj`` remains in its
source dtype because K3 invokes it inside an autowrapped method after the
decoder loop; all decoder-layer residual projections remain INT8 targets.

Example::

    python examples/streaming_oneshot/kimi_k3_wNa8.py \
        --model-id /Users/ang/models/K3 \
        --dataset-id lmms-lab/flickr30k \
        --dataset-split test \
        --text-dataset-id HuggingFaceH4/ultrachat_200k \
        --text-dataset-split train_sft \
        --output-dir /Users/ang/models/K3-WNA8

If quantization completes but publication fails, repeat the same command with
``--finalize-only``. Existing target shards are fingerprint-checked and reused;
missing static tensors are copied without rerunning quantization.

The K3 remote model code requires its normal runtime dependencies (including
``tiktoken``, ``einops``, ``fla-core``, ``flash-attn`` and ``torchvision``).
This repository does not need the full 96-shard checkpoint for the static MXFP4
decoder checks, but a complete checkpoint is required to run the end-to-end
streaming job.
"""

from __future__ import annotations

import argparse
import json
import re
from functools import partial
from pathlib import Path
from typing import Any

import torch
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from datasets import Dataset, load_dataset
from llmcompressor.modeling.kimi_k3 import patch_kimi_k3_transformers_compat
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.streaming import KimiK3WeightMaterializer, streaming_oneshot
from llmcompressor.utils import ImatrixFallbackStats

_EXPERT_TARGET_PATTERN = (
    r"re:^language_model\.model\.layers\.[0-9]+"
    r"\.block_sparse_moe\.experts\.[0-9]+\.w[123]$"
)
_EXPERT_TARGET = re.compile(
    r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\."
    r"(\d+)\.(w[123])$"
)
_OTHER_LINEAR_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "f_a_proj",
    "f_b_proj",
    "b_proj",
    "g_proj",
    "g_a_proj",
    "g_b_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "routed_expert_down_proj",
    "routed_expert_up_proj",
    "self_attention_res_proj",
    "mlp_res_proj",
)
_OTHER_LINEAR_RE = re.compile(
    r"^language_model\.model\.(?:.*\.)?(?:"
    + "|".join(map(re.escape, _OTHER_LINEAR_SUFFIXES))
    + r")$"
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _is_local_json(path: str) -> bool:
    candidate = Path(path).expanduser()
    return candidate.is_file() or candidate.suffix.lower() in {".json", ".jsonl"}


def _load_source(source_id: str, split: str, samples: int) -> Dataset:
    if _is_local_json(source_id):
        return load_dataset("json", data_files=source_id, split=f"train[:{samples}]")
    return load_dataset(source_id, split=f"{split}[:{samples}]")


def load_calibration_datasets(
    dataset_id: str,
    dataset_split: str,
    text_dataset_id: str | None,
    text_dataset_split: str,
    num_calibration_samples: int,
    text_calibration_samples: int,
    seed: int = 42,
) -> tuple[Dataset, list[dict[str, Any]]]:
    """Load multimodal and text calibration data from independent splits."""
    multimodal_dataset = _load_source(
        dataset_id,
        dataset_split,
        num_calibration_samples,
    ).shuffle(seed=seed)

    if text_dataset_id is None or text_calibration_samples == 0:
        return multimodal_dataset, []

    text_dataset = _load_source(
        text_dataset_id,
        text_dataset_split,
        text_calibration_samples,
    ).shuffle(seed=seed)
    return multimodal_dataset, list(text_dataset)


def _normalize_text_content(content: Any) -> list[dict[str, str]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, dict):
        content = [content]

    normalized = []
    for item in content if isinstance(content, list) else []:
        if isinstance(item, str):
            normalized.append({"type": "text", "text": item})
        elif isinstance(item, dict):
            text = item.get("text")
            if text is None and item.get("type") == "reasoning":
                text = item.get("reasoning")
            if isinstance(text, str):
                normalized.append({"type": "text", "text": text})
    return normalized


def _format_text_example(example: dict[str, Any]) -> list[dict[str, Any]]:
    if "messages" not in example:
        if isinstance(example.get("text"), str):
            return [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": example["text"]}],
                }
            ]
        raise ValueError("Text calibration examples must contain 'messages' or 'text'")

    messages = []
    for message in example["messages"]:
        if not isinstance(message, dict) or "role" not in message:
            raise ValueError("Text calibration messages must contain a role")
        content = _normalize_text_content(message.get("content", ""))
        if content:
            messages.append({"role": message["role"], "content": content})
    if not messages:
        raise ValueError("Text calibration example contains no usable messages")
    return messages


def _build_multimodal_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    if "image" not in example or "caption" not in example:
        raise ValueError(
            "Multimodal calibration examples must contain 'image' and 'caption'"
        )
    caption = example["caption"]
    if isinstance(caption, (list, tuple)):
        if not caption:
            raise ValueError("Multimodal calibration example has an empty caption")
        caption = caption[0]
    if not isinstance(caption, str):
        raise ValueError("Multimodal calibration caption must be a string")

    return [
        {
            "role": "user",
            "content": [
                # K3 expects image data under the key matching the content type.
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": "What does the image show?"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": caption}],
        },
    ]


def _preprocess_multimodal_example(
    example: dict[str, Any],
    index: int,
    *,
    processor: Any,
    text_examples: list[dict[str, Any]],
    max_sequence_length: int,
) -> dict[str, Any]:
    messages = _build_multimodal_messages(example)
    if text_examples:
        messages.extend(_format_text_example(text_examples[index % len(text_examples)]))
    return processor(
        messages=messages,
        padding=False,
        max_length=max_sequence_length,
        truncation=True,
    )


def _collate_multimodal_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if len(batch) != 1:
        raise ValueError("Kimi K3 multimodal calibration requires batch-size=1")
    return {
        key: torch.tensor(
            value,
            dtype=torch.bfloat16 if key == "pixel_values" else None,
        )
        for key, value in batch[0].items()
    }


def _index_weight_map(model_id: Path) -> dict[str, str]:
    index_path = model_id / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing Kimi K3 index: {index_path}")
    with index_path.open(encoding="utf-8") as file:
        index = json.load(file)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid weight_map in {index_path}")
    return weight_map


def _expert_targets_from_index(
    model_id: Path, config: Any
) -> tuple[str, set[str]]:
    """Validate native MXFP4 expert keys and return a structural target regex."""
    weight_map = _index_weight_map(model_id)
    all_scale_keys = {key for key in weight_map if key.endswith(".weight_scale")}
    scale_keys = {
        key
        for key in all_scale_keys
        if _EXPERT_TARGET.match(key.removesuffix(".weight_scale"))
    }
    if not scale_keys:
        raise ValueError("Kimi K3 index contains no routed-expert weight_scale keys")
    unexpected_scales = all_scale_keys - scale_keys
    if unexpected_scales:
        raise ValueError(
            "Kimi K3 index contains weight_scale tensors outside routed experts: "
            f"{sorted(unexpected_scales)[:3]}"
        )

    targets = {key.removesuffix(".weight_scale") for key in scale_keys}
    all_packed_targets = {
        key.removesuffix(".weight_packed")
        for key in weight_map
        if key.endswith(".weight_packed")
    }
    packed_targets = {
        name for name in all_packed_targets if _EXPERT_TARGET.fullmatch(name)
    }
    unexpected_packed = all_packed_targets - packed_targets
    if unexpected_packed:
        raise ValueError(
            "Kimi K3 index contains weight_packed tensors outside routed experts: "
            f"{sorted(unexpected_packed)[:3]}"
        )
    if targets != packed_targets:
        missing_packed = sorted(targets - packed_targets)
        missing_scale = sorted(packed_targets - targets)
        raise ValueError(
            "Kimi K3 expert weight_scale/weight_packed targets differ: "
            f"missing_packed={missing_packed[:3]}, "
            f"missing_scale={missing_scale[:3]}"
        )

    matches = [_EXPERT_TARGET.fullmatch(name) for name in targets]
    layers = {int(match.group(1)) for match in matches}
    experts = {int(match.group(2)) for match in matches}
    projections = {match.group(3) for match in matches}

    text_config = getattr(config, "text_config", config)
    num_hidden_layers = int(text_config.num_hidden_layers)
    first_moe_layer = int(text_config.first_k_dense_replace)
    moe_layer_freq = int(getattr(text_config, "moe_layer_freq", 1))
    num_experts = int(text_config.num_experts)
    if moe_layer_freq <= 0 or num_experts <= 0:
        raise ValueError(
            "Kimi K3 num_experts and moe_layer_freq must be positive; "
            f"got num_experts={num_experts}, moe_layer_freq={moe_layer_freq}"
        )
    expected_layers = {
        layer
        for layer in range(num_hidden_layers)
        if layer >= first_moe_layer and layer % moe_layer_freq == 0
    }
    if layers != expected_layers:
        raise ValueError(
            "Kimi K3 expert layers disagree with text_config: "
            f"actual={sorted(layers)}, expected={sorted(expected_layers)}"
        )
    expected_experts = set(range(num_experts))
    if experts != expected_experts:
        raise ValueError(
            "Kimi K3 expert indices disagree with text_config: "
            f"actual_count={len(experts)}, expected_count={num_experts}, "
            f"missing={sorted(expected_experts - experts)[:3]}, "
            f"unexpected={sorted(experts - expected_experts)[:3]}"
        )
    if projections != {"w1", "w2", "w3"}:
        raise ValueError(f"Kimi K3 expert projections are incomplete: {projections}")
    expected_count = len(layers) * len(experts) * len(projections)
    if len(targets) != expected_count:
        raise ValueError(
            "Kimi K3 expert weight_scale keys do not form a complete "
            f"layer/expert/projection set: {len(targets)} != {expected_count}"
        )

    return _EXPERT_TARGET_PATTERN, targets


def _other_targets_from_index(
    model_id: Path, expert_targets: set[str]
) -> tuple[str, set[str]]:
    weight_names = {
        key.removesuffix(".weight")
        for key in _index_weight_map(model_id)
        if key.endswith(".weight") and key.startswith("language_model.model.")
    }
    other = {name for name in weight_names if _OTHER_LINEAR_RE.fullmatch(name)}
    if not other:
        raise ValueError("Kimi K3 index contains no non-expert Linear targets")
    overlap = other & expert_targets
    if overlap:
        raise ValueError(f"Kimi K3 INT4/INT8 target overlap: {sorted(overlap)[:3]}")
    return (
        (
            r"re:^language_model\.model\.(?:.*\.)?(?:"
            + "|".join(map(re.escape, _OTHER_LINEAR_SUFFIXES))
            + r")$"
        ),
        other,
    )


def _register_k3_model(model_id: Path):
    """Load K3 remote classes and make ``AutoModelForCausalLM.from_config`` work."""
    patch_kimi_k3_transformers_compat()
    config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=True, local_files_only=True
    )
    model_ref = config.auto_map.get("AutoModelForCausalLM")
    if not model_ref:
        raise ValueError("Kimi K3 config has no AutoModelForCausalLM auto_map")
    model_class = get_class_from_dynamic_module(
        model_ref, str(model_id), local_files_only=True
    )
    model_class = patch_kimi_k3_transformers_compat(model_class)
    try:
        AutoConfig.register(config.model_type, type(config))
    except ValueError:
        # Re-running this example in one interpreter is harmless when the same
        # dynamic class is already registered.
        pass
    try:
        AutoModelForCausalLM.register(type(config), model_class)
    except ValueError:
        pass
    return config


def _kda_num_heads(config: Any) -> int:
    text_config = getattr(config, "text_config", None)
    linear_attn_config = getattr(text_config, "linear_attn_config", None)
    if isinstance(linear_attn_config, dict):
        num_heads = linear_attn_config.get("num_heads")
    else:
        num_heads = getattr(linear_attn_config, "num_heads", None)
    if isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads <= 0:
        raise ValueError(
            "Kimi K3 text_config.linear_attn_config.num_heads must be a "
            "positive integer"
        )
    return num_heads


def _int_scheme(num_bits: int, targets: list[str]) -> QuantizationScheme:
    return QuantizationScheme(
        targets=targets,
        weights=QuantizationArgs(
            num_bits=num_bits,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.CHANNEL,
            symmetric=True,
            dynamic=False,
            observer="imatrix_mse",
        ),
        input_activations=QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.TOKEN,
            symmetric=True,
            dynamic=True,
            observer=None,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", "--model_id", type=Path, required=True)
    parser.add_argument("--dataset-id", "--dataset_id", required=True)
    parser.add_argument("--dataset-split", "--dataset_split", default="test")
    parser.add_argument("--text-dataset-id", "--text_dataset_id", default=None)
    parser.add_argument(
        "--text-dataset-split",
        "--text_dataset_split",
        default="train_sft",
    )
    parser.add_argument(
        "--num-calibration-samples",
        "--num_calibration_samples",
        type=positive_int,
        default=32,
    )
    parser.add_argument(
        "--text-calibration-samples",
        "--text_calibration_samples",
        type=non_negative_int,
        default=128,
        help="Number of text conversations to cycle across multimodal samples.",
    )
    parser.add_argument(
        "--max-sequence-length",
        "--max_sequence_length",
        type=positive_int,
        default=2048,
    )
    parser.add_argument("--batch-size", "--batch_size", type=positive_int, default=1)
    parser.add_argument("--output-dir", "--output_dir", type=Path, required=True)
    parser.add_argument("--work-dir", "--work_dir", type=Path, default=None)
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Publish completed work-dir shards without rerunning quantization.",
    )
    parser.add_argument(
        "--moe-calibrate-all-experts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect iMatrix statistics for every K3 routed expert.",
    )
    args = parser.parse_args()
    if args.batch_size != 1:
        parser.error("Kimi K3 multimodal calibration requires --batch-size 1")

    model_id = args.model_id.expanduser().resolve()
    config = _register_k3_model(model_id)
    expert_pattern, expert_targets = _expert_targets_from_index(model_id, config)
    other_pattern, other_targets = _other_targets_from_index(model_id, expert_targets)

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=True,
    )
    multimodal_dataset, text_examples = load_calibration_datasets(
        args.dataset_id,
        args.dataset_split,
        args.text_dataset_id,
        args.text_dataset_split,
        args.num_calibration_samples,
        args.text_calibration_samples,
    )
    dataset = multimodal_dataset.map(
        partial(
            _preprocess_multimodal_example,
            processor=processor,
            text_examples=text_examples,
            max_sequence_length=args.max_sequence_length,
        ),
        with_indices=True,
        remove_columns=multimodal_dataset.column_names,
    )
    calibration_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=_collate_multimodal_batch,
    )

    ignores = [
        r"re:^vision_tower(?:\.|$)",
        r"re:^mm_projector(?:\.|$)",
        r"re:.*\.embed_tokens$",
        r"re:.*\.lm_head$",
    ]
    config_groups = {
        "routed_experts_w4a8": _int_scheme(4, [expert_pattern]),
        "other_linear_w8a8": _int_scheme(8, [other_pattern]),
    }
    recipe = [
        IMatrixGatherer(ignore=ignores),
        QuantizationModifier(config_groups=config_groups, ignore=ignores),
    ]

    with ImatrixFallbackStats():
        result = streaming_oneshot(
            model=model_id,
            model_config=config,
            dataset=calibration_dataloader,
            tokenizer=processor,
            recipe=recipe,
            output_dir=args.output_dir,
            work_dir=args.work_dir,
            num_calibration_samples=args.num_calibration_samples,
            max_seq_length=args.max_sequence_length,
            batch_size=args.batch_size,
            shuffle_calibration_samples=False,
            moe_calibrate_all_experts=args.moe_calibrate_all_experts,
            materializer=KimiK3WeightMaterializer(
                kda_num_heads=_kda_num_heads(config)
            ),
            # Keep adjacent boundaries in memory and publish final shards only.
            checkpoint_progress=False,
            # K3 downstream readers expect packed INT4 tensors in the output.
            pack_to_int8=True,
            overwrite_output=True,
            finalize_only=args.finalize_only,
        )
    processor.save_pretrained(result)
    print(
        f"Saved Kimi K3 WNA8 checkpoint to {result} "
        f"(INT4 routed experts={len(expert_targets)}, "
        f"INT8 other={len(other_targets)})"
    )


if __name__ == "__main__":
    main()
