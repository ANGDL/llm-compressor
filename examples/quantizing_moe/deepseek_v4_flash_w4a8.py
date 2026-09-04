"""Quantize DeepSeek-V4-Flash to WNA8 with iMatrix + RTN.

Routed experts use INT4 weights, while attention, the Lightning Indexer, and
shared experts use INT8 weights. All targeted modules use dynamic INT8 input
activations. This is the Transformers-model equivalent of
``examples/streaming_oneshot/deepseek_v4_wNa8.py``.

Example::

    python examples/quantizing_moe/deepseek_v4_flash_w4a8.py \
        --model-id RedHatAI/DeepSeek-V4-Flash-BF16 \
        --dataset-id HuggingFaceH4/ultrachat_200k \
        --num-calibration-samples 512 \
        --max-sequence-length 8192 \
        --output-dir /ssd2/models/DeepSeek-V4-Flash-w4a8-unpacked

    python -m llmcompressor.utils.pack_int4_to_int8 \
        -i /ssd2/models/DeepSeek-V4-Flash-w4a8-unpacked/ \
        -o /ssd2/models/DeepSeek-V4-Flash-w4a8-v2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4ForCausalLM,
    DeepseekV4PreTrainedModel,
)

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.utils import ImatrixFallbackStats, load_context

# Upstream norms are expected in bfloat16 for this model definition.
DeepseekV4PreTrainedModel._keep_in_fp32_modules_strict = set()

_WO_A_PATTERN = r"re:.*self_attn\.o_a_proj$"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Hugging Face model ID or local BF16 checkpoint path.",
    )
    parser.add_argument(
        "--dataset-id",
        nargs="+",
        default=["HuggingFaceH4/ultrachat_200k"],
        help="Hugging Face dataset ID(s) or local JSON/JSONL path(s).",
    )
    parser.add_argument("--dataset-split", default="train_sft")
    parser.add_argument("--num-calibration-samples", type=positive_int, default=64)
    parser.add_argument("--max-sequence-length", type=positive_int, default=512)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument(
        "--quantize-wo-a",
        "--quantize_wo_a",
        dest="quantize_wo_a",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Quantize self_attn.o_a_proj with the INT8 scheme.",
    )
    parser.add_argument(
        "--use-float32-scale-dtype",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use float32 for weight scales instead of the model dtype.",
    )
    parser.add_argument(
        "--moe-calibrate-all-experts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Send calibration tokens through every routed expert.",
    )
    parser.add_argument(
        "--pipeline",
        choices=("independent", "sequential"),
        default="sequential",
        help="Calibration pipeline; sequential limits accelerator residency.",
    )
    parser.add_argument(
        "--device-map",
        default="cpu",
        help="Transformers device_map used while loading the model.",
    )
    parser.add_argument(
        "--offload-folder",
        type=Path,
        default=Path("./offload_folder"),
        help="Directory used when device_map enables disk offloading.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the compressed checkpoint and tokenizer.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="50GB",
        help="Maximum output safetensors shard size.",
    )
    return parser.parse_args()


def _is_local_json(source_id: str) -> bool:
    candidate = Path(source_id).expanduser()
    return candidate.is_file() or candidate.suffix.lower() in {".json", ".jsonl"}


def _load_source(source_id: str, split: str, samples: int) -> Dataset:
    if _is_local_json(source_id):
        return load_dataset(
            "json",
            data_files=str(Path(source_id).expanduser()),
            split=f"train[:{samples}]",
        )
    return load_dataset(source_id, split=f"{split}[:{samples}]")


def _encode_example(example: dict[str, Any]) -> dict[str, Any]:
    if "messages" in example:
        from llmcompressor.modeling.deepseekv4.encoding.encoding_dsv4 import (
            encode_messages,
        )

        return {
            "text": encode_messages(example["messages"], thinking_mode="thinking")
        }
    if "text" in example:
        return {"text": example["text"]}
    if "input_ids" in example:
        return example
    raise ValueError(
        "Calibration examples must contain 'messages', 'text', or 'input_ids'"
    )


def load_calibration_dataset(
    source_ids: list[str], split: str, samples: int, seed: int = 42
) -> Dataset:
    if samples < len(source_ids):
        raise ValueError(
            "num-calibration-samples must be at least the number of dataset "
            f"sources ({samples} < {len(source_ids)})"
        )

    base, remainder = divmod(samples, len(source_ids))
    parts = []
    for index, source_id in enumerate(source_ids):
        count = base + (index < remainder)
        part = _load_source(source_id, split, count).shuffle(seed=seed)
        columns = set(part.column_names)
        if "input_ids" not in columns:
            part = part.map(_encode_example, remove_columns=part.column_names)
        parts.append(part)

    if len(parts) == 1:
        return parts[0]
    schemas = [tuple(part.column_names) for part in parts]
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise ValueError(
            "All calibration sources must produce the same columns; "
            f"got {schemas}"
        )
    return concatenate_datasets(parts)


def _ignores(*, quantize_wo_a: bool = False) -> list[str]:
    ignores = [
        "lm_head",
        r"re:.*embed_tokens$",
        r"re:.*mlp\.gate$",
        r"re:.*self_attn\.compressor\.(gate_proj|kv_proj)$",
        r"re:.*self_attn\.compressor\.indexer\.(gate_proj|kv_proj)$",
        r"re:.*self_attn\.compressor\.indexer\.scorer\.weights_proj$",
        r"re:.*confidence_head\.proj$",
    ]
    if not quantize_wo_a:
        ignores.append(_WO_A_PATTERN)
    return ignores


def _int_scheme(
    num_bits: int,
    targets: list[str],
    scale_dtype: torch.dtype | None = None,
) -> QuantizationScheme:
    return QuantizationScheme(
        targets=targets,
        weights=QuantizationArgs(
            num_bits=num_bits,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.CHANNEL,
            symmetric=True,
            dynamic=False,
            observer="imatrix_mse",
            scale_dtype=scale_dtype,
        ),
        input_activations=QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.TOKEN,
            symmetric=False,
            dynamic=True,
            observer=None,
        ),
    )


def wna8_config(
    *,
    quantize_wo_a: bool = False,
    scale_dtype: torch.dtype | None = None,
) -> tuple[dict[str, QuantizationScheme], list[str]]:
    """Build the streaming-aligned WNA8 config for Transformers module names."""
    experts = _int_scheme(
        4,
        [r"re:.*mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"],
        scale_dtype,
    )
    other_targets = [
        r"re:.*self_attn\.(q_a_proj|q_b_proj|kv_proj|o_b_proj)$",
        r"re:.*self_attn\.compressor\.indexer\.q_b_proj$",
        r"re:.*mlp\.shared_experts\.(gate_proj|up_proj|down_proj)$",
    ]
    if quantize_wo_a:
        other_targets.append(_WO_A_PATTERN)
    other = _int_scheme(8, other_targets, scale_dtype)

    # The Transformers DeepseekV4PreTrainedModel does not instantiate MTP layers,
    # so the native streaming recipe's MTP target has no counterpart here.
    return {
        "experts_w4a8": experts,
        "other_w8a8": other,
    }, _ignores(quantize_wo_a=quantize_wo_a)


def imatrix_rtn_recipe(
    *,
    quantize_wo_a: bool = False,
    scale_dtype: torch.dtype | None = None,
    pipeline: str = "sequential",
) -> tuple[list[IMatrixGatherer | QuantizationModifier], list[str]]:
    config_groups, ignores = wna8_config(
        quantize_wo_a=quantize_wo_a,
        scale_dtype=scale_dtype,
    )
    gatherer_kwargs = {"ignore": ignores}
    if pipeline == "sequential":
        gatherer_kwargs["attach_by_initialize"] = False
    return [
        IMatrixGatherer(**gatherer_kwargs),
        QuantizationModifier(config_groups=config_groups, ignore=ignores),
    ], ignores


def main() -> None:
    args = parse_args()
    dataset = load_calibration_dataset(
        args.dataset_id,
        args.dataset_split,
        args.num_calibration_samples,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    with load_context(DeepseekV4ForCausalLM):
        model = DeepseekV4ForCausalLM.from_pretrained(
            args.model_id,
            dtype="auto",
            device_map=args.device_map,
            offload_folder=args.offload_folder,
        )
    if not isinstance(model, DeepseekV4PreTrainedModel):
        raise TypeError(
            "Expected a Transformers DeepseekV4PreTrainedModel, got "
            f"{type(model).__module__}.{type(model).__name__}"
        )

    recipe, _ = imatrix_rtn_recipe(
        quantize_wo_a=args.quantize_wo_a,
        scale_dtype=torch.float32 if args.use_float32_scale_dtype else None,
        pipeline=args.pipeline,
    )
    imatrix_fallback_stats = ImatrixFallbackStats()
    imatrix_fallback_stats.install_hooks()
    with imatrix_fallback_stats:
        oneshot(
            model=model,
            processor=tokenizer,
            dataset=dataset,
            recipe=recipe,
            max_seq_length=args.max_sequence_length,
            num_calibration_samples=args.num_calibration_samples,
            sequential_targets=["DeepseekV4DecoderLayer"],
            batch_size=args.batch_size,
            shuffle_calibration_samples=True,
            moe_calibrate_all_experts=args.moe_calibrate_all_experts,
            pipeline=args.pipeline,
            propagate_error=False,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        args.output_dir,
        save_compressed=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved DeepSeek-V4-Flash WNA8 iMatrix+RTN checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
