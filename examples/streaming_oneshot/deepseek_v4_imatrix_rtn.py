"""Stream DeepSeek-V4 calibration and W8A8/WNA8/W4A8 RTN quantization.

The source checkpoint may be the native FP8+FP4 format or an ordinary BF16
checkpoint. ``DeepSeekV4WeightMaterializer`` decodes the former on demand and
casts both formats to BF16 while one traced subgraph is resident. Calibration
boundaries remain in memory by default; ``--checkpoint-progress`` enables the
optional durable recovery path.

Example::

    python examples/streaming_oneshot/deepseek_v4_imatrix_rtn.py \
        --model-id /Users/ang/models/DeepSeek-V4-Pro-Tiny-bf16 \
        --dataset-id /Users/ang/Downloads/llm-demo/datasets/ultrachat_200k \
        --quant-mode w4a8 \
        --output-dir /Users/ang/models/DeepSeek-V4-Pro-Tiny-w4a8
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from compressed_tensors.quantization import (
    QuantizationScheme,
    preset_name_to_scheme,
)
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from transformers import AutoTokenizer

# Importing the native package registers DeepSeek-V4 with Transformers.
import llmcompressor.modeling.deepseekv4  # noqa: F401
from datasets import Dataset, concatenate_datasets, load_dataset
from llmcompressor.modeling.deepseekv4.config import ModelConfig
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.streaming import DeepSeekV4WeightMaterializer, streaming_oneshot


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _is_local_json(path: str) -> bool:
    candidate = Path(path).expanduser()
    return candidate.is_file() or candidate.suffix.lower() in {".json", ".jsonl"}


def _encode_example(example: dict[str, Any]) -> dict[str, Any]:
    """Normalize chat/text examples to the schema expected by the tokenizer."""
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


def _load_source(source_id: str, split: str, samples: int) -> Dataset:
    if _is_local_json(source_id):
        return load_dataset(
            "json", data_files=source_id, split=f"train[:{samples}]"
        )
    return load_dataset(source_id, split=f"{split}[:{samples}]")


def load_calibration_dataset(
    source_ids: list[str], split: str, samples: int, seed: int = 42
) -> Dataset:
    """Load sources, allocate the requested total, and normalize their schemas."""
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


def _ignores() -> list[str]:
    ignores = [
        "lm_head",
        r"re:.*embed$",
        r"re:.*ffn\.gate$",
        r"re:.*attn\.compressor\.wgate$",
        r"re:.*attn\.compressor\.wkv$",
        r"re:.*attn\.indexer\.compressor\.wgate$",
        r"re:.*attn\.indexer\.compressor\.wkv$",
        r"re:.*attn\.indexer\.weights_proj$",
    ]
    return ignores


def _int_scheme(num_bits: int) -> QuantizationScheme:
    return QuantizationScheme(
        targets=["Linear"],
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
            symmetric=False,
            dynamic=True,
            observer=None,
        ),
    )


def _schemes(quant_mode: str):
    ignores = _ignores()
    if quant_mode == "w8a8":
        scheme = preset_name_to_scheme("W8A8", ["Linear"])
        if scheme.weights is None:
            raise RuntimeError("W8A8 preset is missing weight settings")
        # wo_a is grouped/block-diagonal and is excluded from the uniform preset.
        ignores.append(r"re:.*attn\.wo_a$")
        scheme.weights.observer = "imatrix_mse"
        return {"group_0": scheme}, ignores
    if quant_mode == "w4a8":
        return {"group_0": _int_scheme(4)}, ignores
    if quant_mode == "wna8":
        experts = _int_scheme(4)
        experts.targets = [r"re:.*ffn\.experts\.\d+\.(w1|w2|w3)$"]
        other = _int_scheme(8)
        other.targets = [
            r"re:.*attn\.(wq_a|wq_b|wkv|wo_a|wo_b)$",
            r"re:.*attn\.indexer\.wq_b$",
            r"re:.*ffn\.shared_experts\.(w1|w2|w3)$",
            r"re:.*mtp\.\d+\.(e_proj|h_proj)$",
        ]
        return {"experts_w4a8": experts, "other_w8a8": other}, ignores
    raise ValueError(f"Unknown quantization mode: {quant_mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument(
        "--dataset-id",
        nargs="+",
        default=["HuggingFaceH4/ultrachat_200k"],
        help="Hugging Face dataset ID(s) or local JSON/JSONL path(s).",
    )
    parser.add_argument("--dataset-split", default="train_sft")
    parser.add_argument("--num-calibration-samples", type=positive_int, default=32)
    parser.add_argument("--max-sequence-length", type=positive_int, default=2048)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument(
        "--quant-mode",
        choices=("w8a8", "wna8", "w4a8"),
        default="w8a8",
        help="Uniform W8A8, mixed W4A8 experts/W8A8 other, or uniform W4A8.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist boundaries and transactions for crash recovery.",
    )
    parser.add_argument(
        "--moe-calibrate-all-experts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect iMatrix statistics for every routed expert.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=True)
    config = ModelConfig.from_pretrained(args.model_id)
    config.max_batch_size = args.batch_size
    config.max_seq_len = args.max_sequence_length
    dataset = load_calibration_dataset(
        args.dataset_id,
        args.dataset_split,
        args.num_calibration_samples,
    )
    config_groups, ignores = _schemes(args.quant_mode)
    recipe = [
        IMatrixGatherer(ignore=ignores),
        QuantizationModifier(
            config_groups=config_groups,
            ignore=ignores,
        ),
    ]
    result = streaming_oneshot(
        model=args.model_id,
        model_config=config,
        dataset=dataset,
        tokenizer=tokenizer,
        recipe=recipe,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_sequence_length,
        batch_size=args.batch_size,
        moe_calibrate_all_experts=args.moe_calibrate_all_experts,
        materializer=DeepSeekV4WeightMaterializer(),
        checkpoint_progress=args.checkpoint_progress,
        overwrite_output=True,
    )
    tokenizer.save_pretrained(result)
    print(
        f"Saved streaming DeepSeek-V4 {args.quant_mode.upper()} checkpoint to "
        f"{result}"
    )


if __name__ == "__main__":
    main()
