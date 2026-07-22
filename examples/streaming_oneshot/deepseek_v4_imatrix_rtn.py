"""Out-of-core DeepSeek-V4 W8A8 iMatrix RTN quantization.

The input checkpoint can be the original DeepSeek-V4 FP8+FP4 checkpoint or a
BF16 checkpoint. The original format is decoded to BF16 per target. Unlike the
regular DeepSeek-V4 example, this script does not keep the full model resident:
streaming PTQ loads one inferred decoder target at a time, persists calibration
statistics, and writes the final safetensors checkpoint incrementally.

Example:

    python examples/streaming_oneshot/deepseek_v4_imatrix_rtn.py \
        --model-id /Users/ang/models/DeepSeek-V4-Pro-Tiny-bf16 \
        --dataset-id ./calibration_data_dsv4_pro.jsonl \
        --num-calibration-samples 32 \
        --max-sequence-length 2048 \
        --output-dir ./DeepSeek-V4-Pro-Tiny-W8A8-IMatrix-RTN \
        --work-dir ./streaming-work-deepseek-v4

For a full model, use a work directory on a disk with enough space for the
activation/statistics artifacts and staging shards. The source checkpoint is
never modified.
"""

import argparse
import os
from pathlib import Path

from compressed_tensors.quantization import preset_name_to_scheme
from transformers import AutoTokenizer

# Importing the native package registers its model_type with Transformers.
import llmcompressor.modeling.deepseekv4  # noqa: F401
from datasets import concatenate_datasets, load_dataset
from llmcompressor.modeling.deepseekv4.config import ModelConfig
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.streaming import DeepSeekV4WeightMaterializer, streaming_oneshot


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def load_calibration_dataset(dataset_ids: list[str], split: str, samples: int):
    """Load and combine Hugging Face or local JSON/JSONL calibration sources."""
    datasets = []
    for dataset_id in dataset_ids:
        if os.path.isfile(dataset_id) or dataset_id.endswith((".json", ".jsonl")):
            dataset = load_dataset(
                "json",
                data_files=dataset_id,
                split=f"train[:{samples}]",
            )
        else:
            dataset = load_dataset(dataset_id, split=f"{split}[:{samples}]")
        datasets.append(dataset.shuffle(seed=42))

    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def preprocess(example):
    """Format DeepSeek-V4 conversations before tokenizer processing."""
    from llmcompressor.modeling.deepseekv4.encoding.encoding_dsv4 import (
        encode_messages,
    )

    if "messages" in example:
        return {
            "text": encode_messages(
                example["messages"],
                thinking_mode="thinking",
            )
        }
    if "text" in example:
        return {"text": example["text"]}
    raise ValueError("Calibration examples must contain 'messages' or 'text'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument(
        "--dataset-id",
        type=str,
        nargs="+",
        default=["HuggingFaceH4/ultrachat_200k"],
    )
    parser.add_argument("--dataset-split", default="train_sft")
    parser.add_argument("--num-calibration-samples", type=positive_int, default=32)
    parser.add_argument("--max-sequence-length", type=positive_int, default=2048)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
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

    scheme = preset_name_to_scheme("W8A8", ["Linear"])
    if scheme.weights is None:
        raise RuntimeError("W8A8 preset is missing weight quantization settings")
    scheme.weights.observer = "imatrix_mse"

    # Match the exclusions used by the in-core DeepSeek-V4 W8A8 example.
    ignores = [
        "lm_head",
        "re:.*embed$",
        r"re:.*ffn\.gate$",
        r"re:.*attn\.compressor\.wgate$",
        r"re:.*attn\.compressor\.wkv$",
        r"re:.*attn\.indexer\.compressor\.wgate$",
        r"re:.*attn\.indexer\.compressor\.wkv$",
        r"re:.*attn\.indexer\.weights_proj$",
        r"re:.*attn\.wo_a$",
    ]
    recipe = [
        IMatrixGatherer(ignore=ignores),
        QuantizationModifier(
            config_groups={"group_0": scheme},
            ignore=ignores,
        ),
    ]

    result = streaming_oneshot(
        model=args.model_id,
        model_config=config,
        dataset=dataset,
        tokenizer=tokenizer,
        preprocessing_func=preprocess,
        recipe=recipe,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_sequence_length,
        batch_size=args.batch_size,
        moe_calibrate_all_experts=args.moe_calibrate_all_experts,
        device=args.device,
        materializer=DeepSeekV4WeightMaterializer(),
    )
    tokenizer.save_pretrained(result)
    print(f"Saved streaming DeepSeek-V4 checkpoint to {result}")


if __name__ == "__main__":
    main()
