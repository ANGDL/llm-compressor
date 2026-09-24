"""Configurable Qwen3 dense W4A8 streaming smoke entrypoint for CI."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from compressed_tensors.quantization import preset_name_to_scheme

from datasets import load_dataset
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.streaming import streaming_oneshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-id",
        default="HuggingFaceH4/ultrachat_200k",
    )
    parser.add_argument("--dataset-split", default="train_sft")
    parser.add_argument("--num-calibration-samples", type=int, default=16)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--checkpoint-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_calibration_samples <= 0:
        raise ValueError("num-calibration-samples must be positive")
    if args.max_sequence_length <= 0:
        raise ValueError("max-sequence-length must be positive")

    dataset = load_dataset(
        args.dataset_id,
        split=f"{args.dataset_split}[:{args.num_calibration_samples}]",
    )
    scheme = preset_name_to_scheme("W4A8", ["Linear"])
    scheme.weights.observer = "imatrix_mse"
    scheme.weights.scale_dtype = torch.float32
    ignores = ["lm_head"]
    recipe = [
        IMatrixGatherer(ignore=ignores),
        QuantizationModifier(
            config_groups={"group_0": scheme},
            ignore=ignores,
        ),
    ]
    streaming_oneshot(
        model=args.model_id,
        dataset=dataset,
        recipe=recipe,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_sequence_length,
        async_save=not args.checkpoint_progress,
        checkpoint_progress=args.checkpoint_progress,
        overwrite_output=True,
        pack_to_int8=True,
    )


if __name__ == "__main__":
    main()
