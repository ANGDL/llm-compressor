"""Run an isolated Streaming PTQ memory probe for a local checkpoint."""

from __future__ import annotations

import argparse
import json
import resource
import sys
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader

from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import AWQModifier, SmoothQuantModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.streaming import streaming_oneshot


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB while macOS reports bytes.
    return int(value if sys.platform == "darwin" else value * 1024)


def _checkpoint_bytes(checkpoint: Path) -> int:
    return sum(path.stat().st_size for path in checkpoint.glob("*.safetensors"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument(
        "--algorithm", choices=("rtn", "imatrix", "gptq", "awq", "smoothquant"),
        default="imatrix",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoint_bytes = _checkpoint_bytes(args.checkpoint)
    if checkpoint_bytes == 0:
        raise ValueError(f"No safetensors files found in {args.checkpoint}")

    baseline_rss = _peak_rss_bytes()
    sample = {
        "input_ids": torch.arange(1, args.sequence_length + 1),
        "attention_mask": torch.ones(args.sequence_length, dtype=torch.long),
    }
    quantizer = QuantizationModifier(
        scheme="W8A8", targets=["Linear"], ignore=["lm_head"],
        weight_observer="imatrix_mse" if args.algorithm == "imatrix" else None,
    )
    recipe = {
        "rtn": [quantizer],
        "imatrix": [IMatrixGatherer(ignore=["lm_head"]), quantizer],
        "gptq": [GPTQModifier(scheme="W8A8", ignore=["lm_head"])],
        "awq": [AWQModifier(), quantizer],
        "smoothquant": [SmoothQuantModifier(), quantizer],
    }[args.algorithm]
    targets = {}

    def capture(message):
        parts = message.record["message"].split(" | ")
        if len(parts) < 3 or parts[1] not in ("admission", "complete"):
            return
        values = targets.setdefault(parts[0], {"target": parts[0]})
        for part in parts[2:]:
            key, separator, value = part.partition("=")
            if separator:
                if value.isdigit():
                    values[key] = int(value)
                else:
                    values[key] = None if value == "None" else value

    sink = logger.add(capture, filter=lambda record: record["function"] == "streaming")
    try:
        output = streaming_oneshot(
            model=args.checkpoint,
            dataset=DataLoader([sample], batch_size=1),
            dataset_fingerprint="resource-probe-v1",
            recipe=recipe,
            output_dir=args.output_dir,
            work_dir=args.work_dir,
            num_calibration_samples=1,
            max_seq_length=args.sequence_length,
            target_dtype=torch.bfloat16,
            device=args.device,
        )
    finally:
        logger.remove(sink)
    peak_rss = _peak_rss_bytes()
    print(
        json.dumps(
            {
                "baseline_rss_bytes": baseline_rss,
                "checkpoint_bytes": checkpoint_bytes,
                "incremental_peak_bytes": max(0, peak_rss - baseline_rss),
                "output": str(output),
                "peak_rss_bytes": peak_rss,
                "algorithm": args.algorithm,
                "device": args.device,
                "targets": list(targets.values()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
