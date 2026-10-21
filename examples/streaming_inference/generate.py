"""Run native cached generation or fixed-token validation from a local checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from llmcompressor.streaming import (
    DeepSeekV41WeightMaterializer,
    compare_cached_to_recompute,
    load_streaming_model,
)


def token_ids(value):
    return torch.tensor([[int(token) for token in value.split(",")]], dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompt")
    inputs.add_argument(
        "--input-ids", help="Comma-separated token ids, without tokenizer"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument(
        "--layout", choices=["standard", "deepseek-v41"], default="standard"
    )
    parser.add_argument("--preload-module-class", action="append", default=[])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--continuation-ids", help="Compare cache vs recompute on fixed tokens"
    )
    parser.add_argument(
        "--weight-only", action="store_true", help="Bypass activation Q/DQ"
    )
    parser.add_argument("--max-device-weight-bytes", type=int)
    parser.add_argument("--max-preparation-bytes", type=int)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    tokenizer = (
        AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
        if args.prompt is not None or args.layout == "deepseek-v41"
        else None
    )
    input_ids = (
        tokenizer(args.prompt, return_tensors="pt")["input_ids"]
        if args.prompt is not None
        else token_ids(args.input_ids)
    )
    options = {}
    if args.layout == "deepseek-v41":
        options["materializer"] = DeepSeekV41WeightMaterializer()
        options["model_kwargs"] = {
            "tokenizer": tokenizer,
            "engram_checkpoint": args.checkpoint,
        }
    with load_streaming_model(
        args.checkpoint,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        activation_qdq=not args.weight_only,
        preload_module_classes=args.preload_module_class,
        max_device_weight_bytes=args.max_device_weight_bytes,
        max_preparation_bytes=args.max_preparation_bytes,
        **options,
    ) as inference:
        # The draft branch is a separate, teacher-forced calibration diagnostic.
        if hasattr(inference.model.config, "calibrate_dspark"):
            inference.model.config.calibrate_dspark = False
        if args.continuation_ids is not None:
            result = compare_cached_to_recompute(
                inference,
                input_ids,
                token_ids(args.continuation_ids),
                rtol=args.rtol,
                atol=args.atol,
            )
        else:
            generated = inference.generate(
                input_ids,
                use_cache=True,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )
            result = {"sequences": generated.cpu().tolist()}
            if tokenizer is not None:
                result["text"] = tokenizer.decode(
                    generated[0], skip_special_tokens=False
                )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.report is not None:
            args.report.write_text(
                json.dumps(
                    {"result": result, "execution": inference.report()},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        if result.get("passed") is False:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
