"""Run native StreamingInference on a local Qwen3-VL checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import resource
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from llmcompressor.streaming import load_streaming_model

DEFAULT_CHECKPOINT = "/path/to/models/Qwen3-VL-32B-Instruct-W8A16"
DEFAULT_MODEL = "/path/to/models/Qwen3-VL-32B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--processor", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--mode", choices=("forward", "generate"), default="forward")
    parser.add_argument("--max-device-weight-gib", type=float, default=8.0)
    parser.add_argument("--max-preparation-gib", type=float, default=8.0)
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--max-prefetch-gib", type=float, default=8.0)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def make_image() -> Image.Image:
    image = Image.new("RGB", (336, 224), (42, 92, 151))
    draw = ImageDraw.Draw(image)
    draw.rectangle((28, 28, 154, 190), outline=(240, 240, 240), width=6)
    draw.ellipse(
        (184, 56, 300, 172), fill=(220, 80, 60), outline=(255, 255, 255), width=6
    )
    draw.line((0, 112, 336, 112), fill=(40, 220, 120), width=5)
    return image


def inputs_for(processor: Any) -> dict[str, torch.Tensor]:
    image = make_image()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": "Describe the geometric objects and their colors.",
                },
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = processor(
        text=[text],
        images=[image],
        padding=False,
        return_tensors="pt",
    )
    return {
        name: value
        for name, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }


def snapshot(device: torch.device) -> dict[str, float]:
    result = {"rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    if device.type == "cuda":
        result.update(
            {
                "gpu_allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
                "gpu_reserved_mib": torch.cuda.memory_reserved(device) / 2**20,
                "gpu_peak_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            }
        )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(
        args.processor,
        local_files_only=True,
        trust_remote_code=True,
    )
    inputs = inputs_for(processor)
    report: dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "processor": str(Path(args.processor).resolve()),
        "input_shapes": {name: list(value.shape) for name, value in inputs.items()},
        "device": str(device),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with load_streaming_model(
        args.checkpoint,
        device=device,
        dtype=torch.bfloat16,
        model_factory=Qwen3VLForConditionalGeneration,
        preload_module_classes=[
            "Qwen3VLVisionBlock",
            "Qwen3VLTextDecoderLayer",
        ],
        max_device_weight_bytes=int(args.max_device_weight_gib * 2**30),
        max_preparation_bytes=int(args.max_preparation_gib * 2**30),
        prefetch=args.prefetch,
        max_prefetch_bytes=int(args.max_prefetch_gib * 2**30),
    ) as stream:
        report["setup_seconds"] = time.monotonic() - started
        report["streaming_report"] = stream.report()
        inference_started = time.monotonic()
        with torch.inference_mode():
            if args.mode == "forward":
                output = stream(**inputs, use_cache=False)
                next_token = output.logits[:, -1].argmax(dim=-1)
                report["forward_shape"] = list(output.logits.shape)
                report["next_token_id"] = int(next_token.item())
                report["next_token_text"] = processor.decode(
                    [int(next_token.item())], skip_special_tokens=True
                )
            else:
                generated = stream.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
                continuation = generated[0, inputs["input_ids"].shape[-1] :]
                report["generated_ids"] = continuation.tolist()
                report["generated_text"] = processor.decode(
                    continuation,
                    skip_special_tokens=True,
                )
        report["inference_seconds"] = time.monotonic() - inference_started
        report["streaming_report_after"] = stream.report()
        report["memory"] = snapshot(device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return report


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    stream = result["streaming_report_after"]
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "checkpoint",
                    "input_shapes",
                    "setup_seconds",
                    "inference_seconds",
                    "forward_shape",
                    "next_token_id",
                    "next_token_text",
                    "generated_ids",
                    "generated_text",
                    "memory",
                )
            }
            | {
                key: stream.get(key)
                for key in (
                    "tensor_reads",
                    "materializations",
                    "peak_estimated_preparation_bytes",
                    "estimated_peak_weight_bytes",
                    "peak_weight_module",
                    "activation_mode",
                    "prefetch",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(payload + "\n")
