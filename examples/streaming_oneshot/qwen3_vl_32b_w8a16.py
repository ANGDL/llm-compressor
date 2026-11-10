"""End-to-end streaming PTQ validation for a local Qwen3-VL-32B checkpoint.

The default run intentionally uses a small deterministic synthetic image set.
This validates the multimodal execution contract without depending on a network
dataset. Use ``--stage all`` for the complete RTN/W8A16 path, or run individual
stages while diagnosing a failure.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from compressed_tensors.quantization.utils import is_module_quantized
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)

from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.streaming import streaming_oneshot

SCRIPT_VERSION = "qwen3-vl-32b-e2e-v1"
DEFAULT_MODEL = "/path/to/models/Qwen3-VL-32B-Instruct"
DEFAULT_ROOT = "/path/to/streaming-e2e/qwen3-vl-32b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument(
        "--stage",
        choices=("preflight", "prepare", "quantize", "validate", "all"),
        default="all",
    )
    parser.add_argument("--algorithm", choices=("rtn", "gptq"), default="rtn")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--async-save", action="store_true")
    parser.add_argument("--async-h2d", choices=("on", "off"), default="on")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[qwen3-vl-e2e] {message}", flush=True)


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def memory_snapshot(device: torch.device) -> dict[str, Any]:
    result = {
        "rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    if device.type == "cuda":
        result.update(
            {
                "gpu_allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
                "gpu_reserved_mib": torch.cuda.memory_reserved(device) / 2**20,
                "gpu_peak_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            }
        )
    return result


def reset_cuda(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def make_image(index: int, width: int, height: int) -> Image.Image:
    image = Image.new(
        "RGB",
        (width, height),
        ((29 * index) % 255, (67 + 31 * index) % 255, (113 + 17 * index) % 255),
    )
    draw = ImageDraw.Draw(image)
    inset = max(8, min(width, height) // 8)
    draw.rectangle(
        (inset, inset, width // 2, height - inset),
        outline=(240, 240, 240),
        width=max(2, inset // 4),
    )
    draw.ellipse(
        (width // 2, height // 4, width - inset, height - inset),
        fill=((200 + 7 * index) % 255, 80, 60),
        outline=(255, 255, 255),
        width=max(2, inset // 4),
    )
    draw.line(
        (0, height // 2, width, height // 2),
        fill=(40, 220, 120),
        width=max(2, inset // 3),
    )
    return image


def messages_for(image: Image.Image, index: int) -> list[dict[str, Any]]:
    prompts = (
        "Describe the shapes and colors in this image.",
        "What geometric objects are visible, and where are they located?",
        "Summarize the image in one short sentence.",
        "Identify the most prominent color and shape.",
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompts[index % len(prompts)]},
            ],
        }
    ]


def prepare_samples(
    processor: Any, sample_count: int, max_sequence_length: int
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    samples = []
    image_shapes = []
    for index in range(sample_count):
        width = (224, 336, 448, 256)[index % 4]
        height = (224, 224, 256, 384)[index % 4]
        image = make_image(index, width, height)
        messages = messages_for(image, index)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = processor(
            text=[text],
            images=[image],
            padding=False,
            truncation=True,
            max_length=max_sequence_length,
            return_tensors="pt",
        )
        sample = {}
        for name, value in encoded.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Processor field {name!r} is not a tensor")
            sample[name] = value.detach().cpu().contiguous()
        samples.append(sample)
        image_shapes.append([width, height])

    identity = {
        "script_version": SCRIPT_VERSION,
        "sample_count": sample_count,
        "max_sequence_length": max_sequence_length,
        "image_shapes": image_shapes,
        "prompts": [
            messages_for(
                make_image(index, image_shapes[index][0], image_shapes[index][1]),
                index,
            )[0]["content"][1]["text"]
            for index in range(sample_count)
        ],
    }
    return samples, identity


def collate_single(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if len(features) != 1:
        raise ValueError("Qwen3-VL E2E calibration intentionally uses batch_size=1")
    return {name: value.clone() for name, value in features[0].items()}


def calibration_loader(
    samples: list[dict[str, torch.Tensor]],
) -> DataLoader:
    return DataLoader(samples, batch_size=1, shuffle=False, collate_fn=collate_single)


def recipe_for(algorithm: str):
    ignored = [
        r"re:.*visual.*",
        r"re:.*lm_head$",
        r"re:.*embed_tokens$",
        r"re:.*norm$",
    ]
    if algorithm == "gptq":
        return GPTQModifier(
            scheme="W8A16",
            ignore=ignored,
            block_size=128,
        )
    return QuantizationModifier(
        scheme="W8A16",
        targets="Linear",
        ignore=ignored,
    )


def preflight(model_path: Path, device: torch.device) -> dict[str, Any]:
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if getattr(config, "model_type", None) != "qwen3_vl":
        raise ValueError(f"Expected qwen3_vl, got {config.model_type!r}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 32B end-to-end run")
    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors shards under {model_path}")
    return {
        "model": str(model_path),
        "model_type": config.model_type,
        "architectures": getattr(config, "architectures", None),
        "shards": len(shards),
        "checkpoint_bytes": sum(path.stat().st_size for path in shards),
        "device": str(device),
        "device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
    }


def load_model(model_path: Path, device: torch.device):
    log(f"loading model from {model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": str(device)},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def inference_inputs(processor: Any) -> dict[str, torch.Tensor]:
    image = make_image(101, 336, 224)
    messages = messages_for(image, 101)
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


def run_inference(
    model: Any,
    processor: Any,
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, Any]:
    inputs = inference_inputs(processor)
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits[:, -1, :].float().cpu()
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    prompt_length = inputs["input_ids"].shape[-1]
    continuation = generated[0, prompt_length:].cpu()
    return {
        "prompt_length": prompt_length,
        "next_token_top10": torch.topk(logits[0], k=10).indices.tolist(),
        "next_token_logits_top10": torch.topk(logits[0], k=10).values.tolist(),
        "generated_ids": continuation.tolist(),
        "text": processor.decode(continuation, skip_special_tokens=True),
    }


def quantization_scope(model: Any) -> dict[str, Any]:
    quantized_modules = [
        name
        for name, module in model.named_modules()
        if name and is_module_quantized(module)
    ]
    unexpected = [
        name
        for name in quantized_modules
        if not name.startswith("model.language_model.layers.")
    ]
    return {
        "count": len(quantized_modules),
        "unexpected_modules": unexpected,
        "only_text_decoder": not unexpected,
        "sample_modules": quantized_modules[:10],
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> None:
    model_path = Path(args.model).expanduser()
    root = Path(args.root).expanduser()
    output = root / f"output-{args.algorithm}-w8a16"
    work = root / f"work-{args.algorithm}-w8a16"
    report_path = root / f"report-{args.algorithm}-w8a16.json"
    device = torch.device(args.device)
    root.mkdir(parents=True, exist_ok=True)

    if report_path.is_file() and not args.overwrite:
        report = json.loads(report_path.read_text())
        report["arguments"] = vars(args)
        report["resumed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    else:
        report = {
            "script_version": SCRIPT_VERSION,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "arguments": vars(args),
            "model": str(model_path),
        }
    report["preflight"] = preflight(model_path, device)
    log(f"preflight: {json.dumps(report['preflight'], sort_keys=True)}")
    if args.stage == "preflight":
        write_json(report_path, report)
        return

    processor = AutoProcessor.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    samples, calibration_identity = prepare_samples(
        processor, args.samples, args.max_sequence_length
    )
    report["calibration"] = {
        **calibration_identity,
        "fingerprint": sha256_json(calibration_identity),
        "fields": sorted(samples[0]),
        "field_shapes": {name: list(value.shape) for name, value in samples[0].items()},
    }
    write_json(root / f"calibration-{args.algorithm}.json", report["calibration"])
    log(
        f"prepared {len(samples)} image batches; "
        f"fields={report['calibration']['fields']}"
    )
    if args.stage == "prepare":
        write_json(report_path, report)
        return

    if args.stage in ("quantize", "all"):
        if output.exists() and args.overwrite:
            shutil.rmtree(output)
        if work.exists() and args.overwrite:
            shutil.rmtree(work)
        os.environ["LLMCOMPRESSOR_STREAMING_ASYNC_H2D"] = (
            "1" if args.async_h2d == "on" else "0"
        )
        log(f"starting streaming {args.algorithm} W8A16 quantization")
        started = time.monotonic()
        reset_cuda(device)
        result = streaming_oneshot(
            model=model_path,
            model_factory=Qwen3VLForConditionalGeneration,
            tokenizer=processor,
            processor_identity={
                "script_version": SCRIPT_VERSION,
                "processor_class": (
                    f"{type(processor).__module__}.{type(processor).__qualname__}"
                ),
                "calibration": report["calibration"]["fingerprint"],
            },
            dataset=calibration_loader(samples),
            dataset_fingerprint=report["calibration"]["fingerprint"],
            recipe=recipe_for(args.algorithm),
            output_dir=output,
            work_dir=work,
            num_calibration_samples=args.samples,
            max_seq_length=args.max_sequence_length,
            batch_size=1,
            shuffle_calibration_samples=False,
            device=device,
            target_dtype=torch.bfloat16,
            checkpoint_progress=not args.async_save,
            async_save=args.async_save,
            overwrite_output=args.overwrite,
            validate_config=True,
        )
        report["quantize"] = {
            "output": str(result),
            "seconds": time.monotonic() - started,
            "output_bytes": directory_size(output),
            "memory": memory_snapshot(device),
        }
        log(
            f"quantization completed in {report['quantize']['seconds']:.1f}s; "
            f"output_bytes={report['quantize']['output_bytes']}"
        )
        write_json(report_path, report)
        if args.stage == "quantize":
            return

    if args.stage in ("validate", "all"):
        if not output.is_dir():
            raise FileNotFoundError(
                f"Quantized output is missing: {output}; run --stage quantize first"
            )
        if not args.skip_baseline:
            reset_cuda(device)
            baseline = load_model(model_path, device)
            report["baseline"] = run_inference(
                baseline, processor, device, args.max_new_tokens
            )
            del baseline
            gc.collect()
            reset_cuda(device)
        quantized = load_model(output, device)
        report["quantized_scope"] = quantization_scope(quantized)
        report["quantized"] = run_inference(
            quantized, processor, device, args.max_new_tokens
        )
        if "baseline" in report:
            baseline_ids = report["baseline"]["generated_ids"]
            quantized_ids = report["quantized"]["generated_ids"]
            overlap = sum(
                left == right for left, right in zip(baseline_ids, quantized_ids)
            )
            report["comparison"] = {
                "generated_token_overlap": overlap / max(len(baseline_ids), 1),
                "baseline_text": report["baseline"]["text"],
                "quantized_text": report["quantized"]["text"],
                "first_token_matches": (
                    report["baseline"]["next_token_top10"][0]
                    == report["quantized"]["next_token_top10"][0]
                ),
            }
        report["validation"] = {
            "output_has_finalized_marker": (output / "FINALIZED").is_file(),
            "output_bytes": directory_size(output),
            "memory": memory_snapshot(device),
        }
        if not report["quantized_scope"]["only_text_decoder"]:
            raise RuntimeError(
                "Quantized modules escaped the text decoder: "
                f"{report['quantized_scope']['unexpected_modules']}"
            )
        write_json(report_path, report)
        log(f"validation report written to {report_path}")

    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(report_path, report)


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception:
        log("FAILED")
        raise
