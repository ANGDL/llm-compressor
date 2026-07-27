"""Command-line access to streaming PTQ stages."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch
from compressed_tensors.quantization import QuantizationScheme

from .collect import collect_calibration_statistics
from .finalize import finalize_streaming_checkpoint
from .oneshot import streaming_oneshot
from .quantize import quantize_streaming


def _json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _factory(reference: str):
    module_name, separator, attribute = reference.partition(":")
    if not separator:
        raise ValueError("model factory must use module.path:attribute syntax")
    return getattr(importlib.import_module(module_name), attribute)


def _schemes(path: str) -> dict[str, QuantizationScheme]:
    return {
        name: QuantizationScheme.model_validate(value)
        for name, value in _json(path).items()
    }


def _batches(path: str):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, (list, tuple)):
        raise TypeError("calibration batch file must contain a list or tuple")
    return value


def _common_collect(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--model-factory", required=True)
    parser.add_argument("--batches", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--dataset-fingerprint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--algorithms", nargs="+", choices=("gptq", "imatrix"), default=("gptq",)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmcompressor.streaming")
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    _common_collect(collect)

    quantize = commands.add_parser("quantize")
    quantize.add_argument("--checkpoint", required=True)
    quantize.add_argument("--artifacts", required=True)
    quantize.add_argument("--staging", required=True)
    quantize.add_argument("--schemes", required=True)
    quantize.add_argument("--device", default="cpu")
    quantize.add_argument("--dtype", default="bfloat16")
    quantize.add_argument("--no-gptq", action="store_true")

    finalize = commands.add_parser("finalize")
    finalize.add_argument("--checkpoint", required=True)
    finalize.add_argument("--artifacts", required=True)
    finalize.add_argument("--staging", required=True)
    finalize.add_argument("--output", required=True)
    finalize.add_argument("--skip-config-validation", action="store_true")

    run = commands.add_parser("run")
    _common_collect(run)
    run.add_argument("--work-dir", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--schemes", required=True)
    run.add_argument("--no-gptq", action="store_true")
    run.add_argument("--skip-config-validation", action="store_true")
    return parser


def _dtype(name: str) -> torch.dtype:
    try:
        dtype = getattr(torch, name)
    except AttributeError as error:
        raise ValueError(f"Unknown torch dtype {name!r}") from error
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError(f"dtype must be floating-point, got {name!r}")
    return dtype


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "collect":
        collect_calibration_statistics(
            model_factory=_factory(args.model_factory),
            checkpoint=args.checkpoint,
            artifact_dir=args.artifacts,
            calibration_batches=_batches(args.batches),
            targets=args.targets,
            recipe=_json(args.recipe),
            dataset_fingerprint=args.dataset_fingerprint,
            algorithms=args.algorithms,
            device=args.device,
            target_dtype=_dtype(args.dtype),
        )
    elif args.command == "quantize":
        quantize_streaming(
            checkpoint=args.checkpoint,
            artifact_dir=args.artifacts,
            staging_dir=args.staging,
            schemes=_schemes(args.schemes),
            use_gptq=not args.no_gptq,
            device=args.device,
            target_dtype=_dtype(args.dtype),
        )
    elif args.command == "finalize":
        finalize_streaming_checkpoint(
            checkpoint=args.checkpoint,
            artifact_dir=args.artifacts,
            staging_dir=args.staging,
            output_dir=args.output,
            validate_config=not args.skip_config_validation,
        )
    else:
        work = Path(args.work_dir)
        streaming_oneshot(
            model_factory=_factory(args.model_factory),
            checkpoint=args.checkpoint,
            output_dir=args.output,
            work_dir=work,
            calibration_batches=_batches(args.batches),
            targets=args.targets,
            recipe=_json(args.recipe),
            dataset_fingerprint=args.dataset_fingerprint,
            schemes=_schemes(args.schemes),
            algorithms=args.algorithms,
            use_gptq=not args.no_gptq,
            device=args.device,
            target_dtype=_dtype(args.dtype),
            validate_config=not args.skip_config_validation,
        )
