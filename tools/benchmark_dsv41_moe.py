#!/usr/bin/env python3
"""Measure V4.1 MoE dtype and dispatch costs without running full-model PTQ.

Uses synthetic inputs and either synthetic weights or one source checkpoint's
MoE. No checkpoint is changed, and no source FP8/FP4 QDQ is performed. Each
variant uses the same weights, routing, per-expert Linear modules and optional
iMatrix observers. ``current`` is compared exactly against the native-dtype
where loop.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from llmcompressor.modeling.deepseekv41.model import DeepseekV41MoE
from llmcompressor.observers.base import Observer

VARIANTS = ("legacy_fp32", "native_where", "native_counted_where", "current")


def _fp32_linear(module, inputs):
    bias = None if module.bias is None else module.bias.float()
    return F.linear(inputs.float(), module.weight.float(), bias).to(inputs.dtype)


def _native_linear(module, inputs):
    return F.linear(inputs, module.weight, module.bias)


def _where_forward(module, inputs, image_mask=None, *, counted=False):
    shape = inputs.shape
    inputs = inputs.reshape(-1, module.dim)
    mask = None if image_mask is None else image_mask.flatten()
    weights, indices = module.gate(inputs, mask)
    output = torch.zeros_like(inputs, dtype=torch.float32)
    counts = (
        torch.bincount(indices.flatten(), minlength=module.n_routed_experts).tolist()
        if counted
        else None
    )
    for expert_id, expert in enumerate(module.experts):
        if (
            counts is not None
            and not counts[expert_id]
            and not module.calibrate_all_experts
        ):
            continue
        rows, slots = torch.where(indices == expert_id)
        if module.calibrate_all_experts:
            all_weights = inputs.new_ones((inputs.shape[0], 1), dtype=weights.dtype)
            all_weights[rows, 0] = weights[rows, slots]
            values = expert(inputs, all_weights)
            output.index_add_(0, rows, values[rows].float())
        elif rows.numel():
            values = expert(inputs[rows], weights[rows, slots, None])
            output.index_add_(0, rows, values.float())
    return (output + module.shared_experts(inputs)).to(inputs.dtype).reshape(shape)


@contextmanager
def _variant(model, name):
    original_forward = model.forward
    linears = [
        (module, module.forward)
        for module in model.modules()
        if type(module) is nn.Linear
    ]
    try:
        if name != "current":
            model.forward = MethodType(
                lambda self, inputs: _where_forward(
                    self, inputs, counted=name == "native_counted_where"
                ),
                model,
            )
            forward = _fp32_linear if name == "legacy_fp32" else _native_linear
            for module, _ in linears:
                module.forward = MethodType(forward, module)
        yield
    finally:
        model.forward = original_forward
        for module, forward in linears:
            module.forward = forward


@contextmanager
def _observers(model, enabled):
    observers = {}
    try:
        if enabled:
            from compressed_tensors.quantization import QuantizationArgs

            for name, module in model.named_modules():
                if type(module) is not nn.Linear:
                    continue
                observer = Observer.load_from_registry(
                    "imatrix_mse",
                    base_name="weight",
                    args=QuantizationArgs(
                        num_bits=4, strategy="channel", observer="imatrix_mse"
                    ),
                )
                observer.attach(module)
                observers[name] = (module, observer)
        yield observers
    finally:
        for module, observer in observers.values():
            observer.detach(module)


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(model, inputs, device, warmup, repeats):
    for _ in range(warmup):
        model(inputs)
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = model(inputs)
        _synchronize(device)
        times.append(1000 * (time.perf_counter() - start))
    return output, {
        "wall_ms": times,
        "median_ms": statistics.median(times),
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }


def _profile(model, inputs, device):
    # CPU operator counts expose nonzero synchronization and casting without
    # depending on the accelerator vendor's device-profiler implementation.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        model(inputs)
        _synchronize(device)
    selected = {
        "aten::nonzero",
        "aten::where",
        "aten::_to_copy",
        "aten::linear",
        "aten::index",
        "aten::index_select",
        "aten::index_add_",
        "aten::sort",
        "aten::bincount",
    }
    return {
        event.key: {"calls": event.count, "cpu_total_us": event.cpu_time_total}
        for event in profile.key_averages()
        if event.key in selected
    }


def _checkpoint_moe(args, device, dtype):
    from llmcompressor.modeling.deepseekv41 import ModelConfig
    from llmcompressor.streaming import DeepSeekV41WeightMaterializer
    from llmcompressor.streaming.materialization import (
        StreamingDTypePolicy,
        materialize_weights,
    )

    config = ModelConfig.from_pretrained(args.checkpoint, local_files_only=True)
    if (
        getattr(config, "quantization_config", {}).get("quant_method")
        == "compressed-tensors"
    ):
        raise ValueError("--checkpoint must be an original source, not a PTQ output")
    if not 0 <= args.layer_id < config.text_config.n_layers:
        raise ValueError("--layer-id must name a backbone layer in the checkpoint")
    with torch.device("meta"):
        model = DeepseekV41MoE(
            config.text_config,
            vision=config.vision_config.num_hidden_layers > 0,
        ).eval()
    prefix = f"model.layers.{args.layer_id}.ffn"
    materializer = DeepSeekV41WeightMaterializer()
    source = materializer.create_source(str(args.checkpoint))
    policy = StreamingDTypePolicy.for_quantized_modules(
        dtype,
        [
            f"{prefix}.{name}"
            for name, module in model.named_modules()
            if type(module) is nn.Linear
        ],
    )
    values = materialize_weights(
        source,
        [f"{prefix}.{name}" for name in model.state_dict()],
        materializer,
        target_dtype=dtype,
        dtype_policy=policy,
    )
    model.load_state_dict(
        {name.removeprefix(prefix + "."): tensor for name, tensor in values.items()},
        assign=True,
        strict=True,
    )
    del values
    return model.to(device)


@torch.inference_mode()
def benchmark(args):
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    config = SimpleNamespace(
        dim=args.hidden_size,
        moe_inter_dim=args.intermediate_size,
        swiglu_limit=10.0,
        num_experts_per_tok=args.topk,
        n_routed_experts=args.experts,
        dspark_num_experts_per_tok=args.topk,
        dspark_n_routed_experts=args.experts,
        gate_temp=1.0,
        score_func="sqrtsoftplus",
        norm_topk_prob=True,
        route_scale=1.5,
    )
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        if args.checkpoint:
            model = _checkpoint_moe(args, device, dtype)
        else:
            with torch.device(device):
                model = DeepseekV41MoE(config).eval()
                model.gate.weight.normal_(0, 1 / math.sqrt(config.dim))
    finally:
        torch.set_default_dtype(previous_dtype)
    model.calibrate_all_experts = args.calibrate_all_experts
    report = {
        "torch": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        "inputs": "synthetic inputs; "
        + ("source checkpoint weights" if args.checkpoint else "synthetic weights"),
        "model": {
            "hidden_size": model.dim,
            "intermediate_size": model.experts[0].w1.out_features,
            "experts": model.n_routed_experts,
            "topk": model.gate.topk,
        },
        "config": vars(args)
        | {
            "output": str(args.output) if args.output else None,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        },
        "cases": [],
    }
    for tokens in args.tokens:
        inputs = torch.randn(1, tokens, model.dim, device=device, dtype=dtype)
        _, indices = model.gate(inputs.flatten(0, 1))
        counts = torch.bincount(indices.flatten(), minlength=model.n_routed_experts)
        case = {
            "tokens": tokens,
            "active_experts": int((counts > 0).sum()),
            "variants": {},
        }
        baseline = baseline_stats = None
        for name in args.variants:
            with (
                _variant(model, name),
                _observers(model, args.with_imatrix) as observers,
            ):
                output, record = _measure(
                    model, inputs, device, args.warmup, args.repeats
                )
                record["finite"] = bool(torch.isfinite(output).all())
                stats = {
                    name: (
                        observer._imatrix_sum.clone(),
                        observer._imatrix_count.clone(),
                    )
                    for name, (_, observer) in observers.items()
                }
                if name == "native_where":
                    baseline, baseline_stats = output.clone(), stats
                if name == "current" and baseline is not None:
                    record["vs_native_where"] = {
                        "exact": torch.equal(output, baseline),
                        "max_abs": (output.float() - baseline.float())
                        .abs()
                        .max()
                        .item(),
                        "imatrix_exact": (
                            all(
                                torch.equal(total, baseline_stats[name][0])
                                and torch.equal(count, baseline_stats[name][1])
                                for name, (total, count) in stats.items()
                            )
                            if args.with_imatrix
                            else None
                        ),
                    }
                if args.profile:
                    try:
                        record["operators"] = _profile(model, inputs, device)
                    except RuntimeError as error:
                        record["profile_error"] = str(error)
            case["variants"][name] = record
            print(json.dumps({"tokens": tokens, "variant": name, **record}), flush=True)
        report["cases"].append(case)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Load one original checkpoint MoE instead of synthetic weights.",
    )
    parser.add_argument("--layer-id", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--intermediate-size", type=int, default=2304)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--tokens", type=int, nargs="+", default=[512, 8192])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument("--with-imatrix", action="store_true")
    parser.add_argument("--calibrate-all-experts", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        min(
            args.hidden_size,
            args.intermediate_size,
            args.experts,
            args.topk,
            args.repeats,
            *args.tokens,
        )
        <= 0
    ):
        parser.error("Model dimensions, token counts and repeats must be positive")
    if args.topk > args.experts or args.warmup < 0:
        parser.error("topk must not exceed experts and warmup must be nonnegative")
    report = benchmark(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved report to {args.output}")
    comparisons = [
        case["variants"].get("current", {}).get("vs_native_where")
        for case in report["cases"]
    ]
    if any(
        item
        and (not item["exact"] or (args.with_imatrix and not item["imatrix_exact"]))
        for item in comparisons
    ):
        raise SystemExit("Current dispatch differs from the native where baseline")
    if any(
        not record["finite"]
        for case in report["cases"]
        for record in case["variants"].values()
    ):
        raise SystemExit("Nonfinite MoE output")


if __name__ == "__main__":
    main()
