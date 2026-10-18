#!/usr/bin/env python3
"""Direct kernel-level comparison for DeepSeek V4.1.

This tool bypasses models, checkpoints, Transformers and streaming inference.
It calls the official TileLang kernel functions directly, then compares their
outputs with the project's pure PyTorch implementations and independent
operation-order references.

Native FP4 kernels are intentionally excluded. The target GPU only needs native
FP8 support for this comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import torch

try:
    from tools.dsv41_sparse_trace import (
        compare_qk_backends,
        pad_official_inputs,
        reference_qk_backend,
        reference_trace,
        run_official_trace,
        summarize_trace,
        to_cpu,
        trim_trace_heads,
        write_instrumented_source,
    )
    from tools.validate_dsv41_gpu_kernels import (
        ceil_power2,
        compare,
        fp8_reference,
        hc_reference,
        json_safe,
        load_file,
        load_project,
        quant_input,
        randn,
        source_info,
        sparse_reference,
    )
except ModuleNotFoundError:
    from dsv41_sparse_trace import (
        compare_qk_backends,
        pad_official_inputs,
        reference_qk_backend,
        reference_trace,
        run_official_trace,
        summarize_trace,
        to_cpu,
        trim_trace_heads,
        write_instrumented_source,
    )
    from validate_dsv41_gpu_kernels import (
        ceil_power2,
        compare,
        fp8_reference,
        hc_reference,
        json_safe,
        load_file,
        load_project,
        quant_input,
        randn,
        source_info,
        sparse_reference,
    )

OPERATORS = ("hc", "sparse", "fp8", "fp8_gemm")


@contextmanager
def default_dtype(dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def fp8_gemm_reference(a, a_scale, b, b_scale, block_size):
    result = torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float32, device=a.device)
    for start in range(0, a.shape[1], block_size):
        partial = (
            a[:, start : start + block_size].float()
            @ b[:, start : start + block_size].float().T
        )
        result += (
            partial
            * a_scale.float()[:, start // block_size, None]
            * b_scale.float()[:, start // block_size].repeat_interleave(block_size)[
                None
            ]
        )
    return result.bfloat16()


def make_fp8_gemm_inputs(m, n, k, block_size, device, rounded):
    activation = randn((m, k), device, torch.bfloat16)
    _, a, a_scale = fp8_reference(activation, block_size, rounded)
    weights = randn((n, k), device, torch.bfloat16).float()
    blocks = weights.unflatten(0, (-1, block_size)).unflatten(-1, (-1, block_size))
    b_scale = blocks.abs().amax((1, 3)).clamp_min(1e-4) * (1.0 / 448)
    if rounded:
        b_scale = ceil_power2(b_scale)
    expanded = b_scale.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
    b = (weights / expanded).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
    scale_dtype = torch.float8_e8m0fnu if rounded else torch.float32
    return (
        a,
        a_scale.to(scale_dtype),
        b,
        b_scale.to(scale_dtype),
        scale_dtype,
    )


def kernel_sparse_cases(*, quick):
    return [(4, 512, 65)] if quick else [(4, 512, 65), (64, 512, 640)]


class KernelComparator:
    def __init__(
        self,
        official,
        project,
        device,
        seed,
        quick,
        *,
        trace_sparse=False,
        trace_dir=None,
        save_trace_tensors=False,
        progress=None,
    ):
        self.official = official
        self.project = project
        self.device = torch.device(device)
        self.seed = seed
        self.quick = quick
        self.results = []
        self.trace_sparse = trace_sparse
        self.trace_dir = trace_dir
        self.save_trace_tensors = save_trace_tensors
        self.progress = progress
        self.trace_module = None

    def _seed(self, name):
        return (
            self.seed + int.from_bytes(hashlib.sha256(name.encode()).digest()[:4])
        ) % (2**32)

    def case(self, operator, name, function):
        started = time.monotonic()
        seed = self._seed(f"{operator}/{name}")
        try:
            torch.manual_seed(seed)
            with torch.inference_mode():
                result = function()
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
            checks = result.get("checks", {})
            passed = bool(checks) and all(item["passed"] for item in checks.values())
            row = {
                "operator": operator,
                "case": name,
                "status": "passed" if passed else "failed",
                "passed": passed,
                "seed": seed,
                **result,
            }
        except Exception as error:
            row = {
                "operator": operator,
                "case": name,
                "status": "error",
                "passed": False,
                "seed": seed,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        row["elapsed_seconds"] = round(time.monotonic() - started, 3)
        self.results.append(row)
        if self.progress:
            self.progress(None)
        print(
            f"[{row['status'].upper()}] {operator}/{name} "
            f"({row['elapsed_seconds']:.2f}s)",
            flush=True,
        )
        for label, check in row.get("checks", {}).items():
            if not check["passed"]:
                print(
                    f"  {label}: max_abs={check.get('max_abs')} "
                    f"relative_l2={check.get('relative_l2')}",
                    flush=True,
                )
        trace = row.get("sparse_trace")
        if trace:
            print(
                f"  trace: status={trace['status']} "
                f"attribution_valid={trace['attribution_valid']}",
                flush=True,
            )
            for label in (
                "first_exact_divergence",
                "first_strict_divergence",
                "first_bf16_weight_divergence",
            ):
                point = trace.get(label)
                if point:
                    print(
                        f"  {label}: block={point['block']} stage={point['stage']} "
                        f"max_abs={point['comparison'].get('max_abs')}",
                        flush=True,
                    )

    def hc(self, hc, b, s):
        mixes = randn((b, s, (2 + hc) * hc), self.device)
        scale = torch.tensor([0.1, 0.2, 0.3], device=self.device)
        base = randn(((2 + hc) * hc,), self.device)
        official = self.official.hc_split_sinkhorn(mixes, scale, base, hc, 20, 1e-6)
        project = self.project["kernels"].hc_split(mixes, scale, base, hc, 20, 1e-6)
        reference = hc_reference(mixes, scale, base, hc)
        checks = {}
        for label, actual, expected in zip(
            ("pre", "post", "comb"), official, reference
        ):
            checks[f"official_vs_order_reference/{label}"] = compare(actual, expected)
        for label, actual, expected in zip(("pre", "post", "comb"), official, project):
            checks[f"official_vs_project/{label}"] = compare(actual, expected)
        for label, actual, expected in zip(("pre", "post", "comb"), project, reference):
            checks[f"project_vs_order_reference/{label}"] = compare(actual, expected)
        return {
            "inputs": {
                "mixes": {"shape": list(mixes.shape), "dtype": str(mixes.dtype)},
                "scale": {"shape": list(scale.shape), "dtype": str(scale.dtype)},
                "base": {"shape": list(base.shape), "dtype": str(base.dtype)},
            },
            "checks": checks,
        }

    def sparse(self, h, d, k):
        batch, sequence = 2, 7
        q = randn((batch, sequence, h, d), self.device, torch.bfloat16)
        kv = randn((batch, max(137, k + 3), d), self.device, torch.bfloat16)
        sink = randn((h,), self.device)
        indices = torch.randint(kv.shape[1], (batch, sequence, k), device=self.device)
        indices[:, 1, ::2] = -1
        indices[:, 2] = -1
        indices[:, 3] = kv.shape[1] - 1
        indices[:, 4] = -1
        indices[:, 4, -1] = kv.shape[1] - 1
        if k > 64:
            indices[:, 6, :64] = -1
        indices = indices.int()
        args = q, kv, sink, indices, d**-0.5
        reference = sparse_reference(*args)
        official = self.official.sparse_attn(*args)
        project = self.project["kernels"].sparse_attention(*args)
        official_tail = self.official.sparse_attn(
            q[:, -1:].contiguous(),
            kv,
            sink,
            indices[:, -1:].contiguous(),
            d**-0.5,
        )
        project_tail = self.project["kernels"].sparse_attention(
            q[:, -1:].contiguous(),
            kv,
            sink,
            indices[:, -1:].contiguous(),
            d**-0.5,
        )
        checks = {
            "official_vs_order_reference": compare(official, reference),
            "official_vs_project": compare(official, project),
            "project_vs_order_reference": compare(project, reference),
            "official_full_vs_tail": compare(
                official[:, -1:], official_tail, exact=True
            ),
            "project_full_vs_tail": compare(project[:, -1:], project_tail, exact=True),
            "official_all_masked_zero": compare(
                official[:, 2], torch.zeros_like(official[:, 2]), exact=True
            ),
            "project_all_masked_zero": compare(
                project[:, 2], torch.zeros_like(project[:, 2]), exact=True
            ),
        }
        result = {
            "inputs": {
                "q": {"shape": list(q.shape), "dtype": str(q.dtype)},
                "kv": {"shape": list(kv.shape), "dtype": str(kv.dtype)},
                "sink": {"shape": list(sink.shape), "dtype": str(sink.dtype)},
                "indices": {"shape": list(indices.shape), "dtype": str(indices.dtype)},
            },
            "kernel_contract": {
                "online_block_size": 64,
                "score_dtype": "float32",
                "value_weight_dtype": "bfloat16",
                "accumulator_dtype": "float32",
                "output_dtype": "bfloat16",
            },
            "checks": checks,
        }
        if self.trace_sparse:
            # Preserve unmodified official/project checks before diagnostic JIT.
            if self.progress:
                self.progress(
                    {
                        "operator": "sparse",
                        "case": f"h{h}_d{d}_k{k}",
                        "status": "tracing",
                        **result,
                    }
                )
            try:
                result["sparse_trace"] = self.sparse_diagnostics(
                    args, official, project, reference, checks
                )
            except Exception as error:
                checks["sparse_trace_completed"] = {"passed": False}
                result["sparse_trace"] = {
                    "status": "error",
                    "attribution_valid": False,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
        return result

    def sparse_diagnostics(self, args, official, project, reference, checks):
        if self.trace_module is None:
            path = write_instrumented_source(
                self.official.__file__, self.trace_dir / "official_sparse_trace.py"
            )
            self.trace_module = load_file(path, "_dsv41_sparse_trace_official")
        traced, kernel = run_official_trace(self.trace_module, *args)
        q, kv, sink, indices, scale = args
        padded_q, padded_sink, requested_heads = pad_official_inputs(q, sink)
        expected = trim_trace_heads(
            reference_trace(padded_q, kv, padded_sink, indices, scale),
            requested_heads,
        )
        checks["instrumented_vs_official"] = compare(
            traced["final"]["output"], official, exact=True
        )
        checks["trace_reference_vs_order_reference"] = compare(
            expected["final"]["output"], reference, exact=True
        )
        checks["trace_reference_vs_project"] = compare(
            expected["final"]["output"], project, exact=True
        )
        checks["instrumented_normalized_cast_vs_output"] = compare(
            traced["final"]["normalized"].to(args[0].dtype),
            traced["final"]["output"],
            exact=True,
        )
        nan_elements = sum(
            int(torch.isnan(value).sum())
            for section in [*traced["blocks"], traced["final"]]
            for value in section.values()
        )
        checks["trace_snapshots_written"] = {
            "passed": nan_elements == 0,
            "nan_elements": nan_elements,
        }
        trusted = all(
            checks[label]["passed"]
            for label in (
                "instrumented_vs_official",
                "trace_reference_vs_order_reference",
                "trace_reference_vs_project",
                "instrumented_normalized_cast_vs_output",
                "trace_snapshots_written",
            )
        )
        diagnostics = summarize_trace(
            traced,
            expected,
            checks["official_vs_project"],
            attribution_valid=trusted,
        )
        diagnostics["instrumented_source"] = source_info(
            Path(self.trace_module.__file__)
        )
        diagnostics["trace_helper_source"] = source_info(
            Path(reference_trace.__code__.co_filename)
        )
        diagnostics["original_kernel_unchanged"] = True
        diagnostics["gate"] = (
            "Attribute intermediate differences only if instrumented output exactly "
            "matches official output and the PyTorch trace exactly matches both "
            "project and independent-reference outputs."
        )
        diagnostics["reference_scope"] = (
            "PyTorch operation-order reference, checked against project output; "
            "not instrumentation of the production project function. All-masked "
            "final normalization follows official 0/inf, not calibration-safe 0/1."
        )
        diagnostics["head_padding"] = {
            "requested_heads": requested_heads,
            "official_kernel_heads": padded_q.shape[2],
            "reference_heads": padded_q.shape[2],
            "trace_trimmed_to_heads": requested_heads,
        }
        diagnostics["qk_backend_comparison"] = compare_qk_backends(
            traced, q, kv, indices, scale, requested_heads
        )
        diagnostics["reference_qk_backend"] = reference_qk_backend(padded_q)
        name = f"h{q.shape[2]}_d{q.shape[-1]}_k{indices.shape[-1]}"
        if self.save_trace_tensors:
            path = self.trace_dir / f"{name}.pt"
            torch.save(
                to_cpu(
                    {
                        "q": q,
                        "kv": kv,
                        "sink": sink,
                        "indices": indices,
                        "scale": scale,
                        "official": official,
                        "project": project,
                        "official_trace": traced,
                        "reference_trace": expected,
                    }
                ),
                path,
            )
            diagnostics["tensor_artifact"] = str(path.resolve())
        if hasattr(kernel, "get_kernel_source"):
            try:
                path = self.trace_dir / f"{name}.cu"
                path.write_text(kernel.get_kernel_source())
                diagnostics["generated_kernel_source"] = source_info(path)
            except Exception as error:
                diagnostics["kernel_source_export_error"] = str(error)
        return diagnostics

    def fp8(self, m, n, group):
        x = quant_input(m, n, group, self.device)
        options = {
            "block_size": group,
            "scale_fmt": "ue8m0",
            "scale_dtype": torch.float8_e8m0fnu,
        }
        official_inplace = self.official.act_quant(x.clone(), inplace=True, **options)
        encoded, scales = self.official.act_quant(x.clone(), **options)
        project = self.project["quantization"].fp8_qdq(x, group)
        expected, codes, expected_scales = fp8_reference(x, group, True)
        encoded_qdq = (
            (encoded.float().unflatten(-1, (-1, group)) * scales.float().unsqueeze(-1))
            .flatten(-2)
            .to(x.dtype)
        )
        checks = {
            "official_inplace_vs_reference": compare(
                official_inplace, expected, exact=True
            ),
            "official_project_vs_reference": compare(project, expected, exact=True),
            "official_codes": compare(encoded.float(), codes.float(), exact=True),
            "official_scales": compare(
                scales.float(), expected_scales.to(scales.dtype).float(), exact=True
            ),
            "official_inplace_vs_encoded": compare(
                official_inplace, encoded_qdq, exact=True
            ),
        }
        return {
            "inputs": {"x": {"shape": list(x.shape), "dtype": str(x.dtype)}},
            "kernel_contract": {
                "block_size": group,
                "activation_dtype": "bfloat16",
                "quant_dtype": "float8_e4m3fn",
                "scale_dtype": "float8_e8m0fnu",
            },
            "checks": checks,
        }

    def fp8_gemm(self, m, n, k, group):
        a, a_scale, b, b_scale, scale_dtype = make_fp8_gemm_inputs(
            m, n, k, group, self.device, rounded=True
        )
        reference = fp8_gemm_reference(a, a_scale, b, b_scale, group)
        with default_dtype(torch.bfloat16):
            official = self.official.fp8_gemm(
                a,
                a_scale,
                b,
                b_scale,
                scale_dtype=scale_dtype,
                block_size=group,
            )
        return {
            "inputs": {
                "a": {"shape": list(a.shape), "dtype": str(a.dtype)},
                "b": {"shape": list(b.shape), "dtype": str(b.dtype)},
                "a_scale": {"shape": list(a_scale.shape), "dtype": str(a_scale.dtype)},
                "b_scale": {"shape": list(b_scale.shape), "dtype": str(b_scale.dtype)},
            },
            "kernel_contract": {
                "activation_dtype": "float8_e4m3fn",
                "weight_dtype": "float8_e4m3fn",
                "scale_dtype": "float8_e8m0fnu",
                "accumulator_dtype": "float32",
                "output_dtype": "bfloat16",
            },
            "project_kernel": {
                "status": "not_available",
                "reason": (
                    "Project uses BF16 reference Linear, not a native FP8 GEMM kernel"
                ),
            },
            "checks": {
                "official_vs_block_scaled_reference": compare(official, reference)
            },
        }

    def run(self, operators):
        if "hc" in operators:
            shapes = [(2, 1), (4, 64)] if not self.quick else [(4, 4)]
            for hc in (2, 4):
                for b, s in shapes:
                    self.case(
                        "hc",
                        f"hc{hc}_b{b}_s{s}",
                        lambda hc=hc, b=b, s=s: self.hc(hc, b, s),
                    )
        if "sparse" in operators:
            for h, d, k in kernel_sparse_cases(quick=self.quick):
                self.case(
                    "sparse",
                    f"h{h}_d{d}_k{k}",
                    lambda h=h, d=d, k=k: self.sparse(h, d, k),
                )
        if "fp8" in operators:
            cases = [(5, 128, 32), (33, 256, 128)]
            if self.quick:
                cases = cases[:1]
            for m, n, group in cases:
                self.case(
                    "fp8",
                    f"m{m}_n{n}_g{group}",
                    lambda m=m, n=n, g=group: self.fp8(m, n, g),
                )
        if "fp8_gemm" in operators:
            cases = [(1, 128, 128, 32), (33, 256, 512, 128)]
            if self.quick:
                cases = cases[:1]
            for m, n, k, group in cases:
                self.case(
                    "fp8_gemm",
                    f"m{m}_n{n}_k{k}_g{group}",
                    lambda m=m, n=n, k=k, g=group: self.fp8_gemm(m, n, k, g),
                )


def cpu_self_test(project):
    device = torch.device("cpu")
    torch.manual_seed(20261018)
    mixes = torch.randn(2, 4, 24)
    scale = torch.tensor([0.1, 0.2, 0.3])
    base = torch.randn(24)
    project_hc = project["kernels"].hc_split(mixes, scale, base, 4, 20, 1e-6)
    reference_hc = hc_reference(mixes, scale, base, 4)
    hc_checks = {
        label: compare(actual, expected)
        for label, actual, expected in zip(
            ("pre", "post", "comb"), project_hc, reference_hc
        )
    }
    q = torch.randn(1, 2, 4, 64).bfloat16()
    kv = torch.randn(1, 129, 64).bfloat16()
    sink = torch.randn(4)
    indices = torch.randint(129, (1, 2, 65), dtype=torch.int32)
    sparse_checks = {
        "project_vs_reference": compare(
            project["kernels"].sparse_attention(q, kv, sink, indices, 64**-0.5),
            sparse_reference(q, kv, sink, indices, 64**-0.5),
        )
    }
    traced = reference_trace(q, kv, sink, indices, 64**-0.5)
    sparse_checks["trace_vs_reference"] = compare(
        traced["final"]["output"],
        sparse_reference(q, kv, sink, indices, 64**-0.5),
        exact=True,
    )
    x = quant_input(5, 128, 32, device)
    fp8_checks = {
        "project_vs_reference": compare(
            project["quantization"].fp8_qdq(x, 32),
            fp8_reference(x, 32, True)[0],
            exact=True,
        )
    }
    checks = {
        **{f"hc/{key}": value for key, value in hc_checks.items()},
        **{f"sparse/{key}": value for key, value in sparse_checks.items()},
        **{f"fp8/{key}": value for key, value in fp8_checks.items()},
    }
    return {
        "status": (
            "passed" if all(item["passed"] for item in checks.values()) else "failed"
        ),
        "checks": checks,
        "official_kernel_called": False,
        "native_fp4_called": False,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-kernel", type=Path)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", type=Path, default=Path("dsv41-kernel-level.json"))
    parser.add_argument("--operators", nargs="+", choices=OPERATORS, default=OPERATORS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--trace-sparse",
        action="store_true",
        help="Instrument a copy of the official sparse kernel and compare each block",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        help="Diagnostic source/tensor directory; defaults beside the JSON report",
    )
    parser.add_argument(
        "--save-sparse-tensors",
        action="store_true",
        help="Save inputs, outputs and complete traces as CPU .pt artifacts",
    )
    args = parser.parse_args(argv)
    if not args.self_test and args.official_kernel is None:
        parser.error("--official-kernel is required unless using --self-test")
    if args.trace_sparse and (args.self_test or "sparse" not in args.operators):
        parser.error("--trace-sparse requires CUDA mode and the sparse operator")
    if (args.save_sparse_tensors or args.trace_dir) and not args.trace_sparse:
        parser.error("--trace-dir/--save-sparse-tensors require --trace-sparse")
    return args


def save_report(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv=None):
    args = parse_args(argv)
    report = {
        "schema_version": 2,
        "tool_source": source_info(Path(__file__)),
        "mode": "cpu_self_test" if args.self_test else "direct_official_kernel",
        "status": "running",
        "requested_operators": list(args.operators),
        "quick": args.quick,
        "seed": args.seed,
        "sparse_trace_requested": args.trace_sparse,
        "scope": (
            "Direct official kernel API comparison only; no model, checkpoint, "
            "Transformers, cache or streaming pipeline."
        ),
        "native_fp4": {
            "status": "skipped",
            "official_kernel_called": False,
            "reason": "Target GPU is FP8-only",
        },
        "results": [],
    }
    try:
        project, project_info = load_project(args.project_root)
        report["project"] = project_info
        if not project:
            raise RuntimeError("Project source files are unavailable")
        if args.self_test:
            result = cpu_self_test(project)
            report["results"].append(
                {"operator": "self_test", "case": "references", **result}
            )
        else:
            device = torch.device(args.device)
            if device.type != "cuda" or not torch.cuda.is_available():
                raise RuntimeError("CUDA is required; use --self-test for CPU checks")
            if not args.official_kernel.is_file():
                raise FileNotFoundError(args.official_kernel)
            for name in ("float8_e4m3fn", "float8_e8m0fnu"):
                if not hasattr(torch, name):
                    raise RuntimeError(f"Missing torch.{name}; use torch>=2.10")
            torch.cuda.set_device(device)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            torch.set_float32_matmul_precision("highest")
            properties = torch.cuda.get_device_properties(device)
            report["environment"] = {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "tilelang": importlib.metadata.version("tilelang"),
                "device": str(device),
                "gpu": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "tf32": False,
                "bf16_reduced_precision_reduction": False,
            }
            report["official_kernel"] = source_info(args.official_kernel)
            official = load_file(args.official_kernel, "_dsv41_kernel_level_official")

            def progress(pending):
                if pending is None:
                    report.pop("in_progress", None)
                else:
                    report["in_progress"] = pending
                save_report(report, args.report)

            comparator = KernelComparator(
                official,
                project,
                device,
                args.seed,
                args.quick,
                trace_sparse=args.trace_sparse,
                trace_dir=args.trace_dir
                or args.report.with_name(f"{args.report.stem}-artifacts"),
                save_trace_tensors=args.save_sparse_tensors,
                progress=progress,
            )
            report["results"] = comparator.results
            save_report(report, args.report)
            comparator.run(args.operators)
    except Exception as error:
        report["status"] = "error"
        report["fatal_error"] = f"{type(error).__name__}: {error}"
        report["fatal_traceback"] = traceback.format_exc()
    if "fatal_error" not in report:
        failed = [
            row
            for row in report["results"]
            if row.get("status") in ("failed", "error") or row.get("passed") is False
        ]
        report["status"] = "failed" if failed else "passed"
    report["summary"] = {
        "passed": sum(row.get("status") == "passed" for row in report["results"]),
        "failed": sum(row.get("status") == "failed" for row in report["results"]),
        "error": sum(row.get("status") == "error" for row in report["results"]),
    }
    if args.trace_sparse:
        report["sparse_trace_summary"] = {
            status: sum(
                row.get("sparse_trace", {}).get("status") == status
                for row in report["results"]
            )
            for status in ("completed", "untrusted", "error")
        }
    save_report(report, args.report)
    print(json.dumps({"status": report["status"], **report["summary"]}))
    print(f"Report: {args.report.resolve()}")
    return 2 if "fatal_error" in report else (1 if report["status"] == "failed" else 0)


if __name__ == "__main__":
    raise SystemExit(main())
