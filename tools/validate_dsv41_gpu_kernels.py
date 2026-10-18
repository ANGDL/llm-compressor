#!/usr/bin/env python3
"""Check V4.1 kernels on an FP8 GPU, without a checkpoint or native FP4.

Only PyTorch, TileLang and the supplied official kernel.py are required.
Project implementations are loaded directly from their source files, bypassing
llmcompressor/Transformers imports and disabling official-kernel dispatch.
Use --self-test for CPU reference checks; these do NOT validate CUDA kernels.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.nn import functional as F

OPERATORS = ("hc", "sparse", "fp8", "fp8_gemm", "mhc", "fp4_reference")


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    # TileLang's source inspection expects the defining module to be registered.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_info(path):
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def json_safe(value):
    """Keep reports valid JSON even if a broken kernel produces NaN/Inf."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def load_project(root):
    base = root / "src/llmcompressor/modeling"
    paths = {
        "kernels": base / "deepseekv4/kernels.py",
        "quantization": base / "deepseekv41/quantization.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        return {}, {"available": False, "missing": missing}
    modules = {
        name: load_file(path, f"_dsv41_alignment_project_{name}")
        for name, path in paths.items()
    }
    # This is a private module instance, not a change to the installed package.
    # Never compare an official kernel against a wrapper calling the same kernel.
    modules["kernels"]._load_reference_kernel_module = lambda: None
    return modules, {
        "available": True,
        "dispatch": "pure PyTorch; official backend forcibly disabled",
        "sources": {name: source_info(path) for name, path in paths.items()},
    }


def hc_reference(mixes, scale, base, hc, iters=20, eps=1e-6):
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def _qk_gemm_fp32(q, values_bf16):
    """Match the official BF16-input/FP32-output sparse QK GEMM contract."""
    batch, sequence, heads, dim = q.shape
    keys = values_bf16.shape[2]
    if q.is_cuda and q.dtype == torch.bfloat16 and values_bf16.dtype == torch.bfloat16:
        left = q.reshape(batch * sequence, heads, dim)
        right = values_bf16.transpose(-1, -2).reshape(batch * sequence, dim, keys)
        return torch.bmm(left, right, out_dtype=torch.float32).reshape(
            batch, sequence, heads, keys
        )
    return torch.einsum(
        "bshd,bskd->bshk",
        q.float(),
        values_bf16.float(),
    )


def _value_gemm_fp32(weights_bf16, values_bf16):
    """Match the official BF16-weight/BF16-value FP32 accumulator GEMM."""
    batch, sequence, heads, keys = weights_bf16.shape
    head_dim = values_bf16.shape[-1]
    if (
        weights_bf16.is_cuda
        and weights_bf16.dtype == torch.bfloat16
        and values_bf16.dtype == torch.bfloat16
    ):
        left = weights_bf16.reshape(batch * sequence, heads, keys)
        right = values_bf16.reshape(batch * sequence, keys, head_dim)
        return torch.bmm(left, right, out_dtype=torch.float32).reshape(
            batch, sequence, heads, head_dim
        )
    return torch.einsum(
        "bshk,bskd->bshd",
        weights_bf16.float(),
        values_bf16.float(),
    )


def sparse_reference(q, kv, sink, indices, scale):
    """Emulate official 64-key online softmax and BF16 unnormalized weights."""
    b, s, h, d = q.shape
    maximum = torch.full((b, s, h, 1), -1e30, device=q.device)
    denominator = torch.zeros_like(maximum)
    acc = torch.zeros((b, s, h, d), device=q.device, dtype=torch.float32)
    batches = torch.arange(b, device=q.device).view(b, 1, 1)
    for start in range(0, indices.size(-1), 64):
        idx = indices[..., start : start + 64]
        idx = F.pad(idx, (0, 64 - idx.shape[-1]), value=-1).long()
        values = kv[batches, idx.clamp_min(0)]
        values = values.masked_fill((idx < 0).unsqueeze(-1), 0)
        scores = _qk_gemm_fp32(q, values) * scale
        scores = scores.masked_fill((idx < 0).unsqueeze(2), -torch.inf)
        new_maximum = torch.maximum(maximum, scores.amax(-1, keepdim=True))
        correction = torch.exp(maximum - new_maximum)
        weights = torch.exp(scores - new_maximum)
        denominator = denominator * correction + weights.sum(-1, keepdim=True)
        acc = acc * correction + _value_gemm_fp32(
            weights.bfloat16(),
            values,
        )
        maximum = new_maximum
    denominator = denominator + torch.exp(sink.view(1, 1, h, 1) - maximum)
    return (acc / denominator).to(q.dtype)


def sparse_cases(*, quick, include_small_head_dim=False):
    cases = [(1, 64, 4), (4, 512, 65), (16, 128, 128)]
    if not quick:
        cases += [
            (h, d, k)
            for h in (1, 2, 4, 8, 16)
            for d, k in ((64, 1), (64, 32), (128, 64), (512, 129))
        ]
        # Production Flash shape, including sliding_window=128 + index_topk=512.
        cases += [(64, 512, k) for k in (128, 512, 640)]
    if include_small_head_dim:
        cases += [(h, 32, 1) for h in (1, 2, 4, 8, 16)]
    return cases


def ceil_power2(x):
    # Unlike log2/ceil, frexp preserves the side of a power-of-two boundary.
    mantissa, exponent = torch.frexp(x)
    exponent = exponent - (mantissa == 0.5).int()
    return torch.ldexp(torch.ones_like(x), exponent)


def fp8_reference(x, group, rounded):
    groups = x.float().unflatten(-1, (-1, group))
    amax = groups.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = amax * (1.0 / 448.0)
    if rounded:
        scale = ceil_power2(scale)
    quant = (groups / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    dequant = (quant.float() * scale).flatten(-2).to(x.dtype)
    return dequant, quant.flatten(-2), scale.squeeze(-1)


def fp4_reference(x, group, e4m3):
    """Scalar E2M1 table with IEEE nearest-even ties; no native FP4 casts."""
    groups = x.float().unflatten(-1, (-1, group))
    minimum = 6 * (2**-9 if e4m3 else 2**-126)
    amax = groups.abs().amax(-1, keepdim=True).clamp_min(minimum)
    scale = amax / 6 if e4m3 else amax * (1.0 / 6.0)
    scale = scale.to(torch.float8_e4m3fn).float() if e4m3 else ceil_power2(scale)
    values = (groups / scale).clamp(-6, 6)
    table = values.new_tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
    distances = (values.abs().unsqueeze(-1) - table).abs()
    # Choose even encodings at midpoints independently of project's bucketize.
    nearest = distances == distances.amin(-1, keepdim=True)
    codes = torch.arange(8, device=x.device)
    priority = codes + (codes % 2) * 8
    chosen = priority.expand_as(distances).masked_fill(~nearest, 32).argmin(-1)
    result = table[chosen] * values.sign() * scale
    return result.flatten(-2).to(x.dtype), scale.squeeze(-1)


def bf16_step_distances(actual, expected):
    def ordered(x):
        bits = x.contiguous().view(torch.int16).int() & 0xFFFF
        magnitude = bits & 0x7FFF
        return torch.where(bits & 0x8000 != 0, 0x8000 - magnitude, 0x8000 + bits)

    return (ordered(actual) - ordered(expected)).abs()


def bf16_steps(actual, expected):
    return int(bf16_step_distances(actual, expected).max().item())


def difference_point(actual, expected, index):
    """Describe a flattened index with its original, zero-based coordinates."""
    index = int(index)
    remainder = index
    coordinate = []
    for size in reversed(actual.shape):
        coordinate.append(remainder % size)
        remainder //= size
    a = float(actual.flatten()[index])
    e = float(expected.flatten()[index])
    return {
        "flat_index": index,
        "coordinate": list(reversed(coordinate)),
        "actual": a,
        "reference": e,
        "abs_difference": abs(a - e),
    }


def compare(actual, expected, *, atol=1e-6, rtol=1e-5, exact=False):
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    same_shape = actual.shape == expected.shape
    result = {
        "actual_shape": list(actual.shape),
        "reference_shape": list(expected.shape),
        "actual_dtype": str(actual.dtype),
        "reference_dtype": str(expected.dtype),
        "finite": finite,
        "atol": atol,
        "rtol": rtol,
        "criterion": "exact" if exact else "allclose",
    }
    if not finite or not same_shape:
        return {**result, "passed": False}
    a, e = actual.double(), expected.double()
    difference = (a - e).abs()
    mismatch = difference != 0
    equal = bool(torch.equal(actual, expected))
    close = bool(torch.allclose(a, e, atol=atol, rtol=rtol))
    result.update(
        exact=equal,
        strict_allclose=bool(torch.allclose(a, e, atol=1e-6, rtol=1e-5)),
        allclose=close,
        max_abs=float(difference.max()),
        rms_abs=float(difference.square().mean().sqrt()),
        relative_l2=float(
            difference.norm() / e.norm().clamp_min(torch.finfo(torch.float64).tiny)
        ),
        mismatched_elements=int(mismatch.sum()),
        elements=actual.numel(),
        passed=(equal if exact else close) and actual.dtype == expected.dtype,
    )
    if actual.dtype == expected.dtype == torch.bfloat16:
        steps = bf16_step_distances(actual, expected)
        result["max_bf16_steps"] = int(steps.max())
        if mismatch.any():
            result["largest_bf16_step_difference"] = {
                **difference_point(actual, expected, steps.flatten().argmax()),
                "bf16_steps": result["max_bf16_steps"],
            }
    if mismatch.any():
        index = difference.flatten().argmax()
        result["largest_difference"] = difference_point(actual, expected, index)
    return result


def tensors_info(**tensors):
    return {
        name: {"shape": list(x.shape), "dtype": str(x.dtype)}
        for name, x in tensors.items()
    }


def randn(shape, device, dtype=torch.float32):
    return torch.randn(shape, device=device, dtype=torch.float32).to(dtype)


def quant_input(m, n, group, device):
    x = randn((m, n), device)
    x *= torch.logspace(-6, 2, n // group, device=device).repeat_interleave(group)
    if m > 1:
        x[0].zero_()
    # Every last-row group has scale 1 for FP4: anchor 6, include both signs,
    # all seven midpoints, and their adjacent BF16 values.
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=device)
    ties = ties.bfloat16()
    around = torch.cat(
        [
            torch.nextafter(ties, torch.full_like(ties, -torch.inf)),
            ties,
            torch.nextafter(ties, torch.full_like(ties, torch.inf)),
        ]
    )
    pattern = torch.cat([around, -around, ties.new_tensor([6, -6])])
    if m > 1:
        x[-1] = pattern.float().repeat(math.ceil(n / pattern.numel()))[:n]
        x[-1, group - 1 :: group] = 6
    return x.bfloat16()


@contextmanager
def default_dtype(dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


class Runner:
    def __init__(self, args, report, official=None, project=None):
        self.args = args
        self.report = report
        self.official = official
        self.project = project or {}
        self.device = torch.device("cpu" if args.self_test else args.device)

    def budget(self, actual, expected):
        return compare(
            actual, expected, atol=self.args.bf16_atol, rtol=self.args.bf16_rtol
        )

    def save(self):
        # Keep a partial report if compilation fails, the process is interrupted,
        # or the user stops a long suite. Atomic replacement prevents torn JSON.
        target = self.args.report
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(json_safe(self.report), indent=2, allow_nan=False) + "\n"
        )
        temporary.replace(target)

    def case(self, operator, name, function):
        begin = time.monotonic()
        seed = (
            self.args.seed
            + int.from_bytes(hashlib.sha256(f"{operator}/{name}".encode()).digest()[:4])
        ) % (2**32)
        try:
            torch.manual_seed(seed)
            with torch.inference_mode():
                details = function()
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
            checks = details.get("checks", {})
            passed = bool(checks) and all(x["passed"] for x in checks.values())
            status = "passed" if passed else "failed"
            if details.pop("diagnostic", False):
                status, passed = "diagnostic", None
            result = dict(
                operator=operator, case=name, status=status, passed=passed, **details
            )
        except Exception as error:
            result = {
                "operator": operator,
                "case": name,
                "status": "error",
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        result["elapsed_seconds"] = round(time.monotonic() - begin, 3)
        result["seed"] = seed
        self.report["results"].append(result)
        self.save()
        print(
            f"[{result['status'].upper()}] {operator}/{name} "
            f"({result['elapsed_seconds']:.2f}s)",
            flush=True,
        )
        for label, check in result.get("checks", {}).items():
            if not check["passed"]:
                print(
                    f"  {label}: max_abs={check.get('max_abs')} "
                    f"relative_l2={check.get('relative_l2')}",
                    flush=True,
                )
        if result.get("error"):
            print(f"  {result['error']}", flush=True)
        if self.args.fail_fast and result["passed"] is False:
            raise RuntimeError(f"Stopped at {operator}/{name}; see JSON report")

    def hc(self, b, s, hc):
        mixes = randn((b, s, (2 + hc) * hc), self.device)
        scale = torch.tensor([0.1, 0.2, 0.3], device=self.device)
        base = randn(((2 + hc) * hc,), self.device)
        expected = hc_reference(mixes, scale, base, hc)
        actual = self.official.hc_split_sinkhorn(mixes, scale, base, hc, 20, 1e-6)
        checks = {
            f"official_vs_order_reference/{label}": compare(a, e)
            for label, a, e in zip(("pre", "post", "comb"), actual, expected)
        }
        if self.project:
            project = self.project["kernels"].hc_split(mixes, scale, base, hc, 20, 1e-6)
            checks.update(
                {
                    f"official_vs_project/{label}": compare(a, p)
                    for label, a, p in zip(("pre", "post", "comb"), actual, project)
                }
            )
        return {
            "inputs": tensors_info(mixes=mixes, scale=scale, base=base),
            "checks": checks,
            "comb_row_sum_error": float((actual[2].sum(-1) - 1).abs().max()),
            "comb_column_sum_error": float((actual[2].sum(-2) - 1).abs().max()),
        }

    def sparse(self, h, d, k, b=2, s=7):
        q = randn((b, s, h, d), self.device, torch.bfloat16)
        kv = randn((b, max(137, k + 3), d), self.device, torch.bfloat16)
        sink = randn((h,), self.device)
        idx = torch.randint(kv.size(1), (b, s, k), device=self.device)
        idx[:, 1, ::2] = -1
        idx[:, 2] = -1
        idx[:, 3] = kv.size(1) - 1
        idx[:, 4] = -1
        idx[:, 4, -1] = kv.size(1) - 1
        if k > 64:
            idx[:, 6, :64] = -1
        idx = idx.int()
        args = q, kv, sink, idx, d**-0.5
        expected = sparse_reference(*args)
        actual = self.official.sparse_attn(*args)
        tail = self.official.sparse_attn(
            q[:, -1:].contiguous(), kv, sink, idx[:, -1:].contiguous(), d**-0.5
        )
        checks = {
            "official_vs_online_reference": self.budget(actual, expected),
            "official_all_masked_zero": compare(
                actual[:, 2], torch.zeros_like(actual[:, 2]), exact=True
            ),
            "official_full_vs_last_query": compare(actual[:, -1:], tail, exact=True),
        }
        if self.project:
            project = self.project["kernels"].sparse_attention(*args)
            checks["official_vs_project"] = self.budget(actual, project)
            # This strict check catches algorithm/order mismatches even when
            # both results fit within the broader official GPU BF16 budget.
            checks["project_vs_online_reference"] = compare(project, expected)
            checks["project_all_masked_zero"] = compare(
                project[:, 2], torch.zeros_like(project[:, 2]), exact=True
            )
            project_tail = self.project["kernels"].sparse_attention(
                q[:, -1:].contiguous(),
                kv,
                sink,
                idx[:, -1:].contiguous(),
                d**-0.5,
            )
            checks["project_full_vs_last_query"] = compare(
                project[:, -1:], project_tail, exact=True
            )
        return {
            "inputs": tensors_info(q=q, kv=kv, sink=sink, indices=idx),
            "patterns": [
                "valid_random",
                "partial_mask",
                "all_masked",
                "duplicate_last_position",
                "only_last_position",
                "valid_random",
                "first_64_masked_if_multiblock",
            ],
            "checks": checks,
            "per_head_max_abs": [
                float((actual[:, :, i].float() - expected[:, :, i].float()).abs().max())
                for i in range(h)
            ],
        }

    def fp8(self, m, n, group, fmt):
        x = quant_input(m, n, group, self.device)
        rounded = fmt != "float32"
        storage = torch.float8_e8m0fnu if fmt == "e8m0" else torch.float32
        expected, quant, scale = fp8_reference(x, group, rounded)
        options = dict(
            block_size=group,
            scale_fmt="ue8m0" if rounded else None,
            scale_dtype=storage,
        )
        actual = self.official.act_quant(x.clone(), inplace=True, **options)
        encoded, official_scale = self.official.act_quant(x.clone(), **options)
        encoded_qdq = (
            (
                encoded.float().unflatten(-1, (-1, group))
                * official_scale.float().unsqueeze(-1)
            )
            .flatten(-2)
            .to(x.dtype)
        )
        checks = {
            "official_qdq_vs_order_reference": compare(actual, expected, exact=True),
            "official_scales": compare(
                official_scale.float(), scale.to(storage).float(), exact=True
            ),
            "official_fp8_codes": compare(encoded.float(), quant.float(), exact=True),
            "official_inplace_vs_encoded_qdq": compare(actual, encoded_qdq, exact=True),
        }
        if rounded and self.project:
            checks["official_vs_project"] = compare(
                actual,
                self.project["quantization"].fp8_qdq(x, group),
                exact=True,
            )
        return {"inputs": tensors_info(x=x), "scale_format": fmt, "checks": checks}

    def fp8_gemm(self, m, n, k, group, fmt):
        storage = torch.float8_e8m0fnu if fmt == "e8m0" else torch.float32
        x = randn((m, k), self.device, torch.bfloat16)
        _, a, scale_a = fp8_reference(x, group, fmt == "e8m0")
        w = randn((n, k), self.device, torch.bfloat16).float()
        blocks = w.unflatten(0, (-1, group)).unflatten(-1, (-1, group))
        scale_b = blocks.abs().amax((1, 3)).clamp_min(1e-4) * (1.0 / 448)
        if fmt == "e8m0":
            scale_b = ceil_power2(scale_b)
        expanded_b = scale_b.repeat_interleave(group, 0).repeat_interleave(group, 1)
        b = (w / expanded_b).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
        scale_a, scale_b = scale_a.to(storage), scale_b.to(storage)
        accum = torch.zeros((m, n), device=self.device)
        for start in range(0, k, group):
            partial = (
                a[:, start : start + group].float()
                @ b[:, start : start + group].float().T
            )
            accum += (
                partial
                * scale_a.float()[:, start // group, None]
                * scale_b.float()[:, start // group].repeat_interleave(group)[None]
            )
        # Official wrapper allocates using the global dtype, while its compiled
        # kernel declares BF16 output. Match generate.py's BF16 default explicitly.
        with default_dtype(torch.bfloat16):
            actual = self.official.fp8_gemm(
                a, scale_a, b, scale_b, scale_dtype=storage, block_size=group
            )
        return {
            "inputs": tensors_info(a=a, b=b, scale_a=scale_a, scale_b=scale_b),
            "checks": {
                "official_vs_block_accumulation": self.budget(actual, accum.bfloat16())
            },
        }

    def mhc(self, dim, hc=4):
        """Shape sensitivity diagnostic; not an end-to-end KV cache test."""
        x = randn((1, 33, hc, dim), self.device, torch.bfloat16)
        weight = randn(((2 + hc) * hc, hc * dim), self.device) / math.sqrt(hc * dim)
        scale = torch.tensor([0.1, 0.2, 0.3], device=self.device)
        base = randn(((2 + hc) * hc,), self.device)
        sublayer = randn((1, 33, dim), self.device, torch.bfloat16)

        def run(value):
            flat = value.flatten(2).float()
            rstd = torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-20)
            projection = F.linear(flat, weight) * rstd
            coeff = self.official.hc_split_sinkhorn(
                projection, scale, base, hc, 20, 1e-6
            )
            return projection, coeff

        full_projection, full = run(x)
        tail_projection, tail = run(x[:, -1:].contiguous())
        same_input = self.official.hc_split_sinkhorn(
            full_projection[:, -1:].contiguous(), scale, base, hc, 20, 1e-6
        )

        def merge(coeff):
            _, post, comb = coeff
            return post.unsqueeze(-1) * sublayer[:, -1:].unsqueeze(-2) + (
                comb.unsqueeze(-1) * x[:, -1:].unsqueeze(-2)
            ).sum(2)

        full_tail = tuple(t[:, -1:] for t in full)
        full_merge, tail_merge = merge(full_tail), merge(tail)
        diagnostics = {
            "fp32_projection_full_vs_decode": compare(
                full_projection[:, -1:], tail_projection
            ),
            "fp32_hc_post_full_vs_decode": compare(full_merge, tail_merge),
            "bf16_hc_post_full_vs_decode": compare(
                full_merge.bfloat16(), tail_merge.bfloat16(), exact=True
            ),
        }
        for label, a, e, fixed in zip(
            ("pre", "post", "comb"), full_tail, tail, same_input
        ):
            diagnostics[f"coefficient_full_vs_decode/{label}"] = compare(a, e)
            diagnostics[f"same_projection_different_shape/{label}"] = compare(
                a, fixed, exact=True
            )
        return {
            "diagnostic": True,
            "inputs": tensors_info(x=x, weight=weight),
            "norm_eps": 1e-20,
            "diagnostics": diagnostics,
            "scope": "random mHC inputs; no model, weights, KV cache or logits",
        }

    def fp4_emulated(self, m, n, group, e4m3):
        x = quant_input(m, n, group, self.device)
        expected, scale = fp4_reference(x, group, e4m3)
        checks = {}
        if self.project:
            checks["project_vs_independent_table_reference"] = compare(
                self.project["quantization"].fp4_qdq(x, group, e4m3_scale=e4m3),
                expected,
                exact=True,
            )
        else:
            return {
                "diagnostic": True,
                "scope": "reference only; project source unavailable",
                "finite": bool(torch.isfinite(expected).all()),
            }
        return {
            "inputs": tensors_info(x=x),
            "scale_format": "e4m3" if e4m3 else "e8m0",
            "min_scale": float(scale.min()),
            "native_fp4_executed": False,
            "scope": "software semantics only; does not validate official FP4 kernel",
            "checks": checks,
        }

    def suite(self):
        selected = set(self.args.operators)
        if "hc" in selected:
            shapes = (
                [(1, 1), (1, 5)]
                if self.args.quick
                else [(1, 1), (1, 5), (1, 32), (2, 7), (4, 64)]
            )
            for hc in (2, 4):
                for b, s in shapes:
                    self.case(
                        "hc",
                        f"b{b}_s{s}_hc{hc}",
                        lambda b=b, s=s, hc=hc: self.hc(b, s, hc),
                    )
        if "sparse" in selected:
            for h, d, k in sparse_cases(
                quick=self.args.quick,
                include_small_head_dim=self.args.include_small_head_dim,
            ):
                self.case(
                    "sparse",
                    f"h{h}_d{d}_k{k}",
                    lambda h=h, d=d, k=k: self.sparse(h, d, k),
                )
        if "fp8" in selected:
            shapes = [(5, 128, 32), (33, 256, 128)]
            if not self.args.quick:
                shapes += [(1, 32, 32), (5, 64, 32), (33, 512, 32)]
            for m, n, group in shapes:
                for fmt in ("float32", "pow2_float32", "e8m0"):
                    self.case(
                        "fp8",
                        f"m{m}_n{n}_g{group}_{fmt}",
                        lambda m=m, n=n, g=group, f=fmt: self.fp8(m, n, g, f),
                    )
        if "fp8_gemm" in selected:
            shapes = [(1, 128, 128, 32), (5, 128, 256, 128)]
            if not self.args.quick:
                shapes += [(33, 256, 512, 32), (33, 256, 512, 128)]
            for m, n, k, group in shapes:
                for fmt in ("float32", "e8m0"):
                    self.case(
                        "fp8_gemm",
                        f"m{m}_n{n}_k{k}_g{group}_{fmt}",
                        lambda m=m, n=n, k=k, g=group, f=fmt: self.fp8_gemm(
                            m, n, k, g, f
                        ),
                    )
        if "mhc" in selected:
            dims = [5120] if self.args.quick else [128, 5120]
            for dim in dims:
                self.case("mhc", f"dim{dim}", lambda d=dim: self.mhc(d))
        if "fp4_reference" in selected:
            for e4m3 in (False, True):
                self.case(
                    "fp4_reference",
                    f"e4m3_scale_{e4m3}",
                    lambda e=e4m3: self.fp4_emulated(5, 128, 32, e),
                )


def cpu_self_test(project):
    """Analytic checks for references, not a substitute for official GPU tests."""
    x = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]]])
    kv = torch.tensor([[[2.0, 3.0]]])
    idx = torch.tensor([[[0, 0], [-1, -1], [0, -1]]], dtype=torch.int32)
    actual = sparse_reference(x, kv, torch.zeros(1), idx, 1.0)
    expected = torch.tensor(
        [
            [
                [[4 / (2 + math.exp(-2)), 6 / (2 + math.exp(-2))]],
                [[0.0, 0.0]],
                [[2 / (1 + math.exp(-5)), 3 / (1 + math.exp(-5))]],
            ]
        ]
    )
    checks = {"sparse_sink_duplicates_all_masked": compare(actual, expected)}
    pre, post, comb = hc_reference(
        torch.zeros(1, 1, 24), torch.ones(3), torch.zeros(24), 4
    )
    checks["hc_pre_zero"] = compare(pre, torch.full_like(pre, 0.500001))
    checks["hc_post_zero"] = compare(post, torch.ones_like(post))
    checks["hc_comb_zero"] = compare(comb, torch.full_like(comb, 1 / (4 + 4e-6)))
    boundary = torch.tensor([0.5, 1.0, 2.0, 4.0])
    checks["power2_boundaries"] = compare(ceil_power2(boundary), boundary, exact=True)
    if project:
        for group in (32, 128):
            x = quant_input(5, 256, group, "cpu")
            checks[f"fp8_project_group{group}"] = compare(
                project["quantization"].fp8_qdq(x, group),
                fp8_reference(x, group, True)[0],
                exact=True,
            )
            for e4m3 in (False, True):
                checks[f"fp4_project_group{group}_e4m3_{e4m3}"] = compare(
                    project["quantization"].fp4_qdq(x, group, e4m3_scale=e4m3),
                    fp4_reference(x, group, e4m3)[0],
                    exact=True,
                )
    return {"checks": checks, "native_gpu_operators_executed": False}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-kernel", type=Path, help="Official inference/kernel.py"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", type=Path, default=Path("dsv41-gpu-alignment.json"))
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--operators", nargs="+", choices=OPERATORS, default=OPERATORS)
    parser.add_argument("--quick", action="store_true", help="Smaller shape matrix")
    parser.add_argument(
        "--include-small-head-dim",
        action="store_true",
        help="Also probe d=32; known official TileLang warp-layout compile limitation",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="CPU references only")
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--no-project",
        action="store_true",
        help="Official vs independent references only",
    )
    parser.add_argument("--bf16-atol", type=float, default=0.002)
    parser.add_argument("--bf16-rtol", type=float, default=0.015625)
    args = parser.parse_args(argv)
    if not args.self_test and args.official_kernel is None:
        parser.error("--official-kernel is required unless using --self-test")
    if any(
        not math.isfinite(value) or value < 0
        for value in (args.bf16_atol, args.bf16_rtol)
    ):
        parser.error("Comparison tolerances must be finite and nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    report = {
        "schema_version": 1,
        "status": "running",
        "mode": "cpu_reference_self_test"
        if args.self_test
        else "official_cuda_alignment",
        "seed": args.seed,
        "requested_operators": list(args.operators),
        "quick": args.quick,
        "include_small_head_dim": args.include_small_head_dim,
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "python": sys.version,
        },
        "comparison_policy": {
            "fp32_atol": 1e-6,
            "fp32_rtol": 1e-5,
            "bf16_atol": args.bf16_atol,
            "bf16_rtol": args.bf16_rtol,
            "quantization": "exact codes, scales and BF16 Q/DQ",
            "note": "BF16 budget pass does not imply exact or strict alignment",
        },
        "native_fp4": {
            "status": "skipped",
            "reason": "Target GPU supports native FP8, not native FP4",
            "official_fp4_kernel_called": False,
        },
        "results": [],
    }
    runner = Runner(args, report)
    try:
        torch.manual_seed(args.seed)
        project, info = (
            ({}, {"available": False, "disabled": True})
            if args.no_project
            else load_project(args.project_root)
        )
        runner.project, report["project"] = project, info
        print(
            "Project comparison: "
            + ("enabled (pure PyTorch)" if project else "unavailable/disabled"),
            flush=True,
        )
        if args.self_test:
            runner.case("self_test", "cpu_references", lambda: cpu_self_test(project))
        else:
            if runner.device.type != "cuda" or not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is required; --self-test checks CPU references only"
                )
            if not args.official_kernel.is_file():
                raise FileNotFoundError(args.official_kernel)
            # E8M0 is a scale storage dtype, not a requirement for native FP4.
            for name in ("float8_e4m3fn", "float8_e8m0fnu"):
                if not hasattr(torch, name):
                    raise RuntimeError(
                        f"Missing torch.{name}; use official torch>=2.10"
                    )
            torch.cuda.set_device(runner.device)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            torch.set_float32_matmul_precision("highest")
            props = torch.cuda.get_device_properties(runner.device)
            report["environment"].update(
                device=str(runner.device),
                gpu=props.name,
                compute_capability=[props.major, props.minor],
                gpu_memory_bytes=props.total_memory,
                tf32=False,
                bf16_reduced_precision_reduction=False,
                tilelang=importlib.metadata.version("tilelang"),
            )
            if report["environment"]["tilelang"] != "0.1.12":
                print(
                    "Note: official requirements specify tilelang==0.1.12; "
                    f"found {report['environment']['tilelang']}",
                    flush=True,
                )
            report["official_kernel"] = source_info(args.official_kernel)
            runner.official = load_file(
                args.official_kernel, "_dsv41_alignment_official_kernel"
            )
            print(f"GPU: {props.name}; native FP4 disabled", flush=True)
            runner.suite()
    except KeyboardInterrupt:
        report["fatal_error"] = "Interrupted; report contains completed cases only"
    except Exception as error:
        report["fatal_error"] = f"{type(error).__name__}: {error}"
        report["fatal_traceback"] = traceback.format_exc()
    counts = {
        state: sum(row["status"] == state for row in report["results"])
        for state in ("passed", "failed", "error", "diagnostic")
    }
    report["summary"] = counts
    if "fatal_error" in report:
        report["status"], exit_code = "incomplete", 2
    elif counts["error"] or counts["failed"]:
        report["status"], exit_code = "failed", 1
    else:
        report["status"], exit_code = (
            "self_test_passed" if args.self_test else "completed",
            0,
        )
    report["scope"] = (
        "Operator tests and mHC shape diagnostics only. No checkpoint, full-model "
        "KV-cache consistency, INT8/INT4 inference or native FP4 validation."
    )
    runner.save()
    print(json.dumps({"status": report["status"], **counts}, ensure_ascii=False))
    if report.get("fatal_error"):
        print(report["fatal_error"], file=sys.stderr)
    print(f"Report: {args.report.resolve()}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
