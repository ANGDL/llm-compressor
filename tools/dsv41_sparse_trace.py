"""Sparse-attention diagnostic snapshots; no model or TileLang import on CPU.

Instrument the supplied official source by inserting only global-memory stores.
The original statements, GEMMs, pipeline stages and thread count are retained.
Instrumentation can still change compiler decisions: its output must be checked
against the unmodified official kernel before attributing intermediate differences.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import torch
from torch.nn import functional as F

try:
    from tools.validate_dsv41_gpu_kernels import compare
except ModuleNotFoundError:
    from validate_dsv41_gpu_kernels import compare


BLOCK_STAGES = (
    "qk",
    "scaled_scores",
    "running_max",
    "rescale",
    "exp",
    "block_sum",
    "denominator",
    "bf16_weights",
    "rescaled_accumulator",
    "accumulator",
)
FINAL_STAGES = ("denominator_with_sink", "normalized", "output")

# Match complete AST statements, not substrings which could silently insert a
# snapshot on the wrong side of a GEMM, reduction or precision conversion.
SNAPSHOTS = (
    (
        "T.gemm(q_shared, kv_shared, acc_s, transpose_B=True, "
        "policy=T.GemmWarpPolicy.FullRow)",
        """for i, j in T.Parallel(h, block):
            trace_scores[by, bx, t, 0, i, j] = acc_s[i, j]""",
    ),
    (
        """for i, j in T.Parallel(h, block):
            acc_s[i, j] *= scale""",
        """for i, j in T.Parallel(h, block):
            trace_scores[by, bx, t, 1, i, j] = acc_s[i, j]""",
    ),
    (
        "T.reduce_max(acc_s, scores_max, dim=1, clear=False)",
        """for i in T.Parallel(h):
            trace_rows[by, bx, t, 0, i] = scores_max[i]""",
    ),
    (
        """for i in T.Parallel(h):
            scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])""",
        """for i in T.Parallel(h):
            trace_rows[by, bx, t, 1, i] = scores_scale[i]""",
    ),
    (
        """for i, j in T.Parallel(h, block):
            acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])""",
        """for i, j in T.Parallel(h, block):
            trace_scores[by, bx, t, 2, i, j] = acc_s[i, j]""",
    ),
    (
        "T.reduce_sum(acc_s, scores_sum, dim=1)",
        """for i in T.Parallel(h):
            trace_rows[by, bx, t, 2, i] = scores_sum[i]""",
    ),
    (
        """for i in T.Parallel(h):
            sum_exp[i] = sum_exp[i] * scores_scale[i] + scores_sum[i]""",
        """for i in T.Parallel(h):
            trace_rows[by, bx, t, 3, i] = sum_exp[i]""",
    ),
    (
        "T.copy(acc_s, acc_s_cast)",
        """for i, j in T.Parallel(h, block):
            trace_weights[by, bx, t, i, j] = acc_s_cast[i, j]""",
    ),
    (
        """for i, j in T.Parallel(h, d):
            acc_o[i, j] *= scores_scale[i]""",
        """for i, j in T.Parallel(h, d):
            trace_acc[by, bx, t, 0, i, j] = acc_o[i, j]""",
    ),
    (
        "T.gemm(acc_s_cast, kv_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)",
        """for i, j in T.Parallel(h, d):
            trace_acc[by, bx, t, 1, i, j] = acc_o[i, j]""",
    ),
    (
        """for i in T.Parallel(h):
            sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])""",
        """for i in T.Parallel(h):
            trace_final_denominator[by, bx, i] = sum_exp[i]""",
    ),
    (
        """for i, j in T.Parallel(h, d):
            acc_o[i, j] /= sum_exp[i]""",
        """for i, j in T.Parallel(h, d):
            trace_normalized[by, bx, i, j] = acc_o[i, j]""",
    ),
)
TRACE_ARGUMENTS = """\
trace_scores: T.Tensor[(b, m, num_blocks, 3, h, block), FP32],
trace_rows: T.Tensor[(b, m, num_blocks, 4, h), FP32],
trace_weights: T.Tensor[(b, m, num_blocks, h, block), BF16],
trace_acc: T.Tensor[(b, m, num_blocks, 2, h, d), FP32],
trace_final_denominator: T.Tensor[(b, m, h), FP32],
trace_normalized: T.Tensor[(b, m, h, d), FP32],
"""


def _ast_key(node):
    return ast.dump(node, include_attributes=False)


def instrument_official_source(source):
    """Fail closed if the expected official snapshot boundaries have changed."""
    tree = ast.parse(source)
    factories = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "sparse_attn_kernel"
    ]
    if len(factories) != 1:
        raise ValueError("Expected one official sparse_attn_kernel factory")
    factory = factories[0]
    block_assignments = [
        node
        for node in factory.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "block"
            for target in node.targets
        )
    ]
    if len(block_assignments) != 1 or _ast_key(block_assignments[0].value) != _ast_key(
        ast.Constant(64)
    ):
        raise ValueError("Official sparse trace requires block=64")
    functions = [
        node
        for node in factory.body
        if isinstance(node, ast.FunctionDef) and node.name == "sparse_attn_kernel_"
    ]
    if len(functions) != 1:
        raise ValueError("Expected one official sparse_attn_kernel_ primitive")
    primitive = functions[0]
    if [arg.arg for arg in primitive.args.args] != [
        "q",
        "kv",
        "o",
        "attn_sink",
        "topk_idxs",
    ]:
        raise ValueError("Official sparse kernel argument contract has changed")
    nodes = list(ast.walk(primitive))
    additions = []
    for anchor, snapshot in SNAPSHOTS:
        key = _ast_key(ast.parse(textwrap.dedent(anchor)).body[0])
        matches = [node for node in nodes if _ast_key(node) == key]
        if len(matches) != 1:
            raise ValueError(f"Expected one snapshot boundary: {anchor}")
        node = matches[0]
        additions.append(
            (
                node.end_lineno,
                textwrap.indent(
                    ast.unparse(ast.parse(textwrap.dedent(snapshot))),
                    " " * node.col_offset,
                ),
            )
        )
    last_arg = primitive.args.args[-1]
    if last_arg.lineno != last_arg.end_lineno or not source.splitlines()[
        last_arg.end_lineno - 1
    ].rstrip().endswith(","):
        raise ValueError("Expected official multiline kernel argument layout")
    additions.append(
        (
            last_arg.end_lineno,
            textwrap.indent(TRACE_ARGUMENTS.rstrip(), " " * last_arg.col_offset),
        )
    )
    lines = source.splitlines()
    for offset, addition in sorted(additions, reverse=True):
        lines[offset:offset] = addition.splitlines()
    instrumented = "\n".join(lines) + "\n"
    # Syntax validation does not compile/execute TileLang or claim GPU coverage.
    ast.parse(instrumented)
    return instrumented


def allocate_trace(q, indices):
    b, s, h, d = q.shape
    blocks = (indices.shape[-1] + 63) // 64
    options = {"device": q.device, "dtype": torch.float32}
    # Poison all destinations so incomplete instrumentation cannot appear valid.
    return [
        torch.full((b, s, blocks, 3, h, 64), torch.nan, **options),
        torch.full((b, s, blocks, 4, h), torch.nan, **options),
        torch.full(
            (b, s, blocks, h, 64), torch.nan, device=q.device, dtype=torch.bfloat16
        ),
        torch.full((b, s, blocks, 2, h, d), torch.nan, **options),
        torch.full((b, s, h), torch.nan, **options),
        torch.full((b, s, h, d), torch.nan, **options),
    ]


def unpack_trace(buffers, output, heads):
    scores, rows, weights, acc, denominator, normalized = buffers
    return {
        "blocks": [
            {
                "qk": scores[:, :, block, 0, :heads],
                "scaled_scores": scores[:, :, block, 1, :heads],
                "running_max": rows[:, :, block, 0, :heads],
                "rescale": rows[:, :, block, 1, :heads],
                "exp": scores[:, :, block, 2, :heads],
                "block_sum": rows[:, :, block, 2, :heads],
                "denominator": rows[:, :, block, 3, :heads],
                "bf16_weights": weights[:, :, block, :heads],
                "rescaled_accumulator": acc[:, :, block, 0, :heads],
                "accumulator": acc[:, :, block, 1, :heads],
            }
            for block in range(scores.shape[2])
        ],
        "final": {
            "denominator_with_sink": denominator[:, :, :heads],
            "normalized": normalized[:, :, :heads],
            "output": output[:, :, :heads].contiguous(),
        },
    }


def run_official_trace(module, q, kv, sink, indices, scale):
    """Apply precisely the official public wrapper's head-padding convention."""
    heads = q.shape[2]
    if heads < 16:
        q = F.pad(q, (0, 0, 0, 16 - heads)).contiguous()
        sink = F.pad(sink, (0, 16 - heads)).contiguous()
    buffers = allocate_trace(q, indices)
    output = torch.full_like(q, torch.nan)
    kernel = module.sparse_attn_kernel(q.shape[2], q.shape[-1], scale)
    kernel(q, kv, output, sink, indices, *buffers)
    return unpack_trace(buffers, output, heads), kernel


def pad_official_inputs(q, sink):
    """Match the official wrapper's head padding and return original head count."""
    heads = q.shape[2]
    if heads < 16:
        q = F.pad(q, (0, 0, 0, 16 - heads)).contiguous()
        sink = F.pad(sink, (0, 16 - heads)).contiguous()
    return q, sink, heads


def trim_trace_heads(trace, heads):
    """Trim a padded reference trace back to the public input head count."""
    blocks = []
    for block in trace["blocks"]:
        blocks.append({name: value[:, :, :heads] for name, value in block.items()})
    return {
        "blocks": blocks,
        "final": {name: value[:, :, :heads] for name, value in trace["final"].items()},
    }


QK_BACKENDS = (
    "fp32_einsum",
    "bf16_bmm_out_fp32",
    "bf16_bmm_default_cast",
)
VALUE_GEMM_BACKENDS = (
    "fp32_einsum",
    "bf16_bmm_out_fp32",
    "bf16_bmm_default_cast",
    "bf16_pairwise_fp32",
)
BLOCK_SUM_BACKENDS = (
    "torch_sum",
    "pairwise_sum",
    "sequential_sum",
)


def reference_qk_backend(q):
    """Select the backend that matches the official execution contract."""
    if q.is_cuda and q.dtype == torch.bfloat16:
        return "bf16_bmm_out_fp32"
    return "fp32_einsum"


def qk_product(q, values_bf16, backend):
    """Compute QK with explicit input/accumulator choices for GPU diagnosis."""
    if backend == "fp32_einsum":
        return torch.einsum("bshd,bskd->bshk", q.float(), values_bf16.float())
    batch, sequence, heads, dim = q.shape
    keys = values_bf16.shape[2]
    left = q.reshape(batch * sequence, heads, dim)
    right = values_bf16.transpose(-1, -2).reshape(batch * sequence, dim, keys)
    if backend == "bf16_bmm_out_fp32":
        return torch.bmm(left, right, out_dtype=torch.float32).reshape(
            batch, sequence, heads, keys
        )
    if backend == "bf16_bmm_default_cast":
        return torch.bmm(left, right).float().reshape(batch, sequence, heads, keys)
    raise ValueError(f"Unknown QK backend: {backend}")


def value_gemm_fp32(weights_bf16, values_bf16):
    """Compute BF16 weight/value GEMM with an FP32 output accumulator."""
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


def _pairwise_reduce_last(values):
    """Reduce the last dimension with an explicit adjacent-pair tree."""
    work = values
    while work.shape[-1] > 1:
        if work.shape[-1] % 2:
            work = torch.cat((work, torch.zeros_like(work[..., :1])), dim=-1)
        work = work[..., 0::2] + work[..., 1::2]
    return work[..., 0]


def value_gemm_product(weights_bf16, values_bf16, backend):
    """Evaluate one explicit value-GEMM reduction backend."""
    batch, sequence, heads, keys = weights_bf16.shape
    head_dim = values_bf16.shape[-1]
    if backend == "fp32_einsum":
        return torch.einsum(
            "bshk,bskd->bshd",
            weights_bf16.float(),
            values_bf16.float(),
        )
    if backend == "bf16_bmm_out_fp32":
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
        return value_gemm_product(weights_bf16, values_bf16, "fp32_einsum")
    if backend == "bf16_bmm_default_cast":
        left = weights_bf16.reshape(batch * sequence, heads, keys)
        right = values_bf16.reshape(batch * sequence, keys, head_dim)
        return torch.bmm(left, right).float().reshape(batch, sequence, heads, head_dim)
    if backend == "bf16_pairwise_fp32":
        if keys % 2:
            weights_bf16 = F.pad(weights_bf16, (0, 1), value=0)
            values_bf16 = F.pad(values_bf16, (0, 0, 0, 1), value=0)
        left_weights = weights_bf16[..., 0::2].float()
        right_weights = weights_bf16[..., 1::2].float()
        left_values = values_bf16[..., 0::2, :].float()
        right_values = values_bf16[..., 1::2, :].float()
        pair_sums = left_weights.unsqueeze(-1) * left_values.unsqueeze(
            2
        ) + right_weights.unsqueeze(-1) * right_values.unsqueeze(2)
        pair_sums = torch.movedim(pair_sums, -2, -1)
        return _pairwise_reduce_last(pair_sums)
    raise ValueError(f"Unknown value GEMM backend: {backend}")


def block_sum_product(values, backend):
    """Evaluate one explicit FP32 block-sum reduction backend."""
    if backend == "torch_sum":
        return values.sum(-1)
    if backend == "pairwise_sum":
        return _pairwise_reduce_last(values)
    if backend == "sequential_sum":
        result = torch.zeros_like(values[..., 0])
        for index in range(values.shape[-1]):
            result = result + values[..., index]
        return result
    raise ValueError(f"Unknown block-sum backend: {backend}")


def qk_backend_trace(q, kv, indices, backend):
    """Return masked QK blocks for one explicit PyTorch backend."""
    batch, sequence, _, _ = q.shape
    batches = torch.arange(batch, device=q.device).view(batch, 1, 1)
    blocks = []
    for start in range(0, indices.shape[-1], 64):
        idx = F.pad(
            indices[..., start : start + 64],
            (0, max(0, start + 64 - indices.shape[-1])),
            value=-1,
        ).long()
        values = kv[batches, idx.clamp_min(0)]
        values = values.masked_fill((idx < 0).unsqueeze(-1), 0)
        qk = qk_product(q, values, backend)
        qk = qk.masked_fill((idx < 0).unsqueeze(2), -torch.inf)
        blocks.append(qk)
    return blocks


def _trace_values(kv, indices, block):
    batch = kv.shape[0]
    batches = torch.arange(batch, device=kv.device).view(batch, 1, 1)
    start = block * 64
    idx = F.pad(
        indices[..., start : start + 64],
        (0, max(0, start + 64 - indices.shape[-1])),
        value=-1,
    ).long()
    values = kv[batches, idx.clamp_min(0)]
    return values.masked_fill((idx < 0).unsqueeze(-1), 0)


def compare_block_sum_backends(official_trace, reference):
    """Compare explicit block-sum reduction trees against official snapshots."""
    reports = {}
    for backend in BLOCK_SUM_BACKENDS:
        checks = []
        for block, official_block in enumerate(official_trace["blocks"]):
            values = reference["blocks"][block]["exp"]
            actual = block_sum_product(values, backend)
            checks.append(
                {
                    "block": block,
                    "key_offset": block * 64,
                    "comparison": compare_stage(official_block["block_sum"], actual),
                }
            )
        reports[backend] = {
            "status": "completed",
            "exact": all(item["comparison"]["exact"] for item in checks),
            "max_abs": max(item["comparison"].get("max_abs", 0.0) for item in checks),
            "blocks": checks,
        }
    return {"backends": reports}


def compare_value_gemm_backends(official_trace, q, kv, indices):
    """Isolate value GEMM reduction differences from prior accumulator state."""
    reports = {}
    for backend in VALUE_GEMM_BACKENDS:
        checks = []
        for block, official_block in enumerate(official_trace["blocks"]):
            values = _trace_values(kv, indices, block)
            weights = official_block["bf16_weights"]
            candidate = value_gemm_product(weights, values, backend)
            candidate = official_block["rescaled_accumulator"] + candidate
            checks.append(
                {
                    "block": block,
                    "key_offset": block * 64,
                    "comparison": compare_stage(
                        official_block["accumulator"], candidate
                    ),
                }
            )
        reports[backend] = {
            "status": "completed",
            "exact": all(item["comparison"]["exact"] for item in checks),
            "strict_allclose": all(
                item["comparison"]["strict_allclose"] for item in checks
            ),
            "max_abs": max(item["comparison"].get("max_abs", 0.0) for item in checks),
            "blocks": checks,
        }
    return {"backends": reports}


def reference_trace(q, kv, sink, indices, scale):
    """Snapshot the same PyTorch operation order used by the project reference."""
    b, s, h, d = q.shape
    maximum = torch.full((b, s, h, 1), -1e30, device=q.device)
    denominator = torch.zeros_like(maximum)
    acc = torch.zeros((b, s, h, d), device=q.device, dtype=torch.float32)
    batches = torch.arange(b, device=q.device).view(b, 1, 1)
    blocks = []
    qk_backend = reference_qk_backend(q)
    for start in range(0, indices.shape[-1], 64):
        idx = F.pad(
            indices[..., start : start + 64],
            (0, max(0, start + 64 - indices.shape[-1])),
            value=-1,
        ).long()
        values = kv[batches, idx.clamp_min(0)]
        values = values.masked_fill((idx < 0).unsqueeze(-1), 0)
        qk = qk_product(q, values, qk_backend)
        qk = qk.masked_fill((idx < 0).unsqueeze(2), -torch.inf)
        scores = qk * scale
        next_maximum = torch.maximum(maximum, scores.amax(-1, keepdim=True))
        correction = torch.exp(maximum - next_maximum)
        weights = torch.exp(scores - next_maximum)
        block_sum = weights.sum(-1, keepdim=True)
        denominator = denominator * correction + block_sum
        bf16_weights = weights.bfloat16()
        rescaled = acc * correction
        acc = rescaled + value_gemm_fp32(bf16_weights, values)
        blocks.append(
            {
                "qk": qk,
                "scaled_scores": scores,
                "running_max": next_maximum.squeeze(-1),
                "rescale": correction.squeeze(-1),
                "exp": weights,
                "block_sum": block_sum.squeeze(-1),
                "denominator": denominator.squeeze(-1),
                "bf16_weights": bf16_weights,
                "rescaled_accumulator": rescaled,
                "accumulator": acc,
            }
        )
        maximum = next_maximum
    denominator = denominator + torch.exp(sink.view(1, 1, h, 1) - maximum)
    normalized = acc / denominator
    return {
        "blocks": blocks,
        "final": {
            "denominator_with_sink": denominator.squeeze(-1),
            "normalized": normalized,
            "output": normalized.to(q.dtype),
        },
    }


def compare_stage(actual, expected):
    """Allow matching +/-inf masks, reject NaNs or differing nonfinite values."""
    if actual.shape != expected.shape:
        return compare(actual, expected)
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    matching = finite | ((actual == expected) & ~torch.isnan(actual))
    a = actual.masked_fill(~finite, 0)
    e = expected.masked_fill(~finite, 0)
    check = compare(a, e)
    check["matching_nonfinite_elements"] = int((~finite & matching).sum())
    check["nonfinite_mismatches"] = int((~matching).sum())
    check["mismatched_elements"] += check["nonfinite_mismatches"]
    check["finite_metrics_only"] = True
    if not bool(matching.all()):
        check.update(passed=False, exact=False, strict_allclose=False, allclose=False)
        flat_index = int((~matching).flatten().int().argmax())
        coordinate = []
        remainder = flat_index
        for size in reversed(actual.shape):
            coordinate.append(remainder % size)
            remainder //= size
        check["first_nonfinite_mismatch"] = {
            "coordinate": list(reversed(coordinate)),
            "actual": float(actual.flatten()[flat_index]),
            "reference": float(expected.flatten()[flat_index]),
        }
    return check


def compare_qk_backends(official_trace, q, kv, indices, scale, heads):
    """Compare explicit PyTorch QK backends against official block snapshots."""
    padded_q, _, _ = pad_official_inputs(q, q.new_zeros(heads))
    reports = {}
    for backend in QK_BACKENDS:
        try:
            blocks = qk_backend_trace(padded_q, kv, indices, backend)
            checks = []
            for block, (actual, expected) in enumerate(
                zip(official_trace["blocks"], blocks)
            ):
                checks.append(
                    {
                        "block": block,
                        "key_offset": block * 64,
                        "comparison": compare_stage(
                            actual["qk"], expected[:, :, :heads]
                        ),
                    }
                )
            exact = all(item["comparison"]["exact"] for item in checks)
            max_abs = max(item["comparison"].get("max_abs", 0.0) for item in checks)
            relative_l2 = max(
                item["comparison"].get("relative_l2", 0.0) for item in checks
            )
            reports[backend] = {
                "status": "completed",
                "exact": exact,
                "max_abs": max_abs,
                "max_relative_l2": relative_l2,
                "blocks": checks,
            }
        except Exception as error:
            reports[backend] = {
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }
    completed = {
        name: report
        for name, report in reports.items()
        if report["status"] == "completed"
    }
    best = None
    if completed:
        best = min(
            completed,
            key=lambda name: (
                completed[name]["max_abs"],
                completed[name]["max_relative_l2"],
            ),
        )
    return {"backends": reports, "best_by_max_abs": best}


def summarize_trace(actual, expected, output_check, *, attribution_valid):
    if len(actual["blocks"]) != len(expected["blocks"]):
        raise ValueError("Official/reference trace block counts differ")
    stages = []
    for block, (a, e) in enumerate(zip(actual["blocks"], expected["blocks"])):
        for stage in BLOCK_STAGES:
            stages.append(
                {
                    "block": block,
                    "key_offset": block * 64,
                    "stage": stage,
                    "axes": ["batch", "query", "head"]
                    + (
                        ["key_in_block"]
                        if stage in ("qk", "scaled_scores", "exp", "bf16_weights")
                        else ["channel"]
                        if stage.endswith("accumulator")
                        else []
                    ),
                    "comparison": compare_stage(a[stage], e[stage]),
                }
            )
    for stage in FINAL_STAGES:
        stages.append(
            {
                "block": None,
                "stage": stage,
                "axes": ["batch", "query", "head"]
                + ([] if stage == "denominator_with_sink" else ["channel"]),
                "comparison": compare_stage(
                    actual["final"][stage], expected["final"][stage]
                ),
            }
        )

    def first(predicate):
        return next((row for row in stages if predicate(row)), None)

    points = []
    coordinates = set()
    for field in ("largest_difference", "largest_bf16_step_difference"):
        point = output_check.get(field)
        if point is None or tuple(point["coordinate"]) in coordinates:
            continue
        coordinates.add(tuple(point["coordinate"]))
        b, s, h, d = point["coordinate"]
        index = (b, s, h)
        trajectory = []
        for block, (a, e) in enumerate(zip(actual["blocks"], expected["blocks"])):
            trajectory.append(
                {
                    "block": block,
                    "rows": {
                        stage: {
                            "official": float(a[stage][index]),
                            "reference": float(e[stage][index]),
                        }
                        for stage in (
                            "running_max",
                            "rescale",
                            "block_sum",
                            "denominator",
                        )
                    },
                    "accumulators": {
                        stage: {
                            "official": float(a[stage][b, s, h, d]),
                            "reference": float(e[stage][b, s, h, d]),
                        }
                        for stage in ("rescaled_accumulator", "accumulator")
                    },
                    "weights": compare_stage(
                        a["bf16_weights"][index], e["bf16_weights"][index]
                    ),
                    "scores": {
                        stage: compare_stage(a[stage][index], e[stage][index])
                        for stage in ("qk", "scaled_scores", "exp")
                    },
                }
            )
        points.append(
            {
                "selected_by": field,
                **point,
                "blocks": trajectory,
                "final": {
                    stage: {
                        "official": float(
                            actual["final"][stage][
                                index
                                if stage == "denominator_with_sink"
                                else (b, s, h, d)
                            ]
                        ),
                        "reference": float(
                            expected["final"][stage][
                                index
                                if stage == "denominator_with_sink"
                                else (b, s, h, d)
                            ]
                        ),
                    }
                    for stage in FINAL_STAGES
                },
            }
        )
    return {
        "status": "completed" if attribution_valid else "untrusted",
        "attribution_valid": attribution_valid,
        "comparison": "instrumented official kernel vs PyTorch operation-order trace",
        "first_exact_divergence": first(lambda row: not row["comparison"]["exact"])
        if attribution_valid
        else None,
        "first_strict_divergence": first(lambda row: not row["comparison"]["passed"])
        if attribution_valid
        else None,
        "first_bf16_weight_divergence": first(
            lambda row: (
                row["stage"] == "bf16_weights" and not row["comparison"]["exact"]
            )
        )
        if attribution_valid
        else None,
        "stages": stages,
        "output_points": points,
    }


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    return value


def write_instrumented_source(official_path, output_path):
    official_path = Path(official_path)
    output_path = Path(output_path)
    if output_path.resolve() == official_path.resolve():
        raise ValueError("The instrumented copy must not overwrite official source")
    source = official_path.read_text()
    instrumented = instrument_official_source(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(instrumented)
    return output_path
