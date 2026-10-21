import importlib.util
import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Optional

import torch
from torch.nn import functional as F

REFERENCE_KERNEL_BACKEND_ENV = "DEEPSEEK_V4_KERNEL_BACKEND"
REFERENCE_KERNEL_PATH_ENV = "DEEPSEEK_V4_REFERENCE_KERNEL_PATH"
REFERENCE_MODEL_ENV = "DEEPSEEK_V4_REFERENCE_MODEL_PATH"


def reset_kernel_backend_cache():
    _load_reference_kernel_module.cache_clear()


def _reference_backend_requested() -> bool:
    backend = os.environ.get(REFERENCE_KERNEL_BACKEND_ENV, "").strip().lower()
    print(f"[kernels] {REFERENCE_KERNEL_BACKEND_ENV} = '{backend}'")
    return backend == "reference"


def _candidate_reference_kernel_path() -> Optional[str]:
    explicit_path = os.environ.get(REFERENCE_KERNEL_PATH_ENV)
    print(f"[kernels] {REFERENCE_KERNEL_PATH_ENV} = '{explicit_path}'")
    if explicit_path:
        return explicit_path

    model_path = os.environ.get(REFERENCE_MODEL_ENV)
    print(f"[kernels] {REFERENCE_MODEL_ENV} = '{model_path}'")
    if not model_path:
        return None

    result = str(Path(model_path).with_name("kernel.py"))
    print(f"[kernels] derived kernel path = '{result}'")
    return result


def get_reference_kernel_path() -> Optional[str]:
    path = _candidate_reference_kernel_path()
    if path:
        exists = os.path.exists(path)
        print(f"[kernels] kernel path '{path}' exists={exists}")
        if exists:
            return path
    else:
        print("[kernels] no kernel path candidate")
    return None


def reference_kernel_available() -> bool:
    return _load_reference_kernel_module() is not None


@lru_cache(1)
def _load_reference_kernel_module() -> Optional[ModuleType]:
    if not _reference_backend_requested():
        print(
            "[kernels] backend not set to 'reference', skipping reference kernel module"
        )
        return None

    path = get_reference_kernel_path()
    if path is None:
        print("[kernels] reference kernel path not found, skipping")
        return None

    print(f"[kernels] loading reference kernel module from '{path}'")
    spec = importlib.util.spec_from_file_location(
        "deepseek_v4_reference_kernel_backend", path
    )
    if spec is None or spec.loader is None:
        print(f"[kernels] failed to create spec for '{path}'")
        return None

    spec = importlib.util.spec_from_file_location(
        "deepseek_v4_reference_kernel_backend", path
    )
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"[kernels] failed to load reference kernel module: {e}")
        return None
    print("[kernels] reference kernel module loaded successfully")
    return module


def _can_use_reference_kernels(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and _load_reference_kernel_module() is not None


def _bmm_fp32(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Use an FP32 accumulator on backends without ``bmm(out_dtype=...)``."""

    try:
        return torch.bmm(left, right, out_dtype=torch.float32)
    except (NotImplementedError, RuntimeError) as error:
        message = str(error).lower()
        if "bmm" not in message and "out_dtype" not in message:
            raise
        return torch.bmm(left.float(), right.float())


def _qk_gemm_fp32(q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    """Match the official sparse QK GEMM accumulator contract.

    The V4.1 TileLang kernel consumes BF16 Q/K operands but keeps the GEMM
    accumulator and result in FP32. ``torch.bmm(..., out_dtype=...)`` exposes
    that contract on CUDA. CPU and other backends use an FP32 einsum fallback
    because ``out_dtype`` is not universally supported there.
    """
    batch, sequence, heads, head_dim = q.shape
    keys = kv.shape[2]
    if q.is_cuda and q.dtype == torch.bfloat16 and kv.dtype == torch.bfloat16:
        left = q.reshape(batch * sequence, heads, head_dim)
        right = kv.transpose(-1, -2).reshape(batch * sequence, head_dim, keys)
        return _bmm_fp32(left, right).reshape(
            batch, sequence, heads, keys
        )
    return torch.einsum(
        "bshd,bskd->bshk",
        q.float(),
        kv.float(),
    )


def _value_gemm_fp32(weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Match the official BF16-weight/BF16-value FP32 accumulator GEMM."""
    batch, sequence, heads, keys = weights.shape
    head_dim = values.shape[-1]
    if (
        weights.is_cuda
        and weights.dtype == torch.bfloat16
        and values.dtype == torch.bfloat16
    ):
        left = weights.reshape(batch * sequence, heads, keys)
        right = values.reshape(batch * sequence, keys, head_dim)
        return _bmm_fp32(left, right).reshape(
            batch, sequence, heads, head_dim
        )
    return torch.einsum(
        "bshk,bskd->bshd",
        weights.float(),
        values.float(),
    )


def sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    module = _load_reference_kernel_module()
    if module is not None and _can_use_reference_kernels(q):
        # The reference TileLang kernel declares this small bias as FP32. Keep
        # that ABI stable even when a legacy or custom streaming dtype policy
        # materializes ``attn_sink`` at the model computation dtype.
        attn_sink = attn_sink.to(dtype=torch.float32)
        return module.sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)

    bsz, seqlen, n_heads, head_dim = q.shape
    output_dtype = kv.dtype
    # V4 and V4.1 use 64-key online softmax. BF16 rounding happens inside each
    # block, before the value GEMM; rounding weights after a global softmax is
    # numerically different once more than one block is selected.
    block_size = 64
    kv_len = kv.size(1)
    flat_kv = kv.reshape(bsz * kv_len, head_dim)
    batch_offset = (
        torch.arange(bsz, device=topk_idxs.device, dtype=torch.long) * kv_len
    ).view(bsz, 1, 1)
    # Match V4.1's finite initial maximum, including leading all-masked blocks.
    scores_max = torch.full(
        (bsz, seqlen, n_heads, 1),
        -1e30,
        dtype=torch.float32,
        device=q.device,
    )
    sum_exp = torch.zeros_like(scores_max)
    acc_o = torch.zeros(
        (bsz, seqlen, n_heads, head_dim), dtype=torch.float32, device=q.device
    )
    for start in range(0, topk_idxs.shape[-1], block_size):
        indices = topk_idxs[..., start : start + block_size]
        indices = F.pad(indices, (0, block_size - indices.shape[-1]), value=-1)
        # Gather only this block, never an expanded [batch, seq, kv_len, dim]
        # cache or all selected keys for the full sequence at once.
        flat_indices = indices.clamp_min(0).long() + batch_offset
        gathered_kv = flat_kv.index_select(0, flat_indices.reshape(-1)).view(
            bsz, seqlen, block_size, head_dim
        )
        gathered_kv = gathered_kv.masked_fill((indices < 0).unsqueeze(-1), 0)
        scores = (_qk_gemm_fp32(q, gathered_kv) * softmax_scale).masked_fill(
            (indices < 0).unsqueeze(2), float("-inf")
        )
        next_max = torch.maximum(scores_max, scores.amax(-1, keepdim=True))
        correction = torch.exp(scores_max - next_max)
        exp_scores = torch.exp(scores - next_max)
        sum_exp = sum_exp * correction + exp_scores.sum(-1, keepdim=True)
        # BF16 operands, FP32 value accumulation and rescaling, then BF16 output.
        acc_o = acc_o * correction + _value_gemm_fp32(
            exp_scores.bfloat16(),
            gathered_kv,
        )
        scores_max = next_max

    sink_expanded = attn_sink.view(1, 1, n_heads, 1)
    # The official kernel returns zero for an all-masked row via 0 / inf.
    # Center at the sink for that row to preserve the same output without
    # overflowing exp() during the differentiable calibration forward.
    scores_max = torch.where(
        (topk_idxs >= 0).any(-1, keepdim=True).unsqueeze(2),
        scores_max,
        sink_expanded,
    )
    sum_exp = sum_exp + torch.exp(sink_expanded - scores_max)
    return (acc_o / sum_exp).to(output_dtype)


def hc_split(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    hc_sinkhorn_iters: int,
    hc_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    module = _load_reference_kernel_module()
    if module is not None and _can_use_reference_kernels(mixes):
        return module.hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            hc_mult,
            hc_sinkhorn_iters,
            hc_eps,
        )

    pre_raw = mixes[..., :hc_mult]
    post_raw = mixes[..., hc_mult : 2 * hc_mult]
    comb_raw = mixes[..., 2 * hc_mult :]

    # pre: sigmoid(mixes * scale + base) + eps
    pre = torch.sigmoid(pre_raw * hc_scale[0] + hc_base[:hc_mult]) + hc_eps
    # post: 2 * sigmoid(mixes * scale + base)
    post = 2 * torch.sigmoid(post_raw * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult])
    # comb: affine → row-wise softmax + eps → sinkhorn normalization
    comb = (comb_raw * hc_scale[2] + hc_base[2 * hc_mult :]).unflatten(
        -1, (hc_mult, hc_mult)
    )
    comb = torch.softmax(comb, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)

    return pre, post, comb


@contextmanager
def temporarily_enable_reference_kernels(kernel_path: str):
    previous_backend = os.environ.get(REFERENCE_KERNEL_BACKEND_ENV)
    previous_kernel_path = os.environ.get(REFERENCE_KERNEL_PATH_ENV)
    os.environ[REFERENCE_KERNEL_BACKEND_ENV] = "reference"
    os.environ[REFERENCE_KERNEL_PATH_ENV] = kernel_path
    reset_kernel_backend_cache()
    try:
        yield
    finally:
        if previous_backend is None:
            os.environ.pop(REFERENCE_KERNEL_BACKEND_ENV, None)
        else:
            os.environ[REFERENCE_KERNEL_BACKEND_ENV] = previous_backend
        if previous_kernel_path is None:
            os.environ.pop(REFERENCE_KERNEL_PATH_ENV, None)
        else:
            os.environ[REFERENCE_KERNEL_PATH_ENV] = previous_kernel_path
        reset_kernel_backend_cache()
