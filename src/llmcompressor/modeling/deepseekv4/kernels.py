import importlib.util
import os
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Optional

import torch

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
        print("[kernels] backend not set to 'reference', skipping reference kernel module")
        return None

    path = get_reference_kernel_path()
    if path is None:
        print("[kernels] reference kernel path not found, skipping")
        return None

    print(f"[kernels] loading reference kernel module from '{path}'")
    spec = importlib.util.spec_from_file_location("deepseek_v4_reference_kernel_backend", path)
    if spec is None or spec.loader is None:
        print(f"[kernels] failed to create spec for '{path}'")
        return None

    spec = importlib.util.spec_from_file_location("deepseek_v4_reference_kernel_backend", path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"[kernels] failed to load reference kernel module: {e}")
        return None
    print(f"[kernels] reference kernel module loaded successfully")
    return module


def _can_use_reference_kernels(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and _load_reference_kernel_module() is not None


def sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    module = _load_reference_kernel_module()
    if module is not None and _can_use_reference_kernels(q):
        return module.sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)

    bsz, seqlen, n_heads, head_dim = q.shape
    output_dtype = kv.dtype
    kv_expanded = kv.unsqueeze(1).expand(-1, seqlen, -1, -1)
    gather_indices = topk_idxs.clamp(min=0).long().unsqueeze(-1).expand(-1, -1, -1, head_dim)
    gathered_kv = torch.gather(kv_expanded, 2, gather_indices)
    # scores: [bsz, seqlen, n_heads, topk]
    scores = torch.einsum("bshd,bskd->bshk", q.float(), gathered_kv.float()) * softmax_scale
    mask = topk_idxs.unsqueeze(2) < 0  # [bsz, seqlen, 1, topk]
    scores = scores.masked_fill(mask, float("-inf"))
    # Match tilelang online softmax: max over real scores only (not including sink)
    scores_max = scores.amax(dim=-1, keepdim=True)  # [bsz, seqlen, n_heads, 1]
    scores_max = scores_max.clamp(min=-1e30)  # avoid -inf max when all masked
    exp_scores = torch.exp(scores - scores_max)  # unnormalized exp
    exp_scores = exp_scores.masked_fill(mask, 0.0)
    # Cast to BF16 before value matmul (matches kernel acc_s_cast)
    exp_scores_bf16 = exp_scores.bfloat16()
    # Weighted sum: BF16 weights × BF16 values → FP32 accumulator
    acc_o = torch.einsum("bshk,bskd->bshd", exp_scores_bf16.float(), gathered_kv.float())
    # Total sum includes sink contribution
    sink_expanded = attn_sink.view(1, 1, n_heads, 1)  # [1, 1, n_heads, 1]
    sum_exp = exp_scores.sum(dim=-1, keepdim=True) + torch.exp(sink_expanded - scores_max)
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
