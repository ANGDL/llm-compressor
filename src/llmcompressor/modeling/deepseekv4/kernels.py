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
    return os.environ.get(REFERENCE_KERNEL_BACKEND_ENV, "").strip().lower() == "reference"


def _candidate_reference_kernel_path() -> Optional[str]:
    explicit_path = os.environ.get(REFERENCE_KERNEL_PATH_ENV)
    if explicit_path:
        return explicit_path

    model_path = os.environ.get(REFERENCE_MODEL_ENV)
    if not model_path:
        return None

    return str(Path(model_path).with_name("kernel.py"))


def get_reference_kernel_path() -> Optional[str]:
    path = _candidate_reference_kernel_path()
    if path and os.path.exists(path):
        return path
    return None


def reference_kernel_available() -> bool:
    return _load_reference_kernel_module() is not None


@lru_cache(1)
def _load_reference_kernel_module() -> Optional[ModuleType]:
    if not _reference_backend_requested():
        return None

    path = get_reference_kernel_path()
    if path is None:
        return None

    spec = importlib.util.spec_from_file_location("deepseek_v4_reference_kernel_backend", path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
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
    scores = torch.einsum("bshd,bskd->bshk", q.float(), gathered_kv.float()) * softmax_scale
    scores = scores.masked_fill(topk_idxs.unsqueeze(2) < 0, float("-inf"))
    sink_scores = attn_sink.view(1, 1, n_heads, 1).expand(bsz, seqlen, -1, -1)
    probs = torch.softmax(torch.cat([sink_scores, scores], dim=-1), dim=-1)[..., 1:]
    return torch.einsum("bshk,bskd->bshd", probs, gathered_kv.float()).to(output_dtype)


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

    adjusted = mixes + hc_base
    pre_logits, post_logits, comb_logits = torch.split(
        adjusted, [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
    )
    pre = torch.softmax(pre_logits * hc_scale[0], dim=-1) + hc_eps
    post = torch.softmax(post_logits * hc_scale[1], dim=-1) + hc_eps
    comb = comb_logits.view(*comb_logits.shape[:-1], hc_mult, hc_mult)
    comb = torch.softmax(comb * hc_scale[2], dim=-2)
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
