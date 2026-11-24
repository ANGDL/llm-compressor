"""Optional device-native numerics scoped to streaming modifier execution."""

from contextlib import contextmanager
from threading import RLock

import torch
from loguru import logger

from llmcompressor.modifiers.gptq.helpers import factorize_hessian as torch_factorize
from llmcompressor.utils.hardware_ops import (
    available_hardware_ops,
    optional_hardware_backend,
)

_PATCH_LOCK = RLock()
_PATCH_USERS = 0
_ORIGINAL_FACTORIZE = None
_FALLBACK_WARNED = False
_MAX_FACTORIZE_INDEX = torch.iinfo(torch.int32).max


@optional_hardware_backend("gptq_factorize_hessian", contract_version=1)
def _factorize_hessian(weights, hessians, percdamp, used_rtn_fallback):
    return torch_factorize(weights, hessians, percdamp, used_rtn_fallback)


def factorize_hessian(weights, hessians, percdamp, used_rtn_fallback):
    """Keep CUDA-compatible factorization views within 32-bit index range."""
    batch_size = hessians.shape[0]
    if not hessians.is_cuda or batch_size <= 1:
        return _factorize_hessian(weights, hessians, percdamp, used_rtn_fallback)
    batch_stride = max(hessians.stride(0), weights.stride(0), 1)
    chunk_size = max(1, _MAX_FACTORIZE_INDEX // batch_stride)
    if batch_size <= chunk_size:
        return _factorize_hessian(weights, hessians, percdamp, used_rtn_fallback)

    for weight_chunk, hessian_chunk, fallback_chunk in zip(
        weights.split(chunk_size),
        hessians.split(chunk_size),
        used_rtn_fallback.split(chunk_size),
    ):
        _factorize_hessian(weight_chunk, hessian_chunk, percdamp, fallback_chunk)
    return hessians


@contextmanager
def native_gptq_factorization():
    """Install the optional factorization bridge without editing GPTQ sources."""
    from llmcompressor.modifiers.gptq import gptq_quantize

    global _PATCH_USERS, _ORIGINAL_FACTORIZE, _FALLBACK_WARNED
    if not _FALLBACK_WARNED:
        names = {operator["name"] for operator in available_hardware_ops()}
        if "gptq_factorize_hessian" not in names:
            logger.warning(
                "Native 'gptq_factorize_hessian' is unavailable in the loaded "
                "llmcompressor_kernels profile; streaming GPTQ will use the eager "
                "Hessian factorization fallback, which can make weight updates an "
                "order of magnitude slower. Load the kernel profile that provides "
                "it before quantizing."
            )
        _FALLBACK_WARNED = True
    with _PATCH_LOCK:
        if _PATCH_USERS == 0:
            _ORIGINAL_FACTORIZE = gptq_quantize.factorize_hessian
            gptq_quantize.factorize_hessian = factorize_hessian
        _PATCH_USERS += 1
    try:
        yield
    finally:
        with _PATCH_LOCK:
            _PATCH_USERS -= 1
            if _PATCH_USERS == 0:
                gptq_quantize.factorize_hessian = _ORIGINAL_FACTORIZE
                _ORIGINAL_FACTORIZE = None
