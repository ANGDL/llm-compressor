"""Optional device-native numerics scoped to streaming modifier execution."""

from contextlib import contextmanager
from threading import RLock

from llmcompressor.modifiers.gptq.helpers import factorize_hessian as torch_factorize
from llmcompressor.utils.hardware_ops import optional_hardware_backend

_PATCH_LOCK = RLock()
_PATCH_USERS = 0
_ORIGINAL_FACTORIZE = None


@optional_hardware_backend("gptq_factorize_hessian", contract_version=1)
def factorize_hessian(weights, hessians, percdamp, used_rtn_fallback):
    return torch_factorize(weights, hessians, percdamp, used_rtn_fallback)


@contextmanager
def native_gptq_factorization():
    """Install the optional factorization bridge without editing GPTQ sources."""
    from llmcompressor.modifiers.gptq import gptq_quantize

    global _PATCH_USERS, _ORIGINAL_FACTORIZE
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
