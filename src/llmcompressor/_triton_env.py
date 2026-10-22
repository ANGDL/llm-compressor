"""Configure optional Triton implementations before dependency imports."""

from __future__ import annotations

import os
import sys

_DISABLE_TRITON_ENV = "LLMCOMPRESSOR_DISABLE_TRITON"
_COMPRESSED_TENSORS_EAGER_ENV = "CT_ENFORCE_EAGER"
_FALSE_VALUES = {"", "0", "false", "no", "off"}


def is_triton_disabled() -> bool:
    """Return whether optional Triton implementations are disabled globally.

    Triton is disabled by default because the package may be importable while
    its CUDA kernels are not launchable on the active accelerator. Set the
    environment variable to ``0`` to explicitly force Triton.
    """

    configured = os.getenv(_DISABLE_TRITON_ENV)
    if configured is None:
        return True
    return configured.strip().lower() not in _FALSE_VALUES


def configure_compressed_tensors_triton() -> None:
    """Propagate the Triton switch to ``compressed_tensors``.

    ``compressed_tensors`` selects its Triton quantization kernels when the
    backend module is imported, based on ``CT_ENFORCE_EAGER``. On accelerators
    where Triton imports but cannot load kernels (for example Kunlunxin P800),
    weight compression would otherwise fail with ``CUDA_ERROR_NOT_SUPPORTED``.
    This normally runs before ``compressed_tensors`` is imported. If its
    backend module was already loaded, update its cached switch as well.
    """

    if is_triton_disabled():
        os.environ[_COMPRESSED_TENSORS_EAGER_ENV] = "1"
        impl_backend = sys.modules.get("compressed_tensors.utils.impl_backend")
        if impl_backend is not None:
            impl_backend.ENFORCE_EAGER = "1"
