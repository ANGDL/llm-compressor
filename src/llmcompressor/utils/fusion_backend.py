"""Public contract for optional fused KDA providers.

The open-source package deliberately knows only the ``fusion`` capability.  A
deployment-specific package may provide ``llmcompressor_fusion_backend`` and
implement the protocol there without adding vendor headers or symbols to this
repository.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

FUSION_BACKEND_MODULE = "llmcompressor_fusion_backend"
KDA_BACKEND_ENV = "LLMCOMPRESSOR_KDA_BACKEND"
KDA_BACKENDS = frozenset({"auto", "default", "fusion", "fla", "torch", "fla_reference"})
_FALSE_VALUES = {"", "0", "false", "no", "off"}


@runtime_checkable
class FusionKDAProvider(Protocol):
    """Provider contract implemented by an optional deployment package.

    The tensor contract matches the Transformers KDA functions: q/k/v are
    ``[batch, sequence, heads, dim]``, g is the log-space forget gate,
    beta is already sigmoid-activated, and the return value is
    ``(output, final_state)``.
    """

    name: str

    def is_available(self) -> bool:
        """Return whether the provider can run on the current process/device."""

    def forward(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        """Run fused KDA using the public tensor contract."""


def requested_kda_backend() -> str:
    """Read and validate the public backend selector.

    ``triton`` remains accepted as a compatibility alias for ``fla``; new
    deployments should use the backend names documented in ``KDA_BACKENDS``.
    """

    value = os.getenv(KDA_BACKEND_ENV, "auto").strip().lower()
    if value == "triton":
        value = "fla"
    if value not in KDA_BACKENDS:
        choices = ", ".join(sorted(KDA_BACKENDS))
        raise ValueError(
            f"invalid {KDA_BACKEND_ENV}={value!r}; choose one of {choices}"
        )
    return value


def triton_is_disabled() -> bool:
    """Read the package-wide Triton switch without importing model code."""

    configured = os.getenv("LLMCOMPRESSOR_DISABLE_TRITON")
    return configured is None or configured.strip().lower() not in _FALSE_VALUES


def load_fusion_provider() -> FusionKDAProvider | None:
    """Load the optional provider without exposing its implementation details."""

    try:
        module = importlib.import_module(FUSION_BACKEND_MODULE)
        factory: Callable[[], FusionKDAProvider] | None = getattr(
            module, "get_provider", None
        )
        provider = (
            factory() if factory is not None else getattr(module, "provider", None)
        )
    except Exception:
        return None
    if provider is None or not isinstance(provider, FusionKDAProvider):
        return None
    try:
        return provider if provider.is_available() else None
    except Exception:
        return None


def require_fusion_provider() -> FusionKDAProvider:
    """Return the provider or raise a backend-neutral configuration error."""

    provider = load_fusion_provider()
    if provider is None:
        raise RuntimeError(
            "KDA backend 'fusion' was requested, but no usable fusion provider "
            "is installed for this device"
        )
    return provider


def select_kda_backend(*, default_uses_triton: bool) -> str:
    """Resolve a backend while preserving strict manual selection semantics."""

    choice = requested_kda_backend()
    if choice == "auto":
        if not (default_uses_triton and triton_is_disabled()):
            return "default"
        if load_fusion_provider() is not None:
            return "fusion"
        return "torch"
    if choice == "fla" and triton_is_disabled():
        raise RuntimeError(
            "KDA backend 'fla' was requested, but Triton is disabled by "
            "LLMCOMPRESSOR_DISABLE_TRITON"
        )
    if choice == "fusion":
        require_fusion_provider()
    return choice


def fusion_kda_adapter(provider: FusionKDAProvider, *, recurrent: bool = False):
    """Adapt a provider to the public FLA/Transformers KDA call signature."""

    def adapted(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=64,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        transpose_state_layout=False,
        **kwargs,
    ):
        if transpose_state_layout and initial_state is not None:
            initial_state = initial_state.transpose(-1, -2)
        output, state = provider.forward(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=chunk_size,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            recurrent=recurrent,
            **kwargs,
        )
        if transpose_state_layout and state is not None:
            state = state.transpose(-1, -2)
        return output, state

    return adapted
