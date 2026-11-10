"""Discover optional prebuilt operator providers without changing CT dispatch."""

from __future__ import annotations

import logging
import os
import threading
import warnings
from functools import wraps
from importlib.metadata import entry_points
from typing import Callable

logger = logging.getLogger(__name__)

_ENTRY_POINT_GROUP = "llmcompressor.ops"
_PROTOCOL_VERSION = 1
_PROVIDERS: tuple | None = None
_PROVIDER_LOCK = threading.RLock()
_LOGGED_PROVIDERS: set[tuple[str, str, str]] = set()


def _operators() -> tuple:
    global _PROVIDERS
    with _PROVIDER_LOCK:
        if _PROVIDERS is not None:
            return _PROVIDERS
        operators = []
        for entry in entry_points(group=_ENTRY_POINT_GROUP):
            try:
                provider = entry.load()()
                if provider.protocol_version != _PROTOCOL_VERSION:
                    raise ValueError("incompatible operator provider protocol")
                candidates = tuple(provider.operators)
                for candidate in candidates:
                    if (
                        not isinstance(candidate.name, str)
                        or not isinstance(candidate.backend, str)
                        or not isinstance(candidate.implementation, str)
                        or not isinstance(candidate.contract_version, int)
                        or not isinstance(candidate.priority, int)
                        or candidate.execution_kind != "precompiled_native"
                        or not callable(candidate.supports)
                        or not callable(candidate.invoke)
                    ):
                        raise ValueError("invalid precompiled operator descriptor")
                operators.extend(candidates)
            except Exception as error:
                warnings.warn(
                    f"Cannot load hardware operator provider {entry.name!r}: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        keys = set()
        for candidate in operators:
            key = (
                candidate.name,
                candidate.contract_version,
                candidate.backend,
                candidate.priority,
            )
            if key in keys:
                raise ValueError(f"Conflicting hardware operator registration: {key}")
            keys.add(key)
        _PROVIDERS = tuple(sorted(operators, key=lambda candidate: candidate.priority))
        return _PROVIDERS


def available_hardware_ops() -> tuple[dict, ...]:
    """Describe installed providers without running their operators."""
    return tuple(
        {
            "name": candidate.name,
            "contract_version": candidate.contract_version,
            "backend": candidate.backend,
            "implementation": candidate.implementation,
            "execution_kind": candidate.execution_kind,
        }
        for candidate in _operators()
    )


def _log_selected_provider(candidate) -> None:
    key = (candidate.name, candidate.backend, candidate.implementation)
    with _PROVIDER_LOCK:
        if key in _LOGGED_PROVIDERS:
            return
        _LOGGED_PROVIDERS.add(key)
    logger.info(
        "Using precompiled native operator %r (backend=%r, implementation=%r, "
        "contract_version=%d)",
        candidate.name,
        candidate.backend,
        candidate.implementation,
        candidate.contract_version,
    )


def optional_hardware_backend(name: str, contract_version: int = 1) -> Callable:
    """Try prebuilt providers before the unchanged internal backend entrypoint."""

    def decorate(fallback: Callable) -> Callable:
        eager = getattr(fallback, "__wrapped__", fallback)

        @wraps(fallback)
        def dispatch(*args, **kwargs):
            mode = os.environ.get("LLMCOMPRESSOR_OPS_BACKEND", "auto").strip().lower()
            if not mode:
                raise ValueError("LLMCOMPRESSOR_OPS_BACKEND must not be empty")
            if mode == "torch":
                return eager(*args, **kwargs)
            if mode == "native":
                return fallback(*args, **kwargs)
            for candidate in _operators():
                if (
                    candidate.name != name
                    or candidate.contract_version != contract_version
                ):
                    continue
                if mode != "auto" and candidate.backend != mode:
                    continue
                if candidate.supports(*args, **kwargs):
                    _log_selected_provider(candidate)
                    return candidate.invoke(*args, **kwargs)
            if mode != "auto":
                raise RuntimeError(
                    f"No supported hardware implementation for {name!r} "
                    f"with backend {mode!r}"
                )
            return fallback(*args, **kwargs)

        return dispatch

    return decorate
