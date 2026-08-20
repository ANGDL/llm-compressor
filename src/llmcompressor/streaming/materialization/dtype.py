"""Dtype policies for checkpoint-backed streaming materialization."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Iterable

import torch

_PRESERVABLE_SOURCE_DTYPES = frozenset(
    {torch.float32, torch.float16, torch.bfloat16}
)


@dataclass(frozen=True)
class StreamingDTypePolicy:
    """Resolve a per-tensor materialization dtype.

    A quantized module's execution tensors (weight and optional bias) use
    ``compute_dtype``. Other ordinary floating-point tensors retain their
    checkpoint dtype by default, which preserves FP32 control parameters
    without widening decoded FP8/FP4 tensors. Explicit regex overrides are an
    extension point for model adapters whose runtime or kernel ABI requires a
    dtype independent of checkpoint storage.
    """

    compute_dtype: torch.dtype
    compute_tensor_names: frozenset[str] = field(default_factory=frozenset)
    overrides: tuple[tuple[str, torch.dtype], ...] = ()
    preserve_source_dtype: bool = True

    def __post_init__(self) -> None:
        if not self.compute_dtype.is_floating_point:
            raise TypeError(
                "Streaming compute dtype must be floating-point, got "
                f"{self.compute_dtype}"
            )
        for pattern, dtype in self.overrides:
            if not isinstance(pattern, str) or not pattern:
                raise ValueError("Dtype override patterns must be non-empty strings")
            if not dtype.is_floating_point:
                raise TypeError(
                    f"Dtype override for {pattern!r} must be floating-point, "
                    f"got {dtype}"
                )
            re.compile(pattern)

    @classmethod
    def for_quantized_modules(
        cls,
        compute_dtype: torch.dtype,
        module_names: Iterable[str],
        *,
        overrides: Iterable[tuple[str, torch.dtype]] = (),
    ) -> "StreamingDTypePolicy":
        return cls(
            compute_dtype=compute_dtype,
            compute_tensor_names=frozenset(
                tensor_name
                for name in module_names
                for tensor_name in (f"{name}.weight", f"{name}.bias")
            ),
            overrides=tuple(overrides),
        )

    def resolve(
        self,
        tensor_name: str,
        source_dtype: torch.dtype,
        *,
        fallback_dtype: torch.dtype | None = None,
    ) -> torch.dtype:
        """Return the dtype used to materialize one checkpoint tensor."""
        # This policy is consulted only for model-declared floating state.
        # Integer source storage can therefore represent packed FP4 weights and
        # must be decoded into a floating computation dtype.
        if not source_dtype.is_floating_point:
            return fallback_dtype or self.compute_dtype
        for pattern, dtype in self.overrides:
            if re.fullmatch(pattern, tensor_name):
                return dtype
        if tensor_name in self.compute_tensor_names:
            return fallback_dtype or self.compute_dtype
        if self.preserve_source_dtype and source_dtype in _PRESERVABLE_SOURCE_DTYPES:
            return source_dtype
        return fallback_dtype or self.compute_dtype

    def configuration(self) -> dict[str, object]:
        names_digest = hashlib.sha256(
            "\n".join(sorted(self.compute_tensor_names)).encode("utf-8")
        ).hexdigest()
        return {
            "compute_dtype": str(self.compute_dtype),
            "preserve_source_dtype": self.preserve_source_dtype,
            "compute_tensor_names_sha256": names_digest,
            "compute_tensor_count": len(self.compute_tensor_names),
            "overrides": [
                (pattern, str(dtype)) for pattern, dtype in self.overrides
            ],
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.configuration(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
