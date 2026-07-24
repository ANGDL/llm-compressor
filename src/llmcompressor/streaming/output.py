"""Shared output metadata construction for streaming checkpoints."""

from __future__ import annotations

from collections.abc import Mapping

from compressed_tensors.compressors.format import infer_module_format
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)
from torch import nn

__all__ = ["build_quantization_config", "quantized_module_formats"]


def quantized_module_formats(
    named_modules, *, prefix: str | None = None
) -> dict[str, str]:
    """Return compressed-tensors formats for quantized named modules."""

    from compressed_tensors.quantization.utils import is_module_quantized

    formats = {}
    for module_name, module in named_modules:
        if prefix is not None and not (
            module_name == prefix or module_name.startswith(f"{prefix}.")
        ):
            continue
        if not is_module_quantized(module):
            continue
        scheme = module.quantization_scheme
        formats[module_name] = (
            scheme.format or infer_module_format(type(module), scheme).value
        )
    return formats


def build_quantization_config(
    schemes: Mapping[str, QuantizationScheme] | QuantizationConfig,
) -> dict:
    """Build a compact compressed-tensors config for streamed modules.

    The resolved modifier config is preferred because its target patterns and
    ignore list are the public checkpoint contract. Exact per-module schemes are
    accepted for the low-level boundary API and are coalesced when identical.
    """

    if isinstance(schemes, QuantizationConfig):
        config = schemes.model_copy(deep=True)
    else:
        grouped: dict[str, tuple[QuantizationScheme, list[str]]] = {}
        for module_name, scheme in schemes.items():
            serialized = scheme.model_dump(mode="json", exclude={"targets"})
            signature = repr(serialized)
            if signature not in grouped:
                grouped[signature] = (scheme.model_copy(deep=True), [])
            grouped[signature][1].append(module_name)
        config = QuantizationConfig(
            config_groups={
                f"group_{index}": scheme.model_copy(
                    update={"targets": targets}
                )
                for index, (scheme, targets) in enumerate(grouped.values())
            }
        )

    formats = {
        scheme.format or infer_module_format(nn.Linear, scheme).value
        for scheme in config.config_groups.values()
    }
    config.format = (
        next(iter(formats))
        if len(formats) == 1
        else CompressionFormat.mixed_precision.value
    )
    config.quantization_status = QuantizationStatus.COMPRESSED
    return config.model_dump(mode="json")
