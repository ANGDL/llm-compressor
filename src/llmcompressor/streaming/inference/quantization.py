"""Reference activation quantization without replacing model-specific forwards."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.utils import match_named_modules
from torch import nn


def reference_activation_qdq(module, value, base_name, args):
    """Keep token scales independent for flattened MoE rows as well as B,T,D.

    CT's TOKEN helper preserves axes 0 and 1. On a 2D Linear input it therefore
    falls back to whole-tensor aminmax. Normalize leading token axes explicitly
    rather than changing the third-party implementation or the model forward.
    """
    shape = value.shape
    if args.strategy == "token" and value.ndim < 3:
        value = value.reshape(1, -1, shape[-1])
    return forward_quantize(module, value, base_name, args).reshape(shape)


def resolve_schemes(
    model: nn.Module, config: Mapping[str, Any] | None
) -> tuple[dict[str, QuantizationScheme], str | None]:
    """Resolve checkpoint schemes to exact model modules, without reading weights."""
    if not config:
        return {}, None
    if config.get("quant_method") != "compressed-tensors":
        raise ValueError(
            "Streaming inference needs a checkpoint materializer for source "
            "quantization, or a compressed-tensors PTQ config"
        )
    if config.get("transform_config"):
        raise ValueError("Streaming inference does not yet apply runtime transforms")
    parsed = QuantizationConfig.model_validate(config)
    if parsed.quantization_status != QuantizationStatus.COMPRESSED:
        raise ValueError("Streaming PTQ inference requires a compressed checkpoint")
    if parsed.kv_cache_scheme is not None:
        raise ValueError("Streaming inference does not yet support quantized KV caches")
    schemes = {}
    for group in parsed.config_groups.values():
        matched = list(match_named_modules(model, group.targets, parsed.ignore))
        if not matched:
            raise ValueError(f"Quantization targets match no modules: {group.targets}")
        for name, module in matched:
            if name in schemes:
                raise ValueError(f"Overlapping quantization schemes at {name!r}")
            if not isinstance(module, nn.Linear):
                raise ValueError(f"Quantized inference requires Linear at {name!r}")
            weights = group.weights
            if (
                weights is None
                or weights.type != "int"
                or weights.num_bits not in (4, 8)
                or not weights.symmetric
                or weights.strategy not in ("channel", "tensor")
                or weights.dynamic
                or weights.actorder is not None
            ):
                raise ValueError(
                    f"Unsupported weight scheme at {name!r}: initial support is "
                    "symmetric channel/tensor INT4 or INT8"
                )
            for activation in (group.input_activations, group.output_activations):
                if activation is not None and (
                    activation.type != "int"
                    or activation.num_bits != 8
                    or activation.dynamic is not True
                    or activation.strategy != "token"
                ):
                    raise ValueError(
                        f"Unsupported activation scheme at {name!r}: initial "
                        "support is dynamic token INT8"
                    )
            schemes[name] = deepcopy(group)
    return schemes, parsed.format


def install_activation_qdq(
    model: nn.Module,
    schemes: Mapping[str, QuantizationScheme],
    *,
    enabled: bool,
) -> list:
    """Preserve the actual bound forward and apply activation Q/DQ exactly once."""
    handles = []
    for name, scheme in schemes.items():
        module = model.get_submodule(name)
        module.quantization_scheme = scheme
        module.quantization_status = QuantizationStatus.DECOMPRESSED
        module.quantization_enabled = enabled
        if not enabled:
            continue

        if scheme.input_activations is not None:
            input_name = next(iter(inspect.signature(module.forward).parameters))

            def quantize_input(module, args, kwargs, input_name=input_name):
                if args:
                    value = reference_activation_qdq(
                        module,
                        args[0],
                        "input",
                        module.quantization_scheme.input_activations,
                    )
                    return (value, *args[1:]), kwargs
                if input_name not in kwargs:
                    raise TypeError("Quantized Linear requires an input tensor")
                kwargs = dict(kwargs)
                kwargs[input_name] = reference_activation_qdq(
                    module,
                    kwargs[input_name],
                    "input",
                    module.quantization_scheme.input_activations,
                )
                return args, kwargs

            handles.append(
                module.register_forward_pre_hook(quantize_input, with_kwargs=True)
            )
        if scheme.output_activations is not None:

            def quantize_output(module, _args, output):
                return reference_activation_qdq(
                    module,
                    output,
                    "output",
                    module.quantization_scheme.output_activations,
                )

            handles.append(module.register_forward_hook(quantize_output))
    return handles
