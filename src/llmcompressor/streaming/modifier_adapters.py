"""Narrow algorithm adapters for the transactional subgraph execution loop."""

from __future__ import annotations

import os
import random
import sys
from collections.abc import Iterable, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import numpy as np
import torch
from compressed_tensors.quantization import apply_quantization_config

from llmcompressor.modifiers.autosmooth import AutoSmoothModifier
from llmcompressor.modifiers.qad import QADModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.quantization.calibration import apply_calibration_status
from llmcompressor.modifiers.transform import QuIPModifier, SpinQuantModifier
from llmcompressor.utils.helpers import DisableQuantization

from .modifier_contract import ModifierContractError


def _weight(module: torch.nn.Module) -> torch.Tensor | None:
    scheme = getattr(module, "quantization_scheme", None)
    return (
        getattr(module, "weight", None)
        if scheme is not None and scheme.weights is not None
        else None
    )


def _input_width(module: torch.nn.Module) -> int:
    return next(
        (
            child.weight.shape[-1]
            for child in module.modules()
            if isinstance(child, torch.nn.Linear)
        ),
        0,
    )


def _on_cpu(device: Any) -> bool:
    return (
        isinstance(device, (str, torch.device))
        and torch.device(device).type == "cpu"
    )


def _validate_parameter_scope(model, target, parameters):
    identities = {id(parameter) for parameter in parameters}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if id(parameter) in identities and not name.startswith(target + "."):
            raise ModifierContractError(
                "Streaming adapter cannot modify a parameter shared outside "
                f"target {target!r}: {name!r}"
            )


@dataclass
class StreamingModifierAdapter:
    modifier: Any
    identifier = "streaming-modifier-v1"
    owns_quantization = False
    calibration_outputs_reusable = False
    propagation_mode = "post-update-full-precision"
    minimum_batches = 1

    def contract(self) -> dict[str, Any]:
        return {
            "streaming_adapter": self.identifier,
            "calibration_outputs_reusable": self.calibration_outputs_reusable,
            "propagation_mode": self.propagation_mode,
            "recovery_state": "replay-subgraph",
            "output_ownership": "pipeline",
        }

    def validate_recipe(self, modifiers: Sequence[Any], quantizer: Any) -> None:
        pass

    def validate_targets(self, model: torch.nn.Module, targets: Sequence[str]) -> None:
        pass

    def estimate_memory(
        self, module: torch.nn.Module, num_tokens: int
    ) -> tuple[int, int]:
        raise NotImplementedError

    def estimate_host_memory(self, module: torch.nn.Module, num_tokens: int) -> int:
        return 0

    def before_calibration(
        self, model: torch.nn.Module, targets: Sequence[str]
    ) -> None:
        pass

    def before_target(self, modules: Sequence[torch.nn.Module]) -> None:
        pass

    def after_update(self, modules: Sequence[torch.nn.Module]) -> None:
        pass

    def update_context(self, model: torch.nn.Module):
        return nullcontext()

    def restore_target(self, module: torch.nn.Module) -> None:
        pass

    def abort(self) -> None:
        self.modifier.remove_hooks()


class QADStreamingAdapter(StreamingModifierAdapter):
    identifier = "qad-local-training-v1"
    propagation_mode = "post-update-quantized"
    minimum_batches = 2

    def validate_recipe(self, modifiers: Sequence[Any], quantizer: Any) -> None:
        if (
            type(quantizer) is not QuantizationModifier
            or len(modifiers) != 2
            or modifiers[0] is not quantizer
            or modifiers[1] is not self.modifier
        ):
            raise ModifierContractError(
                "Streaming QAD requires [QuantizationModifier, QADModifier]; "
                "other algorithm combinations require a separate adapter"
            )
        if not _on_cpu(self.modifier.offload_device):
            raise ModifierContractError("Streaming QAD currently requires CPU cache")
        if quantizer.kv_cache_scheme is not None:
            raise ModifierContractError(
                "Streaming QAD does not support KV quantization"
            )
        for scheme in quantizer.resolved_config.config_groups.values():
            if (
                scheme.input_activations is not None
                or scheme.output_activations is not None
            ):
                raise ModifierContractError(
                    "Streaming QAD currently supports weight-only quantization"
                )

    def estimate_memory(self, module, num_tokens):
        weight = _weight(module)
        held = 0 if weight is None else 6 * weight.numel() * torch.float32.itemsize
        scratch = 0
        if module in self.modifier._seq_target_names:
            scratch = 16 * num_tokens * _input_width(module) * torch.float32.itemsize
        return held, scratch

    def estimate_host_memory(self, module, num_tokens):
        weight = _weight(module)
        snapshot = 0 if weight is None else weight.numel() * weight.element_size()
        if module in self.modifier._seq_target_names:
            snapshot += 4 * num_tokens * _input_width(module) * torch.float32.itemsize
        return snapshot

    def validate_targets(self, model, targets):
        for name in targets:
            root = model.get_submodule(name)
            if root not in self.modifier._seq_target_names:
                raise ModifierContractError(f"QAD target {name!r} was not initialized")
            _validate_parameter_scope(
                model, name,
                (
                    weight for module in root.modules()
                    if (weight := _weight(module)) is not None
                ),
            )

    def restore_target(self, module):
        self.modifier._distilled.add(module)

    def abort(self):
        self.modifier._clear_cache()


class AutoSmoothStreamingAdapter(StreamingModifierAdapter):
    identifier = "autosmooth-local-mapping-v1"

    def update_context(self, model):
        return DisableQuantization(model)

    def contract(self):
        return {**super().contract(), "modifies_norms": True}

    def validate_recipe(self, modifiers, quantizer):
        if (
            type(quantizer) is not QuantizationModifier
            or len(modifiers) != 2
            or modifiers[0] is not self.modifier
            or modifiers[1] is not quantizer
        ):
            raise ModifierContractError(
                "Streaming AutoSmooth requires "
                "[AutoSmoothModifier, QuantizationModifier]"
            )
        if quantizer.kv_cache_scheme is not None or any(
            scheme.input_activations is not None
            or scheme.output_activations is not None
            for scheme in quantizer.resolved_config.config_groups.values()
        ):
            raise ModifierContractError(
                "Streaming AutoSmooth currently supports weight-only quantization"
            )

    def validate_targets(self, model, targets):
        if not self.modifier._resolved_mappings:
            raise ModifierContractError("AutoSmooth did not resolve any local mapping")
        for mapping in self.modifier._resolved_mappings:
            names = [mapping.smooth_name, mapping.parent_name, *mapping.balance_names]
            owner = next(
                (
                    target for target in targets
                    if all(
                        name == target or name.startswith(target + ".")
                        for name in names
                    )
                ),
                None,
            )
            if owner is None:
                raise ModifierContractError(
                    "AutoSmooth mapping crosses a streaming target boundary: "
                    f"{names}"
                )
            _validate_parameter_scope(
                model, owner,
                (
                    parameter
                    for layer in [mapping.smooth_layer, *mapping.balance_layers]
                    for parameter in layer.parameters(recurse=False)
                ),
            )

    def _mappings(self, module):
        return [
            mapping for mapping in self.modifier._resolved_mappings
            if mapping.parent is module
        ]

    def _cache_bytes(self, module, num_tokens):
        mappings = self._mappings(module)
        if not mappings:
            return 0
        weight = mappings[0].balance_layers[0].weight
        return num_tokens * weight.shape[-1] * weight.element_size()

    def estimate_memory(self, module, num_tokens):
        mappings = self._mappings(module)
        if not mappings:
            return 0, 0
        held = 0 if _on_cpu(self.modifier.offload_device) else self._cache_bytes(
            module, num_tokens
        )
        held += sum(
            2 * mapping.balance_layers[0].weight.shape[-1] * torch.float32.itemsize
            for mapping in mappings
        )
        weight_copies = max(
            sum(layer.weight.numel() * layer.weight.element_size()
                for layer in mapping.balance_layers)
            for mapping in mappings
        )
        width = max(layer.weight.shape[0]
                    for mapping in mappings for layer in mapping.balance_layers)
        return held, weight_copies + 2 * num_tokens * width * torch.float32.itemsize

    def estimate_host_memory(self, module, num_tokens):
        return (
            self._cache_bytes(module, num_tokens)
            if _on_cpu(self.modifier.offload_device) else 0
        )

    def abort(self):
        super().abort()
        self.modifier._parent_args_cache.clear()
        self.modifier._smooth_activation_scales.clear()
        self.modifier.ended_ = True


class AutoRoundStreamingAdapter(StreamingModifierAdapter):
    identifier = "autoround-independent-block-output-v1"
    owns_quantization = True
    calibration_outputs_reusable = True
    propagation_mode = "calibration-full-precision"

    @contextmanager
    def update_context(self, model):
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        try:
            with DisableQuantization(model):
                yield
        finally:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
            if workspace is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace

    def contract(self):
        return {
            **super().contract(),
            "streaming_adapter": (
                f"{self.identifier};auto-round={version('auto-round')}"
            ),
        }

    def validate_recipe(self, modifiers, quantizer):
        if len(modifiers) != 1 or quantizer is not self.modifier:
            raise ModifierContractError("Streaming AutoRound must be used alone")
        if self.modifier.device_ids is not None:
            raise ModifierContractError(
                "Streaming AutoRound placement belongs to the pipeline; "
                "device_ids must be None"
            )
        if self.modifier.enable_torch_compile:
            raise ModifierContractError(
                "Streaming AutoRound currently requires enable_torch_compile=False"
            )
        if self.modifier.batch_size < 1 or self.modifier.iters < 0:
            raise ModifierContractError("AutoRound batch_size/iters are invalid")
        for scheme in self.modifier.resolved_config.config_groups.values():
            weight = scheme.weights
            if (
                weight is None or weight.type != "int" or weight.num_bits not in (4, 8)
                or not weight.symmetric
                or scheme.input_activations is not None
                or scheme.output_activations is not None
            ):
                raise ModifierContractError(
                    "Streaming AutoRound supports symmetric INT4/INT8 "
                    "weight-only schemes"
                )

    def validate_targets(self, model, targets):
        for name in targets:
            _validate_parameter_scope(
                model, name,
                (
                    weight for module in model.get_submodule(name).modules()
                    if (weight := _weight(module)) is not None
                ),
            )
            for module in model.get_submodule(name).modules():
                weight = _weight(module)
                if weight is not None and (
                    not isinstance(module, torch.nn.Linear)
                    or any(size % 32 for size in weight.shape)
                ):
                    raise ModifierContractError(
                        "Streaming AutoRound requires Linear weight dimensions "
                        "divisible by 32; external layer skipping is not supported"
                    )

    def before_calibration(self, model, targets):
        self.modifier._sequential_targets = {
            type(model.get_submodule(name)).__name__ for name in targets
        }

    def configure_algorithm(self, modifier, root, kwargs):
        self._parameter_layout = [
            (module, name, parameter.dtype, parameter.device)
            for module in root.modules()
            for name, parameter in module.named_parameters(recurse=False)
            if name in ("weight", "bias")
        ]
        parameter = next(
            weight for module in root.modules()
            if (weight := _weight(module)) is not None
        )
        return {
            **kwargs,
            "device_map": (
                parameter.device.index or 0
                if parameter.device.type == "cuda" else str(parameter.device)
            ),
            "model_dtype": parameter.dtype, "amp": False,
            "seed": torch.initial_seed() % (2**32),
        }

    def estimate_memory(self, module, num_tokens):
        weight = _weight(module)
        if weight is None:
            scratch = (
                8 * num_tokens * _input_width(module) * torch.float32.itemsize
                if self.modifier._is_decoding_layer(module) else 0
            )
            return 0, scratch
        return 8 * weight.numel() * torch.float32.itemsize, (
            8 * weight.numel() * torch.float32.itemsize
        )

    def estimate_host_memory(self, module, num_tokens):
        weight = _weight(module)
        held = 0 if weight is None else 4 * weight.numel() * torch.float32.itemsize
        if self.modifier._is_decoding_layer(module):
            held += 4 * num_tokens * _input_width(module) * torch.float32.itemsize
        return held

    def start_quantizer(self, model, modules):
        apply_quantization_config(
            model, self.modifier.resolved_config, allowed_modules=list(modules)
        )
        for module in modules:
            if getattr(module, "quantization_scheme", None) is not None:
                apply_calibration_status(module)

    def before_target(self, modules):
        self.modifier._q_input = None
        self._parameter_layout = []
        self._expected_modules = [
            module for module in modules if _weight(module) is not None
        ]

    def after_update(self, modules):
        self.modifier._q_input = None
        for module, name, dtype, device in self._parameter_layout:
            parameter = getattr(module, name)
            if parameter.dtype != dtype or parameter.device != device:
                module.register_parameter(
                    name,
                    torch.nn.Parameter(
                        parameter.detach().to(device=device, dtype=dtype),
                        requires_grad=parameter.requires_grad,
                    ),
                )
        self._parameter_layout.clear()
        for module in self._expected_modules:
            scheme = getattr(module, "quantization_scheme", None)
            if scheme is None or scheme.weights is None:
                raise ModifierContractError(
                    "AutoRound unexpectedly skipped a requested quantized layer"
                )
            scale = getattr(module, "weight_scale", None)
            if (
                scale is None or scale.is_meta
                or not torch.isfinite(scale).all() or not (scale != 0).all()
            ):
                raise ModifierContractError(
                    "AutoRound output adapter requires nonzero finite "
                    "materialized weight_scale"
                )

    def abort(self):
        super().abort()
        self._parameter_layout = []
        self.modifier._all_module_input.clear()
        self.modifier._capture_hooks.clear()
        self.modifier._q_input = None


def get_modifier_adapter(modifier: Any) -> StreamingModifierAdapter | None:
    adapters = {
        QADModifier: QADStreamingAdapter,
        AutoSmoothModifier: AutoSmoothStreamingAdapter,
    }
    adapter_type = adapters.get(type(modifier))
    if adapter_type is not None:
        return adapter_type(modifier)
    autoround = sys.modules.get("llmcompressor.modifiers.autoround.base")
    if autoround is not None and type(modifier) is autoround.AutoRoundModifier:
        return AutoRoundStreamingAdapter(modifier)
    return None


def validate_streaming_modifiers(modifiers: Sequence[Any], quantizer: Any) -> None:
    for modifier in modifiers:
        if isinstance(modifier, (SpinQuantModifier, QuIPModifier)):
            raise ModifierContractError(
                f"{type(modifier).__name__} requires an offline/online transform "
                "adapter: norm fusion, embedding/head ownership, tied weights and "
                "runtime transform metadata are not yet supported by streaming"
            )
        adapter = get_modifier_adapter(modifier)
        if adapter is not None:
            adapter.validate_recipe(modifiers, quantizer)


def modifier_adapters(modifiers: Iterable[Any]) -> tuple[StreamingModifierAdapter, ...]:
    return tuple(
        adapter
        for modifier in modifiers
        if (adapter := get_modifier_adapter(modifier)) is not None
    )


@contextmanager
def target_rng_scope(seed: int | None, index: int, device: torch.device):
    """Replay target-local stochastic algorithms without advancing caller RNG."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_states = (
        [
            (generator, generator.get_state())
            for generator in torch.cuda.default_generators
        ]
        if torch.cuda.is_initialized() else []
    )
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda" else []
    )
    try:
        with torch.random.fork_rng(devices=devices):
            target_seed = ((seed or 0) + index) % (2**32)
            random.seed(target_seed)
            np.random.seed(target_seed)
            torch.random.default_generator.manual_seed(target_seed)
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(target_seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for generator, state in cuda_states:
            generator.set_state(state)
