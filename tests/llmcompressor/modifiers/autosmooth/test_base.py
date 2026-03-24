import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
)

import llmcompressor.modifiers.autosmooth.base as autosmooth_base
from llmcompressor.modifiers.autosmooth import AutoSmoothModifier
from llmcompressor.modifiers.awq.mappings import ResolvedMapping
from llmcompressor.pipelines.cache import IntermediatesCache


class _ToyParent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quantized_balance = torch.nn.Linear(4, 4, bias=False)
        self.ignored_balance = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantized_balance(x) + self.ignored_balance(x)


@pytest.mark.unit
@torch.no_grad()
def test_compute_best_scale_uses_autosmooth_snapshot_qargs(monkeypatch):
    parent = _ToyParent()
    smooth = torch.nn.LayerNorm(4)

    qargs = QuantizationArgs(
        strategy=QuantizationStrategy.CHANNEL,
        num_bits=8,
    )
    # Simulate quantization config overwritten by another modifier.
    # Runtime qscheme is None, but AutoSmooth has its own captured snapshot.
    parent.quantized_balance.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=None,
    )
    parent.ignored_balance.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=None,
    )

    mapping = ResolvedMapping(
        smooth_name="toy.smooth",
        smooth_layer=smooth,
        balance_layers=[parent.quantized_balance, parent.ignored_balance],
        balance_names=["toy.quantized_balance", "toy.ignored_balance"],
        parent=parent,
        parent_name="toy",
    )

    modifier = AutoSmoothModifier(n_grid=2, duo_scaling=False)
    modifier._autosmooth_target_modules = {parent.quantized_balance}
    modifier._autosmooth_weight_qargs = {parent.quantized_balance: qargs}
    modifier._smooth_activation_scales[mapping.smooth_name] = (torch.ones(4), 1)

    cache = IntermediatesCache(offload_device=None)
    cache.append({"x": torch.randn(2, 4)})
    modifier._parent_args_cache[parent] = cache

    captured = []

    def _fake_load_from_registry(_name, base_name, args, module):
        assert base_name == "weight"
        assert args is not None
        assert module is parent.quantized_balance
        captured.append((module, args))
        return torch.nn.Identity()

    monkeypatch.setattr(
        autosmooth_base.Observer,
        "load_from_registry",
        staticmethod(_fake_load_from_registry),
    )
    monkeypatch.setattr(autosmooth_base, "call_observer", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        autosmooth_base,
        "forward_quantize",
        lambda _module, value, _base_name, _args: value,
    )

    fp16_outputs = modifier._run_samples(parent)
    best_scales = modifier._compute_best_scale(mapping, fp16_outputs)

    assert len(captured) == 1
    assert captured[0][0] is parent.quantized_balance
    assert captured[0][1] is qargs
    assert torch.isfinite(best_scales).all()


@pytest.mark.unit
@torch.no_grad()
def test_compute_best_scale_skips_non_targeted_balance_layers(monkeypatch):
    parent = _ToyParent()
    smooth = torch.nn.LayerNorm(4)

    target_qargs = QuantizationArgs(
        strategy=QuantizationStrategy.CHANNEL,
        num_bits=8,
    )
    non_target_qargs = QuantizationArgs(
        strategy=QuantizationStrategy.CHANNEL,
        num_bits=4,
    )

    parent.quantized_balance.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=target_qargs,
    )
    # Simulate another modifier attaching qscheme to a layer AutoSmooth ignores.
    parent.ignored_balance.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=non_target_qargs,
    )

    mapping = ResolvedMapping(
        smooth_name="toy.smooth",
        smooth_layer=smooth,
        balance_layers=[parent.quantized_balance, parent.ignored_balance],
        balance_names=["toy.quantized_balance", "toy.ignored_balance"],
        parent=parent,
        parent_name="toy",
    )

    modifier = AutoSmoothModifier(n_grid=2, duo_scaling=False)
    modifier._autosmooth_target_modules = {parent.quantized_balance}
    modifier._smooth_activation_scales[mapping.smooth_name] = (torch.ones(4), 1)

    cache = IntermediatesCache(offload_device=None)
    cache.append({"x": torch.randn(2, 4)})
    modifier._parent_args_cache[parent] = cache

    captured_modules = []

    def _fake_load_from_registry(_name, base_name, args, module):
        assert base_name == "weight"
        assert args is not None
        captured_modules.append(module)
        return torch.nn.Identity()

    monkeypatch.setattr(
        autosmooth_base.Observer,
        "load_from_registry",
        staticmethod(_fake_load_from_registry),
    )
    monkeypatch.setattr(autosmooth_base, "call_observer", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        autosmooth_base,
        "forward_quantize",
        lambda _module, value, _base_name, _args: value,
    )

    fp16_outputs = modifier._run_samples(parent)
    best_scales = modifier._compute_best_scale(mapping, fp16_outputs)

    assert captured_modules == [parent.quantized_balance]
    assert torch.isfinite(best_scales).all()


@pytest.mark.unit
@torch.no_grad()
def test_compute_best_scale_errors_on_missing_qargs_for_targeted_layers(monkeypatch):
    parent = _ToyParent()
    smooth = torch.nn.LayerNorm(4)

    parent.quantized_balance.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=None,
    )
    parent.ignored_balance.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=None,
    )

    mapping = ResolvedMapping(
        smooth_name="toy.smooth",
        smooth_layer=smooth,
        balance_layers=[parent.quantized_balance],
        balance_names=["toy.quantized_balance"],
        parent=parent,
        parent_name="toy",
    )

    modifier = AutoSmoothModifier(n_grid=2, duo_scaling=False)
    modifier._autosmooth_target_modules = {parent.quantized_balance}
    modifier._smooth_activation_scales[mapping.smooth_name] = (torch.ones(4), 1)

    cache = IntermediatesCache(offload_device=None)
    cache.append({"x": torch.randn(2, 4)})
    modifier._parent_args_cache[parent] = cache

    monkeypatch.setattr(autosmooth_base, "call_observer", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        autosmooth_base,
        "forward_quantize",
        lambda _module, value, _base_name, _args: value,
    )

    fp16_outputs = modifier._run_samples(parent)
    with pytest.raises(ValueError, match="could not resolve weight quantization args"):
        modifier._compute_best_scale(mapping, fp16_outputs)


@pytest.mark.unit
@torch.no_grad()
def test_compute_best_scale_duo_scaling_uses_snapshot_qargs(monkeypatch):
    parent = _ToyParent()
    smooth = torch.nn.LayerNorm(4)

    qargs = QuantizationArgs(
        strategy=QuantizationStrategy.CHANNEL,
        num_bits=8,
    )
    # Runtime qscheme is absent, snapshot should still drive both observer
    # construction and layer-scale statistics for duo_scaling.
    parent.quantized_balance.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=None,
    )

    mapping = ResolvedMapping(
        smooth_name="toy.smooth",
        smooth_layer=smooth,
        balance_layers=[parent.quantized_balance],
        balance_names=["toy.quantized_balance"],
        parent=parent,
        parent_name="toy",
    )

    modifier = AutoSmoothModifier(n_grid=2, duo_scaling=True)
    modifier._autosmooth_target_modules = {parent.quantized_balance}
    modifier._autosmooth_weight_qargs = {parent.quantized_balance: qargs}
    modifier._smooth_activation_scales[mapping.smooth_name] = (torch.ones(4), 1)

    cache = IntermediatesCache(offload_device=None)
    cache.append({"x": torch.randn(2, 4)})
    modifier._parent_args_cache[parent] = cache

    monkeypatch.setattr(
        autosmooth_base.Observer,
        "load_from_registry",
        staticmethod(lambda *_args, **_kwargs: torch.nn.Identity()),
    )
    monkeypatch.setattr(autosmooth_base, "call_observer", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        autosmooth_base,
        "forward_quantize",
        lambda _module, value, _base_name, _args: value,
    )

    fp16_outputs = modifier._run_samples(parent)
    best_scales = modifier._compute_best_scale(mapping, fp16_outputs)

    assert torch.isfinite(best_scales).all()


@pytest.mark.unit
@torch.no_grad()
def test_compute_best_scale_preserves_existing_weight_scale(monkeypatch):
    parent = _ToyParent()
    smooth = torch.nn.LayerNorm(4)

    qargs = QuantizationArgs(
        strategy=QuantizationStrategy.CHANNEL,
        num_bits=8,
    )
    parent.quantized_balance.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=qargs,
    )
    parent.quantized_balance.weight_scale = torch.nn.Parameter(torch.ones((4, 1)))

    mapping = ResolvedMapping(
        smooth_name="toy.smooth",
        smooth_layer=smooth,
        balance_layers=[parent.quantized_balance],
        balance_names=["toy.quantized_balance"],
        parent=parent,
        parent_name="toy",
    )

    modifier = AutoSmoothModifier(n_grid=2, duo_scaling=False)
    modifier._autosmooth_target_modules = {parent.quantized_balance}
    modifier._smooth_activation_scales[mapping.smooth_name] = (torch.ones(4), 1)

    cache = IntermediatesCache(offload_device=None)
    cache.append({"x": torch.randn(2, 4)})
    modifier._parent_args_cache[parent] = cache

    monkeypatch.setattr(
        autosmooth_base.Observer,
        "load_from_registry",
        staticmethod(lambda *_args, **_kwargs: torch.nn.Identity()),
    )
    monkeypatch.setattr(autosmooth_base, "call_observer", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        autosmooth_base,
        "forward_quantize",
        lambda _module, value, _base_name, _args: value,
    )

    fp16_outputs = modifier._run_samples(parent)
    best_scales = modifier._compute_best_scale(mapping, fp16_outputs)

    assert torch.isfinite(best_scales).all()
    assert hasattr(parent.quantized_balance, "weight_scale")


@pytest.mark.unit
def test_log_error_metrics_handles_empty_metrics():
    modifier = AutoSmoothModifier()
    modifier._error_metrics = []
    modifier._log_error_metrics()
