import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
)

import llmcompressor.modifiers.autosmooth.base as autosmooth_base
from llmcompressor.modifiers.autosmooth import AutoSmoothModifier
from llmcompressor.modifiers.transform.awq.mappings import ResolvedMapping
from llmcompressor.pipelines.cache import IntermediatesCache


class _ToyParent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quantized_balance = torch.nn.Linear(4, 4, bias=False)
        self.ignored_balance = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantized_balance(x) + self.ignored_balance(x)


class _FakeWeightObserver:
    def __init__(self):
        self.args = type("Args", (), {"dynamic": False})()
        self._scale = None
        self._zero_point = None
        self._global_scale = None

    @property
    def has_statistics(self) -> bool:
        return self._scale is not None

    def __call__(self, weight: torch.Tensor):
        self._scale = torch.ones(weight.shape[0], device=weight.device, dtype=weight.dtype)
        self._zero_point = torch.zeros_like(self._scale)
        self._global_scale = torch.ones(1, device=weight.device, dtype=weight.dtype)
        return self

    def get_qparams(self):
        return {
            "scale": self._scale,
            "zero_point": self._zero_point,
            "global_scale": self._global_scale,
        }


@pytest.mark.unit
@pytest.mark.parametrize(
    "n_grid, duo_scaling, expected_len",
    [
        (20, True, 20),
        (10, False, 10),
        (30, "both", 30),
        (2, True, 2),
    ],
)
def test_get_grid_search_params(n_grid, duo_scaling, expected_len):
    autosmooth = AutoSmoothModifier(n_grid=n_grid, duo_scaling=duo_scaling)

    grid_search_params = autosmooth._get_grid_search_params()

    assert (0.0, False) in grid_search_params
    assert all(ratio >= 0.0 for ratio, _ in grid_search_params)
    assert all(ratio <= 1.0 for ratio, _ in grid_search_params)
    assert len(grid_search_params) == expected_len

    n_false = len([
        ratio for ratio, use_duo_scaling in grid_search_params if use_duo_scaling is False
    ])
    n_true = len([
        ratio for ratio, use_duo_scaling in grid_search_params if use_duo_scaling is True
    ])

    if duo_scaling is False:
        assert n_false == n_grid
        assert n_true == 0
    elif duo_scaling is True:
        assert n_false == 1
        assert n_true == expected_len - 1
    else:
        assert abs(n_false - n_true) <= 1


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

    captured_qargs = []

    def _fake_load_from_registry(_name, base_name, args):
        assert base_name == "weight"
        assert args is not None
        captured_qargs.append(args)
        return _FakeWeightObserver()

    monkeypatch.setattr(
        autosmooth_base.Observer,
        "load_from_registry",
        staticmethod(_fake_load_from_registry),
    )
    monkeypatch.setattr(autosmooth_base, "fuse_weight_observers", lambda *args, **kwargs: None)
    monkeypatch.setattr(autosmooth_base, "update_qparams", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        autosmooth_base,
        "forward_quantize",
        lambda _module, value, _base_name, _args: value,
    )

    fp16_outputs = modifier._run_samples(parent)
    best_scales = modifier._compute_best_scale(mapping, fp16_outputs)

    assert captured_qargs == [qargs]
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

    captured_qargs = []

    def _fake_load_from_registry(_name, base_name, args):
        assert base_name == "weight"
        assert args is not None
        captured_qargs.append(args)
        return _FakeWeightObserver()

    monkeypatch.setattr(
        autosmooth_base.Observer,
        "load_from_registry",
        staticmethod(_fake_load_from_registry),
    )
    monkeypatch.setattr(autosmooth_base, "fuse_weight_observers", lambda *args, **kwargs: None)
    monkeypatch.setattr(autosmooth_base, "update_qparams", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        autosmooth_base,
        "forward_quantize",
        lambda _module, value, _base_name, _args: value,
    )

    fp16_outputs = modifier._run_samples(parent)
    best_scales = modifier._compute_best_scale(mapping, fp16_outputs)

    assert captured_qargs == [target_qargs]
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

    monkeypatch.setattr(autosmooth_base, "fuse_weight_observers", lambda *args, **kwargs: None)
    monkeypatch.setattr(autosmooth_base, "update_qparams", lambda *args, **kwargs: None)
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
        staticmethod(lambda *_args, **_kwargs: _FakeWeightObserver()),
    )
    monkeypatch.setattr(autosmooth_base, "fuse_weight_observers", lambda *args, **kwargs: None)
    monkeypatch.setattr(autosmooth_base, "update_qparams", lambda *args, **kwargs: None)
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
        staticmethod(lambda *_args, **_kwargs: _FakeWeightObserver()),
    )
    monkeypatch.setattr(autosmooth_base, "fuse_weight_observers", lambda *args, **kwargs: None)
    monkeypatch.setattr(autosmooth_base, "update_qparams", lambda *args, **kwargs: None)
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
def test_setup_activation_cache_hooks_uses_signed_values_for_minmax():
    parent = torch.nn.Linear(2, 2, bias=False)
    mapping = ResolvedMapping(
        smooth_name="toy.smooth",
        smooth_layer=torch.nn.LayerNorm(2),
        balance_layers=[parent],
        balance_names=["toy.balance"],
        parent=parent,
        parent_name="toy.parent",
    )

    modifier = AutoSmoothModifier(activation_scale_type="minmax")
    modifier.offload_device = None
    modifier._resolved_mappings = [mapping]

    modifier._setup_activation_cache_hooks()
    try:
        parent(torch.tensor([[-3.0, 2.0], [1.0, -4.0]]))
    finally:
        modifier.remove_hooks()

    (min_vals, max_vals), count = modifier._smooth_activation_scales[mapping.smooth_name]

    assert count == 2
    assert torch.equal(min_vals, torch.tensor([-3.0, -4.0]))
    assert torch.equal(max_vals, torch.tensor([1.0, 2.0]))


@pytest.mark.unit
def test_log_error_metrics_handles_empty_metrics():
    modifier = AutoSmoothModifier()
    modifier._error_metrics = []
    modifier._log_error_metrics()


@pytest.mark.unit
def test_capture_quantization_state_preserves_live_qscheme_for_calibration():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
    layer = model[0]
    layer.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            strategy=QuantizationStrategy.CHANNEL,
            num_bits=8,
        ),
    )

    modifier = AutoSmoothModifier()

    modifier._capture_autosmooth_quantization_state(model)

    assert hasattr(layer, "quantization_scheme")

    modifier.start_calibration(model)

    assert hasattr(layer, "weight_observer")
    assert hasattr(layer, "quantization_status")
