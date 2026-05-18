import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
)

import llmcompressor.modifiers.autosmooth.base as autosmooth_base
from llmcompressor.modifiers.autosmooth import AutoSmoothModifier
from llmcompressor.modifiers.transform.awq.mappings import AWQMapping, ResolvedMapping
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
    # Only quantized_balance has snapshot qargs; ignored_balance is non-quantized
    # and participates in grid search with identity pseudo-quantization.
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
    captured_update_modules = []

    def _fake_update_qparams(modules, *_args, **_kwargs):
        captured_update_modules.append(modules)

    monkeypatch.setattr(autosmooth_base, "update_qparams", _fake_update_qparams)
    monkeypatch.setattr(
        autosmooth_base,
        "forward_quantize",
        lambda _module, value, _base_name, _args: value,
    )

    fp16_outputs = modifier._run_samples(parent)
    best_scales = modifier._compute_best_scale(mapping, fp16_outputs)

    assert captured_qargs == [qargs]
    assert captured_update_modules
    assert all(isinstance(modules, list) for modules in captured_update_modules)
    assert torch.isfinite(best_scales).all()


@pytest.mark.unit
@torch.no_grad()
def test_compute_best_scale_non_quantized_uses_identity(monkeypatch):
    """Non-quantized balance layers participate in grid search with identity
    pseudo-quantization (scale then unscale). Only quantized_balance has qargs,
    so only its qargs should appear in observer construction."""
    parent = _ToyParent()
    smooth = torch.nn.LayerNorm(4)

    target_qargs = QuantizationArgs(
        strategy=QuantizationStrategy.CHANNEL,
        num_bits=8,
    )

    parent.quantized_balance.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=target_qargs,
    )
    # ignored_balance has NO quantization_scheme — non-quantized layer
    # that still participates in the mapping for smoothing.

    mapping = ResolvedMapping(
        smooth_name="toy.smooth",
        smooth_layer=smooth,
        balance_layers=[parent.quantized_balance, parent.ignored_balance],
        balance_names=["toy.quantized_balance", "toy.ignored_balance"],
        parent=parent,
        parent_name="toy",
    )

    modifier = AutoSmoothModifier(n_grid=2, duo_scaling=False)
    modifier._autosmooth_weight_qargs = {parent.quantized_balance: target_qargs}
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
def test_compute_best_scale_all_non_quantized_balance_layers_use_identity(monkeypatch):
    """When all balance layers are non-quantized, AutoSmooth still runs
    grid search via identity pseudo-quantization."""
    parent = _ToyParent()
    smooth = torch.nn.LayerNorm(4)

    mapping = ResolvedMapping(
        smooth_name="toy.smooth",
        smooth_layer=smooth,
        balance_layers=[parent.quantized_balance],
        balance_names=["toy.quantized_balance"],
        parent=parent,
        parent_name="toy",
    )

    modifier = AutoSmoothModifier(n_grid=2, duo_scaling=False)
    modifier._smooth_activation_scales[mapping.smooth_name] = (torch.ones(4), 1)

    cache = IntermediatesCache(offload_device=None)
    cache.append({"x": torch.randn(2, 4)})
    modifier._parent_args_cache[parent] = cache

    def _unexpected_observer(*_args, **_kwargs):
        raise AssertionError(
            "weight observer should not be created for non-quantized layers"
        )

    def _unexpected_update_qparams(*_args, **_kwargs):
        raise AssertionError(
            "update_qparams should not run for non-quantized layers"
        )

    monkeypatch.setattr(
        autosmooth_base.Observer,
        "load_from_registry",
        staticmethod(_unexpected_observer),
    )
    monkeypatch.setattr(autosmooth_base, "update_qparams", _unexpected_update_qparams)

    fp16_outputs = modifier._run_samples(parent)
    best_scales = modifier._compute_best_scale(mapping, fp16_outputs)

    assert torch.isfinite(best_scales).all()


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
def test_compute_best_scale_duo_scaling_includes_non_quantized_in_weight_scales(monkeypatch):
    parent = _ToyParent()
    smooth = torch.nn.LayerNorm(4)

    qargs = QuantizationArgs(
        strategy=QuantizationStrategy.CHANNEL,
        num_bits=8,
    )
    parent.quantized_balance.quantization_scheme = QuantizationScheme(
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

    modifier = AutoSmoothModifier(n_grid=2, duo_scaling=True)
    modifier._autosmooth_weight_qargs = {parent.quantized_balance: qargs}
    modifier._smooth_activation_scales[mapping.smooth_name] = (torch.ones(4), 1)

    cache = IntermediatesCache(offload_device=None)
    cache.append({"x": torch.randn(2, 4)})
    modifier._parent_args_cache[parent] = cache

    captured_call: dict[str, list] = {}

    def _fake_compute_layer_scales(layers, layer_names=None, layer_qargs=None):
        captured_call["layers"] = layers
        captured_call["layer_names"] = layer_names
        captured_call["layer_qargs"] = layer_qargs
        return torch.ones(4)

    monkeypatch.setattr(
        autosmooth_base.AutoSmoothModifier,
        "_compute_layer_scales",
        staticmethod(_fake_compute_layer_scales),
    )
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
    assert len(captured_call["layers"]) == 2
    assert captured_call["layer_names"] == [
        "toy.quantized_balance",
        "toy.ignored_balance",
    ]
    assert captured_call["layer_qargs"][0] is qargs
    assert captured_call["layer_qargs"][1] is None


@pytest.mark.unit
@torch.no_grad()
def test_compute_best_scale_duo_scaling_can_exclude_non_quantized_from_weight_scales(monkeypatch):
    parent = _ToyParent()
    smooth = torch.nn.LayerNorm(4)

    qargs = QuantizationArgs(
        strategy=QuantizationStrategy.CHANNEL,
        num_bits=8,
    )
    parent.quantized_balance.quantization_scheme = QuantizationScheme(
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

    modifier = AutoSmoothModifier(
        n_grid=2,
        duo_scaling=True,
        include_nonq_wscale=False,
    )
    modifier._autosmooth_weight_qargs = {parent.quantized_balance: qargs}
    modifier._smooth_activation_scales[mapping.smooth_name] = (torch.ones(4), 1)

    cache = IntermediatesCache(offload_device=None)
    cache.append({"x": torch.randn(2, 4)})
    modifier._parent_args_cache[parent] = cache

    captured_call: dict[str, list] = {}

    def _fake_compute_layer_scales(layers, layer_names=None, layer_qargs=None):
        captured_call["layers"] = layers
        captured_call["layer_names"] = layer_names
        captured_call["layer_qargs"] = layer_qargs
        return torch.ones(4)

    monkeypatch.setattr(
        autosmooth_base.AutoSmoothModifier,
        "_compute_layer_scales",
        staticmethod(_fake_compute_layer_scales),
    )
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
    assert len(captured_call["layers"]) == 1
    assert captured_call["layer_names"] == ["toy.quantized_balance"]
    assert captured_call["layer_qargs"] == [qargs]


@pytest.mark.unit
@torch.no_grad()
def test_compute_best_scale_sanitizes_scales_before_scalesview(monkeypatch):
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

    mapping = ResolvedMapping(
        smooth_name="toy.smooth",
        smooth_layer=smooth,
        balance_layers=[parent.quantized_balance],
        balance_names=["toy.quantized_balance"],
        parent=parent,
        parent_name="toy",
    )

    modifier = AutoSmoothModifier(n_grid=2, duo_scaling=True)
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

    # Force an inf entry in activation scales. The implementation must sanitize
    # scales before constructing _scalesview to avoid propagating inf/nan into
    # temporary weights and loss computation.
    monkeypatch.setattr(
        autosmooth_base.AutoSmoothModifier,
        "_resolve_activation_scales",
        lambda *_args, **_kwargs: torch.tensor([float("inf"), 1.0, 1.0, 1.0]),
    )
    monkeypatch.setattr(
        autosmooth_base.AutoSmoothModifier,
        "_compute_layer_scales",
        staticmethod(lambda *_args, **_kwargs: torch.ones(4)),
    )

    original_compute_loss = autosmooth_base.AutoSmoothModifier._compute_loss

    def _finite_loss(_self, fp16_outputs, int_w_outputs):
        for out in int_w_outputs:
            assert torch.isfinite(out).all(), "int_w_outputs should stay finite"
        return original_compute_loss(
            _self,
            fp16_outputs,
            int_w_outputs,
        )

    monkeypatch.setattr(
        autosmooth_base.AutoSmoothModifier,
        "_compute_loss",
        _finite_loss,
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
@torch.no_grad()
def test_compute_best_scale_does_not_leak_temporary_qparams(monkeypatch):
    parent = _ToyParent()
    smooth = torch.nn.LayerNorm(4)

    qargs = QuantizationArgs(
        strategy=QuantizationStrategy.TENSOR_GROUP,
        group_size=2,
        num_bits=8,
        symmetric=False,
    )
    parent.quantized_balance.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=qargs,
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
    assert not hasattr(parent.quantized_balance, "weight_observer")
    assert not hasattr(parent.quantized_balance, "weight_scale")
    assert not hasattr(parent.quantized_balance, "weight_zero_point")
    assert not hasattr(parent.quantized_balance, "weight_global_scale")


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
def test_set_resolved_mappings_keeps_non_quantized_balance_layers():
    class _ToyMappingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.smooth = torch.nn.LayerNorm(4)
            self.balance = torch.nn.Linear(4, 4, bias=False)

    model = _ToyMappingModel()
    modifier = AutoSmoothModifier(
        mappings=[AWQMapping(smooth_layer="smooth", balance_layers=["balance"])]
    )

    modifier._set_resolved_mappings(model)

    assert len(modifier._resolved_mappings) == 1
    resolved = modifier._resolved_mappings[0]
    assert resolved.smooth_name == "smooth"
    assert resolved.smooth_layer is model.smooth
    assert resolved.balance_layers == [model.balance]


@pytest.mark.unit
def test_capture_quantization_state_tracks_non_quantized_balance_layers():
    class _ToyMappingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.smooth = torch.nn.LayerNorm(4)
            self.balance = torch.nn.Linear(4, 4, bias=False)

    model = _ToyMappingModel()
    modifier = AutoSmoothModifier(
        mappings=[AWQMapping(smooth_layer="smooth", balance_layers=["balance"])]
    )

    modifier._set_resolved_mappings(model)
    modifier._capture_autosmooth_quantization_state(model)

    assert model.balance in modifier._autosmooth_weight_qargs
    assert modifier._autosmooth_weight_qargs[model.balance] is None


@pytest.mark.unit
def test_capture_quantization_state_reads_from_modules():
    """Verify snapshot reads quantization_scheme from modules and deep copies qargs."""
    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
    layer = model[0]

    original_qargs = QuantizationArgs(
        strategy=QuantizationStrategy.CHANNEL,
        num_bits=8,
    )
    layer.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=original_qargs,
    )

    modifier = AutoSmoothModifier()
    modifier._capture_autosmooth_quantization_state(model)

    assert layer in modifier._autosmooth_weight_qargs
    assert (
        modifier._autosmooth_weight_qargs[layer].strategy
        == QuantizationStrategy.CHANNEL
    )

    # Snapshot should be a deep copy — mutating the original must not affect it.
    original_qargs.num_bits = 4
    assert modifier._autosmooth_weight_qargs[layer].num_bits == 8


@pytest.mark.unit
def test_validate_duo_scaling_with_snapshot_rejects_tensor_strategy():
    layer = torch.nn.Linear(4, 4, bias=False)
    modifier = AutoSmoothModifier(duo_scaling=True)
    modifier._autosmooth_weight_qargs = {
        layer: QuantizationArgs(
            strategy=QuantizationStrategy.TENSOR,
            num_bits=8,
        )
    }

    with pytest.raises(ValueError, match="duo_scaling is only supported"):
        modifier._validate_duo_scaling_with_snapshot()


@pytest.mark.unit
def test_resolve_activation_scales_uses_allreduce_for_mean(monkeypatch):
    parent = _ToyParent()
    mapping = ResolvedMapping(
        smooth_name="toy.smooth",
        smooth_layer=torch.nn.LayerNorm(4),
        balance_layers=[parent.quantized_balance],
        balance_names=["toy.quantized_balance"],
        parent=parent,
        parent_name="toy",
    )

    modifier = AutoSmoothModifier(activation_scale_type="mean")
    modifier._smooth_activation_scales[mapping.smooth_name] = (
        torch.tensor([2.0, 4.0, 6.0, 8.0]),
        2,
    )

    calls = []

    def _fake_allreduce_sum(data):
        calls.append(data)
        return [datum * 2 for datum in data]

    monkeypatch.setattr(autosmooth_base, "is_distributed", lambda: True)
    monkeypatch.setattr(autosmooth_base, "_allreduce_data_sum", _fake_allreduce_sum)

    x_scales = modifier._resolve_activation_scales(mapping, torch.device("cpu"))

    assert calls
    assert torch.allclose(x_scales, torch.tensor([2.0, 4.0, 6.0, 8.0]))


@pytest.mark.unit
def test_compute_loss_uses_allreduce_when_distributed(monkeypatch):
    modifier = AutoSmoothModifier()
    fp16_outputs = [torch.tensor([[1.0, -1.0]])]
    int_w_outputs = [torch.tensor([[0.0, 0.0]])]

    calls = []

    def _fake_allreduce_sum(data):
        calls.append(data)
        return data

    monkeypatch.setattr(autosmooth_base, "active_session", lambda: None)
    monkeypatch.setattr(autosmooth_base, "is_distributed", lambda: True)
    monkeypatch.setattr(autosmooth_base, "_allreduce_data_sum", _fake_allreduce_sum)

    loss = modifier._compute_loss(fp16_outputs, int_w_outputs)

    assert calls
    assert loss == pytest.approx(1.0)


@pytest.mark.unit
def test_capture_quantization_state_preserves_live_qscheme_reference():
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
    assert layer in modifier._autosmooth_weight_qargs
