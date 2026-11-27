import gc
from copy import deepcopy
from weakref import ref

import pytest
import torch
from compressed_tensors.quantization import apply_quantization_config
from torch import nn

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.quantization.quantization.scoped import (
    CalibrationModuleIndex,
)


def _model():
    return nn.ModuleDict(
        {
            "layers": nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "gate": nn.Linear(4, 8),
                            "up": nn.Linear(4, 8),
                            "down": nn.Linear(8, 4),
                        }
                    )
                    for _index in range(4)
                ]
            ),
            "head": nn.Linear(4, 4),
        }
    )


def _modifier():
    return QuantizationModifier(
        config_groups={
            "gate": {
                "targets": ["re:layers\\..*\\.gate", "re:layers\\..*\\.up"],
                "weights": {"num_bits": 4, "strategy": "channel"},
            },
            "linear": {
                "targets": ["Linear"],
                "weights": {"num_bits": 8, "strategy": "channel"},
            },
        },
        ignore=["re:.*\\.up", "head"],
    )


def test_scoped_calibration_matches_full_config_schemes_and_qparam_layouts():
    expected = _model()
    actual = deepcopy(expected)
    modifier = _modifier()
    wanted = list(expected["layers"][2].modules())
    apply_quantization_config(
        expected, modifier.resolved_config, allowed_modules=wanted, show_progress=False
    )
    modifier.start_layerwise_calibration(
        actual, list(reversed(list(actual["layers"][2].modules())))
    )
    for name, module in expected.named_modules():
        other = actual.get_submodule(name)
        expected_scheme = getattr(module, "quantization_scheme", None)
        assert getattr(other, "quantization_scheme", None) == expected_scheme
        if expected_scheme is not None:
            assert module.weight_scale.shape == other.weight_scale.shape
            assert module.weight_scale.dtype == other.weight_scale.dtype
            torch.testing.assert_close(
                module.weight_zero_point, other.weight_zero_point
            )
    modifier.end_layerwise_calibration(list(actual["layers"][2].modules()))


def test_cached_scoped_names_avoid_full_traversal_on_later_windows(monkeypatch):
    model = _model()
    modifier = _modifier()
    modifier.start_layerwise_calibration(model, list(model["layers"][0].modules()))
    modifier.end_layerwise_calibration(list(model["layers"][0].modules()))

    def unexpected_traversal(*args, **kwargs):
        raise AssertionError("later linear calibration must use scoped module names")

    monkeypatch.setattr(model, "named_modules", unexpected_traversal)
    modifier.start_layerwise_calibration(model, list(model["layers"][1].modules()))
    assert model["layers"][1]["gate"].quantization_scheme.weights.num_bits == 4
    assert model["layers"][1]["down"].quantization_scheme.weights.num_bits == 8
    assert not hasattr(model["layers"][1]["up"], "quantization_scheme")
    modifier.end_layerwise_calibration(list(model["layers"][1].modules()))


def test_scoped_index_refreshes_for_replaced_modules_and_other_models():
    model = _model()
    index = CalibrationModuleIndex(model)
    replacement = nn.Linear(8, 4)
    model["layers"][1]["down"] = replacement
    scoped = index.scoped_model(model, [replacement])
    assert list(scoped.named_modules()) == [("layers.1.down", replacement)]
    assert scoped.get_submodule("layers.0.gate") is model["layers"][0]["gate"]
    other = _model()
    scoped = index.scoped_model(other, [other["head"]])
    assert list(scoped.named_modules()) == [("head", other["head"])]


def test_scoped_index_does_not_keep_a_finished_model_alive():
    model = _model()
    index = CalibrationModuleIndex(model)
    model_reference = ref(model)
    del model
    gc.collect()
    assert model_reference() is None
    assert len(index.names) == 0


def test_scoped_name_cache_does_not_cache_scheme_or_ignore_selection():
    model = _model()
    modifier = _modifier()
    modifier.start_layerwise_calibration(model, list(model["layers"][0].modules()))
    modifier.end_layerwise_calibration(list(model["layers"][0].modules()))
    modifier.resolved_config.config_groups["gate"].weights.num_bits = 2
    modifier.resolved_config.ignore.append("layers.1.down")
    modules = list(model["layers"][1].modules())
    modifier.start_layerwise_calibration(model, modules)
    assert model["layers"][1]["gate"].quantization_scheme.weights.num_bits == 2
    assert not hasattr(model["layers"][1]["down"], "quantization_scheme")
    modifier.end_layerwise_calibration(modules)


@pytest.mark.parametrize("strategy", ["channel", "tensor_group"])
def test_scoped_calibration_releases_only_current_window_hooks(strategy):
    modifier = QuantizationModifier(
        config_groups={
            "weights": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "strategy": strategy,
                    **(
                        {"group_size": 4, "type": "float"}
                        if strategy == "tensor_group"
                        else {}
                    ),
                },
            }
        }
    )
    model = _model()
    modules = list(model["layers"][0].modules())
    modifier.start_layerwise_calibration(model, modules)
    for module in modules:
        if isinstance(module, nn.Linear):
            assert hasattr(module, "weight_observer")
    modifier.end_layerwise_calibration(modules)
    assert not modifier._calibration_hooks
    assert all(not hasattr(module, "weight_observer") for module in modules)
