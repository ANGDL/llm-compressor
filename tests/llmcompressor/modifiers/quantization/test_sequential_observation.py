from unittest.mock import patch

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStatus,
    initialize_module_for_quantization,
)
from torch import nn

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.quantization.calibration import initialize_observer


class _FakeDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)


def test_initialize_quantization_applies_scale_dtype():
    model = nn.Sequential(nn.Linear(4, 2, bias=False, dtype=torch.bfloat16))
    modifier = QuantizationModifier(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=8,
                    type="int",
                    strategy="channel",
                    symmetric=True,
                    scale_dtype=torch.float32,
                ),
            )
        }
    )

    modifier.initialize_quantization(model)

    assert model[0].weight_scale.dtype == torch.float32


def test_initialize_quantization_rejects_float64_scale_dtype():
    with pytest.raises(ValueError, match="float64.*not supported"):
        modifier = QuantizationModifier(
            config_groups={
                "group_0": QuantizationScheme(
                    targets=["Linear"],
                    weights=QuantizationArgs(scale_dtype=torch.float64),
                )
            }
        )
        modifier.resolve_quantization_config()


def test_initialize_quantization_rejects_unrelated_scale_dtype():
    model = nn.Sequential(nn.Linear(4, 2, bias=False, dtype=torch.bfloat16))
    modifier = QuantizationModifier(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(scale_dtype=torch.float16),
            )
        }
    )

    with pytest.raises(ValueError, match="supported dtypes"):
        modifier.initialize_quantization(model)


def test_sequential_epoch_end_only_observes_passed_modules():
    """Verify only subgraph modules observed, not all model weights."""
    model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10), nn.Linear(10, 10))
    args = QuantizationArgs(num_bits=8, type="int", symmetric=True, strategy="tensor")
    for layer in model:
        initialize_module_for_quantization(
            layer, QuantizationScheme(targets=[], weights=args)
        )
        initialize_observer(layer, "weight")
    modifier = QuantizationModifier(targets="Linear", scheme="FP8")
    state = State(model=model)
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, None)

    # Wrap update_statistics_from_observed to track which observers are called
    with patch.object(
        model[0].weight_observer,
        "update_statistics_from_observed",
        wraps=model[0].weight_observer.update_statistics_from_observed,
    ) as mock0, patch.object(
        model[1].weight_observer,
        "update_statistics_from_observed",
        wraps=model[1].weight_observer.update_statistics_from_observed,
    ) as mock1, patch.object(
        model[2].weight_observer,
        "update_statistics_from_observed",
        wraps=model[2].weight_observer.update_statistics_from_observed,
    ) as mock2:
        modifier.on_sequential_epoch_end(
            state,
            Event(type_=EventType.SEQUENTIAL_EPOCH_END),
            modules=list(model[0].modules()),
        )
        mock0.assert_called_once()  # Only first module should be observed
        mock1.assert_not_called()  # Second module should NOT be observed
        mock2.assert_not_called()  # Third module should NOT be observed


def test_sequential_activation_qparams_only_updated_once_per_module():
    """Verify activation qparams and sync only called once per module per chunk."""
    model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10), nn.Linear(10, 10))
    act_args = QuantizationArgs(
        num_bits=8, type="int", symmetric=True, strategy="tensor"
    )
    for layer in model:
        initialize_module_for_quantization(
            layer, QuantizationScheme(targets=[], input_activations=act_args)
        )
        initialize_observer(layer, "input")
        layer.input_observer(layer.weight)  # Trigger observer to have statistics
    modifier = QuantizationModifier(targets="Linear", scheme="W8A8")
    state = State(model=model)
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, None)

    # Patch get_qparams and sync_activation_stats to track calls
    with patch.object(
        model[0].input_observer,
        "get_qparams",
        wraps=model[0].input_observer.get_qparams,
    ) as qp_mock0, patch.object(
        model[1].input_observer,
        "get_qparams",
        wraps=model[1].input_observer.get_qparams,
    ) as qp_mock1, patch.object(
        model[2].input_observer,
        "get_qparams",
        wraps=model[2].input_observer.get_qparams,
    ) as qp_mock2, patch.object(
        model[0].input_observer, "sync_activation_stats", return_value=[]
    ) as sync_mock0, patch.object(
        model[1].input_observer, "sync_activation_stats", return_value=[]
    ) as sync_mock1, patch.object(
        model[2].input_observer, "sync_activation_stats", return_value=[]
    ) as sync_mock2, patch(
        "llmcompressor.modifiers.quantization.quantization.mixin.is_distributed",
        return_value=True,
    ):
        # Process chunks sequentially
        modifier.on_sequential_epoch_end(
            state,
            Event(type_=EventType.SEQUENTIAL_EPOCH_END),
            modules=list(model[0].modules()),
        )
        modifier.on_sequential_epoch_end(
            state,
            Event(type_=EventType.SEQUENTIAL_EPOCH_END),
            modules=list(model[1].modules()),
        )
        modifier.on_sequential_epoch_end(
            state,
            Event(type_=EventType.SEQUENTIAL_EPOCH_END),
            modules=list(model[2].modules()),
        )

        # Each observer's methods should be called exactly once
        assert qp_mock0.call_count == 1
        assert qp_mock1.call_count == 1
        assert qp_mock2.call_count == 1
        assert sync_mock0.call_count == 1
        assert sync_mock1.call_count == 1
        assert sync_mock2.call_count == 1


def test_nested_parent_modules_produce_valid_global_scale():
    """When SEQUENTIAL_EPOCH_END receives parent modules (as the pipeline does),
    child Linear layers must get finite, positive global_scale values —
    not uninitialized torch.empty garbage."""
    model = nn.Sequential(_FakeDecoderLayer())
    tg = QuantizationArgs(
        num_bits=4, type="float", symmetric=True, group_size=16, strategy="tensor_group"
    )
    scheme = QuantizationScheme(
        targets=[],
        weights=tg,
        input_activations=QuantizationArgs(**{**tg.model_dump(), "dynamic": "local"}),
    )
    for child in (model[0].q_proj, model[0].k_proj, model[0].v_proj):
        initialize_module_for_quantization(child, scheme)

    modifier = QuantizationModifier(targets="Linear", scheme="NVFP4")
    state = State(model=model)
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, None)

    for child in (model[0].q_proj, model[0].k_proj, model[0].v_proj):
        child.weight_observer(child.weight)
        child.input_observer(torch.randn(2, 64))

    modifier.on_sequential_epoch_end(
        state,
        Event(type_=EventType.SEQUENTIAL_EPOCH_END),
        modules=list(model.modules()),
    )

    for name, param in model.named_parameters():
        if "global_scale" in name:
            assert (
                torch.isfinite(param).all() and param.item() > 0
            ), f"{name} not valid: {param.item()}"


def test_layerwise_quantization_only_initializes_and_freezes_passed_modules():
    model = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 256))
    modifier = QuantizationModifier(targets="Linear", scheme="W8A16")
    state = State(model=model)
    state.layerwise_decompression = True

    modifier.on_initialize(state)
    modifier.on_calibration_start(state, None)

    assert not hasattr(model[0], "quantization_scheme")
    assert not hasattr(model[1], "quantization_scheme")

    modules = list(model[0].modules())
    with patch(
        "llmcompressor.modifiers.quantization.quantization.mixin.fuse_weight_observers"
    ) as fuse_weight_observers:
        modifier.start_layerwise_calibration(model, modules)

    fuse_weight_observers.assert_called_once_with(model, modules)

    assert model[0].quantization_status == QuantizationStatus.CALIBRATION
    assert hasattr(model[0], "weight_observer")
    assert not hasattr(model[1], "quantization_scheme")

    modifier.on_sequential_epoch_end(
        state,
        Event(type_=EventType.SEQUENTIAL_EPOCH_END),
        modules=list(model[0].modules()),
    )

    assert model[0].quantization_status == QuantizationStatus.FROZEN
    assert not hasattr(model[0], "weight_observer")
    assert not hasattr(model[1], "quantization_scheme")


def test_layerwise_quantization_scopes_apply_config_to_passed_modules():
    model = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 256))
    modifier = QuantizationModifier(targets="Linear", scheme="W8A16")
    modules = list(model[0].modules())

    from compressed_tensors.quantization import apply_quantization_config

    with patch(
        "llmcompressor.modifiers.quantization.quantization.mixin.apply_quantization_config",
        wraps=apply_quantization_config,
    ) as apply_config:
        modifier.start_layerwise_calibration(model, modules)

    apply_config.assert_called_once_with(
        model, modifier.resolved_config, allowed_modules=modules
    )
    assert model[0].quantization_status == QuantizationStatus.CALIBRATION
    assert not hasattr(model[1], "quantization_scheme")


def test_layerwise_quantization_preserves_explicit_scale_dtype():
    model = nn.Sequential(nn.Linear(4, 2, bias=False, dtype=torch.bfloat16))
    modifier = QuantizationModifier(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=8,
                    type="int",
                    strategy="channel",
                    symmetric=True,
                    scale_dtype=torch.float32,
                ),
            )
        }
    )
    modifier.start_layerwise_calibration(model, [model[0]])

    assert model[0].weight_scale.dtype == torch.float32
    modifier.end_layerwise_calibration([model[0]])
    assert model[0].weight_scale.dtype == torch.float32


def test_layerwise_quantization_fuses_only_current_group_and_releases_hooks():
    model = nn.Sequential(_FakeDecoderLayer(), _FakeDecoderLayer())
    modifier = QuantizationModifier(targets="Linear", scheme="NVFP4")
    for current_index in range(len(model)):
        modules = list(model[current_index].modules())
        modifier.start_layerwise_calibration(model, modules)
        projections = [
            model[current_index].q_proj,
            model[current_index].k_proj,
            model[current_index].v_proj,
        ]
        for projection in projections:
            assert projection.weight_observer.fusion_handler.is_fused
        for other_index, layer in enumerate(model):
            if other_index != current_index:
                assert not hasattr(layer.q_proj, "weight_observer")
        handles = set(modifier._calibration_hooks)
        assert handles

        modifier.end_layerwise_calibration(modules)

        assert not modifier._calibration_hooks
        assert not modifier._hooks
        for projection in projections:
            assert not hasattr(projection, "weight_observer")
            assert not projection._forward_hooks
            assert not projection._forward_pre_hooks
