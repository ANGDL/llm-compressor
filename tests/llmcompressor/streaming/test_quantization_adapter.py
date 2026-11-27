from contextlib import nullcontext
from copy import deepcopy
from threading import Barrier, current_thread

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from torch import nn

from llmcompressor.core import LifecycleCallbacks, create_session
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import AWQModifier, SmoothQuantModifier
from llmcompressor.streaming.memory_adapters import get_modifier_memory_adapter
from llmcompressor.streaming.modifier_contract import ModifierContractError
from llmcompressor.streaming.parallel.modifier_update import OwnerModuleGroup
from llmcompressor.streaming.pipeline import _modifier_update, _owner_update_adapter
from llmcompressor.streaming.quantization_adapter import (
    QuantizationStreamingMemoryAdapter,
    QuantizationWindowAdapter,
)
from llmcompressor.utils.helpers import DisableQuantization


def _quantizer(observer="minmax", *, static_activations=False):
    weights = QuantizationArgs(
        num_bits=4,
        strategy="channel",
        observer=observer,
        observer_kwargs=(
            {"grid": 3, "patience": 0, "strict": True}
            if observer == "imatrix_mse"
            else {}
        ),
    )
    activation = (
        QuantizationArgs(num_bits=8, strategy="tensor", dynamic=False)
        if static_activations
        else None
    )
    return QuantizationModifier(
        config_groups={
            "weights": QuantizationScheme(
                targets=["Linear"],
                weights=weights,
                input_activations=activation,
                output_activations=activation if observer != "imatrix_mse" else None,
            )
        }
    )


@pytest.mark.parametrize("modifier_type", [QuantizationModifier, GPTQModifier])
def test_quantizers_share_only_the_window_lifecycle(modifier_type):
    modifier = modifier_type(scheme="W8A8")
    adapter = get_modifier_memory_adapter(modifier)
    assert isinstance(adapter, QuantizationWindowAdapter)
    assert adapter.owner_update_scope == "owner_local"
    assert adapter.owner_update_state == "independent"
    assert callable(adapter.update_owner)


@pytest.mark.parametrize(
    "dynamic,strategy,expected",
    [(True, "token", True), (True, "tensor", False), (False, "tensor", False)],
)
def test_rtn_window_coalescing_requires_row_independent_activations(
    dynamic, strategy, expected
):
    modifier = _quantizer()
    scheme = modifier.resolved_config.config_groups["weights"]
    scheme.input_activations = QuantizationArgs(
        num_bits=8, dynamic=dynamic, strategy=strategy
    )
    assert (
        get_modifier_memory_adapter(modifier).supports_independent_expert_windows
        is expected
    )


@pytest.mark.parametrize("transform_type", [AWQModifier, SmoothQuantModifier])
@pytest.mark.parametrize("quantizer_type", [QuantizationModifier, GPTQModifier])
def test_mixed_recipe_requires_all_modifiers_to_participate(
    transform_type, quantizer_type
):
    quantizer = quantizer_type(scheme="W8A8")
    transform = transform_type()
    assert _owner_update_adapter([quantizer], (), [quantizer]) is not None
    assert _owner_update_adapter([quantizer], (), [transform, quantizer]) is None
    assert _owner_update_adapter([quantizer], (), [quantizer, transform]) is None


@pytest.mark.parametrize("transform_type", [AWQModifier, SmoothQuantModifier])
def test_mixed_recipe_uses_the_sequential_callback(monkeypatch, transform_type):
    quantizer = _quantizer()
    modules = (nn.Linear(4, 4),)
    observed = []
    monkeypatch.setattr(
        LifecycleCallbacks,
        "sequential_epoch_end",
        lambda modules: observed.append(modules),
    )
    mode = _modifier_update(
        modules,
        [quantizer],
        owner_groups=[OwnerModuleGroup(torch.device("cpu"), modules)],
        modifiers=[transform_type(), quantizer],
    )
    assert mode == "serial-contract"
    assert observed == [list(modules)]


def _run_update(model, inputs, modifier, *, parallel):
    with create_session() as session:
        session.state.layerwise_calibration = True
        session.initialize(model=model, recipe=[modifier], start=-1, copy_data=False)
        modifier = session.lifecycle.recipe.modifiers[0]
        LifecycleCallbacks.calibration_start()
        modules = tuple(model)
        modifier.start_layerwise_calibration(model, list(modules))
        with torch.no_grad(), DisableQuantization(model):
            for value in inputs:
                model(value)
        groups = tuple(
            OwnerModuleGroup(torch.device(f"cpu:{index}"), (module,))
            for index, module in enumerate(modules)
        )
        mode = _modifier_update(
            modules,
            [modifier],
            owner_groups=groups if parallel else None,
            modifiers=[modifier],
        )
        assert mode == ("owner-local" if parallel else "serial-contract")
        assert not modifier._calibration_hooks
        assert not any(hasattr(module, "weight_observer") for module in modules)
        state = {name: value.clone() for name, value in model.state_dict().items()}
        LifecycleCallbacks.calibration_end()
        session.finalize()
        return state


@pytest.mark.parametrize("observer", ["minmax", "imatrix_mse"])
@pytest.mark.parametrize("static_activations", [False, True])
def test_rtn_owner_updates_match_serial_weights_and_qparams(
    observer, static_activations, monkeypatch
):
    torch.manual_seed(17)
    reference = nn.Sequential(
        nn.Linear(4, 8), nn.Linear(8, 8), nn.Linear(8, 4), nn.Linear(4, 4)
    ).eval()
    parallel = deepcopy(reference)
    inputs = [torch.randn(2, 3, 4), torch.randn(1, 5, 4)]
    expected = _run_update(
        reference,
        inputs,
        _quantizer(observer, static_activations=static_activations),
        parallel=False,
    )
    barrier = Barrier(len(parallel), timeout=10)
    threads = set()
    original_sync = QuantizationModifier.sync_obs_act_stats

    def sync(modifier, modules):
        threads.add(current_thread().name)
        barrier.wait()
        return original_sync(modifier, modules)

    monkeypatch.setattr(QuantizationModifier, "sync_obs_act_stats", sync)
    actual = _run_update(
        parallel,
        inputs,
        _quantizer(observer, static_activations=static_activations),
        parallel=True,
    )
    assert len(threads) == len(parallel)
    assert actual.keys() == expected.keys()
    for name, expected_value in expected.items():
        torch.testing.assert_close(actual[name], expected_value, msg=name)


@pytest.mark.parametrize("state_name", ["weight_observer", "input_scale"])
def test_window_adapter_rejects_quantization_state_shared_across_owners(state_name):
    modules = [nn.Linear(4, 4), nn.Linear(4, 4)]
    shared = nn.Identity() if state_name.endswith("observer") else torch.ones(1)
    for module in modules:
        setattr(module, state_name, shared)
    groups = tuple(
        OwnerModuleGroup(torch.device(f"cpu:{index}"), (module,))
        for index, module in enumerate(modules)
    )
    with pytest.raises(ModifierContractError, match=state_name):
        get_modifier_memory_adapter(_quantizer()).validate_owner_groups(groups)


@pytest.mark.parametrize("fail", [False, True])
def test_rtn_window_cleanup_preserves_other_owner_hooks(fail):
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4)).eval()
    with create_session() as session:
        session.state.layerwise_calibration = True
        session.initialize(
            model=model, recipe=[_quantizer("imatrix_mse")], start=-1, copy_data=False
        )
        modifier = session.lifecycle.recipe.modifiers[0]
        LifecycleCallbacks.calibration_start()
        adapter = get_modifier_memory_adapter(modifier)
        first, second = (model[0],), (model[1],)
        with adapter.expert_window(model, first), torch.no_grad():
            model[0](torch.randn(2, 4))
            observer = model[0].weight_observer
            count = observer._imatrix_count.clone()
            failure_context = pytest.raises(RuntimeError) if fail else nullcontext()
            with failure_context:
                with adapter.expert_window(model, second):
                    model[1](torch.randn(3, 4))
                    if fail:
                        raise RuntimeError("injected failure")
                    adapter.update_owner(torch.device("cpu:1"), second)
            assert observer._imatrix_hook is not None
            assert not hasattr(model[1], "weight_observer")
            adapter.update_owner(torch.device("cpu:0"), first)
            adapter.after_update(first)
            model[0](torch.randn(3, 4))
            assert observer._imatrix_hook is None
            assert not hasattr(observer, "_imatrix_count")
            assert count.item() == 2
        assert not adapter._window_handles
        assert not modifier._calibration_hooks
        assert not model[0]._forward_pre_hooks
        assert not model[1]._forward_pre_hooks
        LifecycleCallbacks.calibration_end()
        session.finalize()


def test_rtn_memory_and_work_estimates_ignore_activation_only_modules():
    weight = nn.Linear(4, 8, device="meta")
    weight.quantization_scheme = QuantizationScheme(
        targets=["Linear"], weights=QuantizationArgs(num_bits=4, strategy="channel")
    )
    activation = nn.Identity()
    activation.quantization_scheme = QuantizationScheme(
        targets=["Identity"], input_activations=QuantizationArgs(num_bits=8)
    )
    adapter = QuantizationStreamingMemoryAdapter(_quantizer())
    assert adapter.estimate_update_memory([weight, activation]) == 4 * 8 * 4
    assert adapter.estimate_update_work([weight, activation]) == 4 * 8
    assert adapter.estimate_batch_memory(activation, 16, "calibration") == 0
    weight.quantization_scheme.weights.observer = "imatrix_mse"
    assert adapter.estimate_batch_memory(weight, 16, "calibration") == 2 * 16 * 4 * 4
    assert adapter.estimate_batch_memory(weight, 16, "propagation") == 0
