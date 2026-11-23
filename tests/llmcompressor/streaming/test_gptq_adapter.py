import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from torch import nn

from llmcompressor.core import LifecycleCallbacks, create_session
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.gptq import helpers as gptq_helpers
from llmcompressor.modifiers.gptq.helpers import (
    iter_batches,
    predict_batch_size,
    quantization_batch_peak_bytes,
)
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.streaming import gptq_adapter as gptq_adapter_module
from llmcompressor.streaming.admission import estimate_modifier_dispatch_memory
from llmcompressor.streaming.calibration import calibration_batch_context
from llmcompressor.streaming.memory_adapters import (
    get_modifier_memory_adapter,
    modifier_calibration_context,
)
from llmcompressor.utils.helpers import DisableQuantization


@pytest.mark.parametrize("fail", [False, True])
def test_scoped_gptq_compatibility_restores_instance_method_and_other_instances(fail):
    modifier = GPTQModifier()
    other = GPTQModifier()
    original = modifier.calibrate_module
    try:
        with modifier_calibration_context([modifier]):
            assert "calibrate_module" in vars(modifier)
            assert "calibrate_module" not in vars(other)
            if fail:
                raise RuntimeError("calibration failure")
    except RuntimeError:
        assert fail
    assert "calibrate_module" not in vars(modifier)
    assert modifier.calibrate_module == original


@pytest.mark.parametrize("ndim", [2, 3])
def test_gptq_coalesced_counts_preserve_original_batch_semantics_and_hook_disable(ndim):
    model = nn.Sequential(nn.Linear(4, 3, bias=False))
    with create_session() as session:
        session.state.layerwise_calibration = True
        session.initialize(
            model=model,
            recipe=[GPTQModifier(scheme="W8A8")],
            start=-1,
            copy_data=False,
        )
        modifier = session.lifecycle.recipe.modifiers[0]
        with modifier_calibration_context([modifier]):
            LifecycleCallbacks.calibration_start()
            modifier.start_layerwise_calibration(model, list(model.modules()))
            shape = (5, 4) if ndim == 2 else (5, 2, 4)
            inputs = torch.randn(shape)
            with DisableQuantization(model), torch.no_grad():
                with calibration_batch_context((2, 3)):
                    model(inputs)
                expected_count = 2 if ndim == 2 else 5
                assert modifier._num_samples[model[0]].item() == expected_count
                snapshot = modifier._hessians[model[0]].clone()
                with HooksMixin.disable_hooks(), calibration_batch_context((2, 3)):
                    model(inputs)
                assert modifier._num_samples[model[0]].item() == expected_count
                torch.testing.assert_close(modifier._hessians[model[0]], snapshot)
            LifecycleCallbacks.sequential_epoch_end(list(model.modules()))
            modifier.end_layerwise_calibration(list(model.modules()))
            LifecycleCallbacks.calibration_end()
        session.finalize()


@pytest.mark.parametrize("batch_size, count", [(None, 1), (2, 2), ("auto", 1)])
def test_gptq_resource_adapter_bounds_compatible_update_batches(batch_size, count):
    model = nn.Sequential(
        *[nn.Linear(8, 4, bias=False, device="meta") for _module in range(3)]
    )
    with create_session() as session:
        session.state.layerwise_calibration = True
        session.initialize(
            model=model,
            recipe=[GPTQModifier(scheme="W8A8", batched_quantization=batch_size)],
            start=-1,
            copy_data=False,
        )
        modifier = session.lifecycle.recipe.modifiers[0]
        adapter = get_modifier_memory_adapter(modifier)
        assert adapter.estimate_update_memory(tuple(model)) == (
            quantization_batch_peak_bytes(model[0], count, modifier.block_size)
        )
        estimate = estimate_modifier_dispatch_memory(
            [modifier],
            model.modules(),
            total_tokens=100,
            dispatch_tokens=11,
        )
        assert estimate.held_bytes >= 3 * 8 * 8 * torch.float32.itemsize
        assert estimate.scratch_bytes >= adapter.estimate_update_memory(tuple(model))
        assert estimate.batch_bytes == (2 * 11 * 8 + 8 * 8) * 4
        assert adapter.estimate_batch_memory(model[0], 11, "propagation") == 0


def test_gptq_auto_update_batch_is_predicted_from_device_bytes():
    model = nn.Sequential(
        *[nn.Linear(8, 4, bias=False, device="meta") for _module in range(40)]
    )
    with create_session() as session:
        session.state.layerwise_calibration = True
        session.initialize(
            model=model,
            recipe=[GPTQModifier(scheme="W8A8", batched_quantization="auto")],
            start=-1,
            copy_data=False,
        )
        modifier = session.lifecycle.recipe.modifiers[0]
        exact_five = quantization_batch_peak_bytes(model[0], 5, modifier.block_size)
        assert (
            predict_batch_size(
                list(model),
                modifier.batched_quantization,
                modifier.block_size,
                exact_five,
            )
            == 5
        )
        assert (
            predict_batch_size(
                list(model),
                modifier.batched_quantization,
                modifier.block_size,
                exact_five - 1,
            )
            == 4
        )


def test_gptq_auto_update_batch_uses_the_largest_shape_group():
    model = nn.Sequential(
        *[nn.Linear(64, 48, bias=False, device="meta") for _module in range(8)],
        *[nn.Linear(32, 48, bias=False, device="meta") for _module in range(8)],
    )
    with create_session() as session:
        session.state.layerwise_calibration = True
        session.initialize(
            model=model,
            recipe=[GPTQModifier(scheme="W8A8", batched_quantization="auto")],
            start=-1,
            copy_data=False,
        )
        modifier = session.lifecycle.recipe.modifiers[0]
        large = next(module for module in model if module.in_features == 64)
        available = quantization_batch_peak_bytes(large, 3, modifier.block_size)

        assert (
            predict_batch_size(
                list(model),
                modifier.batched_quantization,
                modifier.block_size,
                available,
            )
            == 3
        )


def test_gptq_ep4_auto_batches_use_live_budget(monkeypatch):
    modules = [
        nn.Linear(5120, 2304, bias=False, device="meta", dtype=torch.bfloat16)
        for _module in range(132)
    ]
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4, strategy="group", group_size=128, actorder="weight"
        ),
    )
    for module in modules:
        module.quantization_scheme = scheme

    block_size = 128
    reported_budget = int((18.39 + 12.89) * 0.75 * 2**30)
    reduced_budget = reported_budget // 2
    budgets = iter((reported_budget, reduced_budget, reduced_budget))
    monkeypatch.setattr(
        gptq_helpers, "available_batch_bytes", lambda device: next(budgets)
    )
    batches = iter_batches(modules, "auto", block_size, batch_size_cap=len(modules))

    first = next(batches)
    second = next(batches)

    assert 1 < len(second) < len(first) < len(modules)
    for batch, budget in ((first, reported_budget), (second, reduced_budget)):
        assert (
            quantization_batch_peak_bytes(batch[0], len(batch), block_size)
            <= budget
            < quantization_batch_peak_bytes(batch[0], len(batch) + 1, block_size)
        )
    assert first + second + [module for batch in batches for module in batch] == modules


@pytest.mark.parametrize(
    "batched_quantization, expected",
    [(None, 1), (7, 7)],
)
def test_gptq_update_batch_prediction_preserves_explicit_configuration(
    batched_quantization, expected
):
    modules = [nn.Linear(8, 4, bias=False, device="meta") for _module in range(40)]
    assert (
        predict_batch_size(
            modules,
            batched_quantization,
            128,
            available_bytes=1,
        )
        == expected
    )


def test_gptq_update_owner_passes_predicted_batch_cap(monkeypatch):
    model = nn.Sequential(
        *[nn.Linear(8, 4, bias=False, device="meta") for _module in range(40)]
    )
    with create_session() as session:
        session.state.layerwise_calibration = True
        session.initialize(
            model=model,
            recipe=[GPTQModifier(scheme="W8A8", batched_quantization="auto")],
            start=-1,
            copy_data=False,
        )
        modifier = session.lifecycle.recipe.modifiers[0]
        adapter = get_modifier_memory_adapter(modifier)
        captured = {}

        monkeypatch.setattr(
            gptq_adapter_module, "observe", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(
            gptq_adapter_module, "update_qparams", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(
            GPTQModifier,
            "sync_obs_act_stats",
            lambda self, modules: None,
        )
        monkeypatch.setattr(
            GPTQModifier,
            "compress_module_list",
            lambda self, modules, **kwargs: captured.update(kwargs),
        )

        exact_five = quantization_batch_peak_bytes(model[0], 5, modifier.block_size)
        adapter.update_owner(
            torch.device("cpu"),
            tuple(model),
            available_bytes=exact_five,
        )

        assert captured["batch_size_cap"] == 5
