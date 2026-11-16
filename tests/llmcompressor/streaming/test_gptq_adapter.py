import pytest
import torch
from torch import nn

from llmcompressor.core import LifecycleCallbacks, create_session
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.gptq.helpers import quantization_peak_bytes
from llmcompressor.modifiers.utils.hooks import HooksMixin
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
            model=model, recipe=[GPTQModifier(scheme="W8A8")], start=-1,
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


@pytest.mark.parametrize("batch_size, count", [(None, 1), (2, 2), ("auto", 3)])
def test_gptq_resource_adapter_bounds_compatible_update_batches(batch_size, count):
    model = nn.Sequential(
        *[nn.Linear(8, 4, bias=False, device="meta") for _module in range(3)]
    )
    with create_session() as session:
        session.state.layerwise_calibration = True
        session.initialize(
            model=model,
            recipe=[GPTQModifier(scheme="W8A8", batched_quantization=batch_size)],
            start=-1, copy_data=False,
        )
        modifier = session.lifecycle.recipe.modifiers[0]
        adapter = get_modifier_memory_adapter(modifier)
        assert adapter.estimate_update_memory(tuple(model)) == (
            count * quantization_peak_bytes(model[0], modifier.block_size)
        )
        estimate = estimate_modifier_dispatch_memory(
            [modifier], model.modules(), total_tokens=100, dispatch_tokens=11,
        )
        assert estimate.held_bytes >= 3 * 8 * 8 * torch.float32.itemsize
        assert estimate.scratch_bytes >= adapter.estimate_update_memory(tuple(model))
        assert estimate.batch_bytes == (2 * 11 * 8 + 8 * 8) * 4
        assert adapter.estimate_batch_memory(model[0], 11, "propagation") == 0
