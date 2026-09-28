import torch
from compressed_tensors.offload import disable_onloading
from compressed_tensors.quantization import QuantizationMetadata

from llmcompressor.modeling.moe_context import (
    MoECalibrationModule,
    moe_calibration_context,
)
from tests.e2e.e2e_utils import run_oneshot_single
from tests.testing_utils import BaseTestConfig, requires_gpu


class _TestMoE(torch.nn.Module):
    def __init__(self, *, requires_all_experts: bool):
        super().__init__()
        self.requires_all_experts_for_calibration = requires_all_experts


@MoECalibrationModule.register("_TestMoE")
class _TestCalibrationMoE(MoECalibrationModule):
    def __init__(self, original, config, calibrate_all_experts=True):
        super().__init__()
        self.calibrate_all_experts = calibrate_all_experts

    def restore(self, original):
        return original


class _TestMoEModel(torch.nn.Module):
    config = object()

    def __init__(self):
        super().__init__()
        self.optional = _TestMoE(requires_all_experts=False)
        self.required = _TestMoE(requires_all_experts=True)


def test_module_can_require_all_experts_when_global_policy_is_disabled():
    model = _TestMoEModel()

    with moe_calibration_context(model, calibrate_all_experts=False):
        assert not model.optional.calibrate_all_experts
        assert model.required.calibrate_all_experts

    assert isinstance(model.optional, _TestMoE)
    assert isinstance(model.required, _TestMoE)


@requires_gpu(1)
def test_oneshot_integration():
    """
    Tests that moe_calibration_context is called within oneshot
    """
    config = BaseTestConfig(
        cadence="commit",
        model="nm-testing/tinysmokeqwen3moe",
        scheme="NVFP4",
        dataset_id="perfectblend",
        dataset_split="train[:1]",
        num_calibration_samples=1,
        max_seq_length=1,  # not enough tokens to send to all experts w/o context
    )

    model = run_oneshot_single(**config.model_dump())
    _test_qparams(model)


def _test_qparams(model: torch.nn.Module):
    all_qparam_names = QuantizationMetadata.all_qparam_names()

    with disable_onloading():
        all_qparams = [
            (module_name + "." + qparam_name, getattr(module, qparam_name))
            for module_name, module in model.named_modules()
            for qparam_name in all_qparam_names
            if hasattr(module, qparam_name)
        ]
    assert len(all_qparams) > 0, "Model does not have any qparams to test"

    for name, qparam in all_qparams:
        assert isinstance(qparam, torch.Tensor)
        assert qparam._version >= 1, f"{name} was never updated after initialization"
