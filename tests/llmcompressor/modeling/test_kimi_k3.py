import sys
from types import ModuleType
from unittest.mock import patch

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import generic as transformers_generic
from transformers.utils import import_utils
from transformers.utils.output_capturing import OutputRecorder

from llmcompressor.modeling.kimi_k3 import (
    CalibrationKimiK3SparseMoeBlock,
    _patch_flash_attn_varlen_func,
    _patch_transformers_v5_for_checkpoint_code,
    patch_kimi_k3_transformers_compat,
)


def test_kimi_k3_transformers_output_recorder_compat():
    with patch.object(transformers_generic, "OutputRecorder", create=True):
        del transformers_generic.OutputRecorder

        patch_kimi_k3_transformers_compat()
        patch_kimi_k3_transformers_compat()

        assert transformers_generic.OutputRecorder is OutputRecorder


def test_kimi_k3_transformers_tie_weights_compat():
    class CheckpointModel:
        def tie_weights(self):
            self.called = True

    original_tie_weights = CheckpointModel.tie_weights
    patch_kimi_k3_transformers_compat(CheckpointModel)
    patched_tie_weights = CheckpointModel.tie_weights
    patch_kimi_k3_transformers_compat(CheckpointModel)

    model = CheckpointModel()
    model.tie_weights(missing_keys={"lm_head.weight"}, recompute_mapping=False)

    assert model.called
    assert patched_tie_weights is CheckpointModel.tie_weights
    assert patched_tie_weights is not original_tie_weights


def test_kimi_k3_transformers_v5_remote_model_compat():
    calls = []

    def create_causal_mask(*, inputs_embeds, attention_mask, past_key_values):
        calls.append((inputs_embeds, attention_mask, past_key_values))
        return "mask"

    class KimiLinearModel:
        def __init__(self, config):
            config._attn_implementation = "flash_attention_2"
            self._use_flash_attention_2 = True

    linear_module = ModuleType("test_kimi_k3_linear_module")
    linear_module.create_causal_mask = create_causal_mask
    linear_module.KimiLinearModel = KimiLinearModel
    linear_lm_class = type(
        "KimiLinearForCausalLM",
        (),
        {"__module__": linear_module.__name__},
    )
    outer_module = ModuleType("test_kimi_k3_outer_module")
    outer_module.KimiLinearForCausalLM = linear_lm_class

    class CheckpointModel:
        def tie_weights(self):
            pass

    CheckpointModel.__module__ = outer_module.__name__
    with patch.dict(
        sys.modules,
        {
            linear_module.__name__: linear_module,
            outer_module.__name__: outer_module,
        },
    ):
        patch_kimi_k3_transformers_compat(CheckpointModel)
        patch_kimi_k3_transformers_compat(CheckpointModel)

        config = type("Config", (), {"_attn_implementation": "eager"})()
        model = KimiLinearModel(config)
        result = linear_module.create_causal_mask(
            input_embeds="embeds",
            attention_mask="attention",
            cache_position="position",
            past_key_values="cache",
        )

    assert result == "mask"
    assert calls == [("embeds", "attention", "cache")]
    assert config._attn_implementation == "eager"
    assert model._use_flash_attention_2 is False


def test_kimi_k3_flash_attn_compat_drops_unsupported_deterministic_keyword():
    calls = []

    def old_flash_attn(*args, **kwargs):
        calls.append((args, kwargs))
        return "output"

    module = ModuleType("flash_attn")
    module.flash_attn_varlen_func = old_flash_attn
    with patch.dict("sys.modules", {"flash_attn": module}):
        _patch_flash_attn_varlen_func()
        _patch_flash_attn_varlen_func()
        assert module.flash_attn_varlen_func(1, deterministic=True) == "output"

    assert calls == [((1,), {})]


def test_kimi_k3_transformers_v5_flash_attn_dispatch_compat():
    with (
        patch.object(import_utils, "is_torch_fx_available", create=True),
        patch.object(
            PreTrainedModel,
            "_kimi_k3_fa2_compat_patched",
            False,
            create=True,
        ),
        patch.object(PreTrainedModel, "_flash_attn_can_dispatch") as dispatch,
    ):
        del import_utils.is_torch_fx_available
        _patch_transformers_v5_for_checkpoint_code()
        _patch_transformers_v5_for_checkpoint_code()

        model = type(
            "CheckpointK3Model",
            (),
            {
                "_supports_flash_attn": False,
                "_supports_flash_attn_2": True,
            },
        )()
        PreTrainedModel._flash_attn_can_dispatch(model, 2, True)

        assert import_utils.is_torch_fx_available() is hasattr(torch, "fx")
        dispatch.assert_called_once_with(model, 2, True)
        assert model._supports_flash_attn is False


class _FixedGate(nn.Module):
    def forward(self, hidden_states):
        device = hidden_states.device
        return (
            torch.tensor([[0], [1]], device=device),
            torch.ones((2, 1), dtype=torch.float32, device=device),
        )


class _DummyKimiK3Moe(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"num_shared_experts": 1})()
        self.hidden_dim = 2
        self.num_experts = 2
        self.top_k = 1
        self.use_latent_moe = True
        self.gate = _FixedGate()
        self.experts = nn.ModuleList(
            [nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)]
        )
        self.shared_experts = nn.Linear(2, 2, bias=False)
        self.routed_expert_down_proj = nn.Linear(2, 2, bias=False)
        self.routed_expert_norm = nn.Identity()
        self.routed_expert_up_proj = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.experts[0].weight.copy_(torch.eye(2))
            self.experts[1].weight.copy_(2 * torch.eye(2))
            self.shared_experts.weight.copy_(0.5 * torch.eye(2))
            self.routed_expert_down_proj.weight.copy_(torch.eye(2))
            self.routed_expert_up_proj.weight.copy_(torch.eye(2))
        self.to(dtype=torch.bfloat16)


def _run(calibrate_all_experts: bool):
    replacement = CalibrationKimiK3SparseMoeBlock(
        _DummyKimiK3Moe(),
        config=None,
        calibrate_all_experts=calibrate_all_experts,
    )
    seen_tokens = [0, 0]
    hooks = [
        expert.register_forward_pre_hook(
            lambda _module, inputs, index=index: seen_tokens.__setitem__(
                index, inputs[0].shape[0]
            )
        )
        for index, expert in enumerate(replacement.experts)
    ]
    try:
        output = replacement(
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.bfloat16)
        )
    finally:
        for hook in hooks:
            hook.remove()
    return output, seen_tokens


def test_kimi_k3_calibration_preserves_routed_and_shared_output():
    all_expert_output, all_expert_tokens = _run(True)
    routed_output, routed_tokens = _run(False)
    expected = torch.tensor([[[1.5, 3.0], [7.5, 10.0]]], dtype=torch.bfloat16)

    assert torch.equal(all_expert_output, expected)
    assert torch.equal(routed_output, expected)
    assert all_expert_tokens == [2, 2]
    assert routed_tokens == [1, 1]
