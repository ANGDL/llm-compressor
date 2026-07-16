from unittest.mock import patch

import torch
from transformers import PreTrainedModel
from transformers.utils import import_utils

from llmcompressor.modeling.kimi_k25 import (
    CalibrationKimiK25DeepseekV3MoE,
    _patch_checkpoint_tie_weights,
    _patch_transformers_v5_for_checkpoint_code,
    load_kimi_k25_model,
)


def test_loader_forces_checkpoint_implementation():
    sentinel = object()
    config = type(
        "Config",
        (),
        {"auto_map": {"AutoModelForCausalLM": "module.CheckpointModel"}},
    )()

    class CheckpointModel:
        def tie_weights(self):
            pass

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return sentinel

    with (
        patch(
            "llmcompressor.modeling.kimi_k25.AutoConfig.from_pretrained",
            return_value=config,
        ),
        patch(
            "llmcompressor.modeling.kimi_k25.get_class_from_dynamic_module",
            return_value=CheckpointModel,
        ),
        patch.object(
            CheckpointModel,
            "from_pretrained",
            return_value=sentinel,
        ) as from_pretrained,
    ):
        model = load_kimi_k25_model(
            "kimi-checkpoint",
            dtype="auto",
            trust_remote_code=True,
        )

    assert model is sentinel
    from_pretrained.assert_called_once_with(
        "kimi-checkpoint",
        config=config,
        dtype="auto",
    )


def test_checkpoint_transformers_v5_compatibility_patch():
    with (
        patch.object(
            import_utils,
            "is_torch_fx_available",
            create=True,
        ),
        patch.object(PreTrainedModel, "_kimi_k25_fa2_compat_patched", False),
        patch.object(PreTrainedModel, "_flash_attn_can_dispatch") as dispatch,
    ):
        del import_utils.is_torch_fx_available
        _patch_transformers_v5_for_checkpoint_code()

        assert import_utils.is_torch_fx_available() is hasattr(torch, "fx")
        model = type(
            "CheckpointVisionModel",
            (),
            {
                "_supports_flash_attn": False,
                "_supports_flash_attn_2": True,
            },
        )()
        PreTrainedModel._flash_attn_can_dispatch(model, 2, True)
        dispatch.assert_called_once_with(model, 2, True)
    assert model._supports_flash_attn is False


def test_checkpoint_tie_weights_accepts_transformers_v5_argument():
    class CheckpointModel:
        def tie_weights(self):
            self.called = True

    _patch_checkpoint_tie_weights(CheckpointModel)
    model = CheckpointModel()
    model.tie_weights(missing_keys={"lm_head.weight"}, recompute_mapping=False)
    assert model.called


class _Gate(torch.nn.Module):
    def forward(self, hidden_states):
        num_tokens = hidden_states.numel() // hidden_states.shape[-1]
        indices = torch.zeros((num_tokens, 1), dtype=torch.long)
        weights = torch.ones((num_tokens, 1), dtype=hidden_states.dtype)
        return indices, weights


class _Expert(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.call_count = 0

    def forward(self, hidden_states):
        self.call_count += 1
        return hidden_states * self.scale


class _CheckpointMoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = object()
        self.gate = _Gate()
        self.experts = torch.nn.ModuleList([_Expert(2), _Expert(3)])
        self.shared_experts = _Expert(1)


def test_checkpoint_moe_calibrates_all_experts_without_changing_output():
    original = _CheckpointMoE()
    wrapper = CalibrationKimiK25DeepseekV3MoE(
        original,
        config=None,
        calibrate_all_experts=True,
    )
    hidden_states = torch.randn(2, 3, 4)

    output = wrapper(hidden_states)

    assert torch.equal(output, hidden_states * 3)
    assert [expert.call_count for expert in wrapper.experts] == [1, 1]
