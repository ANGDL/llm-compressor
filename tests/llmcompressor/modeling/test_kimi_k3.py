from unittest.mock import patch

import torch
from torch import nn
from transformers.utils import generic as transformers_generic
from transformers.utils.output_capturing import OutputRecorder

from llmcompressor.modeling.kimi_k3 import (
    CalibrationKimiK3SparseMoeBlock,
    patch_kimi_k3_transformers_compat,
)


def test_kimi_k3_transformers_output_recorder_compat():
    with patch.object(transformers_generic, "OutputRecorder", create=True):
        del transformers_generic.OutputRecorder

        patch_kimi_k3_transformers_compat()
        patch_kimi_k3_transformers_compat()

        assert transformers_generic.OutputRecorder is OutputRecorder


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
