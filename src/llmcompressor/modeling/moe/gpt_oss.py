import torch
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

from llmcompressor.modeling.moe.helpers import FusedExpertsProtocol
from llmcompressor.modeling.moe.linear_experts import (
    ExpertMLPWithGate,
    LinearExperts2D,
    _set_parameter_data,
)


class GptOssExpertMLP(ExpertMLPWithGate):
    """
    gate and up occupy the even and odd columns of ``gate_up_proj`` rather than its
    halves, so the generic split would leave each projection holding a mix of both.
    """

    def copy_from_experts_module(self, experts: FusedExpertsProtocol, index: int):
        gate_up = experts.gate_up_proj[index]
        _set_parameter_data(self.gate_proj, "weight", gate_up[:, 0::2].T)
        _set_parameter_data(self.up_proj, "weight", gate_up[:, 1::2].T)
        _set_parameter_data(self.down_proj, "weight", experts.down_proj[index].T)

        gate_up_bias = experts.gate_up_proj_bias[index]
        _set_parameter_data(self.gate_proj, "bias", gate_up_bias[0::2])
        _set_parameter_data(self.up_proj, "bias", gate_up_bias[1::2])
        _set_parameter_data(
            self.down_proj,
            "bias",
            experts.down_proj_bias[index],
        )


class GptOssLinearExperts(LinearExperts2D):
    is_concatenated = False
    is_transposed = True
    has_bias = True
    has_gate = True
    expert_cls_with_gate = GptOssExpertMLP

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        """Clamped SwiGLU of GptOssExperts, over halves rather than even and odd"""
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return (up + 1) * gate * torch.sigmoid(gate * self.alpha)


# register in registry
LinearExperts2D._registry[GptOssExperts] = GptOssLinearExperts
