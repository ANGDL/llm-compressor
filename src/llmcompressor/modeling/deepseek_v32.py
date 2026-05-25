import torch

from llmcompressor.modeling.deepseekv32.config import ModelConfig
from llmcompressor.modeling.deepseekv32.model import DeepseekV32MoE as OriginalDeepseekV32MoE
from llmcompressor.modeling.moe_context import MoECalibrationModule


@MoECalibrationModule.register("DeepseekV32MoE")
class CalibrationDeepseekV32MoE(MoECalibrationModule):
    """
    Calibration version of DeepseekV32MoE that can route all tokens through all experts.
    """

    is_permanent = True

    def __init__(
        self,
        original: OriginalDeepseekV32MoE,
        config: ModelConfig,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.config = config
        self.dim = original.dim
        self.n_routed_experts = original.n_routed_experts
        self.gate = original.gate
        self.experts = original.experts
        self.shared_experts = original.shared_experts
        self.calibrate_all_experts = calibrate_all_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.dim)
        weights, indices = self.gate(hidden_states)
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)
        expert_mask = torch.nn.functional.one_hot(
            indices, num_classes=self.n_routed_experts
        )
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx, expert in enumerate(self.experts):
            if expert is None:
                continue

            token_indices, weight_indices = torch.where(expert_mask[expert_idx])
            has_tokens = token_indices.numel() > 0

            if self.calibrate_all_experts:
                expert_output = expert(hidden_states)
                if has_tokens:
                    expert_weights = weights[token_indices, weight_indices]
                    routed_output = expert_output[token_indices] * expert_weights.unsqueeze(-1)
                    final_hidden_states.index_add_(0, token_indices, routed_output)
            elif has_tokens:
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                expert_weights = weights[token_indices, weight_indices]
                routed_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, routed_output)

        final_hidden_states += self.shared_experts(hidden_states)
        return final_hidden_states.type_as(hidden_states).view(orig_shape)