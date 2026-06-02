import torch

from llmcompressor.modeling.deepseekv4.config import ModelConfig
from llmcompressor.modeling.deepseekv4.model import DeepseekV4MoE as OriginalDeepseekV4MoE
from llmcompressor.modeling.moe_context import MoECalibrationModule


@MoECalibrationModule.register("DeepseekV4MoE")
class CalibrationDeepseekV4MoE(MoECalibrationModule):
    is_permanent = True

    def __init__(
        self,
        original: OriginalDeepseekV4MoE,
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

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.dim)
        flat_ids = None if input_ids is None else input_ids.flatten()
        weights, indices = self.gate(hidden_states, flat_ids)
        output = torch.zeros_like(hidden_states, dtype=torch.float32)
        expert_mask = torch.nn.functional.one_hot(indices, num_classes=self.n_routed_experts)
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_index, expert in enumerate(self.experts):
            if expert is None:
                continue
            token_indices, weight_indices = torch.where(expert_mask[expert_index])
            has_tokens = token_indices.numel() > 0

            if self.calibrate_all_experts:
                expert_output = expert(hidden_states)
                if has_tokens:
                    expert_weights = weights[token_indices, weight_indices]
                    routed = expert_output[token_indices] * expert_weights.unsqueeze(-1)
                    output.index_add_(0, token_indices, routed)
            elif has_tokens:
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                expert_weights = weights[token_indices, weight_indices]
                routed = expert_output * expert_weights.unsqueeze(-1)
                output.index_add_(0, token_indices, routed)

        output += self.shared_experts(hidden_states)
        return output.to(hidden_states.dtype).view(orig_shape)