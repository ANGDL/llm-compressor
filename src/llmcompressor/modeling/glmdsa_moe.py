from typing import Any, cast

import torch

from llmcompressor.modeling.moe_context import MoECalibrationModule
from llmcompressor.utils.dev import skip_weights_initialize


@MoECalibrationModule.register("GlmMoeDsaMoE")
class CalibrationGlmMoeDsaMoE(MoECalibrationModule):
    """
    Calibration version of GlmMoeDsaMoE that can send all tokens to all experts.

    `GlmMoeDsaMoE.experts` is tensor-backed (`GlmMoeDsaNaiveMoe`) instead of an
    nn.ModuleList. We split it into per-expert MLP modules so every expert can
    run independently during calibration and collect quantization statistics.
    """

    is_permanent = True

    def __init__(
        self,
        original: torch.nn.Module,
        config,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.config = config
        expert_cls = cast(type[torch.nn.Module], type(original.shared_experts))
        self.experts = SequentialGlmMoeDsaExperts(
            config,
            original.experts,
            expert_cls=expert_cls,
        )
        self.gate: Any = original.gate
        self.shared_experts: Any = original.shared_experts
        self.calibrate_all_experts = calibrate_all_experts

        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok
        
        del original.experts  # Remove the original tensor-backed experts to save memory
        original.experts = None

    def route_tokens_to_experts(
        self, router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias

        group_scores = (
            router_logits_for_choice.view(
                -1, self.n_group, self.n_routed_experts // self.n_group
            )
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)

        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)

        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator

        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def _moe(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)

        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=self.experts.num_experts
        )
        expert_mask = expert_mask.permute(2, 1, 0)

        for expert_idx, expert_layer in enumerate(self.experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

            if self.calibrate_all_experts:
                expert_out = expert_layer(hidden_states)[token_idx]
            else:
                if token_idx.numel() == 0:
                    continue
                expert_out = expert_layer(hidden_states[token_idx])

            if token_idx.numel() > 0:
                weighted_output = expert_out * topk_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(
                    0, token_idx, weighted_output.to(final_hidden_states.dtype)
                )

        return final_hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape

        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self._moe(hidden_states, topk_indices, topk_weights)

        hidden_states = hidden_states.view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        return original


class SequentialGlmMoeDsaExperts(torch.nn.ModuleList):
    def __init__(self, config, original, expert_cls):
        self.num_experts = original.gate_up_proj.shape[0]

        with skip_weights_initialize():
            super().__init__(
                [
                    expert_cls(config, intermediate_size=config.moe_intermediate_size)
                    for _ in range(self.num_experts)
                ]
            )

        intermediate_size = original.down_proj.shape[-1]
        for expert_idx in range(self.num_experts):
            gate_up = original.gate_up_proj[expert_idx]
            down = original.down_proj[expert_idx]

            gate_proj = gate_up[:intermediate_size]
            up_proj = gate_up[intermediate_size:]

            expert_layer = cast(Any, self[expert_idx])
            expert_layer.gate_proj.weight.data = gate_proj.clone().contiguous()
            expert_layer.up_proj.weight.data = up_proj.clone().contiguous()
            expert_layer.down_proj.weight.data = down.clone().contiguous()
