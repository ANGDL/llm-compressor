from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from llmcompressor.modeling.moe_context import MoECalibrationModule
from llmcompressor.utils.dev import skip_weights_initialize

if TYPE_CHECKING:
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
        DeepseekV4Config,
    )
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4SparseMoeBlock,
    )


@MoECalibrationModule.register("DeepseekV4SparseMoeBlock")
class CalibrationDeepseekV4MoE(MoECalibrationModule):
    """
    Calibration version of DeepseekV4SparseMoeBlock that unpacks batched expert
    weights into individual MLP modules for quantization.

    This module:
    1. Unpacks the packed expert weights (3D -> individual nn.Linear) for calibration
    2. Optionally sends all tokens to all experts during calibration
    3. Stays in unpacked form (permanent) for vLLM compatibility
    """

    is_permanent = True

    def __init__(
        self,
        original: DeepseekV4SparseMoeBlock,
        config: DeepseekV4Config,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok

        self.experts = SequentialDeepseekV4Experts(config, original.experts)
        self.gate = original.gate
        self.shared_experts = original.shared_experts
        self.is_hash = original.is_hash
        self.calibrate_all_experts = calibrate_all_experts
        original.experts = None
        del original.experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        **_,
    ) -> torch.Tensor:
        batch, seq_len, hidden_dim = hidden_states.shape
        residual = hidden_states
        flat = hidden_states.view(-1, hidden_dim)

        if self.is_hash:
            if input_ids is None:
                raise ValueError(
                    "DeepseekV4's hash-routing layers need `input_ids`."
                )
            _, weights, indices = self.gate(hidden_states, input_ids)
        else:
            _, weights, indices = self.gate(hidden_states)

        final_hidden_states = torch.zeros_like(flat, dtype=weights.dtype)
        with torch.no_grad():
            expert_mask = F.one_hot(
                indices, num_classes=self.num_experts
            ).permute(2, 1, 0)

        for i in range(self.num_experts):
            top_k_pos, token_idx = torch.where(expert_mask[i])
            has_tokens = token_idx.numel() > 0

            if self.calibrate_all_experts:
                expert_out_all = self.experts[i](flat)
                if not has_tokens:
                    continue
                expert_out = expert_out_all[token_idx]
            else:
                if not has_tokens:
                    continue
                expert_out = self.experts[i](flat[token_idx])

            weighted_output = expert_out * weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0, token_idx, weighted_output.to(final_hidden_states.dtype)
            )

        routed = final_hidden_states.type(hidden_states.dtype).view(
            batch, seq_len, hidden_dim
        )
        return routed + self.shared_experts(residual)


class _DeepseekV4SwiGLUExpert(torch.nn.Module):
    """
    Routed expert with swiglu_limit clamping, matching DeepseekV4Experts._apply_gate.
    DeepseekV4MLP (used for the shared expert) does not clamp; the packed routed
    experts do via _apply_gate. This wrapper restores that behavior so calibration
    statistics match actual inference.
    """

    def __init__(self, mlp, swiglu_limit: float):
        super().__init__()
        # Transfer the Linear submodules directly so module paths stay as
        # experts[i].gate_proj / up_proj / down_proj (matches quantization configs).
        self.gate_proj = mlp.gate_proj
        self.up_proj = mlp.up_proj
        self.down_proj = mlp.down_proj
        self.act_fn = mlp.act_fn
        self._swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x).clamp(max=self._swiglu_limit)
        up = self.up_proj(x).clamp(min=-self._swiglu_limit, max=self._swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class SequentialDeepseekV4Experts(torch.nn.ModuleList):
    def __init__(self, config: DeepseekV4Config, original_experts):
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
            DeepseekV4MLP,
        )

        num_experts = config.n_routed_experts
        with skip_weights_initialize():
            mlps = [DeepseekV4MLP(config) for _ in range(num_experts)]

        with torch.no_grad():
            for i in range(num_experts):
                gate_up = original_experts.gate_up_proj[i]
                down = original_experts.down_proj[i]

                gate_proj, up_proj = gate_up.chunk(2, dim=0)

                mlps[i].gate_proj.weight.data = gate_proj.clone().contiguous()
                mlps[i].up_proj.weight.data = up_proj.clone().contiguous()
                mlps[i].down_proj.weight.data = down.clone().contiguous()

        super().__init__(
            [_DeepseekV4SwiGLUExpert(mlp, config.swiglu_limit) for mlp in mlps]
        )