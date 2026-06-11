"""
Calibration adapter for Step-3.5-Flash MoE blocks.

The original ``Step3p5MoEMLP`` (loaded via ``trust_remote_code``) stores all
expert weights in three packed ``MoELinear`` modules whose ``weight`` is a
single 3-D ``nn.Parameter`` of shape ``[num_experts, out_features, in_features]``.
Because llmcompressor targets ``nn.Linear`` modules, these packed parameters
are invisible to quantization and would silently remain in BF16.

This module mirrors :mod:`llmcompressor.modeling.glm_moe_dsa` /
:mod:`llmcompressor.modeling.qwen3_5_moe`: it registers a
:class:`MoECalibrationModule` that **unpacks** the packed expert weights into
individual ``Step3p5ExpertMLP`` modules with three ``nn.Linear`` projections.
After replacement the unpacked form is permanent (``is_permanent = True``) so
the saved checkpoint is compatible with downstream runtimes.

Routing delegates to the original ``Step3p5MoEMLP.custom_routing_function``
whenever available, and only falls back to the original softmax + top-k branch
when no custom router is configured.

``need_fp32_gate`` and ``moe_router_scaling_factor`` are honoured.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmcompressor.modeling.moe_context import MoECalibrationModule
from llmcompressor.utils.dev import skip_weights_initialize

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    pass


class Step3p5ExpertMLP(nn.Module):
    """Per-expert MLP using ``nn.Linear`` so quantization can target each Linear.

    Mirrors the math in the original ``Step3p5MoEMLP.get_expert_output``:

        up   = up_proj(x)
        gate = silu(gate_proj(x))
        if limit is not None:
            gate = gate.clamp(max=limit)
            up   = up.clamp(min=-limit, max=limit)
        return down_proj(gate * up)
    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        act_fn,
        swiglu_limit: float | None = None,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(moe_intermediate_size, hidden_size, bias=False)
        self.act_fn = act_fn
        self.limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.up_proj(x)
        gate = self.act_fn(self.gate_proj(x))
        if self.limit is not None:
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
        return self.down_proj(gate * up)


class SequentialStep3p5Experts(nn.ModuleList):
    """Unpack the 3-D ``MoELinear`` weights into individual ``Step3p5ExpertMLP``s.

    Original layout (per ``MoELinear``)::

        original.gate_proj.weight  : [num_experts, moe_intermediate_size, hidden_size]
        original.up_proj.weight    : [num_experts, moe_intermediate_size, hidden_size]
        original.down_proj.weight  : [num_experts, hidden_size, moe_intermediate_size]

    Each slice along ``dim=0`` already matches the ``[out_features, in_features]``
    layout expected by ``nn.Linear.weight``, so we copy without transposing.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        moe_intermediate_size: int,
        act_fn,
        swiglu_limit: float | None,
        original,
    ):
        with skip_weights_initialize():
            super().__init__(
                [
                    Step3p5ExpertMLP(
                        hidden_size=hidden_size,
                        moe_intermediate_size=moe_intermediate_size,
                        act_fn=act_fn,
                        swiglu_limit=swiglu_limit,
                    )
                    for _ in range(num_experts)
                ]
            )

        gate_w = original.gate_proj.weight.data  # [E, I, H]
        up_w = original.up_proj.weight.data      # [E, I, H]
        down_w = original.down_proj.weight.data  # [E, H, I]

        for i in range(num_experts):
            self[i].gate_proj.weight.data = gate_w[i].clone().contiguous()
            self[i].up_proj.weight.data = up_w[i].clone().contiguous()
            self[i].down_proj.weight.data = down_w[i].clone().contiguous()


@MoECalibrationModule.register("Step3p5MoEMLP")
class CalibrationStep3p5MoEMLP(MoECalibrationModule):
    """Calibration version of ``Step3p5MoEMLP`` with experts unpacked into ``nn.Linear``.

    Behaviour matches the original module bit-for-bit when
    ``calibrate_all_experts=False``.  When ``calibrate_all_experts=True``
    (the default), every expert is run on the full token batch so its
    Hessian / activation statistics see all calibration data, but the routed
    output is built from the same routing weights as the original.

    The replacement is permanent (the unpacked layout is what gets saved),
    matching the choice made for GLM/Qwen MoE adapters.
    """

    is_permanent = True

    def __init__(
        self,
        original: nn.Module,
        config,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size

        original_custom_routing_function = getattr(
            original, "custom_routing_function", None
        )
        self.need_fp32_gate = getattr(config, "need_fp32_gate", False)
        self.routed_scaling_factor = getattr(config, "moe_router_scaling_factor", 1.0)

        # Carry over the router Linear (typically excluded from quantization
        # via the standard `re:.*moe.gate$` ignore pattern).
        self.gate = original.gate

        # Preserve optional router bias and rebind bound custom routers to this
        # module so device placement follows the calibration replacement.
        if hasattr(original, "router_bias"):
            self.router_bias = original.router_bias
        if getattr(original_custom_routing_function, "__self__", None) is original:
            self.custom_routing_function = original_custom_routing_function.__func__.__get__(
                self, type(self)
            )
        else:
            self.custom_routing_function = original_custom_routing_function

        # Unpack the packed 3-D experts into individual ``nn.Linear`` modules.
        swiglu_limit: float | None = original.limit  # type: ignore[assignment]
        self.experts = SequentialStep3p5Experts(
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.moe_intermediate_size,
            act_fn=original.act_fn,
            swiglu_limit=swiglu_limit,
            original=original,
        )

        self.calibrate_all_experts = calibrate_all_experts

        # Free the original packed parameters now that they have been copied.
        original.up_proj = None
        original.gate_proj = None
        original.down_proj = None

    def _route(self, hidden_states: torch.Tensor):
        if self.need_fp32_gate:
            router_logits = torch.matmul(
                hidden_states.to(torch.float32),
                self.gate.weight.t().to(torch.float32),
            )
        else:
            router_logits = self.gate(hidden_states)

        if self.custom_routing_function:
            routing_weights, indices = self.custom_routing_function(
                router_logits, self.top_k, renormalize=True
            )
        else:
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, indices = torch.topk(
                routing_weights, self.top_k, dim=-1
            )

        routing_weights = routing_weights * self.routed_scaling_factor
        return routing_weights, indices

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        routing_weights, selected_experts = self._route(hidden_states)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # (num_experts, top_k, num_tokens)
        expert_mask = F.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        for expert_idx, expert_layer in enumerate(self.experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            has_tokens = top_x.numel() > 0

            if self.calibrate_all_experts:
                expert_out_all = expert_layer(hidden_states)
                if not has_tokens:
                    continue
                expert_out = expert_out_all[top_x]
            else:
                if not has_tokens:
                    continue
                expert_out = expert_layer(hidden_states[top_x])

            current_hidden_states = expert_out * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )

        return final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
