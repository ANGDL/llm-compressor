from unittest.mock import patch

import torch
from transformers import PretrainedConfig

from llmcompressor.pipelines.sequential.plan import trace_sequential_plan


class _SequentialTarget(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1


class _DataDependentTargetCall(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.target = _SequentialTarget()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.numel() > 0:
            value = self.target(value)
        return value


class _ModelWithHiddenSequentialTarget(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig()
        self.device = torch.device("cpu")
        self.visible_target = _SequentialTarget()
        self.data_dependent = _DataDependentTargetCall()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.visible_target(value)
        return self.data_dependent(value)


def test_trace_plan_warns_when_matched_target_is_hidden_by_autowrap():
    model = _ModelWithHiddenSequentialTarget()

    with patch("llmcompressor.pipelines.sequential.plan.logger.warning") as warning:
        plan = trace_sequential_plan(
            model,
            {"value": torch.ones(1)},
            sequential_targets=["_SequentialTarget"],
        )

    assert plan.target_names == ("visible_target",)
    messages = [call.args[0] for call in warning.call_args_list]
    message = next(
        message for message in messages if message.startswith("Sequential tracing left")
    )
    assert "1/2 matched target modules outside all subgraphs" in message
    assert "data_dependent.target" in message
    assert "pipeline='independent'" in message
