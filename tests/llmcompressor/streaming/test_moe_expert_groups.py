from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from llmcompressor.streaming import (
    ExpertGroupExecutor,
    ExpertGroupPlan,
    SafetensorsWeightSource,
    SubgraphWeightSession,
    build_meta_model,
    execute_expert_group,
)


class RecordingExpert(nn.Module):
    def __init__(self, hidden_size: int, offset: float):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj.weight.data.copy_(
            torch.eye(hidden_size) * (offset + 1.0)
        )
        self.input_shapes: list[tuple[int, ...]] = []

    def forward(self, hidden_states):
        self.input_shapes.append(tuple(hidden_states.shape))
        return self.proj(hidden_states)


class ExpertsModel(nn.Module):
    def __init__(self, num_experts=4, hidden_size=3):
        super().__init__()
        self.experts = nn.ModuleList(
            [RecordingExpert(hidden_size, float(index)) for index in range(num_experts)]
        )


def _full_forward(experts, hidden_states, top_k_index, top_k_weights):
    output = torch.zeros_like(hidden_states)
    for expert_index, expert in enumerate(experts):
        mask = top_k_index == expert_index
        top_k_pos, token_indices = torch.where(mask.transpose(0, 1))
        expert_output = expert(hidden_states[token_indices])
        weighted_output = expert_output * top_k_weights[
            token_indices, top_k_pos, None
        ]
        output.index_add_(0, token_indices, weighted_output)
    return output


def test_plan_partitions_experts_in_stable_order():
    plan = ExpertGroupPlan.from_num_experts("experts", 5, 2)

    assert [group.indices for group in plan.groups] == [(0, 1), (2, 3), (4,)]
    assert [group.module_names for group in plan.groups] == [
        ("experts.0", "experts.1"),
        ("experts.2", "experts.3"),
        ("experts.4",),
    ]


@pytest.mark.parametrize("group_size", [1, 2, 4])
def test_group_execution_matches_full_expert_execution(group_size):
    torch.manual_seed(7)
    reference = ExpertsModel()
    grouped = ExpertsModel()
    grouped.load_state_dict(reference.state_dict())
    hidden_states = torch.randn(6, 3)
    top_k_index = torch.tensor(
        [[0, 1], [2, 0], [3, 1], [2, 3], [1, 0], [3, 2]],
        dtype=torch.long,
    )
    top_k_weights = torch.tensor(
        [[0.6, 0.4], [0.2, 0.8], [0.7, 0.3], [0.5, 0.5], [0.1, 0.9], [0.8, 0.2]]
    )
    plan = ExpertGroupPlan.from_module("experts", grouped.experts, group_size)
    output = torch.zeros_like(hidden_states)
    for group in plan.groups:
        execute_expert_group(
            grouped.experts,
            hidden_states,
            top_k_index,
            top_k_weights,
            group,
            output=output,
        )
    expected = _full_forward(
        reference.experts, hidden_states, top_k_index, top_k_weights
    )

    torch.testing.assert_close(output, expected)


def test_calibrate_all_experts_runs_empty_routes_on_full_inputs():
    model = ExpertsModel(num_experts=2)
    hidden_states = torch.randn(4, 3)
    top_k_index = torch.tensor([[0], [0], [0], [0]])
    top_k_weights = torch.ones(4, 1)
    plan = ExpertGroupPlan.from_num_experts("experts", 2, 1)

    for group in plan.groups:
        execute_expert_group(
            model.experts,
            hidden_states,
            top_k_index,
            top_k_weights,
            group,
            calibrate_all_experts=True,
        )

    assert model.experts[0].input_shapes == [(4, 3)]
    assert model.experts[1].input_shapes == [(4, 3)]


def test_group_executor_loads_only_one_group_and_restores_meta(tmp_path):
    torch.manual_seed(11)
    reference = ExpertsModel()
    checkpoint = tmp_path / "model.safetensors"
    save_file(reference.state_dict(), checkpoint)
    model = build_meta_model(ExpertsModel)
    session = SubgraphWeightSession(model, SafetensorsWeightSource(checkpoint))
    plan = ExpertGroupPlan.from_num_experts("experts", 4, 2)
    executor = ExpertGroupExecutor(model, session, plan)

    hidden_states = torch.randn(4, 3)
    top_k_index = torch.tensor([[0], [1], [2], [3]])
    top_k_weights = torch.ones(4, 1)

    with executor.loaded_group(plan.groups[0], device="cpu", dtype=torch.float32):
        assert not model.experts[0].proj.weight.is_meta
        assert not model.experts[1].proj.weight.is_meta
        assert model.experts[2].proj.weight.is_meta
        partial = torch.zeros_like(hidden_states)
        execute_expert_group(
            model.experts,
            hidden_states,
            top_k_index,
            top_k_weights,
            plan.groups[0],
            output=partial,
        )
        assert torch.count_nonzero(partial[2:]) == 0

    assert all(parameter.is_meta for parameter in model.parameters())

    output = executor.execute(
        hidden_states,
        top_k_index,
        top_k_weights,
        device="cpu",
        dtype=torch.float32,
    )
    expected = _full_forward(
        reference.experts, hidden_states, top_k_index, top_k_weights
    )
    torch.testing.assert_close(output, expected)
    assert all(parameter.is_meta for parameter in model.parameters())


def test_routing_rejects_unowned_expert_indices():
    model = ExpertsModel(num_experts=2)
    group = ExpertGroupPlan.from_num_experts("experts", 2, 1).groups[0]

    with pytest.raises(ValueError, match="outside the routed expert range"):
        execute_expert_group(
            model.experts,
            torch.randn(1, 3),
            torch.tensor([[2]]),
            torch.ones(1, 1),
            group,
        )
