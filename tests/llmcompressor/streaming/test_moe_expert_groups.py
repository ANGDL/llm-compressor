from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from llmcompressor.core import active_session
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.streaming import (
    DiskBoundaryActivationStore,
    ExpertGroupExecutor,
    ExpertGroupModifierAdapter,
    ExpertGroupPlan,
    InMemoryBoundaryActivationStore,
    RoutedExpertBatch,
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


def test_group_executor_callback_runs_while_group_is_resident(tmp_path):
    reference = ExpertsModel()
    checkpoint = tmp_path / "model.safetensors"
    save_file(reference.state_dict(), checkpoint)
    model = build_meta_model(ExpertsModel)
    session = SubgraphWeightSession(model, SafetensorsWeightSource(checkpoint))
    plan = ExpertGroupPlan.from_num_experts("experts", 4, 2)
    executor = ExpertGroupExecutor(model, session, plan)
    seen = []
    completed = []

    def callback(group, loaded, experts, hidden_states, top_k_index, top_k_weights):
        seen.append(
            (
                group.indices,
                loaded.module_names,
                tuple(
                    not experts[index].proj.weight.is_meta
                    for index in group.indices
                ),
                hidden_states.shape,
                top_k_index.shape,
                top_k_weights.shape,
            )
        )

    def after_group(
        group,
        loaded,
        experts,
        hidden_states,
        top_k_index,
        top_k_weights,
        group_output,
    ):
        completed.append(
            (
                group.indices,
                loaded.module_names,
                group_output.shape,
                all(
                    not experts[index].proj.weight.is_meta
                    for index in group.indices
                ),
            )
        )

    executor.execute(
        torch.randn(4, 3),
        torch.tensor([[0], [1], [2], [3]]),
        torch.ones(4, 1),
        device="cpu",
        dtype=torch.float32,
        on_group=callback,
        after_group=after_group,
    )

    assert seen == [
        ((0, 1), ("experts.0", "experts.1"), (True, True), (4, 3), (4, 1), (4, 1)),
        ((2, 3), ("experts.2", "experts.3"), (True, True), (4, 3), (4, 1), (4, 1)),
    ]
    assert completed == [
        ((0, 1), ("experts.0", "experts.1"), (4, 3), True),
        ((2, 3), ("experts.2", "experts.3"), (4, 3), True),
    ]
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


class GatedExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 8, bias=False)
        self.up_proj = nn.Linear(8, 8, bias=False)
        self.down_proj = nn.Linear(8, 8, bias=False)

    def forward(self, hidden_states):
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )


class GatedExpertsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([GatedExpert() for _ in range(5)])


def _routed_inputs(store):
    for batch_index, num_tokens in enumerate((7, 3)):
        store.put(
            0,
            batch_index,
            RoutedExpertBatch(
                batch_index,
                torch.randn(num_tokens, 8),
                torch.tensor([[0, 1], [2, 0], [3, 1], [1, 2], [0, 3], [1, 0], [2, 1]])[
                    :num_tokens
                ],
                torch.softmax(torch.randn(num_tokens, 2), dim=-1),
            ),
        )


def _calibrate_groups(tmp_path, group_size, inputs, outputs, modifier, save_group):
    model = build_meta_model(GatedExpertsModel)
    session = SubgraphWeightSession(
        model, SafetensorsWeightSource(tmp_path / "experts.safetensors")
    )
    executor = ExpertGroupExecutor(
        model, session, ExpertGroupPlan.from_num_experts("experts", 5, group_size)
    )
    executor.calibrate(
        inputs=inputs,
        outputs=outputs,
        input_boundary=0,
        output_boundary=1,
        modifier=modifier,
        save_group=save_group,
        device="cpu",
        dtype=torch.float32,
        calibrate_all_experts=False,
    )
    return model


def _expert_modifier(modifier_type):
    return modifier_type(
        config_groups={
            "experts": {
                "targets": [r"re:experts\.\d+\.(gate|up|down)_proj$"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "strategy": "group",
                    "group_size": 4,
                    "symmetric": True,
                },
            }
        },
        **({"batched_quantization": None} if modifier_type is GPTQModifier else {}),
    )


@pytest.mark.parametrize("modifier_type", [GPTQModifier, QuantizationModifier])
@pytest.mark.parametrize("group_size", [1, 2])
def test_group_calibration_matches_all_resident_gated_experts(
    tmp_path, modifier_type, group_size
):
    torch.manual_seed(31)
    save_file(GatedExpertsModel().state_dict(), tmp_path / "experts.safetensors")
    inputs = DiskBoundaryActivationStore(tmp_path / "routed-inputs")
    _routed_inputs(inputs)
    modifier = ExpertGroupModifierAdapter(_expert_modifier(modifier_type))
    original_session = active_session()
    snapshots = []
    propagated = []
    for size in (5, group_size):
        saved = {}
        outputs = DiskBoundaryActivationStore(tmp_path / f"outputs-{size}")

        def save_group(group, loaded):
            assert sum(
                not next(expert.parameters()).is_meta
                for expert in loaded.model.experts
            ) == len(group.indices)
            saved.update(
                (name, tensor.detach().cpu().clone())
                for name, tensor in loaded.state_tensors()
            )

        model = _calibrate_groups(
            tmp_path, size, inputs, outputs, modifier, save_group
        )
        assert all(parameter.is_meta for parameter in model.parameters())
        assert not any(
            hasattr(module, "weight_observer") for module in model.modules()
        )
        assert active_session() is original_session
        snapshots.append(saved)
        propagated.append([outputs.get(1, index) for index in (0, 1)])

    assert snapshots[0].keys() == snapshots[1].keys()
    for name, reference in snapshots[0].items():
        torch.testing.assert_close(
            snapshots[1][name], reference, rtol=0, atol=0, msg=name
        )
    for reference, grouped in zip(propagated[0], propagated[1]):
        torch.testing.assert_close(grouped, reference, rtol=0, atol=0)


def test_group_failure_cleans_hooks_meta_and_partial_output(tmp_path):
    save_file(GatedExpertsModel().state_dict(), tmp_path / "experts.safetensors")
    inputs = InMemoryBoundaryActivationStore()
    outputs = InMemoryBoundaryActivationStore()
    _routed_inputs(inputs)
    adapter = ExpertGroupModifierAdapter(_expert_modifier(GPTQModifier))
    captured = []

    class FailingAdapter:
        @contextmanager
        def calibration(self, model):
            captured.append(model)
            with adapter.calibration(model):
                yield
                raise RuntimeError("injected calibration failure")

    with pytest.raises(RuntimeError, match="injected calibration failure"):
        _calibrate_groups(
            tmp_path, 2, inputs, outputs, FailingAdapter(), lambda *_: None
        )

    assert outputs.batch_indices(1) == ()
    assert all(parameter.is_meta for parameter in captured[0].parameters())
    assert not any(
        module._forward_hooks or module._forward_pre_hooks
        for module in captured[0].modules()
    )
    assert not any(
        hasattr(module, "weight_observer") for module in captured[0].modules()
    )
