import pytest
import torch

from llmcompressor.modeling.moe.routing import (
    group_expert_routes,
    select_expert_routes,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tokens,topk,experts", [(0, 2, 5), (7, 3, 5), (13, 1, 8)])
def test_grouped_routes_match_where_order_and_weights(dtype, tokens, topk, experts):
    generator = torch.Generator().manual_seed(43)
    indices = torch.randint(experts - 1, (tokens, topk), generator=generator)
    weights = torch.randn(tokens, topk, generator=generator).to(dtype)
    grouped = group_expert_routes(weights, indices, experts)
    assert all(isinstance(offset, int) for offset in grouped[2])
    for expert_id in range(experts):
        expected_rows, slots = torch.where(indices == expert_id)
        rows, selected_weights = select_expert_routes(grouped, expert_id)
        assert torch.equal(rows, expected_rows)
        assert torch.equal(selected_weights, weights[expected_rows, slots, None])


def test_grouped_routes_round_trip_boundary_without_device_offsets():
    from llmcompressor.streaming.activations import InMemoryBoundaryActivationStore

    indices = torch.tensor([[2, 1], [1, 0], [0, 2]])
    weights = torch.randn(3, 2)
    grouped = group_expert_routes(weights, indices, 4)
    store = InMemoryBoundaryActivationStore()
    store.put(0, 0, {"routes": grouped})
    restored = store.get(0, 0)["routes"]
    assert restored[2] == grouped[2]
    assert all(isinstance(offset, int) for offset in restored[2])
    for expert_id in range(4):
        for actual, expected in zip(
            select_expert_routes(restored, expert_id),
            select_expert_routes(grouped, expert_id),
        ):
            assert torch.equal(actual, expected)


def test_grouped_routes_support_meta_tracing():
    grouped = group_expert_routes(
        torch.empty(7, 2, device="meta", dtype=torch.bfloat16),
        torch.empty(7, 2, device="meta", dtype=torch.long),
        5,
    )
    assert grouped[0].is_meta and grouped[0].shape == (14,)
    assert grouped[1].is_meta and grouped[1].shape == (14, 1)
    assert grouped[2] == (0,) * 6
    rows, weights = select_expert_routes(grouped, 3)
    assert rows.shape == (0,) and weights.shape == (0, 1)


@pytest.mark.parametrize("experts,shape", [(0, (3, 2)), (4, (3,))])
def test_grouped_routes_reject_invalid_layout(experts, shape):
    with pytest.raises(ValueError):
        group_expert_routes(torch.ones(shape), torch.zeros(shape).long(), experts)
