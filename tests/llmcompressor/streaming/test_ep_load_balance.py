from __future__ import annotations

import threading

import pytest
import torch

from llmcompressor.streaming import (
    AsyncResidentWindowScheduler,
    ExpertCostProfile,
    OwnerMemoryProfile,
    RoutePlan,
    RoutePlanCollection,
    build_fixed_owner_placement,
    build_load_balanced_placement,
    plan_resident_windows,
)
from llmcompressor.streaming.parallel.expert import ExpertParallelPlan
from llmcompressor.streaming.parallel.load_balance import plan_staging_bytes
from llmcompressor.streaming.parallel.modifier_update import (
    OwnerModuleGroup,
    OwnerUpdateScheduler,
    partition_modules_by_owner,
)


def test_route_plan_preserves_token_rows_slots_and_weights():
    plan = RoutePlan.from_tensors(
        torch.tensor([[0, 2], [1, 0], [2, 1]]),
        torch.tensor([[0.7, 0.3], [0.2, 0.8], [0.4, 0.6]]),
        num_experts=3,
        original_shape=(3, 4),
        batch_index=4,
    )

    assert plan.batch_index == 4
    assert plan.token_counts == (2, 2, 2)
    assert plan.total_assignments == 6
    assert plan.route(0).token_indices.tolist() == [0, 1]
    assert plan.route(0).slot_indices.tolist() == [0, 1]
    torch.testing.assert_close(
        plan.route(0).weights,
        torch.tensor([0.7, 0.8]),
    )
    assert plan.to_metadata()["original_shape"] == (3, 4)


def test_route_plan_collection_aggregates_work():
    first = RoutePlan.from_tensors(
        torch.tensor([[0], [1], [0]]),
        torch.ones(3, 1),
        num_experts=2,
        batch_index=0,
    )
    second = RoutePlan.from_tensors(
        torch.tensor([[1], [1]]),
        torch.ones(2, 1),
        num_experts=2,
        batch_index=1,
    )
    collection = RoutePlanCollection((first, second))

    assert collection.token_counts == (2, 3)
    assert collection.total_tokens == 5
    assert collection.work_units({0: 2.0, 1: 3.0}) == (4.0, 9.0)


def test_load_balanced_placement_prefers_weighted_route_work():
    costs = {
        "experts.0": ExpertCostProfile(storage_bytes=10, route_tokens=100),
        "experts.1": ExpertCostProfile(storage_bytes=10, route_tokens=90),
        "experts.2": ExpertCostProfile(storage_bytes=10, route_tokens=10),
        "experts.3": ExpertCostProfile(storage_bytes=10, route_tokens=10),
    }

    placement = build_load_balanced_placement(costs, 2)

    assert placement.owner_by_name["experts.0"] != placement.owner_by_name["experts.1"]
    assert max(placement.owner_route_tokens) - min(
        placement.owner_route_tokens
    ) <= 10
    assert placement.placement.ep_size == 2


def test_load_balanced_placement_rejects_expert_without_memory_fit():
    costs = {
        "experts.0": ExpertCostProfile(
            storage_bytes=8,
            persistent_state_bytes=5,
        )
    }
    memory = (
        OwnerMemoryProfile(usable_bytes=12),
        OwnerMemoryProfile(usable_bytes=12),
    )

    with pytest.raises(MemoryError, match="experts.0"):
        build_load_balanced_placement(costs, 2, owner_memory=memory)


def test_fixed_owner_placement_balances_work_under_resident_limits():
    costs = {
        "experts.0": ExpertCostProfile(
            storage_bytes=6, persistent_state_bytes=2, route_tokens=100
        ),
        "experts.1": ExpertCostProfile(
            storage_bytes=6, persistent_state_bytes=2, route_tokens=90
        ),
        "experts.2": ExpertCostProfile(
            storage_bytes=6, persistent_state_bytes=2, route_tokens=10
        ),
        "experts.3": ExpertCostProfile(
            storage_bytes=6, persistent_state_bytes=2, route_tokens=10
        ),
    }
    memory = (
        OwnerMemoryProfile(usable_bytes=20),
        OwnerMemoryProfile(usable_bytes=20),
    )

    placement = build_fixed_owner_placement(
        costs,
        2,
        owner_memory=memory,
    )

    assert placement.policy == "fixed-owner-v1"
    assert all(
        storage + persistent <= memory[owner].usable_bytes
        for owner, (storage, persistent) in enumerate(
            zip(
                placement.owner_storage_bytes,
                placement.owner_persistent_bytes,
            )
        )
    )
    assert max(placement.owner_route_tokens) - min(
        placement.owner_route_tokens
    ) <= 10


def test_resident_windows_split_by_memory_and_report_activation_batches():
    costs = {
        "experts.0": ExpertCostProfile(storage_bytes=6, route_tokens=20),
        "experts.1": ExpertCostProfile(storage_bytes=6, route_tokens=5),
        "experts.2": ExpertCostProfile(storage_bytes=6, route_tokens=4),
        "experts.3": ExpertCostProfile(storage_bytes=6, route_tokens=2),
    }
    memory = (
        OwnerMemoryProfile(
            usable_bytes=20,
            dispatch_bytes_per_token=1,
        ),
        OwnerMemoryProfile(
            usable_bytes=20,
            dispatch_bytes_per_token=1,
        ),
    )
    placement = build_load_balanced_placement(costs, 2, owner_memory=memory)
    windows = plan_resident_windows(placement, costs, memory)

    assert {name for window in windows for name in window.expert_names} == set(
        costs
    )
    assert all(window.resident_bytes <= 20 for window in windows)
    assert max(window.activation_batches for window in windows) >= 2


def test_resident_windows_keep_capacity_for_a_token_before_loading():
    costs = {
        "experts.0": ExpertCostProfile(storage_bytes=5, route_tokens=3),
        "experts.1": ExpertCostProfile(storage_bytes=5, route_tokens=3),
    }
    placement = build_load_balanced_placement(costs, 1)
    windows = plan_resident_windows(
        placement, costs,
        (OwnerMemoryProfile(usable_bytes=10, dispatch_bytes_per_token=1),),
    )
    assert len(windows) == 2
    assert all(window.max_dispatch_tokens == 3 for window in windows)


def test_device_staging_only_reserves_memory_for_paged_owners():
    costs = {
        f"experts.{index}": ExpertCostProfile(
            storage_bytes=8, persistent_state_bytes=4, route_tokens=10
        )
        for index in range(8)
    }
    placement = build_load_balanced_placement(costs, 2)
    memory = (
        OwnerMemoryProfile(usable_bytes=74, fixed_bytes=10, dispatch_bytes_per_token=1),
        OwnerMemoryProfile(usable_bytes=50, fixed_bytes=10, dispatch_bytes_per_token=1),
    )
    windows = plan_resident_windows(placement, costs, memory)
    staging = plan_staging_bytes(windows, costs, memory, {0, 1})
    buffered = plan_resident_windows(placement, costs, memory, staging_bytes=staging)

    assert {0: 0, **staging} == {0: 0, 1: 20}
    assert [window for window in windows if window.owner_rank == 0] == [
        window for window in buffered if window.owner_rank == 0
    ]
    assert {name for window in buffered for name in window.expert_names} == set(costs)
    for window in buffered:
        owner = memory[window.owner_rank]
        stage = staging.get(window.owner_rank, 0)
        assert (
            owner.fixed_bytes + stage + window.resident_bytes
            + owner.dispatch_bytes_per_token * window.max_dispatch_tokens
            <= owner.usable_bytes
        )
        if stage:
            assert sum(
                costs[name].storage_bytes for name in window.expert_names
            ) <= stage


def test_async_window_scheduler_runs_owners_concurrently():
    costs = {
        "experts.0": ExpertCostProfile(storage_bytes=1, route_tokens=4),
        "experts.1": ExpertCostProfile(storage_bytes=1, route_tokens=4),
    }
    placement = build_load_balanced_placement(costs, 2)
    memory = (
        OwnerMemoryProfile(usable_bytes=10),
        OwnerMemoryProfile(usable_bytes=10),
    )
    windows = plan_resident_windows(placement, costs, memory)
    barrier = threading.Barrier(2)
    events: list[tuple[str, int, int]] = []
    lock = threading.Lock()

    def prefetch(window):
        with lock:
            events.append(("prefetch", window.owner_rank, window.window_index))
        return window.expert_names

    def compute(window, state):
        assert state == window.expert_names
        barrier.wait(timeout=5)
        with lock:
            events.append(("compute", window.owner_rank, window.window_index))
        return len(state)

    def update(window, _state, result):
        with lock:
            events.append(("update", window.owner_rank, result))

    def release(window, _state, result):
        with lock:
            events.append(("release", window.owner_rank, result))

    results = AsyncResidentWindowScheduler().run(
        windows,
        prefetch=prefetch,
        compute=compute,
        update=update,
        release=release,
    )

    assert [window.owner_rank for window, _result in results] == [0, 1]
    assert [kind for kind, _owner, _value in events].count("compute") == 2
    assert [kind for kind, _owner, _value in events].count("release") == 2


@pytest.mark.parametrize("next_prefetch_fails", [False, True])
def test_async_windows_release_pending_state_and_preserve_compute_failure(
    next_prefetch_fails,
):
    costs = {
        f"experts.{index}": ExpertCostProfile(storage_bytes=8, route_tokens=1)
        for index in range(3)
    }
    placement = build_load_balanced_placement(costs, 1)
    windows = plan_resident_windows(
        placement, costs, (OwnerMemoryProfile(usable_bytes=10),)
    )
    next_ready = threading.Event()
    discarded = []

    def prefetch(window):
        if window.window_index == 1:
            next_ready.set()
            if next_prefetch_fails:
                raise ValueError("prefetch failed")
        return window.window_index

    def compute(_window, _state):
        assert next_ready.wait(timeout=5)
        raise RuntimeError("compute failed")

    with pytest.raises(RuntimeError, match="compute failed"):
        AsyncResidentWindowScheduler().run(
            windows, prefetch=prefetch, compute=compute, discard=discarded.append
        )
    assert sorted(discarded) == ([0] if next_prefetch_fails else [0, 1])


def test_owner_update_scheduler_runs_disjoint_groups_concurrently():
    modules = (torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    groups = (
        OwnerModuleGroup(torch.device("cpu:0"), (modules[0],)),
        OwnerModuleGroup(torch.device("cpu:1"), (modules[1],)),
    )
    barrier = threading.Barrier(2)
    entered = []
    lock = threading.Lock()

    def update(owner, group):
        with lock:
            entered.append((owner, group))
        barrier.wait(timeout=5)
        return len(group)

    result = OwnerUpdateScheduler().run(groups, update=update)

    assert [owner for owner, _value in result] == [
        torch.device("cpu:0"),
        torch.device("cpu:1"),
    ]
    assert sorted(len(group) for _owner, group in entered) == [1, 1]


def test_partition_modules_by_owner_preserves_declared_order():
    modules = (torch.nn.Linear(2, 2), torch.nn.ReLU(), torch.nn.Linear(2, 2))
    groups = partition_modules_by_owner(
        modules,
        ("cpu:0", "cpu:1"),
        {
            modules[0]: "cpu:1",
            modules[1]: "cpu:0",
            modules[2]: "cpu:1",
        },
    )

    assert [group.owner for group in groups] == [
        torch.device("cpu:0"),
        torch.device("cpu:1"),
    ]
    assert groups[0].modules == (modules[1],)
    assert groups[1].modules == (modules[0], modules[2])


class _TinyExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2))


class _TinyMoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = torch.nn.ModuleList([_TinyExpert() for _ in range(4)])


def test_expert_parallel_plan_accepts_route_profile():
    model = _TinyMoE()
    targets = tuple(f"experts.{index}" for index in range(4))
    plan = ExpertParallelPlan.from_model(
        model,
        targets,
        ("cpu:0", "cpu:1"),
        route_tokens={
            "experts.0": 100,
            "experts.1": 90,
            "experts.2": 10,
            "experts.3": 10,
        },
    )

    assert plan.placement_policy == "activation-aware-v1"
    assert plan.route_profile_fingerprint
    assert plan.owner_by_name["experts.0"] != plan.owner_by_name["experts.1"]
