from __future__ import annotations

import threading

import pytest
import torch

from llmcompressor.streaming.admission import DeviceMemoryBudget
from llmcompressor.streaming.checkpoint import SafetensorsWeightSource
from llmcompressor.streaming.loading import (
    HostMemoryBudget,
    SubgraphPrefetcher,
    SubgraphWeightSession,
)
from llmcompressor.streaming.loading.target import build_meta_model
from tests.llmcompressor.streaming.test_subgraph_weight_session import (
    SessionModel,
    assert_all_meta,
    checkpoint_for,
    subgraph_for,
)


def test_prefetch_materializes_without_installing_shared_model(tmp_path, monkeypatch):
    reference = SessionModel()
    checkpoint = checkpoint_for(reference, tmp_path)
    model = build_meta_model(SessionModel)
    session = SubgraphWeightSession(
        model, SafetensorsWeightSource(checkpoint)
    )
    plan = session.plan(subgraph_for("layers.0"))
    original_prepare = session.prepare
    started = threading.Event()
    release = threading.Event()

    def blocking_prepare(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(session, "prepare", blocking_prepare)
    prefetcher = SubgraphPrefetcher(session)
    required = session.estimate_preparation(plan, dtype=torch.float32)
    reservation = HostMemoryBudget(
        safety_reserve_bytes=0,
        system_available=lambda: 1 << 30,
        cgroup_available=lambda: None,
    ).reserve("test", required)
    try:
        prefetcher.submit(
            plan,
            dtype=torch.float32,
            reservation=reservation,
            label="test/weight_materialization",
        )
        assert started.wait(timeout=5)
        assert_all_meta(model)
        release.set()
        prepared = prefetcher.take()
        assert_all_meta(model)
        assert prepared.device.type == "cpu"
        prepared.move_to("cpu")
        with session.installed(prepared):
            assert not next(model.layers[0].parameters()).is_meta
    finally:
        release.set()
        prefetcher.close(suppress_errors=True)

    assert_all_meta(model)


def test_prefetch_allows_only_one_pending_result(tmp_path):
    reference = SessionModel()
    checkpoint = checkpoint_for(reference, tmp_path)
    session = SubgraphWeightSession(
        build_meta_model(SessionModel), SafetensorsWeightSource(checkpoint)
    )
    plan = session.plan(subgraph_for("layers.0"))
    required = session.estimate_preparation(plan, dtype=torch.float32)
    budget = HostMemoryBudget(
        safety_reserve_bytes=0,
        system_available=lambda: 1 << 30,
        cgroup_available=lambda: None,
    )

    with SubgraphPrefetcher(session) as prefetcher:
        prefetcher.submit(
            plan,
            dtype=torch.float32,
            reservation=budget.reserve("first", required),
            label="first",
        )
        second = budget.reserve("second", required)
        with pytest.raises(RuntimeError, match="already pending"):
            prefetcher.submit(
                plan,
                dtype=torch.float32,
                reservation=second,
                label="second",
            )
        assert second.closed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prefetch_can_enqueue_bounded_cuda_transfer(tmp_path):
    reference = SessionModel()
    checkpoint = checkpoint_for(reference, tmp_path)
    model = build_meta_model(SessionModel)
    session = SubgraphWeightSession(
        model, SafetensorsWeightSource(checkpoint)
    )
    plan = session.plan(subgraph_for("layers.0"))
    prepared_bytes = session.estimate_preparation(plan, dtype=torch.float32)
    materialized_bytes = session.estimate_materialized_bytes(
        plan, dtype=torch.float32
    )
    host_budget = HostMemoryBudget(
        safety_reserve_bytes=0,
        system_available=lambda: 1 << 30,
        cgroup_available=lambda: None,
    )
    device_budget = DeviceMemoryBudget(
        available_bytes=1 << 40, safety_reserve_fraction=0
    )
    prefetcher = SubgraphPrefetcher(session)
    try:
        prepared = host_budget.reserve("prepared", prepared_bytes)
        device_reservation = device_budget.reserve(
            "device", materialized_bytes, device="cuda"
        )
        transfer_reservation = host_budget.reserve(
            "pinned", materialized_bytes
        )
        prefetcher.submit(
            plan,
            dtype=torch.float32,
            reservation=prepared,
            label="test/cuda_transfer",
            device="cuda",
            device_reservation=device_reservation,
            transfer_reservation=transfer_reservation,
        )
        loaded = prefetcher.take()
        assert loaded.device.type == "cuda"
        with session.installed(loaded):
            assert not next(model.layers[0].parameters()).is_meta
    finally:
        prefetcher.close(suppress_errors=True)
    assert_all_meta(model)
