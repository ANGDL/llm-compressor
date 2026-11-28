from __future__ import annotations

import threading
from contextlib import nullcontext

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


def test_prefetch_reuses_one_transfer_stream_per_device(monkeypatch):
    created = []

    class Stream:
        def __init__(self, *, device):
            self.device = torch.device(device)
            created.append(self)

    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    prefetcher = SubgraphPrefetcher(session=None)
    try:
        first = prefetcher._transfer_stream_for("cuda:0")
        second = prefetcher._transfer_stream_for("cuda:0")
        other = prefetcher._transfer_stream_for("cuda:1")
    finally:
        prefetcher._transfer_streams.clear()
        prefetcher._executor.shutdown(wait=True)

    assert first is second
    assert first is not other
    assert len(created) == 2


def test_prefetch_creates_copy_stream_during_setup(monkeypatch):
    created = []

    class Stream:
        def __init__(self, *, device):
            self.device = torch.device(device)
            created.append(self)

    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    prefetcher = SubgraphPrefetcher(session=None, device="cuda:0")
    try:
        assert len(created) == 1
        assert prefetcher._transfer_stream_for("cuda:0") is created[0]
    finally:
        prefetcher._transfer_streams.clear()
        prefetcher._executor.shutdown(wait=True)


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
        assert len(loaded._transfer_sources) == 1
        storages = {
            value.untyped_storage().data_ptr()
            for target in loaded.targets
            for values in (target.parameter_values, target.buffer_values)
            for value in values.values()
        }
        assert len(storages) == sum(
            len(target.parameter_values) + len(target.buffer_values)
            for target in loaded.targets
        )
        with session.installed(loaded):
            assert not next(model.layers[0].parameters()).is_meta
            for name, tensor in reference.layers[0].state_dict().items():
                torch.testing.assert_close(
                    model.layers[0].state_dict()[name].cpu(), tensor
                )
    finally:
        prefetcher.close(suppress_errors=True)
    assert_all_meta(model)


@pytest.mark.parametrize("fail_pinning", [False, True])
def test_async_weight_staging_allocates_per_dtype_and_cleans_up(
    tmp_path, monkeypatch, fail_pinning
):
    def factory():
        model = SessionModel()
        model.layers[0].register_buffer("integer", torch.arange(4))
        return model

    reference = factory()
    checkpoint = checkpoint_for(reference, tmp_path)
    session = SubgraphWeightSession(
        build_meta_model(factory), SafetensorsWeightSource(checkpoint)
    )
    plan = session.plan(subgraph_for("layers.0"))
    budget = HostMemoryBudget(
        safety_reserve_bytes=0, system_available=lambda: 1 << 30,
        cgroup_available=lambda: None,
    )
    required = session.estimate_preparation(plan, dtype=torch.float32)
    prepared = session.prepare(
        plan, dtype=torch.float32, reservation=budget.reserve("prepared", required)
    )
    expected = {
        name: tensor.clone()
        for target in prepared.targets
        for values in (target.parameter_values, target.buffer_values)
        for name, tensor in values.items()
    }
    sizes = []
    waits = []
    original_empty = torch.empty

    class Event:
        def synchronize(self):
            waits.append("event-sync")

    class Stream:
        def synchronize(self):
            waits.append("stream-sync")

        def record_event(self):
            return Event()

        def wait_event(self, event):
            waits.append(("wait-event", event))

    def empty(*args, **kwargs):
        if kwargs.pop("pin_memory", False):
            sizes.append((args[0], kwargs["dtype"]))
            if fail_pinning and len(sizes) == 2:
                raise RuntimeError("pinned allocation failed")
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **_kwargs: Stream())
    execution_stream = Stream()
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda _device=None: execution_stream
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.Tensor, "to", lambda tensor, **_kwargs: tensor.clone())
    transfer = budget.reserve("pinned", prepared.nbytes)
    if fail_pinning:
        with pytest.raises(RuntimeError, match="pinned allocation"):
            prepared.move_to_async("cuda:0", transfer_reservation=transfer)
        assert waits
        assert prepared._transfer_stream is None
    else:
        prepared.move_to_async("cuda:0", transfer_reservation=transfer)
        assert {dtype for _numel, dtype in sizes} == {torch.float32, torch.int64}
        assert len(sizes) == len(prepared._transfer_sources) == 2
        assert prepared._transfer_stream is not None
        for target in prepared.targets:
            for values in (target.parameter_values, target.buffer_values):
                for name, value in values.items():
                    torch.testing.assert_close(value, expected[name])
        prepared.wait_for_transfer()
        assert any(
            isinstance(item, tuple) and item[0] == "wait-event" for item in waits
        )
        assert waits[-1] == "event-sync"
        assert prepared._transfer_stream is None
        assert transfer.closed
        prepared.close()
    assert not prepared.targets
    assert not budget._reservations
    assert_all_meta(session.model)
