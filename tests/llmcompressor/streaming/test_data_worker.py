from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event, current_thread
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from torch.fx import Graph

from llmcompressor.pipelines.sequential.helpers import Subgraph
from llmcompressor.pipelines.sequential.plan import SequentialExecutionPlan
from llmcompressor.streaming import initial_boundary as initial_boundary_module
from llmcompressor.streaming.activations import (
    DiskBoundaryActivationStore,
    InMemoryBoundaryActivationStore,
)
from llmcompressor.streaming.activations.transfer import (
    packed_transfer_bytes,
    prepare_tree_transfer,
)
from llmcompressor.streaming.admission import DeviceMemoryBudget
from llmcompressor.streaming.checkpoint import SafetensorsWeightSource
from llmcompressor.streaming.data_worker import CalibrationDataWorker
from llmcompressor.streaming.execution import freeze_calibration_inputs
from llmcompressor.streaming.initial_boundary import (
    InitialBoundaryWriter,
    initial_boundary_cleanup,
)
from llmcompressor.streaming.loading import (
    HostMemoryBudget,
    HostMemoryError,
    SubgraphWeightSession,
    TargetWeightLoader,
    build_meta_model,
)
from llmcompressor.streaming.tracing import TracedBoundaryAdapter


def _budget(available=2**20):
    return HostMemoryBudget(
        safety_reserve_bytes=0,
        system_available=lambda: available,
        cgroup_available=lambda: None,
    )


def test_worker_overlaps_production_and_preserves_order():
    budget = _budget()
    second_started = Event()
    reads = []

    def produce(index):
        assert current_thread().name.startswith("streaming-data")
        reads.append(index)
        if index == 1:
            second_started.set()
        return torch.full((4,), index)

    with CalibrationDataWorker(
        3, produce, lambda _index: 64, host_budget=budget
    ) as worker:
        worker.prefetch()
        with worker.take() as first:
            worker.prefetch()
            assert second_started.wait(timeout=5)
            assert reads == [0, 1]
            assert first.value.tolist() == [0] * 4
        with worker.take() as second:
            assert second.value.tolist() == [1] * 4
        worker.prefetch()
        with worker.take() as third:
            assert third.index == 2
        with pytest.raises(StopIteration):
            worker.take()
    assert not budget._reservations


def test_worker_admits_before_producing_and_defers_extra_batch():
    budget = _budget(63)
    reads = []
    with CalibrationDataWorker(
        1,
        lambda index: reads.append(index),
        lambda _index: 64,
        host_budget=budget,
    ) as worker:
        worker.prefetch()
        assert not reads
        with pytest.raises(HostMemoryError):
            worker.take()
    assert not budget._reservations


def test_worker_can_produce_cpu_data_with_a_bounded_preprocessing_callback():
    budget = _budget()
    threads = []

    def prepare(value):
        threads.append(current_thread().name)
        return {"value": value * 2, "offset": 3}

    with CalibrationDataWorker(
        1,
        lambda _index: torch.arange(4),
        lambda _index: 64,
        host_budget=budget,
        prepare=prepare,
    ) as worker:
        worker.prefetch()
        with worker.take() as batch:
            torch.testing.assert_close(batch.value["value"], torch.arange(4) * 2)
            assert batch.value["offset"] == 3
    assert threads[0].startswith("streaming-data")
    assert not budget._reservations


def test_worker_discards_speculative_input_without_skipping_its_batch():
    budget = _budget()
    reads = []

    def produce(index):
        reads.append(index)
        return torch.ones(4) * index

    with CalibrationDataWorker(
        2, produce, lambda _index: 32, host_budget=budget
    ) as worker:
        worker.prefetch()
        with worker.take() as first:
            worker.prefetch()
            worker.discard_prefetch()
            assert reads == [0, 1]
            assert len(budget._reservations) == 1
            assert first.index == 0
        with worker.take() as second:
            assert second.index == 1
            torch.testing.assert_close(second.value, torch.ones(4))
    assert reads == [0, 1, 1]
    assert not budget._reservations


def test_worker_rejects_non_cpu_production_and_releases_its_admission():
    budget = _budget()
    with pytest.raises(TypeError, match="require CPU"):
        with CalibrationDataWorker(
            1,
            lambda _index: torch.ones(4, device="meta"),
            lambda _index: 32,
            host_budget=budget,
        ) as worker:
            worker.prefetch()
            worker.take()
    assert not budget._reservations


@pytest.mark.parametrize("consumer_failure", [False, True])
def test_worker_shutdown_releases_unconsumed_batches_and_preserves_errors(
    consumer_failure,
):
    budget = _budget()
    started = Event()

    def produce(index):
        if index == 1:
            started.set()
            raise ValueError("read failed")
        return torch.ones(4)

    with pytest.raises(
        RuntimeError if consumer_failure else ValueError,
        match="consumer failed" if consumer_failure else "read failed",
    ):
        with CalibrationDataWorker(
            2, produce, lambda _index: 32, host_budget=budget
        ) as worker:
            worker.prefetch()
            with worker.take():
                worker.prefetch()
                assert started.wait(timeout=5)
                if consumer_failure:
                    raise RuntimeError("consumer failed")
    assert not budget._reservations
    assert not worker._executor._threads or all(
        not thread.is_alive() for thread in worker._executor._threads
    )


def test_frozen_owned_reads_isolate_mutations_and_keep_metadata(tmp_path):
    value = torch.arange(12).reshape(3, 4)
    batch = {"value": value, "alias": value, "view": value.t(), "offset": 3}
    frozen = freeze_calibration_inputs([batch], tmp_path)
    expected = frozen[0]
    owned = frozen.read_owned(0)
    assert owned["value"] is owned["alias"]
    owned["value"].zero_()
    torch.testing.assert_close(frozen.read_owned(0)["value"], expected["value"])
    assert frozen.batch_bytes == (2 * value.nbytes,)
    assert frozen.batch_transfer_bytes == (packed_transfer_bytes(batch),)
    assert frozen.estimate_read_bytes(0) == 4 * value.nbytes


@dataclass
class _Box:
    tensor: torch.Tensor


def test_frozen_worker_bytes_include_dataclasses(tmp_path):
    frozen = freeze_calibration_inputs([{"box": _Box(torch.ones(8))}], tmp_path)
    assert frozen.batch_bytes == (32,)
    assert frozen.estimate_read_bytes(0) == 64


@pytest.mark.parametrize("disk", [False, True])
def test_output_writer_commits_in_order_and_isolates_reused_cpu_outputs(
    tmp_path, disk, monkeypatch
):
    budget = _budget()
    store = (
        DiskBoundaryActivationStore(tmp_path, host_budget=budget)
        if disk
        else InMemoryBoundaryActivationStore(host_budget=budget)
    )
    entered = Event()
    release = Event()
    recorded = []
    original_put = store.put

    def put(boundary, index, value, **kwargs):
        recorded.append((index, current_thread().name))
        if index == 0:
            entered.set()
            assert release.wait(timeout=5)
        return original_put(boundary, index, value, **kwargs)

    monkeypatch.setattr(store, "put", put)
    reusable = torch.arange(8)
    with InitialBoundaryWriter(
        store, host_budget=budget, device=torch.device("cpu"), enabled=True
    ) as writer:
        try:
            writer.submit(0, {"value": reusable, "alias": reusable})
            assert entered.wait(timeout=5)
            reusable.add_(10)
            writer.submit(1, {"value": reusable, "alias": reusable})
            reusable.add_(10)
            assert [index for index, _thread in recorded] == [0]
        finally:
            release.set()
    assert [index for index, _thread in recorded] == [0, 1]
    assert all(thread.startswith("streaming-boundary") for _index, thread in recorded)
    for index in range(2):
        result = store.get(0, index)
        torch.testing.assert_close(result["value"], torch.arange(8) + index * 10)
        assert result["value"] is result["alias"]
    store.delete(0)
    assert not budget._reservations


def test_output_writer_applies_backpressure_before_third_forward(monkeypatch):
    budget = _budget()
    store = InMemoryBoundaryActivationStore(host_budget=budget)
    entered = Event()
    release = Event()
    ready = Event()
    original_put = store.put

    def put(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return original_put(*args, **kwargs)

    monkeypatch.setattr(store, "put", put)
    with (
        InitialBoundaryWriter(
            store, host_budget=budget, device=torch.device("cpu"), enabled=True
        ) as writer,
        ThreadPoolExecutor(max_workers=1) as caller,
    ):
        try:
            writer.submit(0, torch.ones(4))
            assert entered.wait(timeout=5)
            writer.submit(1, torch.ones(4))

            def request_slot():
                ready.set()
                writer.wait_for_slot()

            pending = caller.submit(request_slot)
            assert ready.wait(timeout=5)
            assert not pending.done()
        finally:
            release.set()
        pending.result(timeout=5)
    store.delete(0)
    assert not budget._reservations


def test_output_worker_failure_releases_reservations_and_preserves_main_error(
    monkeypatch,
):
    budget = _budget()
    store = InMemoryBoundaryActivationStore(host_budget=budget)
    entered = Event()

    def put(*args, **kwargs):
        entered.set()
        raise ValueError("commit failed")

    monkeypatch.setattr(store, "put", put)
    with pytest.raises(RuntimeError, match="main failed"):
        with InitialBoundaryWriter(
            store, host_budget=budget, device=torch.device("cpu"), enabled=True
        ) as writer:
            writer.submit(0, torch.ones(4))
            assert entered.wait(timeout=5)
            raise RuntimeError("main failed")
    assert not budget._reservations
    assert all(not thread.is_alive() for thread in writer._executor._threads)


def test_output_commit_error_reaches_consumer_and_prevents_later_commits(monkeypatch):
    budget = _budget()
    store = InMemoryBoundaryActivationStore(host_budget=budget)
    entered = Event()
    release = Event()
    committed = []

    def put(_boundary, index, _value, **_kwargs):
        committed.append(index)
        entered.set()
        assert release.wait(timeout=5)
        raise ValueError("first commit failed")

    monkeypatch.setattr(store, "put", put)
    with pytest.raises(ValueError, match="first commit failed"):
        with InitialBoundaryWriter(
            store, host_budget=budget, device=torch.device("cpu"), enabled=True
        ) as writer:
            try:
                writer.submit(0, torch.ones(4))
                assert entered.wait(timeout=5)
                writer.submit(1, torch.ones(4))
            finally:
                release.set()
    assert committed == [0]
    assert not budget._reservations


def test_output_writer_falls_back_when_extra_snapshot_does_not_fit():
    budget = _budget(16)
    store = InMemoryBoundaryActivationStore(host_budget=budget)
    with InitialBoundaryWriter(
        store, host_budget=budget, device=torch.device("cpu"), enabled=True
    ) as writer:
        writer.submit(0, torch.ones(4))
        assert writer._fallbacks == 1
    torch.testing.assert_close(store.get(0, 0), torch.ones(4))
    store.delete(0)
    assert not budget._reservations


@pytest.mark.parametrize("disk", [False, True])
def test_failed_collection_releases_already_committed_initial_boundary(tmp_path, disk):
    budget = _budget()
    store = (
        DiskBoundaryActivationStore(tmp_path, host_budget=budget)
        if disk
        else InMemoryBoundaryActivationStore(host_budget=budget)
    )
    with pytest.raises(ValueError, match="collection failed"):
        with initial_boundary_cleanup(store):
            store.put(0, 0, torch.ones(4))
            raise ValueError("collection failed")
    assert store.batch_indices(0) == ()
    assert not budget._reservations


def test_output_worker_waits_for_copy_and_orders_the_next_compute(monkeypatch):
    budget = _budget()
    store = InMemoryBoundaryActivationStore(host_budget=budget)
    compute_ordered = Event()
    copy_ready = Event()

    class Transfer:
        event = object()

        def __init__(self, value):
            self.value = value

        def wait(self):
            assert compute_ordered.is_set()
            copy_ready.set()
            return self.value

        def close(self):
            self.value = None

    def transfer(value, **kwargs):
        assert kwargs["wait_stream"] is compute
        return Transfer(value)

    compute = SimpleNamespace(wait_event=lambda _event: compute_ordered.set())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: compute)
    monkeypatch.setattr(initial_boundary_module, "transfer_tree_to_cpu", transfer)
    with InitialBoundaryWriter(
        store, host_budget=budget, device=torch.device("cpu"), enabled=True
    ) as writer:
        writer._stream = object()
        writer.submit(0, torch.arange(4))
    assert copy_ready.is_set()
    torch.testing.assert_close(store.get(0, 0), torch.arange(4))
    store.delete(0)
    assert not budget._reservations


def test_output_admission_discards_prefetch_before_serial_retry():
    budget = _budget(64)
    store = InMemoryBoundaryActivationStore(host_budget=budget)
    speculative = budget.reserve("speculative input", 64)
    with InitialBoundaryWriter(
        store,
        host_budget=budget,
        device=torch.device("cpu"),
        enabled=True,
        release_input_prefetch=speculative.close,
    ) as writer:
        writer.submit(0, torch.ones(8))
        assert speculative.closed
        assert writer._fallbacks == 1
    torch.testing.assert_close(store.get(0, 0), torch.ones(8))
    store.delete(0)
    assert not budget._reservations


def _adapter(tmp_path, forward, device="cpu"):
    reference = nn.Sequential(nn.Linear(4, 4))
    checkpoint = tmp_path / "model.safetensors"
    save_file(reference.state_dict(), checkpoint)
    model = build_meta_model(lambda: nn.Sequential(nn.Linear(4, 4)))
    graph = Graph()
    value = graph.placeholder("value")
    output = graph.call_function(forward, (value,))
    graph.output({"hidden": output, "alias": output})
    prefix = Subgraph(graph, {"value"}, set())
    graph = Graph()
    hidden = graph.placeholder("hidden")
    graph.output({"result": graph.call_module("0", (hidden,))})
    target = Subgraph(graph, {"hidden"}, {"hidden"})
    source = SafetensorsWeightSource(checkpoint)
    return TracedBoundaryAdapter(
        model,
        SequentialExecutionPlan((prefix, target), ("0",), (1,)),
        TargetWeightLoader(model, source),
        SubgraphWeightSession(model, source),
        torch.device(device),
        torch.float32,
        (target,),
        _prefix_runtime_modules=(),
    )


def test_initial_reader_runs_next_batch_during_current_prefix(tmp_path, monkeypatch):
    second_read = Event()
    current_forward = Event()
    batch_threads = []
    batches = [{"value": torch.full((2, 4), float(index))} for index in range(3)]
    frozen = freeze_calibration_inputs(batches, tmp_path / "inputs")
    original_read = frozen.read_owned

    def read(index):
        batch_threads.append(current_thread().name)
        if index == 1:
            assert current_forward.wait(timeout=5)
            second_read.set()
        return original_read(index)

    monkeypatch.setattr(frozen, "read_owned", read)

    def forward(value):
        assert current_thread().name == "MainThread"
        if value[0, 0].item() == 0:
            current_forward.set()
            assert second_read.wait(timeout=5)
        return value + 1

    adapter = _adapter(tmp_path, forward)
    actual = list(adapter.calibration_boundaries(frozen))
    assert len(batch_threads) == 3
    assert all(thread.startswith("streaming-data") for thread in batch_threads)
    for batch, boundary in zip(batches, actual):
        torch.testing.assert_close(boundary["hidden"], batch["value"] + 1)
        assert boundary["hidden"] is boundary["alias"]


def test_initial_reader_corruption_propagates_and_releases_prefix_state(tmp_path):
    batches = [{"value": torch.randn(2, 4)} for _index in range(3)]
    frozen = freeze_calibration_inputs(batches, tmp_path / "inputs")
    frozen.store._batch_path(0, 1).write_bytes(b"corrupt")
    adapter = _adapter(tmp_path, torch.sin)
    budget = _budget()
    adapter._initial_host_budget = budget
    with pytest.raises(RuntimeError, match="Cannot read committed"):
        list(adapter.calibration_boundaries(frozen))
    assert not budget._reservations
    assert all(parameter.is_meta for parameter in adapter.model.parameters())


def test_noncontiguous_prefix_inputs_keep_their_original_strides(tmp_path):
    batch = {"value": torch.randn(4, 4).t()}
    frozen = freeze_calibration_inputs([batch], tmp_path / "inputs")
    adapter = _adapter(tmp_path, torch.sin)
    adapter.device = torch.device("cuda:0")
    with adapter._prepared_calibration_batches(frozen) as inputs:
        value = next(inputs)
        assert isinstance(value, dict)
        assert value["value"].stride() == batch["value"].stride()
        torch.testing.assert_close(value["value"], batch["value"])
    assert adapter._initial_data_worker is None


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_initial_workers_preserve_numerics_and_cleanup_on_early_exit(
    tmp_path, monkeypatch, enabled
):
    monkeypatch.setenv("LLMCOMPRESSOR_STREAMING_ASYNC_INITIAL_BOUNDARY", enabled)
    batches = [{"value": torch.randn(2, 4)} for _index in range(3)]
    frozen = freeze_calibration_inputs(batches, tmp_path / "inputs")
    adapter = _adapter(tmp_path, torch.sin)
    budget = _budget()
    adapter._initial_host_budget = budget
    iterator = adapter.calibration_boundaries(frozen)
    try:
        first = next(iterator)
        torch.testing.assert_close(first["hidden"], batches[0]["value"].sin())
    finally:
        iterator.close()
    assert not budget._reservations
    assert all(parameter.is_meta for parameter in adapter.model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_initial_async_transfers_preserve_aliases_and_reused_gpu_buffers(
    tmp_path, monkeypatch
):
    batches = [{"value": torch.full((2, 4), float(index))} for index in range(3)]
    frozen = freeze_calibration_inputs(batches, tmp_path / "inputs")
    reusable = torch.empty((2, 4), device="cuda:0")

    def forward(value):
        reusable.copy_(value + 2)
        return reusable

    adapter = _adapter(tmp_path, forward, "cuda:0")
    budget = _budget()
    adapter._initial_host_budget = budget
    store = InMemoryBoundaryActivationStore(host_budget=budget)
    with InitialBoundaryWriter(
        store, host_budget=budget, device=adapter.device, enabled=True
    ) as writer:
        iterator = adapter.calibration_boundaries(frozen)
        try:
            for index in range(len(batches)):
                writer.wait_for_slot()
                writer.submit(index, next(iterator))
        finally:
            iterator.close()
    for index, batch in enumerate(batches):
        actual = store.get(0, index)
        torch.testing.assert_close(actual["hidden"], batch["value"] + 2)
        assert actual["hidden"] is actual["alias"]
    store.delete(0)
    assert not budget._reservations


def test_prepared_transfer_preserves_nested_aliases_on_cpu():
    value = torch.randn(4, 4).t()
    batch = {"value": value, "alias": value, "empty": torch.empty(0)}
    prepared = prepare_tree_transfer(batch)
    with prepared.transfer(
        torch.device("cpu"), stream=None, device_budget=DeviceMemoryBudget()
    ) as transfer:
        actual = transfer.wait()
        torch.testing.assert_close(actual["value"], value)
        assert actual["value"] is actual["alias"]
        assert actual["empty"].shape == (0,)
