from dataclasses import dataclass

import pytest
import torch

from llmcompressor.streaming import (
    DiskBoundaryActivationStore,
    InMemoryBoundaryActivationStore,
)


@dataclass
class BoundaryOutput:
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    metadata: dict[str, object]


@dataclass
class OutputWithCache:
    hidden_states: torch.Tensor
    past_key_values: object | None


def make_value(seed=0):
    generator = torch.Generator().manual_seed(seed)
    hidden_states = torch.randn(2, 3, 4, generator=generator).to(torch.bfloat16)
    return {
        "model_output": BoundaryOutput(
            hidden_states=hidden_states,
            attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
            position_ids=torch.tensor([[0, 1, 2], [0, 1, 2]]),
            metadata={"causal": True, "layer": seed},
        ),
        "auxiliary": (torch.ones(1), [None, "reference"]),
    }


def assert_values_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert torch.equal(left, right)
    elif hasattr(left, "__dataclass_fields__"):
        assert type(left) is type(right)
        for name in left.__dataclass_fields__:
            assert_values_equal(getattr(left, name), getattr(right, name))
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            assert_values_equal(left_item, right_item)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_values_equal(left[key], right[key])
    else:
        assert left == right


@pytest.fixture(params=["memory", "disk"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryBoundaryActivationStore()
    return DiskBoundaryActivationStore(tmp_path / "boundaries")


def test_memory_and_disk_round_trip_nested_values(store):
    value = make_value()

    store.put(1, 0, value)
    loaded = store.get(1, 0)

    assert_values_equal(value, loaded)
    assert loaded["model_output"].hidden_states.dtype == torch.bfloat16
    assert store.contains(1, 0)
    assert store.batch_indices(1) == (0,)


def test_put_detaches_and_snapshots_tensors(store):
    tensor = torch.arange(4.0, requires_grad=True)
    value = {"hidden_states": tensor}

    store.put(0, 0, value)
    with torch.no_grad():
        tensor.add_(10)
    loaded = store.get(0, 0)

    assert not loaded["hidden_states"].requires_grad
    assert torch.equal(loaded["hidden_states"], torch.arange(4.0))

    loaded["hidden_states"].add_(100)
    reloaded = store.get(0, 0)
    assert torch.equal(reloaded["hidden_states"], torch.arange(4.0))


def test_iterates_batches_in_stable_order(store):
    store.put(3, 2, {"hidden_states": torch.tensor([2])})
    store.put(3, 0, {"hidden_states": torch.tensor([0])})
    store.put(3, 1, {"hidden_states": torch.tensor([1])})

    observed = [
        (batch, value["hidden_states"].item())
        for batch, value in store.iter_boundary(3)
    ]

    assert observed == [(0, 0), (1, 1), (2, 2)]


def test_delete_removes_consumed_boundary_only(store):
    store.put(1, 0, make_value(1))
    store.put(2, 0, make_value(2))

    store.delete(1)

    assert not store.contains(1, 0)
    assert store.contains(2, 0)
    with pytest.raises(KeyError, match="not committed"):
        store.get(1, 0)


@pytest.mark.parametrize(
    "bad_value",
    [
        object(),
        {"unsupported": {1, 2}},
        {"unsupported_key": {torch.tensor(1): "value"}},
    ],
)
def test_rejects_unsupported_values(store, bad_value):
    with pytest.raises(TypeError, match="Unsupported"):
        store.put(0, 0, bad_value)


@pytest.mark.parametrize(
    "cache_value",
    [
        {"past_key_values": ((torch.ones(1), torch.ones(1)),)},
        {"kv_cache": object()},
    ],
)
def test_rejects_kv_cache_by_default(store, cache_value):
    with pytest.raises(TypeError, match="KV cache"):
        store.put(0, 0, cache_value)


def test_rejects_kv_cache_in_dataclass(store):
    value = OutputWithCache(torch.ones(1), past_key_values=object())

    with pytest.raises(TypeError, match="KV cache"):
        store.put(0, 0, value)


def test_disk_ignores_partial_temporary_files(tmp_path):
    store = DiskBoundaryActivationStore(tmp_path / "boundaries")
    directory = store.root / "boundary-00001"
    directory.mkdir()
    (directory / ".batch-00000.pt.deadbeef.tmp").write_bytes(b"partial")

    assert not store.contains(1, 0)
    assert store.batch_indices(1) == ()
    with pytest.raises(KeyError, match="not committed"):
        store.get(1, 0)


def test_corrupt_committed_disk_batch_is_reported(tmp_path):
    store = DiskBoundaryActivationStore(tmp_path / "boundaries")
    path = store.root / "boundary-00000" / "batch-00000.pt"
    path.parent.mkdir()
    path.write_bytes(b"corrupt")

    assert store.contains(0, 0)
    with pytest.raises(RuntimeError, match="Cannot read committed"):
        store.get(0, 0)


def test_twenty_boundaries_keep_only_adjacent_disk_data(tmp_path):
    store = DiskBoundaryActivationStore(tmp_path / "boundaries")
    peak_bytes = 0
    boundary_sizes = []

    for boundary in range(20):
        store.put(
            boundary,
            0,
            {"hidden_states": torch.ones(128, 128, dtype=torch.bfloat16)},
        )
        boundary_sizes.append(store.disk_bytes())
        peak_bytes = max(peak_bytes, store.disk_bytes())
        if boundary >= 1:
            store.delete(boundary - 1)

    single_boundary_bytes = boundary_sizes[0]
    assert peak_bytes <= single_boundary_bytes * 2 + 4096
    assert len([path for path in store.root.iterdir() if path.is_dir()]) == 1
    assert store.contains(19, 0)


def test_memory_reports_only_live_boundary_tensor_bytes():
    store = InMemoryBoundaryActivationStore()
    store.put(0, 0, {"hidden_states": torch.ones(8, dtype=torch.float32)})
    first_size = store.tensor_bytes()
    store.put(1, 0, {"hidden_states": torch.ones(8, dtype=torch.float32)})

    assert store.tensor_bytes() == first_size * 2
    store.delete(0)
    assert store.tensor_bytes() == first_size
