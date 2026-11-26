from dataclasses import dataclass

import pytest
import torch

from llmcompressor.streaming import (
    DiskBoundaryActivationStore,
    HostMemoryBudget,
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


def test_round_trip_torch_dtype_metadata(store):
    value = {"dtype": torch.bfloat16, "hidden_states": torch.ones(2)}

    store.put(0, 0, value)
    loaded = store.get(0, 0)

    assert loaded["dtype"] is torch.bfloat16
    assert torch.equal(loaded["hidden_states"], value["hidden_states"])


def test_meta_read_preserves_metadata_and_aliases_without_cloning(store, monkeypatch):
    tensor = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    value = {
        "hidden_states": tensor,
        "alias": tensor,
        "nested": BoundaryOutput(
            hidden_states=tensor,
            attention_mask=torch.ones(2, 3, dtype=torch.bool),
            position_ids=torch.arange(3).expand(2, 3),
            metadata={"dtype": torch.bfloat16},
        ),
    }
    store.put(0, 0, value)

    def unexpected_clone(*_args, **_kwargs):
        raise AssertionError("metadata reads must not clone activation payloads")

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "clone", unexpected_clone)
        metadata = store.get(0, 0, device="meta")
    assert metadata["hidden_states"].is_meta
    assert metadata["hidden_states"] is metadata["alias"]
    assert metadata["hidden_states"] is metadata["nested"].hidden_states
    assert metadata["hidden_states"].shape == tensor.shape
    assert metadata["hidden_states"].dtype == tensor.dtype
    assert metadata["hidden_states"].nbytes == tensor.nbytes
    assert metadata["nested"].attention_mask.is_meta
    assert metadata["nested"].attention_mask.dtype == torch.bool
    assert metadata["nested"].metadata["dtype"] is torch.bfloat16
    assert_values_equal(store.get(0, 0), value)


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


def test_execution_metadata_charges_selected_aliases_once_and_excludes_carry():
    store = InMemoryBoundaryActivationStore()
    hidden = torch.ones(3, 4)
    store.put(0, 0, {
        "hidden": hidden,
        "alias": hidden,
        "carry": torch.ones(1000, 4),
    })
    metadata = store.execution_metadata(0, 0, {"hidden"})
    assert set(metadata.values) == {"hidden", "alias"}
    assert metadata.values["hidden"] is metadata.values["alias"]
    assert metadata.tensor_bytes == hidden.nbytes
    executed = store.get_for_execution(0, 0, {"hidden"})
    assert set(executed) == set(metadata.values)


def test_execution_alias_cache_is_invalidated_when_boundary_batch_changes():
    store = InMemoryBoundaryActivationStore()
    hidden = torch.ones(3, 4)
    store.put(0, 0, {"hidden": hidden, "alias": hidden})
    assert set(store.execution_metadata(0, 0, {"hidden"}).values) == {
        "hidden", "alias"
    }
    store.put(0, 0, {"hidden": hidden, "alias": hidden.clone()})
    assert set(store.execution_metadata(0, 0, {"hidden"}).values) == {"hidden"}


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


def test_commit_propagated_batch_consumes_only_matching_input(store):
    store.put(0, 0, {"hidden_states": torch.tensor([0])})
    store.put(0, 1, {"hidden_states": torch.tensor([1])})

    store.commit_propagated_batch(
        0, 1, 0, {"hidden_states": torch.tensor([10])}
    )

    assert not store.contains(0, 0)
    assert store.contains(0, 1)
    assert store.contains(1, 0)


def test_failed_propagated_snapshot_retains_input_batch(store, monkeypatch):
    store.put(0, 0, {"hidden_states": torch.tensor([0])})

    def fail_put(*_args, **_kwargs):
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(store, "put", fail_put)
    with pytest.raises(RuntimeError, match="snapshot failed"):
        store.commit_propagated_batch(
            0, 1, 0, {"hidden_states": torch.tensor([10])}
        )

    assert store.contains(0, 0)


def test_memory_store_rejects_complete_non_cpu_boundaries():
    with pytest.raises(ValueError, match="stored on CPU"):
        InMemoryBoundaryActivationStore("meta")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cross_device_boundary_read_does_not_clone_after_copy(monkeypatch):
    store = InMemoryBoundaryActivationStore()
    store.put(0, 0, {"hidden_states": torch.ones(4)})
    original_clone = torch.Tensor.clone
    clone_calls = 0

    def recording_clone(self, *args, **kwargs):
        nonlocal clone_calls
        clone_calls += 1
        return original_clone(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", recording_clone)
    loaded = store.get(0, 0, device="cuda")

    assert loaded["hidden_states"].device.type == "cuda"
    assert clone_calls == 0


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


def test_memory_store_deduplicates_equal_boundary_tensors():
    store = InMemoryBoundaryActivationStore(deduplicate_tensors=True)
    mask = torch.tril(torch.ones(32, 32))

    store.put(0, 0, {"mask_0": mask, "mask_1": mask.clone()})

    assert store.tensor_bytes() == mask.untyped_storage().nbytes()
    loaded = store.get(0, 0)
    assert loaded["mask_0"].data_ptr() == loaded["mask_1"].data_ptr()


def test_memory_budget_counts_distinct_overlapping_views():
    budget = HostMemoryBudget(
        safety_reserve_bytes=0,
        system_available=lambda: 1 << 20,
        cgroup_available=lambda: None,
    )
    store = InMemoryBoundaryActivationStore(host_budget=budget)
    source = torch.arange(8, dtype=torch.float32)

    store.put(0, 0, {"left": source[:6], "right": source[2:]})

    assert store.tensor_bytes() == 2 * 6 * source.element_size()


def test_selective_execution_carries_unread_snapshots_without_copy(monkeypatch):
    store = InMemoryBoundaryActivationStore(deduplicate_tensors=True)
    store.put(0, 0, {"active": torch.ones(3), "carry": torch.arange(128)})
    stored_carry = store._boundaries[0][0]["carry"]
    original_clone = torch.Tensor.clone

    def check_clone(tensor, *args, **kwargs):
        assert tensor is not stored_carry
        return original_clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", check_clone)
    active = store.get_for_execution(0, 0, {"active"})
    assert set(active) == {"active"}
    active["active"].add_(2)
    assert torch.equal(store._boundaries[0][0]["active"], torch.ones(3))
    store.commit_propagated_batch(
        0, 1, 0, {"result": active["active"]}, consumed_names={"active"}
    )
    assert not store.contains(0, 0)
    assert store._boundaries[1][0]["carry"] is stored_carry
    assert set(store._boundaries[1][0]) == {"carry", "result"}
    assert torch.equal(store._boundaries[1][0]["result"], torch.full((3,), 3.0))


def test_routing_commit_reuses_source_aliases_without_snapshot_copy():
    store = InMemoryBoundaryActivationStore(deduplicate_tensors=True)
    store.put(0, 0, {"carry": torch.arange(128)})
    stored_carry = store._boundaries[0][0]["carry"]

    store.commit_propagated_batch(
        0,
        1,
        0,
        {"carry": stored_carry},
        consumed_names=(),
        reuse_source_aliases=True,
    )

    assert store._boundaries[1][0]["carry"] is stored_carry
    assert store.tensor_bytes() == stored_carry.numel() * stored_carry.element_size()


def test_selective_execution_expands_transitive_nested_aliases():
    store = InMemoryBoundaryActivationStore(deduplicate_tensors=True)
    first, second = torch.ones(3), torch.arange(3)
    store.put(0, 0, {
        "active": first,
        "nested": {"first": first, "second": second},
        "transitive": [second],
        "equal_content_alias": first.clone(),
        "dataclass": BoundaryOutput(first, torch.ones(1), torch.arange(1), {}),
        "carry": torch.zeros(100),
    })
    value = store.get_for_execution(0, 0, {"active"})
    assert set(value) == {
        "active", "nested", "transitive", "equal_content_alias", "dataclass"
    }
    assert value["active"] is value["nested"]["first"]
    assert value["active"] is value["equal_content_alias"]
    assert value["active"] is value["dataclass"].hidden_states
    assert value["transitive"][0] is value["nested"]["second"]
    value["active"].add_(5)
    value["transitive"][0].add_(9)
    del value["active"]
    store.commit_propagated_batch(0, 1, 0, value, consumed_names={"active"})
    restored = store.get(1, 0)
    assert torch.equal(restored["nested"]["first"], torch.full((3,), 6.0))
    assert torch.equal(restored["transitive"][0], torch.arange(3) + 9)
    restored["carry"].add_(100)
    assert torch.equal(store.get(1, 0)["carry"], torch.zeros(100))


def test_selective_execution_preserves_existing_distinct_view_snapshots():
    store = InMemoryBoundaryActivationStore()
    source = torch.arange(5.0)
    store.put(0, 0, {"active": source[:3], "other": source[2:]})
    value = store.get_for_execution(0, 0, {"active"})
    assert set(value) == {"active"}
    value["active"].add_(10)
    store.commit_propagated_batch(0, 1, 0, value, consumed_names=())
    assert torch.equal(store.get(1, 0)["other"], torch.arange(2.0, 5.0))


def test_partial_commit_preserves_content_deduplication_with_private_carry():
    store = InMemoryBoundaryActivationStore(deduplicate_tensors=True)
    store.put(0, 0, {"active": torch.ones(3), "carry": torch.full((3,), 2.0)})
    value = store.get_for_execution(0, 0, {"active"})
    store.commit_propagated_batch(
        0, 1, 0, {"result": value["active"] + 1}, consumed_names={"active"}
    )
    assert store.tensor_bytes() == 3 * torch.tensor(0.0).element_size()
    selected = store.get_for_execution(1, 0, {"result"})
    assert set(selected) == {"carry", "result"}
    assert selected["carry"] is selected["result"]
    selected["result"].add_(5)
    assert torch.equal(store.get(1, 0)["carry"], torch.full((3,), 2.0))


def test_selective_execution_rejects_missing_inputs_and_handles_empty_group(store):
    store.put(0, 0, {"active": torch.ones(3)})
    if isinstance(store, InMemoryBoundaryActivationStore):
        with pytest.raises(KeyError, match="Missing boundary inputs"):
            store.get_for_execution(0, 0, {"missing"})
        assert store.get_for_execution(0, 0, ()) == {}
    else:
        assert_values_equal(
            store.get_for_execution(0, 0, ()), store.get(0, 0)
        )
    with pytest.raises(KeyError, match="not committed"):
        store.get_for_execution(99, 0, ())


def test_partial_propagation_retains_carry_and_consumes_only_current_batch(store):
    for batch in range(2):
        store.put(0, batch, {
            "active": torch.tensor([batch]),
            "carry": BoundaryOutput(
                torch.ones(2), torch.ones(1), torch.arange(2), {"layer": 0}
            ),
        })
    for boundary in range(3):
        for batch in range(2):
            original = store.get(boundary, batch)
            value = store.get_for_execution(boundary, batch, {"active"})
            result = {"active": value["active"] + 1}
            store.commit_propagated_batch(
                boundary, boundary + 1, batch, result, consumed_names={"active"}
            )
            assert not store.contains(boundary, batch)
            assert_values_equal(
                store.get(boundary + 1, batch)["carry"], original["carry"]
            )
        assert store.batch_indices(boundary) == ()
    assert store.get(3, 0)["active"].item() == 3
    assert store.get(3, 1)["active"].item() == 4


def test_partial_commit_failure_keeps_source_destination_and_reservations():
    budget = HostMemoryBudget(
        safety_reserve_bytes=0, system_available=lambda: 1024,
        cgroup_available=lambda: None,
    )
    store = InMemoryBoundaryActivationStore(host_budget=budget)
    store.put(0, 0, {"carry": torch.arange(8)})
    store.put(1, 0, {"previous": torch.ones(2)})
    reservations = set(budget._reservations)
    with pytest.raises(TypeError, match="Unsupported"):
        store.commit_propagated_batch(
            0, 1, 0, {"bad": object()}, consumed_names=()
        )
    assert store.contains(0, 0)
    assert set(store.get(1, 0)) == {"previous"}
    assert budget._reservations == reservations
    with pytest.raises(MemoryError, match="Insufficient host memory"):
        store.commit_propagated_batch(
            0, 1, 0, {"large": torch.ones(300)}, consumed_names=()
        )
    assert budget._reservations == reservations
    store.commit_propagated_batch(
        0, 1, 0, {"result": torch.ones(3)}, consumed_names=()
    )
    assert not store.contains(0, 0)
    assert len(budget._reservations) == 1
    reservation = next(iter(budget._reservations))
    assert reservation.reserved_bytes == 0
    assert reservation.committed_bytes == store.tensor_bytes()
    store.delete(1)
    assert not budget._reservations
