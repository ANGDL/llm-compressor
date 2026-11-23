from dataclasses import dataclass

import pytest
import torch

from llmcompressor.streaming.activations.transfer import (
    transfer_tree,
    transfer_tree_to_cpu,
)
from llmcompressor.streaming.admission import DeviceMemoryBudget, DeviceMemoryError
from llmcompressor.streaming.loading.host_memory import (
    HostMemoryBudget,
    HostMemoryError,
)


@dataclass
class TensorBox:
    value: torch.Tensor
    label: str


def test_transfer_tree_packs_nested_values_and_preserves_aliases():
    base = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    value = {
        "pair": [base, base.t()],
        "boxed": TensorBox(base, "shared"),
        "empty": torch.empty(0, 3),
        "none": None,
    }

    with transfer_tree(value, "cpu") as transfer:
        result = transfer.wait()

        assert result["pair"][0] is result["boxed"].value
        assert result["boxed"].label == "shared"
        assert result["empty"].shape == (0, 3)
        assert result["none"] is None
        torch.testing.assert_close(result["pair"][0], base)
        torch.testing.assert_close(result["pair"][1], base.t())
        assert transfer.tensor_count == 3
        assert transfer.total_bytes >= base.numel() * base.element_size()


def test_transfer_tree_handles_values_without_tensors():
    value = {"value": None, "items": [1, "two", 3.0]}
    with transfer_tree(value, "cpu") as transfer:
        assert transfer.wait() == value
        assert transfer.tensor_count == 0
        assert transfer.total_bytes == 0


def test_transfer_tree_admits_aligned_bytes_and_releases_staging():
    value = [torch.ones(1), torch.zeros(1)]
    device_budget = DeviceMemoryBudget(available_bytes=260, safety_reserve_fraction=0)
    host_budget = HostMemoryBudget(
        safety_reserve_bytes=0,
        system_available=lambda: 259,
        cgroup_available=lambda: None,
    )
    with pytest.raises(HostMemoryError):
        transfer_tree(
            value, "cpu", device_budget=device_budget, host_budget=host_budget
        )
    assert not device_budget._reservations
    assert not host_budget._reservations

    host_budget._system_available = lambda: 260
    transfer = transfer_tree(
        value, "cpu", device_budget=device_budget, host_budget=host_budget
    )
    with transfer:
        assert transfer.total_bytes == 260
        assert device_budget.usable_bytes("cpu") == 0
        assert sum(item.committed_bytes for item in host_budget._reservations) == 260
        with pytest.raises(DeviceMemoryError):
            transfer_tree(value, "cpu", device_budget=device_budget)
    assert transfer.value is None
    assert not host_budget._reservations
    assert device_budget.usable_bytes("cpu") == 260


def test_transfer_tree_to_cpu_keeps_cpu_values():
    value = {"a": torch.arange(8, dtype=torch.float32), "b": None}
    transfer = transfer_tree_to_cpu(value)
    result = transfer.wait()
    assert result["a"] is value["a"]
    assert transfer.tensor_count == 1
    assert transfer.total_bytes == 0
    transfer.close()
    assert transfer.value is None


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="coalesced asynchronous transfer requires CUDA-compatible device",
)
def test_transfer_tree_orders_h2d_producer_and_d2h_streams():
    value = {"a": torch.arange(64, dtype=torch.float32), "b": torch.ones(4, 8)}
    input_stream = torch.cuda.Stream(device=0)
    producer = torch.cuda.Stream(device=0)
    output_stream = torch.cuda.Stream(device=0)
    host_budget = HostMemoryBudget(
        safety_reserve_bytes=0,
        system_available=lambda: 2**20,
        cgroup_available=lambda: None,
    )

    with transfer_tree(value, "cuda:0", stream=input_stream) as transfer:
        with torch.cuda.stream(producer):
            result = {name: tensor + 5 for name, tensor in transfer.wait().items()}
        carried_cpu = torch.arange(2)
        copied = transfer_tree_to_cpu(
            {"carried": carried_cpu, **result},
            stream=output_stream,
            wait_stream=producer,
            host_budget=host_budget,
        )
        try:
            host = copied.wait()
            torch.testing.assert_close(host["a"], value["a"] + 5)
            torch.testing.assert_close(host["b"], value["b"] + 5)
            assert host["carried"] is carried_cpu
            assert host["a"].is_pinned()
            assert copied.tensor_count == 2
            assert copied.source_value is None
            assert sum(item.committed_bytes for item in host_budget._reservations) == (
                copied.total_bytes
            )
        finally:
            copied.close()
        assert not host_budget._reservations


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_transfer_tree_moves_all_empty_tensors_to_the_destination():
    empty = torch.empty((0, 3))
    with transfer_tree(empty, "cuda:0") as transfer:
        device_tensor = transfer.wait()
        assert device_tensor.device == torch.device("cuda:0")
        copied = transfer_tree_to_cpu(device_tensor)
        try:
            host_tensor = copied.wait()
            assert host_tensor.device.type == "cpu"
            assert host_tensor.shape == empty.shape
        finally:
            copied.close()
