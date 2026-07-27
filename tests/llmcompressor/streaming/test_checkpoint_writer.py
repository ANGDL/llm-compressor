from __future__ import annotations

import json
import threading

import pytest
import torch
from safetensors import safe_open

from llmcompressor.streaming import (
    ArtifactCompatibilityError,
    StreamingCheckpointWriter,
)
from llmcompressor.streaming.checkpoint import (
    AsyncDirectSafetensorsWriter,
    DirectSafetensorsWriter,
)
from llmcompressor.streaming.checkpoint import writer as writer_module


def read_shard(path):
    with safe_open(path, framework="pt", device="cpu") as file:
        return {name: file.get_tensor(name) for name in file.keys()}


def test_transactions_are_independent_from_output_shards(tmp_path):
    writer = StreamingCheckpointWriter(tmp_path, run_fingerprint="run-a")
    with writer.transaction("subgraph-00000") as transaction:
        transaction.write_tensor(
            "layers.0.weight", torch.arange(4), output_shard="model-1.safetensors"
        )
        transaction.write_tensor(
            "shared.scale", torch.ones(2), output_shard="model-2.safetensors"
        )
        transaction.mark_quantized("layers.0", "int-quantized")
        transaction.commit()

    with writer.transaction("subgraph-00001") as transaction:
        transaction.write_tensor(
            "layers.1.weight", torch.arange(3), output_shard="model-1.safetensors"
        )
        transaction.commit()

    states = writer.assemble_shards()

    assert set(states) == {"model-1.safetensors", "model-2.safetensors"}
    first = read_shard(tmp_path / "shards/model-1.safetensors")
    assert set(first) == {"layers.0.weight", "layers.1.weight"}
    assert torch.equal(first["layers.0.weight"], torch.arange(4))
    assert states["model-1.safetensors"]["module_formats"] == {
        "layers.0": "int-quantized"
    }


def test_uncommitted_transaction_is_removed_and_replayed(tmp_path):
    writer = StreamingCheckpointWriter(tmp_path, run_fingerprint="run-a")
    with writer.transaction("subgraph-00000") as transaction:
        transaction.write_tensor(
            "partial", torch.ones(2), output_shard="model.safetensors"
        )

    assert not (tmp_path / "transactions/subgraph-00000.pending").exists()
    assert not writer.is_transaction_complete("subgraph-00000")

    with writer.transaction("subgraph-00000") as transaction:
        transaction.write_tensor(
            "complete", torch.zeros(2), output_shard="model.safetensors"
        )
        transaction.commit()

    assert writer.is_transaction_complete("subgraph-00000")


def test_corrupt_committed_payload_is_not_resumable(tmp_path):
    writer = StreamingCheckpointWriter(tmp_path, run_fingerprint="run-a")
    with writer.transaction("subgraph-00000") as transaction:
        record = transaction.write_tensor(
            "weight", torch.ones(2), output_shard="model.safetensors"
        )
        transaction.commit()

    committed_payload = (
        tmp_path / "transactions/subgraph-00000/tensors" / record.payload.name
    )
    committed_payload.write_bytes(b"corrupt")

    assert not writer.is_transaction_complete("subgraph-00000")
    assert writer.has_transaction("subgraph-00000")
    with pytest.raises(RuntimeError, match="corrupt"):
        writer.assemble_shards()


def test_duplicate_tensor_across_transactions_is_rejected(tmp_path):
    writer = StreamingCheckpointWriter(tmp_path, run_fingerprint="run-a")
    for index in range(2):
        with writer.transaction(f"subgraph-{index:05d}") as transaction:
            transaction.write_tensor(
                "weight", torch.tensor(index), output_shard="model.safetensors"
            )
            transaction.commit()

    with pytest.raises(ValueError, match="multiple transactions"):
        writer.assemble_shards()


def test_writer_rejects_a_different_run(tmp_path):
    StreamingCheckpointWriter(tmp_path, run_fingerprint="run-a")

    with pytest.raises(ArtifactCompatibilityError, match="different quantization"):
        StreamingCheckpointWriter(tmp_path, run_fingerprint="run-b")


def test_committed_boundary_is_validated_and_reloaded(tmp_path):
    writer = StreamingCheckpointWriter(tmp_path, run_fingerprint="run-a")
    with writer.transaction("subgraph-00000") as transaction:
        transaction.write_tensor(
            "weight", torch.ones(2), output_shard="model.safetensors"
        )
        transaction.write_boundary(
            0, {"hidden": torch.arange(3)}, boundary=1
        )
        transaction.commit()

    boundary, values = writer.committed_boundary("subgraph-00000")
    assert boundary == 1
    assert torch.equal(values[0]["hidden"], torch.arange(3))

    boundary_path = next(
        (tmp_path / "transactions/subgraph-00000/boundaries").iterdir()
    )
    boundary_path.write_bytes(b"corrupt")
    assert not writer.is_transaction_complete("subgraph-00000")
    assert writer.committed_boundary("subgraph-00000") is None


def test_transaction_metadata_contains_payload_checksum(tmp_path):
    writer = StreamingCheckpointWriter(tmp_path, run_fingerprint="run-a")
    with writer.transaction("subgraph-00000") as transaction:
        transaction.write_tensor(
            "weight", torch.ones(2), output_shard="model.safetensors"
        )
        transaction.commit()

    metadata = json.loads(
        (tmp_path / "transactions/subgraph-00000/metadata.json").read_text()
    )
    assert len(metadata["records"][0]["sha256"]) == 64


def test_writer_preserves_bfloat16_storage(tmp_path):
    writer = StreamingCheckpointWriter(tmp_path, run_fingerprint="run-a")
    expected = torch.tensor([1.5, -2.25], dtype=torch.bfloat16)
    with writer.transaction("subgraph-00000") as transaction:
        transaction.write_tensor(
            "weight", expected, output_shard="model.safetensors"
        )
        transaction.commit()

    writer.assemble_shards()
    actual = read_shard(tmp_path / "shards/model.safetensors")["weight"]
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected)


def test_direct_writer_publishes_final_shard_without_tensor_payloads(tmp_path):
    writer = DirectSafetensorsWriter(tmp_path, run_fingerprint="run-a")
    output = writer.write_shard(
        "subgraph-00000",
        {"layers.0.weight": torch.arange(4)},
        quantized_modules={"layers.0": "int-quantized"},
    )

    assert output == tmp_path / "model-subgraph-00000.safetensors"
    assert torch.equal(
        read_shard(output)["layers.0.weight"], torch.arange(4)
    )
    assert not (tmp_path / "transactions").exists()
    assert not list(tmp_path.rglob("*.bin"))


def test_async_direct_writer_returns_before_disk_write_finishes(tmp_path):
    direct = DirectSafetensorsWriter(tmp_path, run_fingerprint="run-a")
    original = direct.write_shard
    started = threading.Event()
    release = threading.Event()

    def blocked_write(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        return original(*args, **kwargs)

    direct.write_shard = blocked_write
    writer = AsyncDirectSafetensorsWriter(direct)
    source = torch.ones(4)
    try:
        output = writer.write_shard("subgraph-00000", {"weight": source})
        assert started.wait(timeout=5)
        assert not output.exists()
        source.zero_()
        release.set()
        writer.close()
    finally:
        release.set()
        writer.close()

    assert torch.equal(read_shard(output)["weight"], torch.ones(4))


def test_async_direct_writer_propagates_background_failure(tmp_path):
    direct = DirectSafetensorsWriter(tmp_path, run_fingerprint="run-a")

    def failed_write(*_args, **_kwargs):
        raise OSError("disk full")

    direct.write_shard = failed_write
    writer = AsyncDirectSafetensorsWriter(direct)
    writer.write_shard("subgraph-00000", {"weight": torch.ones(4)})

    with pytest.raises(RuntimeError, match="subgraph-00000"):
        writer.close()


def test_direct_writer_publishes_state_only_after_header_validation(
    tmp_path, monkeypatch
):
    writer = DirectSafetensorsWriter(tmp_path, run_fingerprint="run-a")
    monkeypatch.setattr(
        writer_module,
        "_validate_safetensors_header",
        lambda *_args: (_ for _ in ()).throw(ValueError("invalid header")),
    )

    with pytest.raises(ValueError, match="invalid header"):
        writer.write_shard("subgraph-00000", {"weight": torch.ones(4)})

    assert not list(tmp_path.glob("*.safetensors"))
    assert not list((tmp_path / ".streaming-state").glob("*.json"))
