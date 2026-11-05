from __future__ import annotations

import json
import multiprocessing
import os
import time
from pathlib import Path

import pytest
import torch

from llmcompressor.streaming import (
    BoundaryCompactor,
    ParallelContext,
    PlacementPlan,
    SingleParticipantCoordinator,
    StreamingCheckpointWriter,
    StreamingWorkDirectoryLock,
    build_resume_plan,
)
from llmcompressor.streaming.checkpoint.coordinator import (
    CommittedTransactionCorruptionError,
)


def _hold_lock(path: str, ready: str) -> None:
    with StreamingWorkDirectoryLock(path):
        Path(ready).write_text("ready", encoding="utf-8")
        multiprocessing.Event().wait(10)


def _commit_with_fault(root: str, ready: str, release: str) -> None:
    os.environ["LLMCOMPRESSOR_STREAMING_FAULT_AT"] = (
        "transaction.after_rename"
    )
    os.environ["LLMCOMPRESSOR_STREAMING_FAULT_READY"] = ready
    os.environ["LLMCOMPRESSOR_STREAMING_FAULT_RELEASE"] = release
    coordinator = SingleParticipantCoordinator(root, run_fingerprint="run-a")
    writer = StreamingCheckpointWriter(
        root,
        run_fingerprint="run-a",
        coordinator=coordinator,
    )
    with writer.transaction("subgraph-00000") as transaction:
        transaction.write_tensor(
            "weight",
            torch.ones(2),
            output_shard="model.safetensors",
        )
        transaction.commit()


def test_single_device_parallel_contract_is_explicit():
    context = ParallelContext.single_device()
    assert context.to_dict()["ep_size"] == 1
    assert context.participants == ("rank-00000",)
    assert context.is_coordinator

    placement = PlacementPlan.from_mapping({"layer.0.weight": 0})
    assert placement.owner("layer.0.weight") == 0
    assert placement.fingerprint


def test_work_directory_lock_rejects_second_process(tmp_path):
    ready = tmp_path / "ready"
    process = multiprocessing.Process(
        target=_hold_lock,
        args=(str(tmp_path / "work"), str(ready)),
    )
    process.start()
    try:
        for _ in range(100):
            if ready.exists():
                break
            time.sleep(0.01)
        assert ready.exists()
        with pytest.raises(RuntimeError, match="already locked"):
            StreamingWorkDirectoryLock(tmp_path / "work").acquire()
    finally:
        process.terminate()
        process.join(timeout=5)


def test_recovery_accepts_local_transaction_after_process_kill(tmp_path):
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    process = multiprocessing.Process(
        target=_commit_with_fault,
        args=(str(tmp_path), str(ready), str(release)),
    )
    process.start()
    try:
        for _ in range(500):
            if ready.exists():
                break
            time.sleep(0.01)
        assert ready.exists()
        process.terminate()
        process.join(timeout=5)
        assert process.exitcode != 0
    finally:
        release.write_text("release", encoding="utf-8")
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    assert (tmp_path / "transactions/subgraph-00000").is_dir()
    assert not (tmp_path / "commits/subgraph-00000.json").exists()
    coordinator = SingleParticipantCoordinator(tmp_path, run_fingerprint="run-a")
    writer = StreamingCheckpointWriter(
        tmp_path,
        run_fingerprint="run-a",
        coordinator=coordinator,
    )
    assert writer.is_transaction_complete("subgraph-00000")
    assert (tmp_path / "commits/subgraph-00000.json").is_file()


def test_boundary_metadata_records_structure_aliases_and_checksum(tmp_path):
    writer = StreamingCheckpointWriter(
        tmp_path,
        run_fingerprint="run-a",
        coordinator=SingleParticipantCoordinator(
            tmp_path,
            run_fingerprint="run-a",
        ),
    )
    shared = torch.arange(4)
    with writer.transaction("subgraph-00000") as transaction:
        transaction.write_tensor(
            "weight",
            torch.ones(2),
            output_shard="model.safetensors",
        )
        transaction.write_boundary(
            0,
            {"left": shared, "right": shared},
            boundary=1,
            expected_batch_count=1,
        )
        transaction.commit()

    metadata = json.loads(
        (tmp_path / "transactions/subgraph-00000/metadata.json").read_text()
    )
    boundary = metadata["boundary"]
    assert boundary["expected_batch_count"] == 1
    assert boundary["batches"][0]["summary"]["aliases"]["value.left"] == 0
    assert boundary["batches"][0]["summary"]["aliases"]["value.right"] == 0
    assert writer.is_transaction_complete("subgraph-00000")
    assert (tmp_path / "commits/subgraph-00000.json").is_file()


def test_boundary_content_corruption_is_rejected(tmp_path):
    writer = StreamingCheckpointWriter(tmp_path, run_fingerprint="run-a")
    with writer.transaction("subgraph-00000") as transaction:
        transaction.write_tensor(
            "weight",
            torch.ones(2),
            output_shard="model.safetensors",
        )
        transaction.write_boundary(
            0,
            {"hidden": torch.ones(2)},
            boundary=1,
        )
        transaction.commit()

    path = next((tmp_path / "transactions/subgraph-00000/boundaries").glob("*.pt"))
    value = torch.load(path, weights_only=False)
    value["hidden"].add_(1)
    torch.save(value, path)
    assert not writer.is_transaction_complete("subgraph-00000")


def test_single_participant_commit_requires_prepared_manifest(tmp_path):
    coordinator = SingleParticipantCoordinator(
        tmp_path,
        run_fingerprint="run-a",
    )
    with pytest.raises(CommittedTransactionCorruptionError):
        coordinator.validate_commit("subgraph-00000")


def test_resume_plan_accepts_only_contiguous_transactions(tmp_path):
    writer = StreamingCheckpointWriter(tmp_path, run_fingerprint="run-a")
    for index in range(2):
        with writer.transaction(f"subgraph-{index:05d}") as transaction:
            transaction.write_tensor(
                f"weight-{index}",
                torch.ones(2),
                output_shard="model.safetensors",
            )
            transaction.commit()

    plan = build_resume_plan(writer)
    assert plan.transaction_ids == ("subgraph-00000", "subgraph-00001")
    assert plan.next_transaction_id == "subgraph-00002"


def test_boundary_compaction_records_legal_retirement(tmp_path):
    writer = StreamingCheckpointWriter(tmp_path, run_fingerprint="run-a")
    for index in range(3):
        with writer.transaction(f"subgraph-{index:05d}") as transaction:
            transaction.write_tensor(
                f"weight-{index}",
                torch.ones(2),
                output_shard="model.safetensors",
            )
            transaction.write_boundary(
                0,
                {"hidden": torch.full((2,), index)},
                boundary=index + 1,
            )
            transaction.commit()

    retired = BoundaryCompactor(tmp_path, keep_recent=2).compact(
        ("subgraph-00000", "subgraph-00001", "subgraph-00002")
    )

    assert retired == ("subgraph-00000",)
    assert not (
        tmp_path / "transactions/subgraph-00000/boundaries"
    ).exists()
    assert (tmp_path / "retirements/subgraph-00000.json").is_file()
    assert writer.is_transaction_complete("subgraph-00000")
    with pytest.raises(RuntimeError, match="compacted"):
        next(writer.iter_committed_boundary("subgraph-00000"))
    assert build_resume_plan(writer).transaction_ids == (
        "subgraph-00000",
        "subgraph-00001",
        "subgraph-00002",
    )
