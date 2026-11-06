"""Prepare or measure a fixed-width committed recovery chain in one process."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import weakref
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch

from llmcompressor.streaming.checkpoint import StreamingCheckpointWriter
from llmcompressor.streaming.compaction import BoundaryCompactor
from llmcompressor.streaming.recovery import build_resume_plan


def prepare(root: Path, depth: int, batch_count: int, width: int) -> None:
    if root.exists():
        raise FileExistsError(f"Recovery fixture already exists: {root}")
    writer = StreamingCheckpointWriter(root, run_fingerprint="resource-recovery-v1")
    boundary = {"hidden_states": torch.ones(1, width, 64)}
    transaction_ids = []
    for index in range(depth):
        transaction_id = f"subgraph-{index:05d}"
        with writer.transaction(transaction_id) as transaction:
            transaction.write_tensor(
                f"layers.{index}.weight",
                torch.ones(64, 64),
                output_shard="model.safetensors",
            )
            for batch_index in range(batch_count):
                transaction.write_boundary(
                    batch_index,
                    boundary,
                    boundary=index + 1,
                    expected_batch_count=batch_count,
                )
            transaction.commit()
        transaction_ids.append(transaction_id)
        BoundaryCompactor(root).compact(tuple(transaction_ids))


def measure(root: Path) -> dict:
    writer = StreamingCheckpointWriter(root, run_fingerprint="resource-recovery-v1")
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = perf_counter()
    plan = build_resume_plan(writer)
    scan_seconds = perf_counter() - started
    references = []
    peak_live_batches = 0
    frontier_bytes = 0
    frontier_batches = 0
    frontier_id = plan.transaction_ids[-1]
    started = perf_counter()
    for _index, boundary in writer.iter_committed_boundary(
        frontier_id, metadata=plan.metadata[-1]
    ):
        tensor = boundary["hidden_states"]
        references = [reference for reference in references if reference() is not None]
        references.append(weakref.ref(tensor))
        peak_live_batches = max(peak_live_batches, len(references))
        frontier_bytes += tensor.nbytes
        frontier_batches += 1
        del tensor, boundary
    frontier_seconds = perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024
    retained_bytes = sum(
        path.stat().st_size
        for path in root.glob("transactions/*/boundaries/*")
        if path.is_file()
    )
    return {
        "pid": os.getpid(),
        "depth": len(plan.transaction_ids),
        "baseline_rss_bytes": int(baseline * scale),
        "peak_rss_bytes": int(peak * scale),
        "incremental_peak_bytes": int(max(0, peak - baseline) * scale),
        "scan_stats": asdict(plan.scan_stats),
        "scan_seconds": scan_seconds,
        "frontier_seconds": frontier_seconds,
        "frontier_batches": frontier_batches,
        "frontier_tensor_bytes": frontier_bytes,
        "peak_live_batches": peak_live_batches,
        "retained_boundary_bytes": retained_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "measure"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--batch-count", type=int, default=4)
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args()
    if min(args.depth, args.batch_count, args.width) < 1:
        parser.error("depth, batch-count and width must be positive")
    if args.mode == "prepare":
        prepare(args.root, args.depth, args.batch_count, args.width)
    else:
        print(json.dumps(measure(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
