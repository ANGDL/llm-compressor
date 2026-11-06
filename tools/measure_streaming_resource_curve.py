"""Measure isolated Streaming PTQ resource points and emit a JSON curve."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class ResourceCase:
    label: str
    checkpoint: Path


def parse_case_spec(spec: str) -> ResourceCase:
    label, separator, checkpoint = spec.partition("=")
    if (
        not separator
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label)
        or not checkpoint
    ):
        raise ValueError(
            f"Invalid case {spec!r}; expected LABEL=CHECKPOINT_DIRECTORY"
        )
    return ResourceCase(label=label, checkpoint=Path(checkpoint))


def parse_probe_output(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        required = {
            "checkpoint_bytes",
            "incremental_peak_bytes",
            "output",
        }
        if required <= value.keys():
            return value
    raise ValueError("resource probe did not emit a valid JSON measurement")


def build_report(
    measurements: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    successful = [
        measurement
        for measurement in measurements
        if measurement.get("returncode") == 0
        and "checkpoint_bytes" in measurement
    ]
    return {
        "schema": 1,
        "cases": list(measurements),
        "curve": {
            "checkpoint_bytes": [
                measurement["checkpoint_bytes"]
                for measurement in successful
            ],
            "incremental_peak_bytes": [
                measurement["incremental_peak_bytes"]
                for measurement in successful
            ],
        },
    }


def run_recovery_case(
    depth: int,
    *,
    probe: Path,
    work_root: Path,
    batch_count: int,
    width: int,
) -> dict[str, Any]:
    root = work_root / f"depth{depth}"
    measurement: dict[str, Any] = {"label": f"depth{depth}", "depth": depth}
    for mode in ("prepare", "measure"):
        result = subprocess.run(
            [
                sys.executable, str(probe), mode, str(root),
                "--depth", str(depth), "--batch-count", str(batch_count),
                "--width", str(width),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        measurement["returncode"] = result.returncode
        if result.returncode:
            measurement.update(
                error=f"recovery {mode} failed",
                stdout=result.stdout,
                stderr=result.stderr,
            )
            return measurement
        if mode == "measure":
            try:
                measurement.update(json.loads(result.stdout.splitlines()[-1]))
            except (ValueError, IndexError) as error:
                measurement.update(returncode=1, error=str(error))
    return measurement


def run_case(
    case: ResourceCase,
    *,
    probe: Path,
    work_root: Path,
    output_root: Path,
    sequence_length: int,
    algorithm: str = "imatrix",
    device: str = "cpu",
) -> dict[str, Any]:
    work_dir = work_root / case.label
    output_dir = output_root / case.label
    result = subprocess.run(
        [
            sys.executable,
            str(probe),
            str(case.checkpoint),
            str(work_dir),
            str(output_dir),
            "--sequence-length",
            str(sequence_length),
            "--algorithm", algorithm,
            "--device", device,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    measurement: dict[str, Any] = {
        "label": case.label,
        "checkpoint": str(case.checkpoint),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    if result.returncode == 0:
        try:
            measurement.update(parse_probe_output(result.stdout))
        except ValueError as error:
            measurement["error"] = str(error)
            measurement["returncode"] = 1
    else:
        measurement["error"] = (
            f"resource probe failed with return code {result.returncode}"
        )
    return measurement


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--case",
        action="append",
        metavar="LABEL=CHECKPOINT_DIRECTORY",
    )
    mode.add_argument("--depth", type=int, action="append")
    parser.add_argument(
        "--probe",
        type=Path,
        default=Path(__file__).parents[1]
        / "tests/llmcompressor/streaming/resource_probe.py",
    )
    parser.add_argument("--work-root", type=Path, default=Path("resource-work"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("resource-output")
    )
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument(
        "--algorithm", choices=("rtn", "imatrix", "gptq", "awq", "smoothquant"),
        default="imatrix",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-count", type=int, default=4)
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args(argv)

    if args.depth:
        if min(*args.depth, args.batch_count, args.width) < 1:
            parser.error("depth, batch-count and width must be positive")
        if len(set(args.depth)) != len(args.depth):
            parser.error("depth values must be unique")
        measurements = [
            run_recovery_case(
                depth,
                probe=Path(__file__).parents[1]
                / "tests/llmcompressor/streaming/recovery_resource_probe.py",
                work_root=args.work_root,
                batch_count=args.batch_count,
                width=args.width,
            )
            for depth in args.depth
        ]
    else:
        try:
            cases = [parse_case_spec(spec) for spec in args.case]
        except ValueError as error:
            parser.error(str(error))
        if len({case.label for case in cases}) != len(cases):
            parser.error("case labels must be unique")
        measurements = [
            run_case(
                case,
                probe=args.probe,
                work_root=args.work_root,
                output_root=args.output_root,
                sequence_length=args.sequence_length,
                algorithm=args.algorithm,
                device=args.device,
            )
            for case in cases
        ]
    report = build_report(measurements)
    if args.depth:
        successful = [item for item in measurements if item["returncode"] == 0]
        report["curve"] = {
            name: [item[name] for item in successful]
            for name in (
                "depth", "incremental_peak_bytes", "peak_rss_bytes",
                "retained_boundary_bytes", "peak_live_batches",
            )
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(item["returncode"] == 0 for item in measurements) else 1


if __name__ == "__main__":
    raise SystemExit(main())
