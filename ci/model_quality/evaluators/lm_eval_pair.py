"""Evaluate base and compressed models in isolated lm-eval subprocesses."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ci.model_quality.metrics import compare_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--compressed-model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--model-backend", default="vllm")
    parser.add_argument("--model-args-json", default="{}")
    parser.add_argument("--num-fewshot", type=int)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--fewshot-as-multiturn", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _metric_specs(raw: str) -> list[dict[str, Any]]:
    value = json.loads(raw)
    if not isinstance(value, list) or not value:
        raise ValueError("metrics-json must be a non-empty list")
    for index, spec in enumerate(value):
        if not isinstance(spec, dict) or not isinstance(spec.get("name"), str):
            raise ValueError(f"metrics-json[{index}] must contain a metric name")
        if spec.get("direction") not in {"higher", "lower"}:
            raise ValueError(f"metrics-json[{index}] has invalid direction")
    return value


def _run_one(args: argparse.Namespace, model: str, output: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "ci.model_quality.evaluators.lm_eval_single",
        "--model-path",
        model,
        "--model-backend",
        args.model_backend,
        "--model-args-json",
        args.model_args_json,
        "--task",
        args.task,
        "--batch-size",
        args.batch_size,
        "--seed",
        str(args.seed),
        "--output",
        str(output),
    ]
    if args.num_fewshot is not None:
        command.extend(["--num-fewshot", str(args.num_fewshot)])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.apply_chat_template:
        command.append("--apply-chat-template")
    if args.fewshot_as_multiturn:
        command.append("--fewshot-as-multiturn")
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    specs = _metric_specs(args.metrics_json)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    base_path = args.work_dir / "base.json"
    compressed_path = args.work_dir / "compressed.json"
    _run_one(args, args.base_model, base_path)
    _run_one(args, args.compressed_model, compressed_path)

    base = json.loads(base_path.read_text())
    compressed = json.loads(compressed_path.read_text())
    metrics = []
    for spec in specs:
        name = spec["name"]
        if name not in base["metrics"] or name not in compressed["metrics"]:
            raise KeyError(f"metric {name!r} is missing from lm-eval results")
        metrics.append(
            {
                **spec,
                "base_value": base["metrics"][name],
                "compressed_value": compressed["metrics"][name],
            }
        )

    payload = {
        "schema_version": 1,
        "suite_id": f"lm-eval:{args.task}",
        "evaluator": "lm-evaluation-harness",
        "task": args.task,
        "metrics": metrics,
        "base_metadata": {
            "versions": base.get("versions"),
            "n-shot": base.get("n-shot"),
            "runtime_versions": base.get("runtime_versions"),
        },
        "compressed_metadata": {
            "versions": compressed.get("versions"),
            "n-shot": compressed.get("n-shot"),
            "runtime_versions": compressed.get("runtime_versions"),
        },
        "config": compressed.get("config"),
    }
    normalized = compare_evaluation(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n")
    return 1 if normalized["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
