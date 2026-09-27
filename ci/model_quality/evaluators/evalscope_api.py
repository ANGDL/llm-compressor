"""Run EvalScope against an OpenAI-compatible endpoint and normalize metrics."""

from __future__ import annotations

import argparse
import glob
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from ci.model_quality.metrics import compare_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--generation-config", default="{}")
    parser.add_argument("--dataset-args", default="{}")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    baseline = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument("--reference-values-json")
    baseline.add_argument("--base-url")
    parser.add_argument("--base-model")
    parser.add_argument("--reference-id")
    return parser.parse_args()


def metric_specs(raw: str) -> list[dict[str, Any]]:
    value = json.loads(raw)
    if not isinstance(value, list) or not value:
        raise ValueError("metrics-json must be a non-empty list")
    for index, spec in enumerate(value):
        if not isinstance(spec, dict) or not isinstance(spec.get("name"), str):
            raise ValueError(f"metrics-json[{index}] must contain a metric name")
        if spec.get("direction") not in {"higher", "lower"}:
            raise ValueError(f"metrics-json[{index}] has invalid direction")
    return value


def run_evalscope(
    args: argparse.Namespace, *, api_url: str, model: str, work_dir: Path
) -> dict[str, float]:
    from evalscope import TaskConfig, run_task

    work_dir.mkdir(parents=True, exist_ok=True)
    task = TaskConfig(
        model=model,
        api_url=api_url.rstrip("/") + "/v1",
        api_key=args.api_key,
        eval_type="openai_api",
        datasets=[args.dataset],
        dataset_args=json.loads(args.dataset_args),
        generation_config=json.loads(args.generation_config),
        eval_batch_size=args.eval_batch_size,
        limit=args.limit,
        seed=args.seed,
        work_dir=str(work_dir),
        no_timestamp=True,
    )
    run_task(task)
    return load_report_metrics(work_dir, model, args.dataset)


def load_report_metrics(work_dir: Path, model: str, dataset: str) -> dict[str, float]:
    candidates = glob.glob(str(work_dir / "reports" / "**" / "*.json"), recursive=True)
    for candidate in candidates:
        document = json.loads(Path(candidate).read_text(encoding="utf-8"))
        report_name = str(document.get("name") or document.get("report_name") or "")
        if report_name and dataset not in report_name:
            continue
        metrics = {}
        for metric in document.get("metrics", []):
            identity = metric.get("identity") or {}
            name = (
                metric.get("name")
                or metric.get("legacy_name")
                or identity.get("key")
                or identity.get("name")
            )
            score = metric.get("score")
            if isinstance(name, str) and isinstance(score, (int, float)):
                metrics[name] = float(score)
        if metrics:
            return metrics
    raise ValueError(
        f"EvalScope report for model={model!r}, dataset={dataset!r} has no metrics"
    )


def reference_metrics(raw: str) -> dict[str, float]:
    value = json.loads(raw)
    if not isinstance(value, dict) or not value:
        raise ValueError("reference-values-json must be a non-empty object")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in value.values()
    ):
        raise ValueError("reference-values-json values must be numbers")
    return {str(key): float(item) for key, item in value.items()}


def main() -> int:
    args = parse_args()
    specs = metric_specs(args.metrics_json)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    compressed = run_evalscope(
        args,
        api_url=args.api_url,
        model=args.model,
        work_dir=args.work_dir / "compressed",
    )
    if args.reference_values_json:
        base = reference_metrics(args.reference_values_json)
        base_metadata = {
            "source": "reference-values",
            "reference_id": args.reference_id,
        }
    else:
        base_model = args.base_model or args.model
        base = run_evalscope(
            args,
            api_url=args.base_url,
            model=base_model,
            work_dir=args.work_dir / "base",
        )
        base_metadata = {
            "source": "endpoint",
            "url": args.base_url,
            "model": base_model,
        }
    metrics = []
    for spec in specs:
        name = spec["name"]
        if name not in base or name not in compressed:
            raise KeyError(f"metric {name!r} is missing from EvalScope results")
        metrics.append(
            {
                **spec,
                "base_value": base[name],
                "compressed_value": compressed[name],
            }
        )
    payload = {
        "schema_version": 1,
        "suite_id": f"evalscope:{args.dataset}",
        "evaluator": "evalscope/openai_api",
        "task": args.dataset,
        "metrics": metrics,
        "base_metadata": base_metadata,
        "compressed_metadata": {"url": args.api_url, "model": args.model},
        "runtime_versions": {
            package: importlib.metadata.version(package)
            for package in ("evalscope", "openai")
            if package
            in {
                distribution.metadata["Name"]
                for distribution in importlib.metadata.distributions()
            }
        },
        "config": {
            "limit": args.limit,
            "seed": args.seed,
            "eval_batch_size": args.eval_batch_size,
            "generation_config": json.loads(args.generation_config),
            "dataset_args": json.loads(args.dataset_args),
        },
    }
    normalized = compare_evaluation(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 1 if normalized["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
