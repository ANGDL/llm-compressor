"""Evaluate base and compressed OpenAI-compatible endpoints with lm-eval."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from ci.model_quality.metrics import compare_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    baseline = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument("--base-url")
    baseline.add_argument("--reference-values-json")
    parser.add_argument("--reference-id")
    parser.add_argument("--compressed-url", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--compressed-model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--task", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--num-fewshot", type=int)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-gen-toks", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def run_endpoint(
    args: argparse.Namespace, *, base_url: str, model_name: str
) -> dict[str, Any]:
    import lm_eval

    model_args = {
        "base_url": base_url.rstrip("/") + "/v1/completions",
        "model": model_name,
        "tokenized_requests": False,
        "tokenizer_backend": None,
        "tokenizer": args.tokenizer,
        "num_concurrent": args.batch_size,
        "max_gen_toks": args.max_gen_toks,
        "api_key": args.api_key,
    }
    results = lm_eval.simple_evaluate(
        model="local-completions",
        model_args=",".join(
            f"{key}={str(value).lower() if isinstance(value, bool) else value}"
            for key, value in model_args.items()
            if value is not None
        ),
        tasks=[args.task],
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=1,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )
    return {
        "metrics": {
            key: value
            for key, value in results["results"][args.task].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        },
        "versions": results.get("versions", {}),
        "n-shot": results.get("n-shot", {}),
    }


def package_versions() -> dict[str, str]:
    versions = {}
    for package in ("lm-eval", "openai", "requests"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def main() -> int:
    args = parse_args()
    specs = metric_specs(args.metrics_json)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.reference_values_json:
        reference_values = json.loads(args.reference_values_json)
        if not isinstance(reference_values, dict) or not reference_values:
            raise ValueError("reference-values-json must be a non-empty object")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in reference_values.values()
        ):
            raise ValueError("reference-values-json values must be numbers")
        base = {
            "metrics": reference_values,
            "reference_id": args.reference_id,
            "source": "reference-values",
        }
    else:
        base = run_endpoint(args, base_url=args.base_url, model_name=args.base_model)
    (args.work_dir / "base.json").write_text(
        json.dumps(base, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    compressed = run_endpoint(
        args, base_url=args.compressed_url, model_name=args.compressed_model
    )
    (args.work_dir / "compressed.json").write_text(
        json.dumps(compressed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
        "suite_id": f"lm-eval-api:{args.task}",
        "evaluator": "lm-evaluation-harness/local-completions",
        "task": args.task,
        "metrics": metrics,
        "base_metadata": {
            "url": args.base_url,
            "model": args.base_model,
            "versions": base.get("versions"),
            "source": base.get("source", "endpoint"),
            "reference_id": base.get("reference_id"),
        },
        "compressed_metadata": {
            "url": args.compressed_url,
            "model": args.compressed_model,
            "versions": compressed.get("versions"),
        },
        "runtime_versions": package_versions(),
        "config": {
            "num_fewshot": args.num_fewshot,
            "limit": args.limit,
            "batch_size": args.batch_size,
            "max_gen_toks": args.max_gen_toks,
            "seed": args.seed,
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
