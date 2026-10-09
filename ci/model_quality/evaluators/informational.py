"""Informational evaluation: run a tool, record all scores, never auto-gate.

The quantized model is served by the runtime-smoke step; this discovers that
one served model via ``GET /v1/models`` (one service == one model), runs the
selected tool (lm_eval or evalscope) against it, then writes a normalized
result with reference value 1 and ``min_recovery`` 0 for every metric — so the
evaluate stage always PASSES and simply records the scores. A human reviews the
scores and decides whether the model is acceptable.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from ci.model_quality.metrics import compare_evaluation


def _discover_model(api_url: str, api_key: str) -> str:
    request = urllib.request.Request(
        api_url.rstrip("/") + "/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        document = json.load(response)
    data = document.get("data") or []
    if not data or not data[0].get("id"):
        raise SystemExit(f"no served model found at {api_url}/v1/models")
    return str(data[0]["id"])


def _scores(tool: str, args: argparse.Namespace, model: str) -> dict[str, float]:
    if tool == "evalscope":
        from ci.model_quality.evaluators.evalscope_api import run_evalscope

        namespace = SimpleNamespace(
            api_key=args.api_key,
            dataset=args.dataset,
            dataset_args="{}",
            generation_config="{}",
            eval_batch_size=args.batch_size,
            limit=args.limit,
            seed=args.seed,
        )
        return run_evalscope(
            namespace,
            api_url=args.api_url,
            model=model,
            work_dir=args.work_dir / "compressed",
        )
    from ci.model_quality.evaluators.lm_eval_api_pair import run_endpoint

    namespace = SimpleNamespace(
        tokenizer=None,
        batch_size=args.batch_size,
        max_gen_toks=args.max_gen_toks,
        api_key=args.api_key,
        task=args.dataset,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        seed=args.seed,
    )
    return run_endpoint(namespace, base_url=args.api_url, model_name=model)["metrics"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", choices=["lm_eval", "evalscope"], required=True)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--limit", type=float)
    parser.add_argument("--num-fewshot", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-gen-toks", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    model = _discover_model(args.api_url, args.api_key)
    scores = _scores(args.tool, args, model)
    if not scores:
        raise SystemExit(
            f"{args.tool} produced no metrics for dataset {args.dataset!r}"
        )

    # Reference value 1 + min_recovery 0 ⇒ every metric passes; recovery equals
    # the raw score, so the recorded numbers are the model's actual scores.
    metrics = [
        {
            "name": name,
            "direction": "higher",
            "min_recovery": 0.0,
            "base_value": 1.0,
            "compressed_value": float(score),
        }
        for name, score in scores.items()
    ]
    payload = {
        "schema_version": 1,
        "suite_id": f"{args.tool}:{args.dataset}",
        "evaluator": f"{args.tool}/openai_api",
        "task": args.dataset,
        "metrics": metrics,
        "base_metadata": {
            "source": "reference-values",
            "note": "informational: reference=1, no automatic gate",
        },
        "compressed_metadata": {"url": args.api_url, "model": model},
    }
    normalized = compare_evaluation(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
