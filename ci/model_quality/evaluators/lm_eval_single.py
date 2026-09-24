"""Run one lm-evaluation-harness model and persist selected raw metrics."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-backend", default="vllm")
    parser.add_argument("--model-args-json", default="{}")
    parser.add_argument("--task", required=True)
    parser.add_argument("--num-fewshot", type=int)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--fewshot-as-multiturn", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _model_args(model_path: str, raw: str) -> str:
    extra = json.loads(raw)
    if not isinstance(extra, dict):
        raise ValueError("model-args-json must be an object")
    values: dict[str, Any] = {"pretrained": model_path, **extra}
    encoded = []
    for key, value in values.items():
        if isinstance(value, bool):
            value = str(value).lower()
        elif isinstance(value, (dict, list)):
            value = json.dumps(value, separators=(",", ":"))
        encoded.append(f"{key}={value}")
    return ",".join(encoded)


def _package_exists(package: str) -> bool:
    try:
        importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def main() -> int:
    args = parse_args()
    import lm_eval

    results = lm_eval.simple_evaluate(
        model=args.model_backend,
        model_args=_model_args(args.model_path, args.model_args_json),
        tasks=[args.task],
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )
    task_result = results["results"][args.task]
    payload = {
        "task": args.task,
        "metrics": {
            key: value
            for key, value in task_result.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        },
        "versions": results.get("versions", {}),
        "n-shot": results.get("n-shot", {}),
        "config": {
            "model_backend": args.model_backend,
            "num_fewshot": args.num_fewshot,
            "limit": args.limit,
            "batch_size": args.batch_size,
            "apply_chat_template": args.apply_chat_template,
            "fewshot_as_multiturn": args.fewshot_as_multiturn,
            "seed": args.seed,
        },
        "runtime_versions": {
            package: importlib.metadata.version(package)
            for package in ("lm-eval", "vllm", "torch", "transformers")
            if _package_exists(package)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
