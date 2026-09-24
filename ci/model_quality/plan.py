"""CLI for model selection and Buildkite pipeline generation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .buildkite import render_pipeline
from .config import RUN_MODES, ConfigError, load_model_config
from .planner import build_execution_plan
from .state import atomic_write_json, runs_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--git-sha", default=os.getenv("BUILDKITE_COMMIT", "local"))
    parser.add_argument(
        "--max-gpu-hours",
        type=float,
        default=float(os.getenv("MAX_GPU_HOURS", "80")),
    )
    parser.add_argument("--model-filter", default=os.getenv("MODEL_FILTER", "all"))
    parser.add_argument(
        "--priority-override",
        default=os.getenv("PRIORITY_OVERRIDE", "none"),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--attempt-id")
    parser.add_argument(
        "--run-mode",
        choices=RUN_MODES,
        default=os.getenv("RUN_MODE", "quantize"),
    )
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--pipeline-output", type=Path)
    parser.add_argument("--persist-plan", action="store_true")
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.run_mode in {"eval_only", "upload_only"} and not args.run_id:
        print(
            f"--run-id is required for {args.run_mode} so an existing artifact is used",
            file=sys.stderr,
        )
        return 2
    try:
        config = load_model_config(args.config)
        plan = build_execution_plan(
            config,
            git_sha=args.git_sha,
            max_gpu_hours=args.max_gpu_hours,
            model_filter=args.model_filter,
            priority_override=args.priority_override,
            run_mode=args.run_mode,
            run_id=args.run_id,
            attempt_id=args.attempt_id,
        )
        pipeline = render_pipeline(plan, config_path=str(args.config))
    except ConfigError as error:
        print(f"model quality configuration error: {error}", file=sys.stderr)
        return 2

    plan_json = json.dumps(plan, indent=2, sort_keys=True)
    if args.persist_plan:
        atomic_write_json(runs_root() / plan["run_id"] / "execution-plan.json", plan)
    if args.plan_output:
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(plan_json + "\n", encoding="utf-8")
    else:
        print(plan_json)

    if args.pipeline_output:
        args.pipeline_output.parent.mkdir(parents=True, exist_ok=True)
        args.pipeline_output.write_text(pipeline, encoding="utf-8")
    if args.upload:
        subprocess.run(
            ["buildkite-agent", "pipeline", "upload"],
            input=pipeline,
            text=True,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
