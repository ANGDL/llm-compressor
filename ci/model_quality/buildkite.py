"""Buildkite dynamic pipeline rendering."""

from __future__ import annotations

import json
import re
import shlex
from typing import Any

import yaml


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")


def render_pipeline(plan: dict[str, Any], *, config_path: str) -> str:
    """Render selected jobs as a staged Buildkite pipeline."""

    steps: list[dict[str, Any]] = []
    terminal_keys = []
    for job in plan["selected"]:
        model_id = job["id"]
        prefix = _key(model_id)
        common = [
            "python3",
            "-m",
            "ci.model_quality.stage",
            "--config",
            config_path,
            "--model",
            model_id,
            "--run-id",
            plan["run_id"],
            "--attempt-id",
            plan["attempt_id"],
            "--run-mode",
            plan["run_mode"],
            "--git-sha",
            plan["git_sha"],
            "--fingerprint",
            job["fingerprint"],
            "--evaluation-fingerprint",
            job["evaluation_fingerprint"],
        ]

        previous = None
        stages_by_mode = {
            "quantize": ("preflight", "quantize", "validate", "runtime-smoke"),
            "quantize_and_eval": (
                "preflight",
                "quantize",
                "validate",
                "runtime-smoke",
                "evaluate",
            ),
            "eval_only": ("preflight", "validate", "runtime-smoke", "evaluate"),
            # Revalidate and reconstruct the report before publishing an existing run.
            "upload_only": ("preflight", "validate", "runtime-smoke"),
        }
        for stage in stages_by_mode[plan["run_mode"]]:
            key = f"{prefix}-{stage}"
            step: dict[str, Any] = {
                "label": f":gear: {model_id} | {stage}",
                "key": key,
                "command": shlex.join(common + ["--stage", stage]),
                "agents": {"queue": "model-quality-gpu"},
                "concurrency": 1,
                "concurrency_group": "llm-compressor/model-quality-host",
            }
            if previous is not None:
                step["depends_on"] = previous
            steps.append(step)
            previous = key

        report_key = f"{prefix}-report"
        report_step: dict[str, Any] = {
            "label": f":memo: {model_id} | report",
            "key": report_key,
            "command": shlex.join(common + ["--stage", "report"]),
            "allow_dependency_failure": True,
            "agents": {"queue": "model-quality-gpu"},
            "concurrency": 1,
            "concurrency_group": "llm-compressor/model-quality-host",
        }
        if previous is not None:
            report_step["depends_on"] = previous
        steps.append(report_step)
        previous = report_key
        upload_requested = job["upload_enabled"] or plan["run_mode"] == "upload_only"
        if upload_requested:
            publish_key = f"{prefix}-publish"
            steps.append(
                {
                    "label": f":arrow_up: {model_id} | publish",
                    "key": publish_key,
                    "command": shlex.join(common + ["--stage", "publish"]),
                    "agents": {"queue": "model-quality-gpu"},
                    "concurrency": 1,
                    "concurrency_group": "llm-compressor/model-quality-host",
                }
            )
            if previous is not None:
                steps[-1]["depends_on"] = previous
            terminal_keys.append(publish_key)
        else:
            terminal_keys.append(previous)

    for job in plan["deferred"]:
        model_id = job["id"]
        steps.append(
            {
                "label": f":hourglass_flowing_sand: {model_id} | deferred",
                "key": f"{_key(model_id)}-deferred",
                "command": (
                    "echo "
                    + shlex.quote(f"{model_id}: DEFERRED_BUDGET - {job['reason']}")
                ),
                "agents": {"queue": "model-quality-gpu"},
            }
        )

    summary_command = [
        "python3",
        "-m",
        "ci.model_quality.stage",
        "--config",
        config_path,
        "--run-id",
        plan["run_id"],
        "--attempt-id",
        plan["attempt_id"],
        "--stage",
        "aggregate",
        "--run-mode",
        plan["run_mode"],
        "--git-sha",
        plan["git_sha"],
        "--models-json",
        json.dumps([job["id"] for job in plan["selected"]]),
    ]
    summary: dict[str, Any] = {
        "label": ":bar_chart: model quality summary",
        "key": "model-quality-summary",
        "command": shlex.join(summary_command),
        "agents": {"queue": "model-quality-gpu"},
    }
    terminal_keys = [key for key in terminal_keys if key is not None]
    if terminal_keys:
        summary["depends_on"] = terminal_keys
        summary["allow_dependency_failure"] = True
    steps.append(summary)
    return yaml.safe_dump({"steps": steps}, sort_keys=False)
