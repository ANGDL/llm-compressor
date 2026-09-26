"""Executor adapters that turn a reviewed plan into queued work."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..buildkite import render_pipeline
from ..state import utc_now
from .audit import ValidationError


@dataclass(frozen=True)
class LaunchSpec:
    """Everything an executor adapter needs to schedule a reviewed plan."""

    run_id: str
    attempt_id: str
    run_mode: str
    git_sha: str
    plan: dict[str, Any]
    inference: dict[str, Any] | None = None
    quantization_argv: list[str] | None = None


@dataclass(frozen=True)
class DispatchResult:
    executor: str
    dispatched: bool
    mode: str
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "executor": self.executor,
            "dispatched": self.dispatched,
            "mode": self.mode,
            "detail": self.detail,
        }


class ExecutorAdapter(Protocol):
    name: str

    def dispatch(
        self, spec: LaunchSpec, jobs: list[dict[str, Any]]
    ) -> DispatchResult: ...

    def cancel(self, *, run_id: str, reason: str) -> DispatchResult: ...


class QueueExecutor:
    """Record jobs for the local worker; never start a process in-request."""

    name = "queue"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()

    def dispatch(self, spec: LaunchSpec, jobs: list[dict[str, Any]]) -> DispatchResult:
        return DispatchResult(
            executor=self.name,
            dispatched=False,
            mode="queued",
            detail={
                "queued_jobs": [job["job_id"] for job in jobs],
                "worker_command": (
                    "python3 -m ci.model_quality.web.worker "
                    f"--runs-root {self.root} --once"
                ),
            },
        )

    def cancel(self, *, run_id: str, reason: str) -> DispatchResult:
        return DispatchResult(
            executor=self.name,
            dispatched=False,
            mode="record-only",
            detail={"run_id": run_id, "reason": reason},
        )


class DryRunExecutor(QueueExecutor):
    """Queues nothing; used to validate the control plane end to end."""

    name = "dry-run"

    def dispatch(self, spec: LaunchSpec, jobs: list[dict[str, Any]]) -> DispatchResult:
        return DispatchResult(
            executor=self.name,
            dispatched=False,
            mode="dry-run",
            detail={"queued_jobs": [job["job_id"] for job in jobs]},
        )


class BuildkiteExecutor:
    """Render the reviewed plan as a Buildkite pipeline.

    Uploading is opt-in: the adapter renders and stores the YAML by default so
    a misconfigured deployment cannot start GPU work, and only calls
    ``buildkite-agent pipeline upload`` when explicitly enabled.
    """

    name = "buildkite"

    def __init__(
        self,
        root: str | Path,
        *,
        config_path: str | Path,
        upload: bool = False,
        agent_command: tuple[str, ...] = ("buildkite-agent", "pipeline", "upload"),
        timeout_seconds: float = 60.0,
        env: dict[str, str] | None = None,
        runner=subprocess.run,
    ) -> None:
        self.root = Path(root).expanduser()
        self.config_path = str(config_path)
        self.upload = bool(upload)
        self.agent_command = tuple(agent_command)
        self.timeout_seconds = float(timeout_seconds)
        self.env = env
        self.runner = runner

    def dispatch(self, spec: LaunchSpec, jobs: list[dict[str, Any]]) -> DispatchResult:
        pipeline = render_pipeline(spec.plan, config_path=self.config_path)
        directory = self.root / "_jobs" / "pipelines"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{spec.run_id}-{spec.attempt_id}.yml"
        path.write_text(pipeline, encoding="utf-8")
        detail: dict[str, Any] = {
            "pipeline_path": str(path),
            "step_count": pipeline.count("label:"),
        }
        if not self.upload:
            return DispatchResult(
                executor=self.name,
                dispatched=False,
                mode="render-only",
                detail=detail,
            )
        completed = self.runner(
            list(self.agent_command),
            input=pipeline,
            text=True,
            capture_output=True,
            env=self.env,
            timeout=self.timeout_seconds,
        )
        detail.update(
            {
                "returncode": completed.returncode,
                "stdout": (completed.stdout or "")[-2000:],
                "stderr": (completed.stderr or "")[-2000:],
                "uploaded_at": utc_now(),
            }
        )
        if completed.returncode != 0:
            return DispatchResult(
                executor=self.name,
                dispatched=False,
                mode="upload-failed",
                detail=detail,
            )
        return DispatchResult(
            executor=self.name, dispatched=True, mode="uploaded", detail=detail
        )

    def cancel(self, *, run_id: str, reason: str) -> DispatchResult:
        token = os.getenv("MODEL_QUALITY_BUILDKITE_TOKEN")
        api_url = os.getenv("MODEL_QUALITY_BUILDKITE_API_URL")
        if not token or not api_url:
            return DispatchResult(
                executor=self.name,
                dispatched=False,
                mode="record-only",
                detail={
                    "run_id": run_id,
                    "reason": reason,
                    "message": (
                        "Buildkite cancellation needs "
                        "MODEL_QUALITY_BUILDKITE_TOKEN and "
                        "MODEL_QUALITY_BUILDKITE_API_URL; the intent is recorded"
                    ),
                },
            )
        return DispatchResult(
            executor=self.name,
            dispatched=False,
            mode="not-implemented",
            detail={
                "run_id": run_id,
                "reason": reason,
                "message": (
                    "Buildkite REST cancellation is not wired up in this "
                    "deployment; cancel the build in Buildkite and the recorded "
                    "intent keeps the audit trail complete"
                ),
            },
        )


def create_executor(
    name: str,
    *,
    root: str | Path,
    config_path: str | Path | None,
    upload: bool = False,
) -> ExecutorAdapter:
    """Build the configured executor adapter."""

    normalized = (name or "queue").strip().lower()
    if normalized in {"dry-run", "dry_run", "none"}:
        return DryRunExecutor(root)
    if normalized == "queue":
        return QueueExecutor(root)
    if normalized == "buildkite":
        if config_path is None:
            raise ValidationError(
                "the buildkite executor needs a model quality --config path"
            )
        return BuildkiteExecutor(root, config_path=config_path, upload=upload)
    raise ValidationError(f"unknown executor: {name!r}")


def describe_pipeline(pipeline: str) -> dict[str, Any]:
    """Summarize a rendered pipeline for API responses and logs."""

    labels = [
        line.split("label:", 1)[1].strip()
        for line in pipeline.splitlines()
        if "label:" in line
    ]
    return {"step_count": len(labels), "labels": labels}


def dump_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"
