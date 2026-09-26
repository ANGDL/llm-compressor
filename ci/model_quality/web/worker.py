"""Execute queued model quality jobs with an allowlisted stage CLI.

The web API never spawns a process inside a request. It records jobs; this
worker drains the queue and is the only component that runs the reviewed
``ci.model_quality.stage`` command line.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from .audit import AuditLog, ValidationError
from .jobs import JobStore, ReservationLedger, settle_model_group
from .ops import NotificationService
from .store import NotFoundError, safe_component

STAGE_LOG_FILENAMES = {
    "preflight": "preflight.log",
    "quantize": "quantization.log",
    "validate": "validation.log",
    "runtime-smoke": "runtime-smoke.log",
    "evaluate": "evaluation.log",
    "report": "report.log",
    "publish": "publish.log",
}
QUALITY_STAGES = {"evaluate", "report"}


def stage_argv(
    job: dict[str, Any],
    stage: str,
    *,
    config_path: str | Path,
    argv_prefix: tuple[str, ...],
) -> list[str]:
    """Build the fixed, positional stage command for one job stage."""

    argv = [
        *argv_prefix,
        "--config",
        str(config_path),
        "--model",
        job["model_id"],
        "--run-id",
        job["run_id"],
        "--attempt-id",
        job["attempt_id"],
        "--stage",
        stage,
        "--run-mode",
        str(job.get("details", {}).get("run_mode", "quantize")),
        "--git-sha",
        str(job.get("details", {}).get("git_sha", "local")),
    ]
    if job.get("artifact_fingerprint"):
        argv += ["--fingerprint", str(job["artifact_fingerprint"])]
    if job.get("evaluation_fingerprint"):
        argv += ["--evaluation-fingerprint", str(job["evaluation_fingerprint"])]
    return argv


def classify_failure(stage: str, returncode: int, message: str) -> str:
    if returncode == 124:
        return "RESOURCE_TIMEOUT"
    if "No such file or directory" in message:
        return "RESOURCE_UNAVAILABLE"
    if stage in QUALITY_STAGES:
        return "QUALITY_GATE_FAILED"
    return "SCRIPT_FAILED"


def claim_reservation(ledger: ReservationLedger, job: dict[str, Any]) -> None:
    """Show the reservation as QUEUED while its first job is running."""

    reservation_id = job.get("reservation_id")
    if not reservation_id:
        return
    try:
        ledger.mark_queued(str(reservation_id), job["job_id"])
    except (NotFoundError, ValidationError):
        return


def execute_job(
    job: dict[str, Any],
    *,
    root: str | Path,
    config_path: str | Path,
    jobs: JobStore,
    ledger: ReservationLedger,
    audit: AuditLog | None = None,
    notifications: NotificationService | None = None,
    argv_prefix: tuple[str, ...] = ("python3", "-m", "ci.model_quality.stage"),
    env: dict[str, str] | None = None,
    timeout_seconds: float | None = None,
    runner=subprocess.run,
) -> dict[str, Any]:
    """Run every stage of a queued job and record the outcome."""

    root = Path(root).expanduser()
    stages = list(job.get("stages") or [])
    if not stages:
        raise ValidationError(f"job {job['job_id']!r} has no stages to run")
    model_dir = (
        root
        / safe_component(job["run_id"], "run_id")
        / safe_component(job["model_id"], "model_id")
    )
    log_dir = model_dir / "logs" / safe_component(job["attempt_id"], "attempt_id")
    log_dir.mkdir(parents=True, exist_ok=True)
    run_env = {**os.environ, **(env or {})}
    run_env["MODEL_QUALITY_RUNS_ROOT"] = str(root)
    run_env.pop("BUILDKITE", None)

    claim_reservation(ledger, job)
    current = jobs.update(job, status="RUNNING")
    exit_code = 0
    failed_stage = None
    message = ""
    for stage in stages:
        argv = stage_argv(
            current, stage, config_path=config_path, argv_prefix=argv_prefix
        )
        log_path = log_dir / STAGE_LOG_FILENAMES.get(stage, f"{stage}.log")
        try:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"$ {shlex.join(argv)}\n")
                log.flush()
                completed = runner(
                    argv,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=run_env,
                    cwd=str(root),
                    timeout=timeout_seconds,
                )
            exit_code = int(completed.returncode)
        except subprocess.TimeoutExpired:
            exit_code = 124
            message = f"{stage} exceeded its timeout"
        except FileNotFoundError as error:
            exit_code = 127
            message = f"{stage} could not start: {error}"
        if exit_code != 0:
            failed_stage = stage
            message = message or f"{stage} exited with {exit_code}"
            break

    if failed_stage is None:
        finished = jobs.update(
            current,
            status="SUCCEEDED",
            exit_code=0,
            details={
                **current.get("details", {}),
                "log_dir": str(log_dir),
                "completed_stages": stages,
            },
        )
        settle_model_group(
            jobs, ledger, finished, reason="completed", cancel_active=False
        )
        return finished

    failure_class = classify_failure(failed_stage, exit_code, message)
    finished = jobs.update(
        current,
        status="FAILED",
        exit_code=exit_code,
        failure_class=failure_class,
        details={
            **current.get("details", {}),
            "log_dir": str(log_dir),
            "failed_stage": failed_stage,
            "message": message,
        },
    )
    settle_model_group(
        jobs,
        ledger,
        finished,
        reason=f"failed: {failure_class}",
        cancel_active=True,
    )
    audit = audit or AuditLog(root)
    audit.record(
        actor="worker",
        action="job.failed",
        target=f"{job['run_id']}/{job['model_id']}/{job['job_id']}",
        result="FAILED",
        details={
            "stage": failed_stage,
            "failure_class": failure_class,
            "exit_code": exit_code,
            "message": message,
        },
    )
    notifications = notifications or NotificationService(root)
    event = "run.publish_failed" if failed_stage == "publish" else "run.failed"
    notifications.notify(
        event,
        {
            "run_id": job["run_id"],
            "model_id": job["model_id"],
            "job_id": job["job_id"],
            "stage": failed_stage,
            "failure_class": failure_class,
            "exit_code": exit_code,
        },
    )
    return finished


def run_once(
    *,
    root: str | Path,
    config_path: str | Path,
    jobs: JobStore | None = None,
    ledger: ReservationLedger | None = None,
    notifications: NotificationService | None = None,
    argv_prefix: tuple[str, ...] = ("python3", "-m", "ci.model_quality.stage"),
    env: dict[str, str] | None = None,
    timeout_seconds: float | None = None,
    capacity_gpu_hours: float = 80.0,
) -> dict[str, Any] | None:
    """Execute the oldest queued job, or return ``None`` when idle."""

    jobs = jobs or JobStore(root)
    ledger = ledger or ReservationLedger(root, capacity_gpu_hours=capacity_gpu_hours)
    queued = jobs.queued()
    if not queued:
        return None
    return execute_job(
        queued[0],
        root=root,
        config_path=config_path,
        jobs=jobs,
        ledger=ledger,
        notifications=notifications,
        argv_prefix=argv_prefix,
        env=env,
        timeout_seconds=timeout_seconds,
    )


def parse_argv_prefix(value: str) -> tuple[str, ...]:
    tokens = tuple(shlex.split(value))
    if not tokens:
        raise ValidationError("stage CLI must not be empty")
    return tokens


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run queued model quality jobs")
    parser.add_argument("--runs-root", default=os.getenv("MODEL_QUALITY_RUNS_ROOT"))
    parser.add_argument("--config", required=True, help="model quality manifest")
    parser.add_argument("--once", action="store_true", help="run one job and exit")
    parser.add_argument("--max-jobs", type=int, default=0, help="0 means unlimited")
    parser.add_argument(
        "--stage-cli",
        default=os.getenv(
            "MODEL_QUALITY_WEB_STAGE_CLI", "python3 -m ci.model_quality.stage"
        ),
    )
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument(
        "--capacity-gpu-hours",
        type=float,
        default=float(os.getenv("MODEL_QUALITY_GPU_HOUR_CAPACITY", "80")),
    )
    args = parser.parse_args(argv)
    if not args.runs_root:
        print("--runs-root or MODEL_QUALITY_RUNS_ROOT is required", file=sys.stderr)
        return 2
    jobs = JobStore(args.runs_root)
    ledger = ReservationLedger(
        args.runs_root, capacity_gpu_hours=args.capacity_gpu_hours
    )
    prefix = parse_argv_prefix(args.stage_cli)
    notifications = NotificationService(args.runs_root)
    completed = 0
    while True:
        result = run_once(
            root=args.runs_root,
            config_path=args.config,
            jobs=jobs,
            ledger=ledger,
            notifications=notifications,
            argv_prefix=prefix,
            timeout_seconds=args.timeout_seconds,
            capacity_gpu_hours=args.capacity_gpu_hours,
        )
        if result is None:
            break
        completed += 1
        print(
            f"{result['job_id']} {result['status']} "
            f"({result.get('failure_class') or 'ok'})"
        )
        if args.once or (args.max_jobs and completed >= args.max_jobs):
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
