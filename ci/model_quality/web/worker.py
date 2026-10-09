"""Execute queued model quality jobs with an allowlisted stage CLI.

The web API never spawns a process inside a request. It records jobs; this
worker drains the queue and is the only component that runs the reviewed
``ci.model_quality.stage`` command line.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from ..state import write_stage_result
from .audit import AuditLog, ValidationError
from .jobs import (
    TERMINAL_JOB_STATUSES,
    JobStore,
    ReservationLedger,
    settle_model_group,
)
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


def job_values(job: dict[str, Any], root: Path) -> dict[str, str]:
    model_dir = root / job["run_id"] / job["model_id"]
    reports_dir = model_dir / "attempts" / job["attempt_id"] / "reports"
    return {
        "{run_id}": job["run_id"],
        "{model_id}": job["model_id"],
        "{run_dir}": str(model_dir),
        "{output_dir}": str(model_dir / "model"),
        "{work_dir}": str(model_dir / "work"),
        "{reports_dir}": str(reports_dir),
        "{source_path}": str(job.get("details", {}).get("source_path", "")),
    }


def resolve_tokens(values: list[Any], replacements: dict[str, str]) -> list[str]:
    result = []
    for value in values:
        token = str(value)
        for placeholder, replacement in replacements.items():
            token = token.replace(placeholder, replacement)
        result.append(token)
    return result


def _effective_config(
    job: dict[str, Any], root: Path, config_path: str | Path
) -> str | Path:
    """Prefer a run-local manifest so a script-first run executes its own argv.

    ``control.ControlPlane._launch`` writes ``<run_dir>/model-config.yaml`` for
    plans that carry a synthesized config. When present, the stage CLI reloads
    ``workflow.quantize`` from it instead of the worker's global ``--config``.
    """

    run_id = job.get("run_id")
    if run_id:
        run_cfg = root / str(run_id) / "model-config.yaml"
        if run_cfg.is_file():
            return run_cfg
    return config_path


def execution_argv(
    job: dict[str, Any],
    stage: str,
    *,
    root: Path,
    config_path: str | Path,
    argv_prefix: tuple[str, ...],
    container_runtime: tuple[str, ...],
) -> list[str]:
    if stage == "runtime-smoke" and job.get("container_name"):
        values = job_values(job, root)
        arguments = resolve_tokens(
            job.get("details", {}).get("inference_arguments", []), values
        )
        # The built-in harness wraps the user's serve script: it injects the
        # run-scoped model/report paths as env vars (model_dir / MODEL_DIR /
        # output_dir / reports_dir), launches the serve script unchanged, waits
        # for 127.0.0.1:<port>, probes it, and writes the PASS result JSON. This
        # keeps existing env-based serve scripts working without rewriting them.
        harness = str(
            Path(__file__).resolve().parents[1]
            / "entrypoints"
            / "runtime_smoke_harness.sh"
        )
        port = str(job.get("details", {}).get("inference_port", 8025))
        smoke = [
            "bash",
            harness,
            str(job["script"]),
            port,
            values["{reports_dir}"],
            values["{output_dir}"],
            *arguments,
        ]
        if container_runtime == ("direct",):
            return smoke
        return [*container_runtime, "exec", str(job["container_name"]), *smoke]
    effective_config = _effective_config(job, root, config_path)
    return stage_argv(job, stage, config_path=effective_config, argv_prefix=argv_prefix)


def _stage_artifact_fingerprint(job: dict[str, Any], root: Path) -> str | None:
    """Return the artifact identity the other stages record for this run.

    ``ci.model_quality.stage`` rebinds the planner fingerprint carried by the
    job to the finalized fingerprint that preflight wrote into the run's input
    manifest. The container smoke runs the harness directly instead of that
    CLI, so it has to apply the same rebinding -- recording the planner value
    here makes the report discard the smoke as stale and fail the run.
    """

    manifest_path = (
        Path(root)
        / safe_component(job["run_id"], "run_id")
        / safe_component(job["model_id"], "model_id")
        / "input-manifest.json"
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return job.get("artifact_fingerprint")
    if isinstance(manifest, dict) and manifest.get("finalized_artifact_fingerprint"):
        return str(manifest["finalized_artifact_fingerprint"])
    return job.get("artifact_fingerprint")


def record_inference_result(job: dict[str, Any], root: Path) -> None:
    result_file = job.get("result_file")
    if not result_file:
        raise ValidationError("inference job has no result_file")
    resolved = resolve_tokens([result_file], job_values(job, root))[0]
    result_path = Path(resolved)
    if not result_path.is_file():
        raise ValidationError(f"inference result is missing: {result_path}")
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"invalid inference result: {error}") from error
    if payload.get("status") != "PASS":
        raise ValidationError("inference result status is not PASS")
    write_stage_result(
        job["run_id"],
        job["model_id"],
        "runtime-smoke",
        {"status": "PASS", "result": payload, "container_name": job["container_name"]},
        attempt_id=job["attempt_id"],
        artifact_fingerprint=_stage_artifact_fingerprint(job, root),
        evaluation_fingerprint=job.get("evaluation_fingerprint"),
    )


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


# 128 + SIGTERM: distinct from a plain non-zero exit so a canceled stage is
# never mistaken for an organic failure when the record is read back.
_CANCELED_RETURNCODE = 143


class _ProcessResult:
    """Expose ``returncode`` like ``subprocess.CompletedProcess`` does."""

    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


def _terminate_process_group(proc: subprocess.Popen, grace_seconds: float) -> None:
    """SIGTERM the stage's process group, escalating to SIGKILL after a grace."""

    try:
        group = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=grace_seconds)
            return
        except subprocess.TimeoutExpired:
            continue


def run_cancellable(
    argv: list[str],
    *,
    is_canceled: Callable[[], bool],
    poll_interval: float = 1.0,
    grace_seconds: float = 10.0,
    timeout: float | None = None,
    **popen_kwargs: Any,
):
    """Run a stage, killing its process group when the job is canceled.

    The on-disk job record is the cancellation signal: ``cancel_job`` flips the
    status to CANCELED, and this runner notices on the next poll and sends
    SIGTERM (then SIGKILL after a grace period) to the whole session so a
    RUNNING stage is actually stopped — not merely marked. ``start_new_session``
    gives the child its own process group so the signal reaches any helpers it
    spawned. Raises ``subprocess.TimeoutExpired`` like ``subprocess.run`` so the
    caller classifies a timeout identically.
    """

    deadline = None if timeout is None else time.monotonic() + timeout
    proc = subprocess.Popen(argv, start_new_session=True, **popen_kwargs)
    while True:
        wait_for = poll_interval
        if deadline is not None:
            wait_for = min(poll_interval, max(0.0, deadline - time.monotonic()))
        try:
            return _ProcessResult(proc.wait(timeout=wait_for))
        except subprocess.TimeoutExpired:
            pass
        if is_canceled():
            _terminate_process_group(proc, grace_seconds)
            return _ProcessResult(_CANCELED_RETURNCODE)
        if deadline is not None and time.monotonic() >= deadline:
            _terminate_process_group(proc, grace_seconds)
            raise subprocess.TimeoutExpired(argv, timeout)


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
    container_runtime: tuple[str, ...] = ("docker",),
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
    repository_root = str(Path(__file__).parents[3])
    existing_pythonpath = run_env.get("PYTHONPATH")
    run_env["PYTHONPATH"] = (
        repository_root
        if not existing_pythonpath
        else repository_root + os.pathsep + existing_pythonpath
    )
    evaluation_command = job.get("details", {}).get("evaluation_command")
    if evaluation_command:
        run_env["MODEL_QUALITY_EVALUATION_COMMAND_JSON"] = json.dumps(
            evaluation_command
        )

    claim_reservation(ledger, job)
    current = jobs.update(job, status="RUNNING")

    def _is_canceled() -> bool:
        try:
            latest = jobs.get(job["run_id"], job["model_id"], job["job_id"])
        except NotFoundError:
            return False
        return latest.get("status") == "CANCELED"

    # Only the built-in runner learns to kill a live subprocess; an injected
    # runner (tests, alternate executors) is used verbatim so the seam holds.
    run_stage = runner
    if runner is subprocess.run:
        run_stage = functools.partial(run_cancellable, is_canceled=_is_canceled)

    exit_code = 0
    failed_stage = None
    message = ""
    for stage in stages:
        # A Stop between stages lands as a CANCELED record; honor it before
        # spending the next stage's reservation.
        if _is_canceled():
            break
        argv = execution_argv(
            current,
            stage,
            root=root,
            config_path=config_path,
            argv_prefix=argv_prefix,
            container_runtime=container_runtime,
        )
        log_path = log_dir / STAGE_LOG_FILENAMES.get(stage, f"{stage}.log")
        try:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"$ {shlex.join(argv)}\n")
                log.flush()
                completed = run_stage(
                    argv,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=run_env,
                    cwd=repository_root,
                    timeout=timeout_seconds,
                )
            exit_code = int(completed.returncode)
        except subprocess.TimeoutExpired:
            exit_code = 124
            message = f"{stage} exceeded its timeout"
        except FileNotFoundError as error:
            exit_code = 127
            message = f"{stage} could not start: {error}"
        except OSError as error:
            # e.g. the log directory became unwritable; treat as a transient
            # infrastructure failure so the job is settled (and its reservation
            # released) instead of being left RUNNING forever.
            exit_code = 1
            message = f"{stage} could not run: {error}"
        except Exception as error:  # noqa: BLE001 - never leak a RUNNING job
            exit_code = 1
            message = f"{stage} failed unexpectedly: {error}"
        if exit_code != 0:
            failed_stage = stage
            message = message or f"{stage} exited with {exit_code}"
            break
        if stage == "runtime-smoke" and current.get("container_name"):
            try:
                record_inference_result(current, root)
            except ValidationError as error:
                exit_code = 1
                failed_stage = stage
                message = str(error)
                break
            except Exception as error:  # noqa: BLE001 - never leak a RUNNING job
                exit_code = 1
                failed_stage = stage
                message = f"runtime-smoke result could not be recorded: {error}"
                break

    # Cancellation may have landed while the last stage ran (the runner kills
    # the subprocess) or between stages. The on-disk record is authoritative:
    # never overwrite a CANCELED job back to SUCCEEDED/FAILED — just release its
    # reservation and report the canceled record.
    latest = jobs.get(job["run_id"], job["model_id"], job["job_id"])
    if latest.get("status") == "CANCELED":
        settle_model_group(
            jobs, ledger, latest, reason="canceled", cancel_active=True
        )
        return latest

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


def _dependencies_settled(job: dict[str, Any], jobs: JobStore) -> bool:
    """True when every job created before ``job`` for the same attempt is done.

    A single worker drains the queue oldest-first, which is what makes the
    per-run stage order hold. Deployments that split the queue across workers
    (for example one worker per job kind) would otherwise start ``report``
    while the runtime smoke is still bringing the server up, fail it on the
    missing stages, and cancel the very stage the operator asked to re-run.
    """

    def order(entry: dict[str, Any]) -> tuple[str, str]:
        return (str(entry.get("created_at") or ""), str(entry.get("job_id") or ""))

    mine = order(job)
    siblings = jobs.list_jobs(
        job["run_id"], model_id=job["model_id"], attempt_id=job.get("attempt_id")
    )
    return all(
        other.get("status") in TERMINAL_JOB_STATUSES or order(other) >= mine
        for other in siblings
    )


def run_once(
    *,
    root: str | Path,
    config_path: str | Path,
    jobs: JobStore | None = None,
    ledger: ReservationLedger | None = None,
    notifications: NotificationService | None = None,
    argv_prefix: tuple[str, ...] = ("python3", "-m", "ci.model_quality.stage"),
    container_runtime: tuple[str, ...] = ("docker",),
    env: dict[str, str] | None = None,
    timeout_seconds: float | None = None,
    capacity_gpu_hours: float = 80.0,
    kinds: frozenset[str] | None = None,
) -> dict[str, Any] | None:
    """Execute the oldest queued job, or return ``None`` when idle."""

    jobs = jobs or JobStore(root)
    ledger = ledger or ReservationLedger(root, capacity_gpu_hours=capacity_gpu_hours)
    queued = [
        job for job in jobs.queued() if kinds is None or job.get("kind") in kinds
    ]
    ready = next((job for job in queued if _dependencies_settled(job, jobs)), None)
    if ready is None:
        return None
    return execute_job(
        ready,
        root=root,
        config_path=config_path,
        jobs=jobs,
        ledger=ledger,
        notifications=notifications,
        argv_prefix=argv_prefix,
        container_runtime=container_runtime,
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
        "--container-runtime",
        default=os.getenv("MODEL_QUALITY_CONTAINER_RUNTIME", "docker"),
    )
    parser.add_argument("--poll-seconds", type=float, default=0.0)
    parser.add_argument(
        "--kinds",
        default="",
        help="comma-separated job kinds handled by this worker; empty means all",
    )
    parser.add_argument(
        "--capacity-gpu-hours",
        type=float,
        default=float(os.getenv("MODEL_QUALITY_GPU_HOUR_CAPACITY", "80")),
    )
    args = parser.parse_args(argv)
    if not args.runs_root:
        print("--runs-root or MODEL_QUALITY_RUNS_ROOT is required", file=sys.stderr)
        return 2
    # In-process helpers such as ``write_stage_result`` resolve the runs root
    # from the environment while the worker drains ``--runs-root``. Keep both on
    # the same directory; otherwise stage records land in the relative default
    # instead of the configured store and the report sees them as missing.
    os.environ["MODEL_QUALITY_RUNS_ROOT"] = str(args.runs_root)
    jobs = JobStore(args.runs_root)
    ledger = ReservationLedger(
        args.runs_root, capacity_gpu_hours=args.capacity_gpu_hours
    )
    prefix = parse_argv_prefix(args.stage_cli)
    container_runtime = parse_argv_prefix(args.container_runtime)
    notifications = NotificationService(args.runs_root)
    kinds = frozenset(token.strip() for token in args.kinds.split(",") if token.strip())
    completed = 0
    while True:
        result = run_once(
            root=args.runs_root,
            config_path=args.config,
            jobs=jobs,
            ledger=ledger,
            notifications=notifications,
            argv_prefix=prefix,
            container_runtime=container_runtime,
            timeout_seconds=args.timeout_seconds,
            capacity_gpu_hours=args.capacity_gpu_hours,
            kinds=kinds or None,
        )
        if result is None:
            if args.poll_seconds > 0:
                import time

                time.sleep(args.poll_seconds)
                continue
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
