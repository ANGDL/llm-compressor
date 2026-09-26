"""Job records and GPU-hour reservations for controlled execution."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from ..state import atomic_write_json, utc_now
from .audit import (
    ConflictError,
    ValidationError,
    exclusive_lock,
    timestamp_token,
)
from .store import NotFoundError, read_json, safe_component

JOB_STATUSES = (
    "CREATED",
    "QUEUED",
    "ALLOCATED",
    "RUNNING",
    "SUCCEEDED",
    "FAILED",
    "EXPIRED",
    "CANCELED",
)
TERMINAL_JOB_STATUSES = frozenset({"SUCCEEDED", "FAILED", "EXPIRED", "CANCELED"})
ACTIVE_JOB_STATUSES = frozenset({"CREATED", "QUEUED", "ALLOCATED", "RUNNING"})

JOB_KINDS = (
    "quantize",
    "inference",
    "evaluate",
    "report",
    "publish",
    "backup",
    "restore",
)

# Resource and handoff failures are retryable; a quality failure is not,
# because retrying it would only re-run a gate that already rejected the
# artifact. They must never be conflated in the UI.
FAILURE_CLASSES = (
    "RESOURCE_UNAVAILABLE",
    "RESOURCE_TIMEOUT",
    "CONTAINER_UNAVAILABLE",
    "SCRIPT_FAILED",
    "ARTIFACT_HANDOFF_FAILED",
    "DEPENDENCY_FAILED",
    "CONFIG_ERROR",
    "QUALITY_GATE_FAILED",
    "CANCELED",
)
RETRYABLE_FAILURE_CLASSES = frozenset(
    {
        "RESOURCE_UNAVAILABLE",
        "RESOURCE_TIMEOUT",
        "CONTAINER_UNAVAILABLE",
        "ARTIFACT_HANDOFF_FAILED",
    }
)


def _timestamp(value: Any) -> float:
    if not isinstance(value, str):
        return float("-inf")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return float("-inf")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


class JobStore:
    """Record quantization, inference, evaluation, and publish jobs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()

    def _jobs_dir(self, run_id: str, model_id: str) -> Path:
        return (
            self.root
            / safe_component(run_id, "run_id")
            / safe_component(model_id, "model_id")
            / "jobs"
        )

    def create(
        self,
        *,
        run_id: str,
        model_id: str,
        attempt_id: str,
        kind: str,
        executor: str,
        command: list[str] | None = None,
        queue: str | None = None,
        gpu_count: int | None = None,
        gpu_hours: float | None = None,
        stages: list[str] | None = None,
        container_name: str | None = None,
        script: str | None = None,
        result_file: str | None = None,
        artifact_fingerprint: str | None = None,
        evaluation_fingerprint: str | None = None,
        reservation_id: str | None = None,
        status: str = "QUEUED",
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if kind not in JOB_KINDS:
            raise ValidationError(f"unknown job kind: {kind!r}")
        if status not in JOB_STATUSES:
            raise ValidationError(f"unknown job status: {status!r}")
        safe_component(attempt_id, "attempt_id")
        job_id = f"job-{timestamp_token()}-{uuid4().hex[:8]}"
        job = {
            "schema_version": 1,
            "job_id": job_id,
            "run_id": run_id,
            "model_id": model_id,
            "attempt_id": attempt_id,
            "kind": kind,
            "status": status,
            "executor": executor,
            "queue": queue,
            "gpu_count": gpu_count,
            "gpu_hours": gpu_hours,
            "stages": list(stages or []),
            "command": list(command or []),
            "container_name": container_name,
            "script": script,
            "result_file": result_file,
            "artifact_fingerprint": artifact_fingerprint,
            "evaluation_fingerprint": evaluation_fingerprint,
            "reservation_id": reservation_id,
            "failure_class": None,
            "exit_code": None,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "details": details or {},
        }
        self._write(job)
        return job

    def _write(self, job: dict[str, Any]) -> None:
        path = self._jobs_dir(job["run_id"], job["model_id"]) / f"{job['job_id']}.json"
        atomic_write_json(path, job)

    def update(self, job: dict[str, Any], **fields: Any) -> dict[str, Any]:
        status = fields.get("status", job["status"])
        if status not in JOB_STATUSES:
            raise ValidationError(f"unknown job status: {status!r}")
        failure_class = fields.get("failure_class", job.get("failure_class"))
        if failure_class is not None and failure_class not in FAILURE_CLASSES:
            raise ValidationError(f"unknown failure class: {failure_class!r}")
        updated = {**job, **fields, "updated_at": utc_now()}
        self._write(updated)
        return updated

    def get(self, run_id: str, model_id: str, job_id: str) -> dict[str, Any]:
        checked = safe_component(job_id, "job_id")
        path = self._jobs_dir(run_id, model_id) / f"{checked}.json"
        value = read_json(path)
        if not isinstance(value, dict):
            raise NotFoundError(f"job {job_id!r} was not found")
        return value

    def iter_jobs(
        self,
        *,
        run_id: str | None = None,
        model_id: str | None = None,
        status: str | None = None,
        kind: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        if run_id is None:
            run_dirs = [path for path in self.root.glob("*") if path.is_dir()]
        else:
            run_dirs = [self.root / safe_component(run_id, "run_id")]
        for run_dir in run_dirs:
            if not run_dir.is_dir() or run_dir.name.startswith("_"):
                continue
            for model_dir in run_dir.glob("*"):
                if not model_dir.is_dir():
                    continue
                if model_id is not None and model_dir.name != model_id:
                    continue
                for path in sorted((model_dir / "jobs").glob("*.json")):
                    value = read_json(path)
                    if not isinstance(value, dict):
                        continue
                    if status is not None and value.get("status") != status:
                        continue
                    if kind is not None and value.get("kind") != kind:
                        continue
                    yield value

    def list_jobs(
        self,
        run_id: str,
        *,
        model_id: str | None = None,
        attempt_id: str | None = None,
        kind: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        jobs = [
            job
            for job in self.iter_jobs(
                run_id=run_id, model_id=model_id, kind=kind, status=status
            )
            if attempt_id is None or job.get("attempt_id") == attempt_id
        ]
        return sorted(
            jobs, key=lambda job: str(job.get("created_at", "")), reverse=True
        )

    def queued(self) -> list[dict[str, Any]]:
        """Return queued jobs oldest first, which is the worker's queue order."""

        jobs = list(self.iter_jobs(status="QUEUED"))
        return sorted(jobs, key=lambda job: str(job.get("created_at", "")))

    def cancel(self, job: dict[str, Any], reason: str) -> dict[str, Any]:
        if job.get("status") in TERMINAL_JOB_STATUSES:
            return job
        return self.update(
            job,
            status="CANCELED",
            failure_class="CANCELED",
            details={**job.get("details", {}), "cancel_reason": reason},
        )


class ReservationLedger:
    """Track GPU-hour reservations so admission cannot double-book capacity."""

    def __init__(self, root: str | Path, *, capacity_gpu_hours: float) -> None:
        if capacity_gpu_hours <= 0:
            raise ValidationError("capacity_gpu_hours must be positive")
        self.root = Path(root).expanduser()
        self.capacity_gpu_hours = float(capacity_gpu_hours)

    @property
    def path(self) -> Path:
        return self.root / "_audit" / "reservations.json"

    @property
    def lock_path(self) -> Path:
        return self.root / "_audit" / "reservations.lock"

    def _load(self) -> dict[str, Any]:
        value = read_json(self.path)
        if not isinstance(value, dict) or not isinstance(
            value.get("reservations"), list
        ):
            return {"schema_version": 1, "reservations": []}
        return value

    def _save(self, value: dict[str, Any]) -> None:
        atomic_write_json(self.path, value)

    def reserve(
        self,
        *,
        run_id: str,
        model_id: str,
        attempt_id: str,
        gpu_hours: float,
        gpu_count: int | None = None,
    ) -> dict[str, Any]:
        if gpu_hours < 0:
            raise ValidationError("gpu_hours must be >= 0")
        safe_component(run_id, "run_id")
        safe_component(model_id, "model_id")
        safe_component(attempt_id, "attempt_id")
        with exclusive_lock(self.lock_path):
            ledger = self._load()
            outstanding = sum(
                float(record["gpu_hours"])
                for record in ledger["reservations"]
                if record.get("state") in {"RESERVED", "QUEUED"}
            )
            if outstanding + gpu_hours > self.capacity_gpu_hours + 1e-9:
                raise ConflictError(
                    "GPU-hour admission rejected: "
                    f"requested {gpu_hours:g} with "
                    f"{self.capacity_gpu_hours - outstanding:g} remaining of "
                    f"{self.capacity_gpu_hours:g}"
                )
            record = {
                "schema_version": 1,
                "reservation_id": (f"res-{timestamp_token()}-{uuid4().hex[:8]}"),
                "run_id": run_id,
                "model_id": model_id,
                "attempt_id": attempt_id,
                "gpu_hours": float(gpu_hours),
                "gpu_count": gpu_count,
                "state": "RESERVED",
                "reason": None,
                "created_at": utc_now(),
                "updated_at": utc_now(),
            }
            ledger["reservations"].append(record)
            self._save(ledger)
            return record

    def _transition(
        self, reservation_id: str, state: str, reason: str | None
    ) -> dict[str, Any]:
        if state not in {"RESERVED", "QUEUED", "RELEASED"}:
            raise ValidationError(f"unknown reservation state: {state!r}")
        with exclusive_lock(self.lock_path):
            ledger = self._load()
            for record in ledger["reservations"]:
                if record.get("reservation_id") == reservation_id:
                    record.update(
                        state=state,
                        reason=reason,
                        updated_at=utc_now(),
                    )
                    self._save(ledger)
                    return record
        raise NotFoundError(f"reservation {reservation_id!r} was not found")

    def mark_queued(self, reservation_id: str, job_id: str) -> dict[str, Any]:
        """Move a reservation from RESERVED to QUEUED once a job starts.

        The transition is idempotent and never resurrects a released
        reservation, so a retried job cannot re-charge finished GPU-hours.
        """

        with exclusive_lock(self.lock_path):
            ledger = self._load()
            for record in ledger["reservations"]:
                if record.get("reservation_id") != reservation_id:
                    continue
                if record.get("state") != "RESERVED":
                    return record
                record.update(state="QUEUED", job_id=job_id, updated_at=utc_now())
                self._save(ledger)
                return record
        raise NotFoundError(f"reservation {reservation_id!r} was not found")

    def release(self, reservation_id: str, *, reason: str) -> dict[str, Any]:
        return self._transition(reservation_id, "RELEASED", reason)

    def release_for_run(self, run_id: str, *, reason: str) -> list[str]:
        released = []
        with exclusive_lock(self.lock_path):
            ledger = self._load()
            for record in ledger["reservations"]:
                if record.get("run_id") != run_id:
                    continue
                if record.get("state") in {"RESERVED", "QUEUED"}:
                    record.update(state="RELEASED", reason=reason, updated_at=utc_now())
                    released.append(str(record.get("reservation_id")))
            if released:
                self._save(ledger)
        return released

    def release_for_job(
        self, job: dict[str, Any], *, reason: str
    ) -> dict[str, Any] | None:
        reservation_id = job.get("reservation_id")
        if not reservation_id:
            return None
        ledger = self._load()
        for record in ledger["reservations"]:
            if (
                record.get("reservation_id") == reservation_id
                and record.get("state") == "RELEASED"
            ):
                return record
        return self.release(reservation_id, reason=reason)

    def summary(self) -> dict[str, Any]:
        ledger = self._load()
        reservations = ledger["reservations"]
        reserved = sum(
            float(record["gpu_hours"])
            for record in reservations
            if record.get("state") in {"RESERVED", "QUEUED"}
        )
        released = sum(
            float(record["gpu_hours"])
            for record in reservations
            if record.get("state") == "RELEASED"
        )
        return {
            "capacity_gpu_hours": self.capacity_gpu_hours,
            "reserved_gpu_hours": reserved,
            "released_gpu_hours": released,
            "available_gpu_hours": max(0.0, self.capacity_gpu_hours - reserved),
            "reservations": sorted(
                reservations,
                key=lambda record: _timestamp(record.get("created_at")),
                reverse=True,
            ),
        }

    def outstanding_for_run(self, run_id: str) -> float:
        return sum(
            float(record["gpu_hours"])
            for record in self._load()["reservations"]
            if record.get("run_id") == run_id
            and record.get("state") in {"RESERVED", "QUEUED"}
        )


def settle_model_group(
    jobs: JobStore,
    ledger: ReservationLedger,
    job: dict[str, Any],
    *,
    reason: str,
    cancel_active: bool,
) -> list[dict[str, Any]]:
    """Finish a model/attempt job group and release its reservation.

    One reservation covers every stage of a model in an attempt, so it may only
    be released once the whole group is terminal. When a job fails or is
    canceled, the remaining stages of that group are blocked by the dependency
    chain and are canceled with it, which releases the unused GPU-hours.
    """

    group = jobs.list_jobs(
        job["run_id"], model_id=job["model_id"], attempt_id=job["attempt_id"]
    )
    if cancel_active:
        for item in group:
            if item.get("job_id") == job.get("job_id"):
                continue
            if item.get("status") in ACTIVE_JOB_STATUSES:
                jobs.cancel(item, reason)
        group = jobs.list_jobs(
            job["run_id"], model_id=job["model_id"], attempt_id=job["attempt_id"]
        )
    if any(item.get("status") in ACTIVE_JOB_STATUSES for item in group):
        return []
    released: list[dict[str, Any]] = []
    for item in group:
        try:
            record = ledger.release_for_job(item, reason=reason)
        except NotFoundError:
            continue
        if record is not None:
            released.append(record)
    return released
