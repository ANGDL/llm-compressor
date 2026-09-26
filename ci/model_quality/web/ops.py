"""Operational capabilities: capacity, trends, schedules, and notifications."""

from __future__ import annotations

import json
import os
import platform
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from ..config import load_model_config
from ..state import atomic_write_json, utc_now
from .audit import (
    ConflictError,
    ValidationError,
    canonical_json,
    exclusive_lock,
    timestamp_token,
)
from .jobs import ACTIVE_JOB_STATUSES, JobStore, ReservationLedger
from .plans import LaunchPolicy
from .security import Principal
from .store import NotFoundError, RunStore, read_json, safe_component

_CRON_FIELDS = (
    ("minute", 0, 59),
    ("hour", 0, 23),
    ("day_of_month", 1, 31),
    ("month", 1, 12),
    ("day_of_week", 0, 6),
)
_MAX_CRON_SEARCH_DAYS = 366


def _key_token(value: str) -> str:
    """Reduce an ISO timestamp to characters allowed in an idempotency key."""

    return "".join(
        character for character in value if character.isalnum() or character in "._:-"
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


def gpu_device_count(env: dict[str, str] | None = None) -> int:
    env = env if env is not None else dict(os.environ)
    configured = env.get("MODEL_QUALITY_GPU_DEVICES")
    if configured and configured.strip().isdigit():
        return int(configured.strip())
    try:
        import torch

        return int(torch.accelerator.device_count())
    except Exception:
        return 0


def capacity_forecast(
    *,
    ledger: ReservationLedger,
    jobs: JobStore | None = None,
    window_hours: float = 168.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Estimate when the outstanding reservations will drain.

    The projection is deliberately conservative: it divides the GPU-hours still
    held by the ledger by the GPU-hours actually released per wall-clock hour
    in the recent past, and reports ``confidence: none`` instead of guessing
    when there is no release history at all.
    """

    if window_hours <= 0:
        raise ValidationError("window_hours must be positive")
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    summary = ledger.summary()
    window_start = moment - timedelta(hours=window_hours)
    released = [
        record
        for record in summary["reservations"]
        if record.get("state") == "RELEASED"
        and _timestamp(record.get("updated_at")) >= window_start.timestamp()
    ]
    observed = sum(float(record["gpu_hours"]) for record in released)
    throughput = observed / window_hours
    pending = float(summary["reserved_gpu_hours"])
    eta_hours = pending / throughput if throughput > 0 else None
    if not released:
        confidence = "none"
    elif len(released) < 3:
        confidence = "low"
    else:
        confidence = "medium"
    queued_jobs = 0
    active_jobs = 0
    if jobs is not None:
        for job in jobs.iter_jobs():
            if job.get("status") == "QUEUED":
                queued_jobs += 1
            if job.get("status") in ACTIVE_JOB_STATUSES:
                active_jobs += 1
    return {
        "generated_at": utc_now(),
        "window_hours": float(window_hours),
        "confidence": confidence,
        "pending_gpu_hours": round(pending, 4),
        "available_gpu_hours": round(summary["available_gpu_hours"], 4),
        "queued_jobs": queued_jobs,
        "active_jobs": active_jobs,
        "observed_releases": len(released),
        "observed_gpu_hours": round(observed, 4),
        "throughput_gpu_hours_per_hour": round(throughput, 4),
        "eta_hours": None if eta_hours is None else round(eta_hours, 4),
        "estimated_drain_at": (
            None
            if eta_hours is None
            else (moment + timedelta(hours=eta_hours)).isoformat()
        ),
    }


def capabilities(
    *,
    config_path: str | Path | None,
    ledger: ReservationLedger,
    policy: LaunchPolicy,
    jobs: JobStore | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return queue, GPU, and capacity information for operators."""

    devices = gpu_device_count(env)
    active = 0
    queued = 0
    if jobs is not None:
        for job in jobs.iter_jobs():
            if job.get("status") in ACTIVE_JOB_STATUSES:
                active += 1
            if job.get("status") == "QUEUED":
                queued += 1
    models: dict[str, int] = {"total": 0, "enabled": 0, "disabled": 0}
    if config_path is not None:
        for model in load_model_config(config_path)["models"]:
            models["total"] += 1
            models["enabled" if model["enabled"] else "disabled"] += 1
    summary = ledger.summary()
    return {
        "generated_at": utc_now(),
        "queues": [
            {
                "name": "model-quality-gpu",
                "gpu_devices": devices,
                "active_jobs": active,
                "queued_jobs": queued,
                "reserved_gpu_hours": summary["reserved_gpu_hours"],
                "available_gpu_hours": summary["available_gpu_hours"],
            }
        ],
        "capacity": {
            "gpu_hour_capacity": summary["capacity_gpu_hours"],
            "reserved_gpu_hours": summary["reserved_gpu_hours"],
            "released_gpu_hours": summary["released_gpu_hours"],
            "available_gpu_hours": summary["available_gpu_hours"],
        },
        "executor": policy.executor,
        "models": models,
        "hosts": [
            {
                "name": socket.gethostname(),
                "platform": platform.platform(),
                "gpu_devices": devices,
            }
        ],
        "reservations": summary["reservations"][:50],
        "forecast": capacity_forecast(ledger=ledger, jobs=jobs),
    }


class TrendService:
    """Derive quality and GPU-hour cost trends from run facts."""

    def __init__(self, store: RunStore, root: str | Path) -> None:
        self.store = store
        self.root = Path(root).expanduser()

    def _runs(self, limit: int) -> list[dict[str, Any]]:
        if limit < 1 or limit > 500:
            raise ValidationError("limit must be between 1 and 500")
        return self.store.list_runs(limit=limit)["items"]

    def quality(
        self, *, model_id: str | None = None, limit: int = 50
    ) -> dict[str, Any]:
        series: list[dict[str, Any]] = []
        for row in self._runs(limit):
            run_id = row["run_id"]
            try:
                run = self.store.get_run(run_id)
            except (NotFoundError, ValueError):
                continue
            for model in run.get("models", []):
                if model_id and model["model_id"] != model_id:
                    continue
                report = read_json(
                    self.root
                    / safe_component(run_id, "run_id")
                    / safe_component(model["model_id"], "model_id")
                    / "reports"
                    / "evaluation.json"
                )
                metrics = []
                if isinstance(report, dict):
                    for metric in report.get("metrics", []):
                        if not isinstance(metric, dict):
                            continue
                        metrics.append(
                            {
                                "name": metric.get("name"),
                                "recovery": metric.get("recovery"),
                                "status": metric.get("status"),
                            }
                        )
                series.append(
                    {
                        "run_id": run_id,
                        "model_id": model["model_id"],
                        "updated_at": row.get("updated_at"),
                        "run_status": run.get("status"),
                        "artifact_fingerprint": model.get("artifact_fingerprint"),
                        "evaluation_fingerprint": model.get("evaluation_fingerprint"),
                        "metrics": metrics,
                    }
                )
        series.sort(key=lambda item: str(item.get("updated_at") or ""))
        return {
            "model_id": model_id,
            "points": series,
            "total": len(series),
            "metric_names": sorted(
                {
                    metric["name"]
                    for point in series
                    for metric in point["metrics"]
                    if metric["name"]
                }
            ),
        }

    def cost(self, *, limit: int = 50) -> dict[str, Any]:
        points = []
        for row in self._runs(limit):
            run_id = row["run_id"]
            run_dir = self.root / safe_component(run_id, "run_id")
            actual_seconds = 0.0
            for path in run_dir.glob("*/state/*.json"):
                value = read_json(path)
                if isinstance(value, dict) and isinstance(
                    value.get("elapsed_seconds"), (int, float)
                ):
                    actual_seconds += float(value["elapsed_seconds"])
            budget = row.get("budget") or {}
            plan = read_json(run_dir / "execution-plan.json")
            deferred = plan.get("deferred", []) if isinstance(plan, dict) else []
            points.append(
                {
                    "run_id": run_id,
                    "updated_at": row.get("updated_at"),
                    "run_mode": row.get("run_mode"),
                    "run_status": row.get("status"),
                    "estimated_gpu_hours": budget.get("selected_gpu_hours"),
                    "max_gpu_hours": budget.get("max_gpu_hours"),
                    "actual_gpu_hours": round(actual_seconds / 3600.0, 4),
                    "deferred_models": len(deferred),
                }
            )
        points.sort(key=lambda item: str(item.get("updated_at") or ""))
        return {
            "points": points,
            "total": len(points),
            "estimated_gpu_hours": round(
                sum(float(point["estimated_gpu_hours"] or 0) for point in points), 4
            ),
            "actual_gpu_hours": round(
                sum(float(point["actual_gpu_hours"] or 0) for point in points), 4
            ),
        }


def _parse_cron_field(value: str, low: int, high: int, name: str) -> frozenset[int]:
    if not value:
        raise ValidationError(f"cron {name} field is empty")
    allowed: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ValidationError(f"cron {name} field has an empty entry")
        step = 1
        if "/" in part:
            part, _, step_text = part.partition("/")
            if not step_text.isdigit() or int(step_text) < 1:
                raise ValidationError(f"cron {name} step must be a positive integer")
            step = int(step_text)
        if part == "*":
            start, end = low, high
        elif "-" in part:
            start_text, _, end_text = part.partition("-")
            if not start_text.isdigit() or not end_text.isdigit():
                raise ValidationError(f"cron {name} range must be numeric")
            start, end = int(start_text), int(end_text)
        elif part.isdigit():
            start = end = int(part)
        else:
            raise ValidationError(f"cron {name} field is not supported: {part!r}")
        if start < low or end > high or start > end:
            raise ValidationError(f"cron {name} must be within {low}-{high}: {part!r}")
        allowed.update(range(start, end + 1, step))
    return frozenset(allowed)


def parse_cron(expression: str) -> dict[str, Any]:
    """Parse a five-field cron expression without external dependencies."""

    if not isinstance(expression, str):
        raise ValidationError("cron must be a string")
    fields = expression.split()
    if len(fields) != 5:
        raise ValidationError("cron must have five fields: m h dom mon dow")
    parsed: dict[str, Any] = {}
    for (name, low, high), value in zip(_CRON_FIELDS, fields):
        parsed[name] = _parse_cron_field(value, low, high, name)
    return {
        **parsed,
        "day_of_month_restricted": fields[2] != "*",
        "day_of_week_restricted": fields[4] != "*",
        "expression": " ".join(fields),
    }


def cron_matches(parsed: dict[str, Any], moment: datetime) -> bool:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    if moment.minute not in parsed["minute"] or moment.hour not in parsed["hour"]:
        return False
    if moment.month not in parsed["month"]:
        return False
    day_matches = moment.day in parsed["day_of_month"]
    weekday = moment.isoweekday() % 7
    weekday_matches = weekday in parsed["day_of_week"]
    if parsed["day_of_month_restricted"] and parsed["day_of_week_restricted"]:
        return day_matches or weekday_matches
    return day_matches and weekday_matches


def next_run(expression: str, after: datetime | None = None) -> str | None:
    parsed = parse_cron(expression)
    moment = after or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.replace(second=0, microsecond=0) + timedelta(minutes=1)
    for _ in range(_MAX_CRON_SEARCH_DAYS * 24 * 60):
        if cron_matches(parsed, moment):
            return moment.isoformat()
        moment += timedelta(minutes=1)
    return None


class ScheduleStore:
    """Persist scheduled runs; the scheduler tick is a separate CLI entry point."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()

    @property
    def path(self) -> Path:
        return self.root / "_ops" / "schedules.json"

    @property
    def lock_path(self) -> Path:
        return self.root / "_ops" / "schedules.lock"

    def _load(self) -> dict[str, Any]:
        value = read_json(self.path)
        if not isinstance(value, dict) or not isinstance(value.get("schedules"), list):
            return {"schema_version": 1, "schedules": []}
        return value

    def list(self) -> dict[str, Any]:
        schedules = sorted(
            self._load()["schedules"],
            key=lambda item: str(item.get("created_at", "")),
            reverse=True,
        )
        return {"items": schedules, "total": len(schedules)}

    def get(self, schedule_id: str) -> dict[str, Any]:
        checked = safe_component(schedule_id, "schedule_id")
        for schedule in self._load()["schedules"]:
            if schedule.get("schedule_id") == checked:
                return schedule
        raise NotFoundError(f"schedule {schedule_id!r} was not found")

    def create(
        self,
        *,
        cron: str,
        plan_request: dict[str, Any],
        actor: str,
        enabled: bool = True,
    ) -> dict[str, Any]:
        if not isinstance(plan_request, dict) or not plan_request:
            raise ValidationError("plan_request must be a non-empty mapping")
        if plan_request.get("run_mode") not in {
            "quantize",
            "quantize_and_eval",
            "eval_only",
            "upload_only",
        }:
            raise ValidationError("plan_request.run_mode is required")
        upcoming = next_run(cron)
        schedule = {
            "schema_version": 1,
            "schedule_id": f"sch-{timestamp_token()}",
            "cron": cron,
            "enabled": bool(enabled),
            "plan_request": plan_request,
            "created_by": actor,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "last_run_at": None,
            "last_occurrence": None,
            "last_run_id": None,
            "next_run_at": upcoming if enabled else None,
        }
        with exclusive_lock(self.lock_path):
            store = self._load()
            store["schedules"].append(schedule)
            atomic_write_json(self.path, store)
        return schedule

    def update(self, schedule_id: str, **fields: Any) -> dict[str, Any]:
        checked = safe_component(schedule_id, "schedule_id")
        with exclusive_lock(self.lock_path):
            store = self._load()
            for index, schedule in enumerate(store["schedules"]):
                if schedule.get("schedule_id") != checked:
                    continue
                updated = {**schedule, **fields, "updated_at": utc_now()}
                store["schedules"][index] = updated
                atomic_write_json(self.path, store)
                return updated
        raise NotFoundError(f"schedule {schedule_id!r} was not found")

    def set_enabled(self, schedule_id: str, enabled: bool) -> dict[str, Any]:
        schedule = self.get(schedule_id)
        upcoming = next_run(schedule["cron"]) if enabled else None
        return self.update(schedule_id, enabled=bool(enabled), next_run_at=upcoming)

    def delete(self, schedule_id: str) -> dict[str, Any]:
        checked = safe_component(schedule_id, "schedule_id")
        with exclusive_lock(self.lock_path):
            store = self._load()
            remaining = [
                schedule
                for schedule in store["schedules"]
                if schedule.get("schedule_id") != checked
            ]
            if len(remaining) == len(store["schedules"]):
                raise NotFoundError(f"schedule {schedule_id!r} was not found")
            store["schedules"] = remaining
            atomic_write_json(self.path, store)
        return {"schedule_id": checked, "deleted": True}

    def due(self, now: datetime | None = None) -> list[dict[str, Any]]:
        moment = now or datetime.now(timezone.utc)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        due = []
        for schedule in self._load()["schedules"]:
            if not schedule.get("enabled"):
                continue
            upcoming = schedule.get("next_run_at")
            if not isinstance(upcoming, str):
                continue
            try:
                parsed = datetime.fromisoformat(upcoming.replace("Z", "+00:00"))
            except ValueError:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            if parsed <= moment:
                due.append(schedule)
        return sorted(due, key=lambda item: str(item.get("next_run_at", "")))


class NotificationService:
    """Record failure notifications and optionally POST them to a webhook."""

    def __init__(self, root: str | Path, *, env: dict[str, str] | None = None) -> None:
        self.root = Path(root).expanduser()
        self.env = env if env is not None else dict(os.environ)

    @property
    def config_path(self) -> Path:
        return self.root / "_ops" / "notifications.json"

    @property
    def log_path(self) -> Path:
        return self.root / "_ops" / "notification-log.jsonl"

    def config(self) -> dict[str, Any]:
        value = read_json(self.config_path)
        if isinstance(value, dict):
            return value
        return {
            "schema_version": 1,
            "enabled": False,
            "webhook_url": None,
            "events": ["run.failed", "job.failed", "run.publish_failed"],
        }

    def log(self, *, limit: int = 100) -> dict[str, Any]:
        if not self.log_path.is_file():
            return {"items": [], "total": 0}
        entries = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        entries.reverse()
        return {"items": entries[:limit], "total": len(entries)}

    def notify(self, event: str, payload: dict[str, Any]) -> dict[str, Any]:
        config = self.config()
        record = {
            "schema_version": 1,
            "recorded_at": utc_now(),
            "event": event,
            "payload": payload,
            "delivery": "recorded",
        }
        webhook = config.get("webhook_url")
        events = config.get("events") or []
        allow_network = str(
            self.env.get("MODEL_QUALITY_NOTIFY_ALLOW_NETWORK", "")
        ).lower() in {"1", "true", "yes", "on"}
        if not config.get("enabled") or not webhook or event not in events:
            record["delivery"] = "disabled"
        elif not allow_network:
            record["delivery"] = "suppressed"
            record["message"] = (
                "set MODEL_QUALITY_NOTIFY_ALLOW_NETWORK=1 to deliver webhooks"
            )
        else:
            body = canonical_json(record).encode("utf-8")
            request = urlrequest.Request(
                str(webhook),
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlrequest.urlopen(request, timeout=5) as response:
                    record["delivery"] = "delivered"
                    record["status_code"] = response.status
            except (urlerror.URLError, OSError, ValueError) as error:
                record["delivery"] = "failed"
                record["error"] = str(error)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(record) + "\n")
        return record


def run_due_schedules(
    *,
    schedules: ScheduleStore,
    control,
    notifications: NotificationService,
    actor: str = "scheduler",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Start every due schedule; each occurrence keeps its own idempotency key."""

    principal = Principal(
        actor=actor, roles=frozenset({"operator"}), authenticated=True
    )
    results = []
    for schedule in schedules.due(now):
        occurrence = str(schedule.get("next_run_at"))
        key = f"schedule:{schedule['schedule_id']}:{_key_token(occurrence)}"
        recorded = (
            schedule.get("last_run_id")
            if schedule.get("last_occurrence") == occurrence
            else None
        )
        if not recorded:
            response = control.idempotency.lookup(key)
            if isinstance(response, dict):
                recorded = response.get("run_id")
        if recorded:
            updated = schedules.update(
                schedule["schedule_id"],
                last_occurrence=occurrence,
                last_run_id=recorded,
                next_run_at=next_run(schedule["cron"]),
            )
            results.append(
                {
                    "schedule_id": updated["schedule_id"],
                    "status": "REPLAYED",
                    "run_id": recorded,
                    "occurrence": occurrence,
                }
            )
            continue
        try:
            preview = control.preview_plan(schedule["plan_request"], principal)
            response = control.start_run(
                {"plan_hash": preview["plan_hash"], "idempotency_key": key}, principal
            )
            schedules.update(
                schedule["schedule_id"],
                last_run_at=utc_now(),
                last_occurrence=occurrence,
                last_run_id=response["run_id"],
                next_run_at=next_run(schedule["cron"]),
            )
            results.append(
                {
                    "schedule_id": schedule["schedule_id"],
                    "status": "STARTED",
                    "run_id": response["run_id"],
                    "occurrence": occurrence,
                }
            )
        except (ConflictError, ValidationError, NotFoundError, OSError) as error:
            schedules.update(
                schedule["schedule_id"],
                last_run_at=utc_now(),
                next_run_at=next_run(schedule["cron"]),
            )
            notifications.notify(
                "schedule.failed",
                {
                    "schedule_id": schedule["schedule_id"],
                    "occurrence": occurrence,
                    "error": str(error),
                },
            )
            results.append(
                {
                    "schedule_id": schedule["schedule_id"],
                    "status": "FAILED",
                    "error": str(error),
                }
            )
    return {"items": results, "total": len(results)}
