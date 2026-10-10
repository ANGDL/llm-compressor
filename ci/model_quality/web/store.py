"""Read-only index over the model quality run store."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..state import runs_root

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_STAGES = (
    "preflight",
    "quantize",
    "validate",
    "runtime-smoke",
    "evaluate",
    "report",
    "publish",
)
_SUCCESS_STATUSES = {"PASS", "PASSED", "SUCCESS", "SUCCEEDED"}
_FAILURE_STATUSES = {"FAIL", "FAILED"}
_ACTIVE_STATUSES = {"CREATED", "QUEUED", "ALLOCATED", "RUNNING"}
# Upper bound on a single log line's width in the API response; wider lines are
# truncated so one giant line can't balloon a polled log fetch to tens of MB.
_MAX_LOG_LINE_CHARS = 4000
# Stage names do not always appear verbatim in the log file name written by
# ``ci.model_quality.executor`` (``quantize`` writes ``quantization.log``).
_STAGE_LOG_NAMES = {
    "preflight": ("preflight.log",),
    "quantize": ("quantization.log",),
    "validate": ("validation.log",),
    "runtime-smoke": ("runtime-smoke.log",),
    "evaluate": ("evaluation.log",),
    "report": ("report.log",),
    "publish": ("publish.log",),
}
# Reverse map (log file name -> stage) used to surface in-flight stages, whose
# state file is not written until they terminate, from their live log file.
_STAGE_FOR_LOG = {
    filename: stage
    for stage, filenames in _STAGE_LOG_NAMES.items()
    for filename in filenames
}


class NotFoundError(LookupError):
    """Raised when a requested run-store object does not exist."""


def safe_component(value: str, name: str) -> str:
    """Return ``value`` when it is a safe single path component."""

    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f"unsafe {name}: {value!r}")
    return value


def read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    """Read optional JSON, returning ``None`` for missing or malformed files."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


class RunStore:
    """Read facts written under ``MODEL_QUALITY_RUNS_ROOT``.

    Optional malformed files are ignored. One partially written run therefore
    cannot make the entire Runs page unavailable.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).expanduser() if root is not None else runs_root()

    _check = staticmethod(safe_component)
    _read_json = staticmethod(read_json)

    @staticmethod
    def _timestamp(value: Any) -> float | None:
        if value is None:
            return None
        try:
            text = str(value).replace("Z", "+00:00")
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        except ValueError:
            return None

    def _run_dir(self, run_id: str) -> Path:
        return self.root / self._check(run_id, "run_id")

    def _model_dir(self, run_id: str, model_id: str) -> Path:
        return self._run_dir(run_id) / self._check(model_id, "model_id")

    def _require_run(self, run_id: str) -> Path:
        path = self._run_dir(run_id)
        if not path.is_dir():
            raise NotFoundError(f"run {run_id!r} was not found")
        return path

    def _iter_run_dirs(self) -> Iterator[Path]:
        if not self.root.is_dir():
            return
        for path in self.root.iterdir():
            if path.is_dir() and _SAFE_COMPONENT.fullmatch(path.name):
                yield path

    def _model_ids(self, run_dir: Path, plan: dict[str, Any]) -> list[str]:
        selected = plan.get("selected", [])
        planned = {
            str(item["id"])
            for item in selected
            if isinstance(item, dict) and item.get("id")
        }
        materialized = {
            path.name
            for path in run_dir.iterdir()
            if path.is_dir()
            and _SAFE_COMPONENT.fullmatch(path.name)
            and path.name not in {"attempt-plans", "events", "logs"}
        }
        return sorted(planned | materialized)

    @staticmethod
    def _status(value: dict[str, Any] | None) -> str:
        if value is None:
            return "PENDING"
        status = value.get("status") or value.get("result") or value.get("state")
        return str(status).upper() if status else "UNKNOWN"

    def _stage_records(self, model_dir: Path) -> dict[str, dict[str, Any] | None]:
        result: dict[str, dict[str, Any] | None] = {}
        for stage in _STAGES:
            value = self._read_json(model_dir / "state" / f"{stage}.json")
            result[stage] = value if isinstance(value, dict) else None
        return result

    def _live_stages(
        self, model_dir: Path, stages: dict[str, dict[str, Any] | None]
    ) -> set[str]:
        """Stages whose last record is terminal but whose log is still growing.

        A stage record is written only when the stage ends, so if the stage's
        log keeps advancing past that record a fresh invocation (a re-run) of
        the same stage is in progress — the persisted FAIL/PASS is now history,
        not the current state. Reading a log never updates its mtime, so this
        only trips on genuine writes, and the recency guard keeps a stage that
        died without recording a terminal state from looking live forever.
        """

        live: set[str] = set()
        now = time.time()
        for stage, record in stages.items():
            if not isinstance(record, dict):
                continue
            if self._status(record) in _ACTIVE_STATUSES:
                continue
            recorded = self._timestamp(record.get("recorded_at"))
            attempt = record.get("attempt_id")
            if (
                recorded is None
                or not attempt
                or not _SAFE_COMPONENT.fullmatch(str(attempt))
            ):
                continue
            log_dir = model_dir / "logs" / str(attempt)
            newest: float | None = None
            for name in _STAGE_LOG_NAMES.get(stage, ()):
                try:
                    mtime = (log_dir / name).stat().st_mtime
                except OSError:
                    continue
                newest = mtime if newest is None else max(newest, mtime)
            if newest is None:
                continue
            if newest > recorded + 2 and now - newest < 180:
                live.add(stage)
        return live

    @staticmethod
    def _first_identity(
        stages: dict[str, dict[str, Any] | None], key: str
    ) -> str | None:
        for stage in reversed(_STAGES):
            value = stages[stage]
            if isinstance(value, dict) and value.get(key):
                return str(value[key])
        return None

    @staticmethod
    def _model_status(statuses: dict[str, str]) -> str:
        values = set(statuses.values()) - {"PENDING"}
        if values & _FAILURE_STATUSES:
            return "FAIL"
        if values & {"CANCELED"}:
            return "CANCELED"
        if values & {"RESOURCE_TIMEOUT", "EXPIRED"}:
            return "RESOURCE_TIMEOUT"
        if values & _ACTIVE_STATUSES:
            return "RUNNING"
        if "WARN" in values:
            return "WARN"
        if values and values <= _SUCCESS_STATUSES | {"SKIPPED"}:
            return "PASS"
        return "PENDING"

    def _canceled_before_start(self, model_dir: Path) -> bool:
        """True when every recorded job was canceled and no stage ever reported.

        ``ControlPlane.cancel`` cancels queued jobs and releases their
        reservations, but a run canceled before its first stage finishes leaves
        no ``state/<stage>.json`` for the status derivation to read. Without this
        signal such a run keeps reporting PENDING after it was canceled.
        """

        statuses = []
        for path in sorted((model_dir / "jobs").glob("*.json")):
            value = self._read_json(path)
            if isinstance(value, dict):
                statuses.append(str(value.get("status") or "").upper())
        return bool(statuses) and all(status == "CANCELED" for status in statuses)

    def _attempts(self, model_dir: Path) -> list[dict[str, Any]]:
        current = self._read_json(model_dir / "current-attempt.json")
        current_id = current.get("attempt_id") if isinstance(current, dict) else None
        root = model_dir / "attempts"
        rows = []
        for attempt_dir in root.iterdir() if root.is_dir() else []:
            if not attempt_dir.is_dir() or not _SAFE_COMPONENT.fullmatch(
                attempt_dir.name
            ):
                continue
            stages: dict[str, dict[str, Any]] = {}
            for path in sorted((attempt_dir / "state").glob("*.json")):
                value = self._read_json(path)
                if isinstance(value, dict):
                    stages[path.stem] = value
            timestamps = [
                str(value["recorded_at"])
                for value in stages.values()
                if value.get("recorded_at")
            ]
            statuses = {name: self._status(value) for name, value in stages.items()}
            rows.append(
                {
                    "attempt_id": attempt_dir.name,
                    "current": attempt_dir.name == current_id,
                    "status": self._model_status(statuses),
                    "stages": statuses,
                    "started_at": min(timestamps) if timestamps else None,
                    "updated_at": max(timestamps) if timestamps else None,
                }
            )
        # A stage writes its ``state/<stage>.json`` only when it terminates, so an
        # in-flight stage has a growing log but no recorded state — and therefore
        # no row above. Surface those running attempts (and any stage that has a
        # log but no state yet) straight from the logs directory so the live log
        # is selectable in the UI during the run instead of only appearing once
        # the stage finishes or fails.
        by_id = {row["attempt_id"]: row for row in rows}
        logs_root = model_dir / "logs"
        for log_dir in logs_root.iterdir() if logs_root.is_dir() else []:
            if not log_dir.is_dir() or not _SAFE_COMPONENT.fullmatch(log_dir.name):
                continue
            newest = None
            running_stages: dict[str, float] = {}
            for log_path in log_dir.rglob("*.log"):
                stage = _STAGE_FOR_LOG.get(log_path.name)
                if stage is None:
                    continue
                try:
                    mtime = log_path.stat().st_mtime
                except OSError:
                    continue
                running_stages[stage] = mtime
                newest = mtime if newest is None else max(newest, mtime)
            if not running_stages:
                continue
            iso = (
                datetime.fromtimestamp(newest, tz=timezone.utc).isoformat()
                if newest is not None
                else None
            )
            row = by_id.get(log_dir.name)
            if row is None:
                row = {
                    "attempt_id": log_dir.name,
                    "current": log_dir.name == current_id,
                    "status": "RUNNING",
                    "stages": {},
                    "started_at": iso,
                    "updated_at": iso,
                }
                rows.append(row)
                by_id[log_dir.name] = row
            # Only fill stages that have no recorded terminal state; never clobber
            # a real status with the synthetic RUNNING marker.
            for stage in running_stages:
                row["stages"].setdefault(stage, "RUNNING")
            if iso is not None:
                row["updated_at"] = row["updated_at"] or iso
                row["started_at"] = row["started_at"] or iso
        return sorted(
            rows,
            key=lambda item: (item["updated_at"] or "", item["attempt_id"]),
            reverse=True,
        )

    @staticmethod
    def _compact_stage_detail(value: dict[str, Any] | None) -> dict[str, Any] | None:
        """Keep status and diagnostics while excluding artifact inventories."""

        if not isinstance(value, dict):
            return None
        keys = (
            "status",
            "reason_code",
            "message",
            "failure_class",
            "attempt_id",
            "artifact_fingerprint",
            "evaluation_fingerprint",
            "recorded_at",
        )
        result = {key: value[key] for key in keys if value.get(key) is not None}
        failures = value.get("failures")
        if isinstance(failures, list):
            result["failures"] = [
                item if isinstance(item, str) else str(item)
                for item in failures[:20]
            ]
        return result

    def _failure_history(
        self, model_dir: Path, current_attempt_id: str | None
    ) -> list[dict[str, Any]]:
        """Return compact immutable stage and job failures for the run detail UI."""

        history: list[dict[str, Any]] = []
        attempts_dir = model_dir / "attempts"
        for attempt_dir in attempts_dir.iterdir() if attempts_dir.is_dir() else []:
            if not attempt_dir.is_dir() or not _SAFE_COMPONENT.fullmatch(attempt_dir.name):
                continue
            for path in sorted((attempt_dir / "state").glob("*.json")):
                value = self._read_json(path)
                if not isinstance(value, dict):
                    continue
                status = self._status(value)
                if status not in _FAILURE_STATUSES | {"RESOURCE_TIMEOUT"}:
                    continue
                detail = self._compact_stage_detail(value) or {}
                history.append(
                    {
                        "stage": path.stem,
                        "status": status,
                        "reason_code": detail.get("reason_code"),
                        "message": detail.get("message"),
                        "failure_class": detail.get("failure_class"),
                        "attempt_id": detail.get("attempt_id") or attempt_dir.name,
                        "recorded_at": detail.get("recorded_at"),
                        "current": attempt_dir.name == current_attempt_id,
                    }
                )
        for path in sorted((model_dir / "jobs").glob("*.json")):
            value = self._read_json(path)
            if not isinstance(value, dict):
                continue
            status = str(value.get("status") or "").upper()
            if status not in {"FAILED", "EXPIRED", "CANCELED"}:
                continue
            details = value.get("details") if isinstance(value.get("details"), dict) else {}
            failure_class = value.get("failure_class") or details.get("failure_class")
            message = details.get("message") or details.get("cancel_reason")
            history.append(
                {
                    "stage": details.get("failed_stage") or (value.get("stages") or [value.get("kind")])[0],
                    "status": status,
                    "reason_code": failure_class or status,
                    "message": message,
                    "failure_class": failure_class,
                    "attempt_id": value.get("attempt_id"),
                    "job_id": value.get("job_id"),
                    "recorded_at": value.get("updated_at") or value.get("created_at"),
                    "current": value.get("attempt_id") == current_attempt_id,
                }
            )
        return sorted(
            history,
            key=lambda item: (self._timestamp(item.get("recorded_at")) or float("-inf"), item.get("attempt_id") or ""),
            reverse=True,
        )

    def _report(self, model_dir: Path) -> dict[str, Any] | None:
        for name in ("summary.json", "evaluation.json", "report.json"):
            value = self._read_json(model_dir / "reports" / name)
            if isinstance(value, dict):
                return value
        return None

    def _publish_summary(self, model_dir: Path) -> dict[str, Any]:
        value = self._read_json(model_dir / "state" / "publish.json")
        if not isinstance(value, dict):
            value = self._read_json(model_dir / "publish.json")
        success = self._read_json(model_dir / "commit-markers" / "SUCCESS.json")
        if not isinstance(success, dict):
            success = self._read_json(model_dir / "SUCCESS.json")
        raw_status = self._status(value if isinstance(value, dict) else None)
        if isinstance(success, dict) and raw_status == "PENDING":
            status = "SUCCESS"
        elif raw_status in _SUCCESS_STATUSES:
            status = "SUCCESS"
        elif raw_status in _FAILURE_STATUSES:
            status = "FAILED"
        elif raw_status in _ACTIVE_STATUSES:
            status = "UPLOADING"
        elif raw_status == "SKIPPED":
            status = "NOT_REQUESTED"
        else:
            status = raw_status
        return {"status": status, "state": value, "success": success}

    def _model_summary(
        self, run_id: str, model_id: str, *, include_details: bool = True
    ) -> dict[str, Any]:
        model_dir = self._model_dir(run_id, model_id)
        stages = self._stage_records(model_dir)
        statuses = {stage: self._status(value) for stage, value in stages.items()}
        for stage in self._live_stages(model_dir, stages):
            statuses[stage] = "RUNNING"
        status = self._model_status(statuses)
        if status == "PENDING" and self._canceled_before_start(model_dir):
            status = "CANCELED"
        if not include_details:
            return {
                "model_id": model_id,
                "status": status,
                "stages": statuses,
                "publish": self._publish_summary(model_dir),
            }
        manifest = self._read_json(model_dir / "artifact-manifest.json")
        current = self._read_json(model_dir / "current-attempt.json")
        return {
            "model_id": model_id,
            "status": status,
            "stages": statuses,
            "stage_details": {
                stage: self._compact_stage_detail(value)
                for stage, value in stages.items()
            },
            "attempt_id": (
                current.get("attempt_id") if isinstance(current, dict) else None
            ),
            "attempts": self._attempts(model_dir),
            "failure_history": [
                {**item, "model_id": model_id}
                for item in self._failure_history(
                    model_dir,
                    current.get("attempt_id") if isinstance(current, dict) else None,
                )
            ],
            "artifact_fingerprint": (
                current.get("artifact_fingerprint")
                if isinstance(current, dict)
                else self._first_identity(stages, "artifact_fingerprint")
            ),
            "evaluation_fingerprint": (
                current.get("evaluation_fingerprint")
                if isinstance(current, dict)
                else self._first_identity(stages, "evaluation_fingerprint")
            ),
            "artifact_manifest": (
                {
                    key: manifest[key]
                    for key in ("schema_version", "artifact_fingerprint", "artifact_content_fingerprint", "content_fingerprint", "file_count")
                    if isinstance(manifest, dict) and key in manifest
                }
                if isinstance(manifest, dict)
                else None
            ),
            "report": self._report(model_dir),
            "publish": self._publish_summary(model_dir),
        }

    @staticmethod
    def _run_status(models: list[dict[str, Any]], aggregate: Any) -> str:
        if isinstance(aggregate, dict):
            aggregate_status = str(aggregate.get("status", "")).upper()
            if aggregate_status in _FAILURE_STATUSES:
                if any(item["publish"]["status"] == "FAILED" for item in models):
                    return "PUBLISH_FAILED"
                return "FAIL"
            if aggregate_status in _SUCCESS_STATUSES:
                return "PASS"
        statuses = {item["status"] for item in models}
        if "FAIL" in statuses:
            return "FAIL"
        if "CANCELED" in statuses:
            return "CANCELED"
        if "RESOURCE_TIMEOUT" in statuses:
            return "RESOURCE_TIMEOUT"
        if "RUNNING" in statuses:
            return "RUNNING"
        if models and statuses <= {"PASS"}:
            return "PASS"
        if "WARN" in statuses:
            return "WARN"
        return "PLANNED"

    @staticmethod
    def _run_publish(models: list[dict[str, Any]]) -> str:
        states = {item["publish"]["status"] for item in models}
        if "FAILED" in states:
            return "FAILED"
        if states and states <= {"SUCCESS"}:
            return "SUCCESS"
        if "UPLOADING" in states:
            return "UPLOADING"
        if "READY" in states:
            return "READY"
        return "NOT_REQUESTED"

    def _updated_at(self, run_dir: Path, plan: dict[str, Any]) -> str | None:
        values = [
            str(plan[key])
            for key in ("updated_at", "created_at", "planned_at")
            if plan.get(key)
        ]
        for path in run_dir.glob("*/state/*.json"):
            value = self._read_json(path)
            if isinstance(value, dict) and value.get("recorded_at"):
                values.append(str(value["recorded_at"]))
        marker = self._read_json(run_dir / "cancel.json")
        if isinstance(marker, dict) and marker.get("recorded_at"):
            values.append(str(marker["recorded_at"]))
        for path in run_dir.glob("*/jobs/*.json"):
            value = self._read_json(path)
            if not isinstance(value, dict):
                continue
            for key in ("updated_at", "finished_at", "created_at"):
                if value.get(key):
                    values.append(str(value[key]))
                    break
        if not values:
            return None
        return max(values, key=lambda value: self._timestamp(value) or float("-inf"))

    def list_runs(
        self,
        *,
        status: str | None = None,
        model: str | None = None,
        since: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        if offset < 0 or limit < 1 or limit > 500:
            raise ValueError("offset must be >= 0 and limit must be between 1 and 500")
        wanted_status = status.upper() if status else None
        since_timestamp = self._timestamp(since) if since else None
        if since and since_timestamp is None:
            raise ValueError("since must be an ISO-8601 timestamp")
        rows = []
        for run_dir in self._iter_run_dirs():
            plan = self._read_json(run_dir / "execution-plan.json")
            if not isinstance(plan, dict):
                continue
            run_id = str(plan.get("run_id", run_dir.name))
            models = [
                self._model_summary(run_id, model_id, include_details=False)
                for model_id in self._model_ids(run_dir, plan)
            ]
            if model and not any(item["model_id"] == model for item in models):
                continue
            aggregate = self._read_json(run_dir / "aggregate-report.json")
            run_status = self._run_status(models, aggregate)
            if wanted_status and run_status != wanted_status:
                continue
            updated = self._updated_at(run_dir, plan)
            if since_timestamp is not None and (
                updated is None
                or (self._timestamp(updated) or float("-inf")) < since_timestamp
            ):
                continue
            counts = {
                key: sum(item["status"] == key for item in models)
                for key in ("PASS", "WARN", "FAIL", "RUNNING", "PENDING")
            }
            rows.append(
                {
                    "run_id": run_id,
                    "attempt_id": plan.get("attempt_id"),
                    "git_sha": plan.get("git_sha"),
                    "run_mode": plan.get("run_mode"),
                    "created_at": plan.get("created_at") or plan.get("planned_at"),
                    "updated_at": updated,
                    "status": run_status,
                    "budget": plan.get("budget", {}),
                    "models": counts,
                    "model_count": len(models),
                    "model_ids": [item["model_id"] for item in models],
                    "publish": self._run_publish(models),
                }
            )
        rows.sort(
            key=lambda item: (
                self._timestamp(item["updated_at"] or item["created_at"])
                or float("-inf"),
                item["run_id"],
            ),
            reverse=True,
        )
        return {
            "items": rows[offset : offset + limit],
            "total": len(rows),
            "offset": offset,
            "limit": limit,
        }

    def get_run(self, run_id: str) -> dict[str, Any]:
        run_dir = self._require_run(run_id)
        plan = self._read_json(run_dir / "execution-plan.json")
        if not isinstance(plan, dict):
            raise NotFoundError(f"run {run_id!r} has no execution plan")
        models = [
            self._model_summary(run_id, model_id)
            for model_id in self._model_ids(run_dir, plan)
        ]
        aggregate = self._read_json(run_dir / "aggregate-report.json")
        result = dict(plan)
        result.update(
            {
                "status": self._run_status(models, aggregate),
                "models": models,
                "publish": self._run_publish(models),
                "aggregate_report": aggregate,
                "updated_at": self._updated_at(run_dir, plan),
            }
        )
        return result

    def get_model(self, run_id: str, model_id: str) -> dict[str, Any]:
        self._require_run(run_id)
        model_dir = self._model_dir(run_id, model_id)
        if not model_dir.is_dir():
            raise NotFoundError(f"model {model_id!r} was not found in run {run_id!r}")
        return self._model_summary(run_id, model_id)

    def get_artifacts(self, run_id: str, model_id: str) -> dict[str, Any]:
        model_dir = self._model_dir(run_id, model_id)
        self.get_model(run_id, model_id)
        manifest = self._read_json(model_dir / "artifact-manifest.json")
        files = manifest.get("files", []) if isinstance(manifest, dict) else []
        return {"model_id": model_id, "manifest": manifest, "files": files}

    def get_publish(self, run_id: str, model_id: str) -> dict[str, Any]:
        self.get_model(run_id, model_id)
        return self._publish_summary(self._model_dir(run_id, model_id))

    def get_logs(
        self,
        run_id: str,
        model_id: str,
        *,
        stage: str | None = None,
        attempt_id: str | None = None,
        offset: int = 0,
        limit: int = 1000,
        tail: bool = False,
    ) -> dict[str, Any]:
        self.get_model(run_id, model_id)
        if offset < 0 or limit < 1 or limit > 10000:
            raise ValueError("invalid log pagination")
        if stage is not None and stage not in _STAGES:
            raise ValueError(f"unknown stage: {stage!r}")
        model_dir = self._model_dir(run_id, model_id)
        if attempt_id:
            checked_attempt = self._check(attempt_id, "attempt_id")
            roots = [
                model_dir / "logs" / checked_attempt,
                model_dir / "attempts" / checked_attempt / "logs",
            ]
        else:
            roots = [model_dir / "logs"]
        files = []
        for root in roots:
            if root.is_file():
                files.append(root)
            elif root.is_dir():
                files.extend(path for path in root.rglob("*") if path.is_file())
        if stage:
            names = _STAGE_LOG_NAMES.get(stage, ())
            files = [
                path
                for path in files
                if path.name in names or stage in path.name or stage in str(path.parent)
            ]
        # Order by mtime so the most recently written (actively-growing) log ends
        # up last; tailing then surfaces the running attempt instead of a stale head.
        def _sort_key(path: Path) -> tuple:
            try:
                return (path.stat().st_mtime, str(path))
            except OSError:
                return (0.0, str(path))

        files = sorted(set(files), key=_sort_key)
        run_dir = self._run_dir(run_id)
        # Tail polls re-read on a 3s timer; decoding and dict-wrapping an entire
        # multi-MB, actively-growing log on every poll stalls the single-threaded
        # server. In tail mode read only a bounded window from each file's end and
        # count totals cheaply (no decode) so a poll stays O(window), not O(file).
        window_bytes = min(8 << 20, max(limit * 256, 1 << 18)) if tail else None
        lines: list[dict[str, Any]] = []
        total = 0
        for path in files:
            try:
                if window_bytes is None:
                    file_total, texts = self._all_capped_lines(path)
                else:
                    file_total, texts = self._tail_capped_lines(path, window_bytes)
            except OSError:
                continue
            relative = str(path.relative_to(run_dir))
            base = max(0, file_total - len(texts))
            for index, text in enumerate(texts, 1):
                lines.append({"path": relative, "line": base + index, "text": text})
            total += file_total
        if tail:
            start = max(0, len(lines) - limit)
            items = lines[start : start + limit]
            return {
                "items": items,
                "total": total,
                "offset": max(0, total - len(items)),
                "limit": limit,
            }
        return {
            "items": lines[offset : offset + limit],
            "total": total,
            "offset": offset,
            "limit": limit,
        }

    @staticmethod
    def _cap_line(text: str) -> str:
        # One pathological line (a stage dumping a huge JSON blob or tensor) can
        # be megabytes wide; cap its width so a single line cannot balloon the
        # polled response. The full log is still on disk.
        if len(text) > _MAX_LOG_LINE_CHARS:
            dropped = len(text) - _MAX_LOG_LINE_CHARS
            return f"{text[:_MAX_LOG_LINE_CHARS]}… [{dropped} more chars truncated]"
        return text

    @classmethod
    def _all_capped_lines(cls, path: Path) -> tuple[int, list[str]]:
        content = path.read_text(encoding="utf-8", errors="replace")
        texts = [cls._cap_line(text) for text in content.splitlines()]
        return len(texts), texts

    @classmethod
    def _tail_capped_lines(cls, path: Path, window_bytes: int) -> tuple[int, list[str]]:
        size = path.stat().st_size
        total = 0
        last = b""
        with path.open("rb") as handle:
            while True:
                buffer = handle.read(1 << 20)
                if not buffer:
                    break
                total += buffer.count(b"\n")
                last = buffer[-1:]
            if size and last and last != b"\n":
                total += 1  # final line without a trailing newline
            if size > window_bytes:
                handle.seek(size - window_bytes)
                handle.readline()  # drop the partial leading line
            else:
                handle.seek(0)
            window = handle.read()
        texts = [cls._cap_line(text) for text in window.decode("utf-8", "replace").splitlines()]
        return total, texts

    def events(self, run_id: str, *, after: int | None = None) -> list[dict[str, Any]]:
        run_dir = self._require_run(run_id)
        event_dir = run_dir / "events"
        values = []
        for path in event_dir.glob("*.json") if event_dir.is_dir() else []:
            value = self._read_json(path)
            if not isinstance(value, dict):
                continue
            sequence = value.get("sequence")
            if after is not None and isinstance(sequence, int) and sequence <= after:
                continue
            values.append(value)
        return sorted(
            values,
            key=lambda item: (
                item.get("sequence") if isinstance(item.get("sequence"), int) else -1,
                str(item.get("recorded_at", "")),
            ),
        )

    def models(self, config_path: str | Path | None = None) -> list[dict[str, Any]]:
        if config_path is None:
            return []
        from ..config import load_model_config

        values = []
        for model in load_model_config(config_path)["models"]:
            values.append(
                {
                    "id": model["id"],
                    "enabled": model["enabled"],
                    "priority": model["priority"],
                    "business_tier": model.get("business_tier"),
                    "resources": model["resources"],
                    "upload_enabled": bool(model.get("upload", {}).get("enabled")),
                    "workflow_revision": model["workflow"]["revision"],
                }
            )
        return values
