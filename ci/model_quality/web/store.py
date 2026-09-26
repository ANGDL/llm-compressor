"""Read-only index over the model quality run store."""

from __future__ import annotations

import json
import re
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
            and path.name not in {"events", "logs"}
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
        return sorted(
            rows,
            key=lambda item: (item["updated_at"] or "", item["attempt_id"]),
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

    def _model_summary(self, run_id: str, model_id: str) -> dict[str, Any]:
        model_dir = self._model_dir(run_id, model_id)
        stages = self._stage_records(model_dir)
        statuses = {stage: self._status(value) for stage, value in stages.items()}
        manifest = self._read_json(model_dir / "artifact-manifest.json")
        current = self._read_json(model_dir / "current-attempt.json")
        return {
            "model_id": model_id,
            "status": self._model_status(statuses),
            "stages": statuses,
            "stage_details": stages,
            "attempt_id": (
                current.get("attempt_id") if isinstance(current, dict) else None
            ),
            "attempts": self._attempts(model_dir),
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
            "artifact_manifest": manifest if isinstance(manifest, dict) else None,
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
                self._model_summary(run_id, model_id)
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
        lines = []
        for path in sorted(set(files)):
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lines.extend(
                {
                    "path": str(path.relative_to(self._run_dir(run_id))),
                    "line": line_number,
                    "text": text,
                }
                for line_number, text in enumerate(content.splitlines(), 1)
            )
        return {
            "items": lines[offset : offset + limit],
            "total": len(lines),
            "offset": offset,
            "limit": limit,
        }

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
