"""Dependency-free HTTP control plane and read-only observer for CI runs."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs, unquote
from wsgiref.simple_server import make_server

from ..config import ConfigError
from .audit import AuditLog, ConflictError, IdempotencyStore, ValidationError
from .backups import BackupService
from .control import ControlPlane
from .executors import create_executor
from .jobs import JobStore, ReservationLedger
from .ops import (
    NotificationService,
    ScheduleStore,
    TrendService,
    capabilities,
    gpu_telemetry,
)
from .plans import LaunchPolicy, introspect_script
from .publishing import PublishService
from .security import (
    AuthenticationError,
    PermissionDenied,
    Principal,
    as_dict,
    principal_from_headers,
)
from .store import NotFoundError, RunStore, read_json
from .ui import asset as ui_asset
from .ui import is_ui_path

MAX_BODY_BYTES = 1024 * 1024


class MethodNotAllowed(Exception):
    """Raised when a known path is called with an unsupported HTTP method."""

    def __init__(self, allow: set[str]) -> None:
        super().__init__("method not allowed; allowed: " + ", ".join(sorted(allow)))
        self.allow = allow


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )


class ModelQualityAPI:
    """WSGI application exposing the model quality control plane."""

    def __init__(
        self,
        store: RunStore | None = None,
        *,
        config_path: str | None = None,
        policy: LaunchPolicy | None = None,
        git_sha: str | None = None,
        executor_upload: bool = False,
        control: ControlPlane | None = None,
        publishing: PublishService | None = None,
        backups: BackupService | None = None,
        schedules: ScheduleStore | None = None,
        notifications: NotificationService | None = None,
        require_token: str | None = None,
    ) -> None:
        self.store = store or RunStore()
        self.root = self.store.root
        self.config_path = config_path or os.getenv("MODEL_QUALITY_CONFIG") or None
        self.git_sha = git_sha or os.getenv("MODEL_QUALITY_GIT_SHA", "local")
        self.policy = policy or LaunchPolicy.from_env()
        self.require_token = require_token
        self.audit = AuditLog(self.root)
        self.idempotency = IdempotencyStore(self.root)
        self.jobs = JobStore(self.root)
        self.ledger = ReservationLedger(
            self.root, capacity_gpu_hours=self.policy.gpu_hour_capacity
        )
        self.trends = TrendService(self.store, self.root)
        self.schedules = schedules or ScheduleStore(self.root)
        self.notifications = notifications or NotificationService(self.root)
        self.executor = create_executor(
            self.policy.executor,
            root=self.root,
            config_path=self.config_path,
            upload=executor_upload,
        )
        self.control = control or self._build_control()
        self.publishing = publishing or PublishService(
            root=self.root,
            config_path=self.config_path,
            store=self.store,
            policy=self.policy,
            executor=self.executor,
            audit=self.audit,
            idempotency=self.idempotency,
            jobs=self.jobs,
        )
        self.backups = backups or BackupService(
            runs_root=self.root,
            backup_root=os.getenv("MODEL_QUALITY_BACKUP_ROOT"),
            prefix=os.getenv("MODEL_QUALITY_BACKUP_PREFIX"),
            retention_days=self.policy.backup_retention_days,
            max_source_bytes=self.policy.backup_max_source_bytes,
            audit=self.audit,
            idempotency=self.idempotency,
        )

    def _build_control(self) -> ControlPlane | None:
        if not self.config_path:
            return None
        return ControlPlane(
            root=self.root,
            store=self.store,
            config_path=self.config_path,
            policy=self.policy,
            executor=self.executor,
            ledger=self.ledger,
            jobs=self.jobs,
            audit=self.audit,
            idempotency=self.idempotency,
            git_sha=self.git_sha,
            executor_upload=False,
        )

    # ------------------------------------------------------------------ WSGI

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]):
        method = environ.get("REQUEST_METHOD", "GET").upper()
        path = unquote(environ.get("PATH_INFO", "/")).rstrip("/") or "/"
        query = parse_qs(environ.get("QUERY_STRING", ""))
        try:
            principal = self._principal(environ)
            if method in {"GET", "HEAD"}:
                value, content_type, headers = self.dispatch(principal, path, query)
            elif method in {"POST", "PUT", "PATCH", "DELETE"}:
                body = self._read_body(environ)
                value, content_type, headers = self.dispatch_write(
                    principal, method, path, query, body
                )
            else:
                return self._respond(
                    start_response,
                    405,
                    {"error": f"method {method} is not supported"},
                    extra_headers=[("Allow", "GET, POST, DELETE")],
                )
        except AuthenticationError as error:
            return self._respond(
                start_response,
                401,
                {"error": str(error)},
                extra_headers=[("WWW-Authenticate", "Bearer")],
            )
        except PermissionDenied as error:
            return self._respond(start_response, 403, {"error": str(error)})
        except MethodNotAllowed as error:
            return self._respond(
                start_response,
                405,
                {"error": str(error)},
                extra_headers=[("Allow", ", ".join(sorted(error.allow)))],
            )
        except (NotFoundError, FileNotFoundError) as error:
            return self._respond(start_response, 404, {"error": str(error)})
        except ConflictError as error:
            return self._respond(start_response, 409, {"error": str(error)})
        except (ValidationError, ConfigError, ValueError) as error:
            return self._respond(start_response, 400, {"error": str(error)})
        except (OSError, RuntimeError) as error:
            return self._respond(start_response, 400, {"error": str(error)})
        except Exception as error:  # pragma: no cover - server boundary
            return self._respond(
                start_response, 500, {"error": f"internal server error: {error}"}
            )
        return self._respond(
            start_response,
            200,
            value,
            content_type=content_type,
            extra_headers=headers,
        )

    def _principal(self, environ: dict[str, Any]) -> Principal:
        headers = {}
        for key, value in environ.items():
            if key.startswith("HTTP_"):
                headers[key[5:].replace("_", "-").lower()] = str(value)
        if environ.get("CONTENT_TYPE"):
            headers["content-type"] = str(environ["CONTENT_TYPE"])
        return principal_from_headers(headers, required_token=self.require_token)

    @staticmethod
    def _read_body(environ: dict[str, Any]) -> dict[str, Any]:
        try:
            length = int(environ.get("CONTENT_LENGTH") or 0)
        except ValueError as error:
            raise ValidationError("invalid Content-Length header") from error
        if length > MAX_BODY_BYTES:
            raise ValidationError("request body exceeds 1 MiB")
        if length <= 0:
            return {}
        stream = environ.get("wsgi.input")
        raw = stream.read(length) if stream is not None else b""
        if not raw:
            return {}
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValidationError(f"request body must be JSON: {error}") from error
        if not isinstance(value, dict):
            raise ValidationError("request body must be a JSON object")
        return value

    @staticmethod
    def _respond(
        start_response: Callable[..., Any],
        status: int,
        value: Any,
        *,
        content_type: str = "application/json",
        extra_headers: list[tuple[str, str]] | None = None,
    ):
        phrases = {
            200: "OK",
            400: "Bad Request",
            401: "Unauthorized",
            403: "Forbidden",
            404: "Not Found",
            405: "Method Not Allowed",
            409: "Conflict",
            500: "Internal Server Error",
            503: "Service Unavailable",
        }
        body = value if isinstance(value, bytes) else _json_bytes(value)
        headers = [
            ("Content-Type", content_type),
            ("Content-Length", str(len(body))),
            ("Cache-Control", "no-store"),
        ]
        headers.extend(extra_headers or [])
        start_response(f"{status} {phrases.get(status, 'Error')}", headers)
        return [body]

    @staticmethod
    def _one(query: dict[str, list[str]], name: str) -> str | None:
        values = query.get(name)
        return values[0] if values else None

    def _require_control(self) -> ControlPlane:
        if self.control is None:
            raise ValidationError(
                "control operations need a model quality manifest; start the "
                "web server with --config"
            )
        return self.control

    # ------------------------------------------------------------ read routes

    def dispatch(
        self, principal: Principal, path: str, query: dict[str, list[str]]
    ) -> tuple[Any, str, list[tuple[str, str]]]:
        if is_ui_path(path):
            return ui_asset(path)
        parts = [part for part in path.split("/") if part]
        if parts == ["api", "health"]:
            return (
                {"status": "ok", "principal": as_dict(principal)},
                "application/json",
                [],
            )
        if parts == ["api", "whoami"]:
            return as_dict(principal), "application/json", []
        if parts == ["api", "models"]:
            return (
                {"items": self.store.models(self.config_path)},
                "application/json",
                [],
            )
        if parts == ["api", "scripts", "introspect"]:
            principal.require("read")
            return (
                introspect_script(self._one(query, "path"), policy=self.policy),
                "application/json",
                [],
            )
        if parts == ["api", "capabilities"]:
            principal.require("read")
            return (
                capabilities(
                    config_path=self.config_path,
                    ledger=self.ledger,
                    policy=self.policy,
                    jobs=self.jobs,
                ),
                "application/json",
                [],
            )
        if parts == ["api", "trends", "quality"]:
            principal.require("read")
            return (
                self.trends.quality(
                    model_id=self._one(query, "model_id"),
                    limit=int(self._one(query, "limit") or 50),
                ),
                "application/json",
                [],
            )
        if parts == ["api", "trends", "cost"]:
            principal.require("read")
            return (
                self.trends.cost(limit=int(self._one(query, "limit") or 50)),
                "application/json",
                [],
            )
        if parts == ["api", "audit"]:
            principal.require("read")
            return (
                self.audit.read(
                    action=self._one(query, "action"),
                    actor=self._one(query, "actor"),
                    target=self._one(query, "target"),
                    offset=int(self._one(query, "offset") or 0),
                    limit=int(self._one(query, "limit") or 100),
                ),
                "application/json",
                [],
            )
        if parts == ["api", "ops", "schedules"]:
            principal.require("read")
            return self.schedules.list(), "application/json", []
        if parts == ["api", "ops", "gpus"]:
            principal.require("read")
            return gpu_telemetry(), "application/json", []
        if parts == ["api", "ops", "notifications"]:
            principal.require("read")
            return (
                {
                    "config": self.notifications.config(),
                    "log": self.notifications.log(
                        limit=int(self._one(query, "limit") or 100)
                    ),
                },
                "application/json",
                [],
            )
        if len(parts) == 3 and parts[:2] == ["api", "plans"]:
            return (
                self._require_control().get_plan(parts[2], principal),
                "application/json",
                [],
            )
        if parts == ["api", "runs"]:
            principal.require("read")
            return (
                self.store.list_runs(
                    status=self._one(query, "status"),
                    model=self._one(query, "model"),
                    since=self._one(query, "since"),
                    offset=int(self._one(query, "offset") or 0),
                    limit=int(self._one(query, "limit") or 50),
                ),
                "application/json",
                [],
            )
        if len(parts) >= 3 and parts[:2] == ["api", "runs"]:
            return self._dispatch_run_read(principal, parts, query)
        raise NotFoundError(f"unknown API path: {path}")

    def _dispatch_run_read(
        self, principal: Principal, parts: list[str], query: dict[str, list[str]]
    ) -> tuple[Any, str, list[tuple[str, str]]]:
        principal.require("read")
        run_id = parts[2]
        if len(parts) == 3:
            return self.store.get_run(run_id), "application/json", []
        if parts[3] == "plan-request" and len(parts) == 4:
            run_dir = self.store.root / run_id
            plan = read_json(run_dir / "execution-plan.json")
            source = plan.get("source_request") if isinstance(plan, dict) else None
            if not source:
                raise NotFoundError(
                    f"run {run_id!r} has no clonable plan request"
                )
            return source, "application/json", []
        if parts[3] == "events" and len(parts) == 4:
            after_value = self._one(query, "after")
            events = self.store.events(
                run_id, after=int(after_value) if after_value is not None else None
            )
            payload = b"".join(
                (
                    f"id: {event.get('sequence')}\n".encode("utf-8")
                    if event.get("sequence") is not None
                    else b""
                )
                + b"event: state\ndata: "
                + _json_bytes(event).rstrip(b"\n")
                + b"\n\n"
                for event in events
            )
            return (
                payload,
                "text/event-stream; charset=utf-8",
                [("X-Accel-Buffering", "no")],
            )
        if parts[3] == "jobs" and len(parts) == 4:
            return (
                self._require_control().list_jobs(
                    run_id,
                    model_id=self._one(query, "model_id"),
                    attempt_id=self._one(query, "attempt_id"),
                    principal=principal,
                ),
                "application/json",
                [],
            )
        if parts[3] == "backups" and len(parts) == 4:
            self.store.get_run(run_id)
            return self.backups.list(run_id), "application/json", []
        if parts[3] == "artifacts" and len(parts) == 6 and parts[5] == "access":
            model_id = self._one(query, "model_id")
            if not model_id:
                raise ValidationError("model_id is required for artifact access")
            return (
                self.publishing.artifact_access(run_id, parts[4], model_id, principal),
                "application/json",
                [],
            )
        if parts[3] == "models" and len(parts) >= 5:
            model_id = parts[4]
            if len(parts) == 5:
                return self.store.get_model(run_id, model_id), "application/json", []
            suffix = parts[5:]
            if suffix == ["artifacts"]:
                return (
                    self.store.get_artifacts(run_id, model_id),
                    "application/json",
                    [],
                )
            if suffix == ["publish"]:
                return self.store.get_publish(run_id, model_id), "application/json", []
            if suffix == ["logs"]:
                return (
                    self.store.get_logs(
                        run_id,
                        model_id,
                        stage=self._one(query, "stage"),
                        attempt_id=self._one(query, "attempt_id"),
                        offset=int(self._one(query, "offset") or 0),
                        limit=int(self._one(query, "limit") or 1000),
                        tail=(self._one(query, "tail") or "").lower()
                        in ("1", "true", "yes"),
                    ),
                    "application/json",
                    [],
                )
            if suffix == ["jobs"]:
                return (
                    self._require_control().list_jobs(
                        run_id,
                        model_id=model_id,
                        attempt_id=self._one(query, "attempt_id"),
                        principal=principal,
                    ),
                    "application/json",
                    [],
                )
        raise NotFoundError(f"unknown API path: {'/'.join(parts)}")

    # ----------------------------------------------------------- write routes

    def dispatch_write(
        self,
        principal: Principal,
        method: str,
        path: str,
        query: dict[str, list[str]],
        body: dict[str, Any],
    ) -> tuple[Any, str, list[tuple[str, str]]]:
        parts = [part for part in path.split("/") if part]
        control = self._require_control
        if parts == ["api", "plans", "preview"]:
            if method != "POST":
                raise MethodNotAllowed({"POST"})
            return (
                control().preview_plan(body, principal),
                "application/json",
                [],
            )
        if parts == ["api", "runs"]:
            if method != "POST":
                raise MethodNotAllowed({"POST"})
            return control().start_run(body, principal), "application/json", []
        if parts == ["api", "ops", "schedules"]:
            if method != "POST":
                raise MethodNotAllowed({"GET", "POST"})
            principal.require("schedule")
            cron = body.get("cron")
            plan_request = body.get("plan_request")
            if not isinstance(cron, str):
                raise ValidationError("cron is required")
            schedule = self.schedules.create(
                cron=cron,
                plan_request=plan_request,
                actor=principal.actor,
                enabled=bool(body.get("enabled", True)),
            )
            self.audit.record(
                actor=principal.actor,
                action="schedule.create",
                target=schedule["schedule_id"],
                result="ACCEPTED",
                request=body,
                details={"cron": cron, "next_run_at": schedule["next_run_at"]},
            )
            return schedule, "application/json", []
        if parts == ["api", "ops", "notifications", "test"]:
            if method != "POST":
                raise MethodNotAllowed({"POST"})
            principal.require("ops.read")
            return (
                self.notifications.notify(
                    str(body.get("event") or "test"), body.get("payload") or {}
                ),
                "application/json",
                [],
            )
        if len(parts) == 4 and parts[:2] == ["api", "ops"] and parts[2] == "schedules":
            if method != "DELETE":
                raise MethodNotAllowed({"DELETE"})
            principal.require("schedule")
            schedule_id = parts[3]
            result = self.schedules.delete(schedule_id)
            self.audit.record(
                actor=principal.actor,
                action="schedule.delete",
                target=schedule_id,
                result="ACCEPTED",
                request=body,
            )
            return result, "application/json", []
        if len(parts) == 5 and parts[:2] == ["api", "ops"] and parts[2] == "schedules":
            if method != "POST":
                raise MethodNotAllowed({"POST"})
            principal.require("schedule")
            schedule_id, action = parts[3], parts[4]
            if action not in {"enable", "disable"}:
                raise NotFoundError(f"unknown schedule action: {action!r}")
            result = self.schedules.set_enabled(schedule_id, action == "enable")
            self.audit.record(
                actor=principal.actor,
                action=f"schedule.{action}",
                target=schedule_id,
                result="ACCEPTED",
                request=body,
            )
            return result, "application/json", []
        if len(parts) >= 4 and parts[:2] == ["api", "runs"]:
            return self._dispatch_run_write(principal, method, parts, query, body)
        if len(parts) == 4 and parts[:2] == ["api", "backups"]:
            if method != "POST":
                raise MethodNotAllowed({"POST"})
            backup_id = parts[2]
            action = parts[3]
            if action == "restore":
                return (
                    self.backups.restore(backup_id, body, principal),
                    "application/json",
                    [],
                )
        if len(parts) == 5 and parts[:2] == ["api", "backups"]:
            if method != "POST":
                raise MethodNotAllowed({"POST"})
            backup_id, action, sub = parts[2], parts[3], parts[4]
            if action == "restore" and sub == "preview":
                principal.require("restore")
                return (
                    self.backups.restore_preview(backup_id, body),
                    "application/json",
                    [],
                )
        raise NotFoundError(f"unknown API path: {path}")

    def _dispatch_run_write(
        self,
        principal: Principal,
        method: str,
        parts: list[str],
        query: dict[str, list[str]],
        body: dict[str, Any],
    ) -> tuple[Any, str, list[tuple[str, str]]]:
        control = self._require_control
        run_id = parts[2]
        if len(parts) == 4:
            if method != "POST":
                raise MethodNotAllowed({"POST"})
            action = parts[3]
            if action == "retry":
                return (
                    control().retry(run_id, body, principal),
                    "application/json",
                    [],
                )
            if action == "evaluate":
                return (
                    control().evaluate(run_id, body, principal),
                    "application/json",
                    [],
                )
            if action == "cancel":
                return (
                    control().cancel(run_id, body, principal),
                    "application/json",
                    [],
                )
            if action == "backup":
                return (
                    self.backups.create(run_id, body, principal),
                    "application/json",
                    [],
                )
        if len(parts) == 5 and parts[3] in {"publish", "backup"}:
            if method != "POST":
                raise MethodNotAllowed({"POST"})
            action, sub = parts[3], parts[4]
            if action == "backup" and sub == "preview":
                principal.require("backup")
                return (
                    self.backups.preview(run_id, body.get("scope")),
                    "application/json",
                    [],
                )
            if action == "publish":
                model_id = self._single_model(run_id, body, sub)
                if sub == "preview":
                    principal.require("publish")
                    return (
                        self.publishing.preview(run_id, model_id),
                        "application/json",
                        [],
                    )
        if len(parts) >= 6 and parts[3] == "models":
            if method != "POST":
                raise MethodNotAllowed({"POST"})
            model_id = parts[4]
            suffix = parts[5:]
            if suffix == ["publish", "preview"]:
                principal.require("publish")
                return (
                    self.publishing.preview(run_id, model_id),
                    "application/json",
                    [],
                )
            if suffix == ["publish"]:
                return (
                    self.publishing.publish(run_id, model_id, body, principal),
                    "application/json",
                    [],
                )
            if len(parts) == 8 and parts[5] == "jobs" and parts[7] == "cancel":
                return (
                    control().cancel_job(run_id, model_id, parts[6], body, principal),
                    "application/json",
                    [],
                )
            if len(parts) == 8 and parts[5] == "jobs" and parts[7] == "retry":
                return (
                    control().retry_job(run_id, model_id, parts[6], body, principal),
                    "application/json",
                    [],
                )
            if len(parts) == 8 and parts[5] == "stages" and parts[7] == "rerun":
                return (
                    control().rerun_stage(
                        run_id, model_id, parts[6], body, principal
                    ),
                    "application/json",
                    [],
                )
            if len(parts) == 8 and parts[5] == "stages" and parts[7] == "skip":
                return (
                    control().skip_stage(
                        run_id, model_id, parts[6], body, principal
                    ),
                    "application/json",
                    [],
                )
        raise NotFoundError(f"unknown API path: {'/'.join(parts)}")

    def _single_model(self, run_id: str, body: dict[str, Any], action: str) -> str:
        model_id = body.get("model_id")
        if isinstance(model_id, str) and model_id:
            return model_id
        models = self.store.get_run(run_id).get("models", [])
        if len(models) == 1:
            return str(models[0]["model_id"])
        raise ValidationError(
            f"model_id is required for /publish/{action} when a run has "
            f"{len(models)} models"
        )


def create_app(
    store: RunStore | None = None,
    *,
    config_path: str | None = None,
    policy: LaunchPolicy | None = None,
    git_sha: str | None = None,
    executor_upload: bool = False,
    **kwargs: Any,
) -> ModelQualityAPI:
    """Create the WSGI application used by local deployments and tests."""

    return ModelQualityAPI(
        store,
        config_path=config_path,
        policy=policy,
        git_sha=git_sha,
        executor_upload=executor_upload,
        **kwargs,
    )


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    store: RunStore | None = None,
    config_path: str | None = None,
    git_sha: str | None = None,
    executor_upload: bool = False,
) -> None:
    app = create_app(
        store,
        config_path=config_path,
        git_sha=git_sha,
        executor_upload=executor_upload,
    )
    with make_server(host, port, app) as server:
        print(f"model-quality web listening on http://{host}:{port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the model quality API")
    parser.add_argument(
        "--host", default=os.getenv("MODEL_QUALITY_WEB_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("MODEL_QUALITY_WEB_PORT", "8000"))
    )
    parser.add_argument("--runs-root")
    parser.add_argument("--config", default=os.getenv("MODEL_QUALITY_CONFIG") or None)
    parser.add_argument("--git-sha", default=os.getenv("MODEL_QUALITY_GIT_SHA"))
    parser.add_argument(
        "--executor-upload",
        action="store_true",
        help="allow the Buildkite adapter to upload the rendered pipeline",
    )
    args = parser.parse_args()
    serve(
        args.host,
        args.port,
        store=RunStore(args.runs_root) if args.runs_root else None,
        config_path=args.config,
        git_sha=args.git_sha,
        executor_upload=args.executor_upload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
