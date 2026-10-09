"""Backup and restore of run facts, artifacts, logs, and source checkpoints."""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..state import atomic_write_json, utc_now
from .audit import (
    AuditLog,
    ConflictError,
    IdempotencyStore,
    ValidationError,
    timestamp_token,
)
from .security import Principal
from .store import NotFoundError, read_json, safe_component

BACKUP_STATUSES = ("REQUESTED", "RUNNING", "VERIFIED", "FAILED", "EXPIRED")
_LOG_STAGE_NAMES = {
    "quantize": ("quantization.log",),
    "evaluate": ("evaluation.log",),
    "runtime-smoke": ("runtime-smoke.log",),
    "publish": ("publish.log",),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_manifest_join(base: Path, relative: Any) -> Path:
    """Join a manifest-supplied relative path, refusing traversal escapes.

    ``manifest.json`` is not covered by the write-once immutability guard, so a
    corrupt or tampered manifest could otherwise point ``entry["path"]`` at an
    absolute path or ``../`` sequence and make an admin-triggered restore read
    or write outside the backup/target directory.
    """

    if not isinstance(relative, str) or not relative:
        raise ValidationError(f"backup manifest path is invalid: {relative!r}")
    candidate = (base / relative).resolve()
    base_resolved = base.resolve()
    if candidate != base_resolved and base_resolved not in candidate.parents:
        raise ValidationError(
            f"backup manifest path escapes its directory: {relative!r}"
        )
    return candidate


def _directory_size(path: Path, *, limit: int | None = None) -> int:
    total = 0
    for file in sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    ):
        total += file.stat().st_size
        if limit is not None and total > limit:
            return total
    return total


class BackupService:
    """Copy run facts into an immutable backup prefix and restore them again."""

    def __init__(
        self,
        *,
        runs_root: str | Path,
        backup_root: str | Path | None = None,
        prefix: str | None = None,
        retention_days: int = 30,
        max_source_bytes: int = 0,
        audit: AuditLog | None = None,
        idempotency: IdempotencyStore | None = None,
    ) -> None:
        self.runs_root = Path(runs_root).expanduser()
        self.backup_root = (
            Path(backup_root).expanduser()
            if backup_root is not None
            else self.runs_root / "_backups"
        )
        self.prefix = prefix or f"file://{self.backup_root}"
        self.retention_days = int(retention_days)
        self.max_source_bytes = int(max_source_bytes)
        self.audit = audit or AuditLog(self.runs_root)
        self.idempotency = idempotency or IdempotencyStore(self.runs_root)

    # ------------------------------------------------------------------ scope

    @staticmethod
    def normalize_scope(scope: Any) -> dict[str, Any]:
        if scope is None:
            scope = {}
        if not isinstance(scope, dict):
            raise ValidationError("scope must be a mapping")
        logs = scope.get("logs", False)
        if logs not in (True, False) and not isinstance(logs, list):
            raise ValidationError("scope.logs must be true, false, or a list of stages")
        if isinstance(logs, list):
            unknown = sorted(set(map(str, logs)) - set(_LOG_STAGE_NAMES))
            if unknown:
                raise ValidationError(f"unknown log stages: {', '.join(unknown)}")
        return {
            "metadata": bool(scope.get("metadata", True)),
            "artifact": bool(scope.get("artifact", True)),
            "logs": logs,
            "source_checkpoint": bool(scope.get("source_checkpoint", False)),
        }

    def _model_ids(self, run_dir: Path) -> list[str]:
        return sorted(
            path.name
            for path in run_dir.iterdir()
            if path.is_dir()
            and path.name not in {"events", "attempt-plans", "logs"}
            and not path.name.startswith("_")
        )

    def _scope_files(
        self, run_id: str, scope: dict[str, Any]
    ) -> tuple[list[tuple[Path, str]], list[dict[str, str]]]:
        run_dir = self.runs_root / safe_component(run_id, "run_id")
        if not run_dir.is_dir():
            raise NotFoundError(f"run {run_id!r} was not found")
        files: list[tuple[Path, str]] = []
        risks: list[dict[str, str]] = []

        def add(path: Path) -> None:
            if path.is_file():
                files.append((path, str(path.relative_to(run_dir))))

        if scope["metadata"]:
            for name in (
                "execution-plan.json",
                "run.json",
                "cancel.json",
                "aggregate-report.json",
            ):
                add(run_dir / name)
            for pattern in ("events/*.json", "attempt-plans/*.json"):
                for path in sorted(run_dir.glob(pattern)):
                    add(path)
            for model_id in self._model_ids(run_dir):
                model_dir = run_dir / model_id
                for name in (
                    "input-manifest.json",
                    "artifact-manifest.json",
                    "current-attempt.json",
                ):
                    add(model_dir / name)
                for pattern in (
                    "state/*.json",
                    "reports/*",
                    "attempts/*/state/*.json",
                    "attempts/*/reports/*",
                ):
                    for path in sorted(model_dir.glob(pattern)):
                        add(path)

        if scope["artifact"]:
            for model_id in self._model_ids(run_dir):
                for path in sorted((run_dir / model_id / "model").rglob("*")):
                    add(path)

        logs = scope["logs"]
        if logs:
            for model_id in self._model_ids(run_dir):
                log_dir = run_dir / model_id / "logs"
                for path in sorted(log_dir.rglob("*")) if log_dir.is_dir() else []:
                    if not path.is_file():
                        continue
                    if isinstance(logs, list):
                        names = {
                            name for stage in logs for name in _LOG_STAGE_NAMES[stage]
                        }
                        if path.name not in names:
                            continue
                    add(path)

        if scope["source_checkpoint"]:
            source = self._source_checkpoint(run_dir)
            if source is None:
                risks.append(
                    {
                        "code": "SOURCE_CHECKPOINT_UNKNOWN",
                        "severity": "error",
                        "message": (
                            "no input manifest recorded a source checkpoint path"
                        ),
                    }
                )
            elif not source.is_dir():
                risks.append(
                    {
                        "code": "SOURCE_CHECKPOINT_MISSING",
                        "severity": "error",
                        "message": f"source checkpoint is not available: {source}",
                    }
                )
            else:
                if self.max_source_bytes <= 0:
                    risks.append(
                        {
                            "code": "SOURCE_CHECKPOINT_NOT_ALLOWED",
                            "severity": "error",
                            "message": (
                                "backing up a source checkpoint needs "
                                "MODEL_QUALITY_BACKUP_MAX_SOURCE_BYTES"
                            ),
                        }
                    )
                else:
                    size = _directory_size(source, limit=self.max_source_bytes)
                    if size > self.max_source_bytes:
                        risks.append(
                            {
                                "code": "SOURCE_CHECKPOINT_TOO_LARGE",
                                "severity": "error",
                                "message": (
                                    f"source checkpoint needs more than "
                                    f"{self.max_source_bytes} bytes"
                                ),
                            }
                        )
                    else:
                        for path in sorted(
                            candidate
                            for candidate in source.rglob("*")
                            if candidate.is_file()
                        ):
                            files.append((path, f"source-checkpoint/{path.name}"))
        return files, risks

    @staticmethod
    def _source_checkpoint(run_dir: Path) -> Path | None:
        for model_id in sorted(
            path.name for path in run_dir.iterdir() if path.is_dir()
        ):
            manifest = read_json(run_dir / model_id / "input-manifest.json")
            if isinstance(manifest, dict):
                model = manifest.get("model")
                if isinstance(model, dict):
                    path = model.get("source", {}).get("path")
                    if isinstance(path, str) and path:
                        return Path(path).expanduser()
            preflight = read_json(run_dir / model_id / "state" / "preflight.json")
            if isinstance(preflight, dict) and preflight.get("source_path"):
                return Path(str(preflight["source_path"])).expanduser()
        return None

    # ---------------------------------------------------------------- preview

    def preview(self, run_id: str, scope: Any) -> dict[str, Any]:
        normalized = self.normalize_scope(scope)
        files, risks = self._scope_files(run_id, normalized)
        backup_id = f"{timestamp_token()}_{run_id}"
        expires_at = datetime.now(timezone.utc) + timedelta(days=self.retention_days)
        errors = [risk for risk in risks if risk["severity"] == "error"]
        return {
            "run_id": run_id,
            "backup_id": backup_id,
            "scope": normalized,
            "file_count": len(files),
            "total_bytes": sum(path.stat().st_size for path, _ in files),
            "target_prefix": f"{self.prefix}/{backup_id}",
            "retention": {
                "days": self.retention_days,
                "expires_at": expires_at.isoformat(),
            },
            "estimated_seconds": max(1, len(files) // 50),
            "ready": not errors,
            "risks": risks,
            "files": [
                {"path": relative, "size_bytes": path.stat().st_size}
                for path, relative in files
            ],
        }

    # ----------------------------------------------------------------- create

    def create(
        self, run_id: str, request: dict[str, Any], principal: Principal
    ) -> dict[str, Any]:
        principal.require("backup")
        preview = self.preview(run_id, request.get("scope"))
        if not preview["ready"]:
            raise ConflictError(
                "backup preview is not ready: "
                + "; ".join(
                    risk["message"]
                    for risk in preview["risks"]
                    if risk["severity"] == "error"
                )
            )
        payload = {
            "run_id": run_id,
            "scope": preview["scope"],
            "backup_id": preview["backup_id"],
        }
        explicit = request.get("idempotency_key")
        key = explicit or f"run.backup:{preview['backup_id']}"
        claim = self.idempotency.begin(key, payload)
        if claim["replayed"]:
            return claim["response"]
        try:
            record = self._create(run_id, preview, principal)
        except Exception as error:
            self.idempotency.fail(key, str(error))
            raise
        self.audit.record(
            actor=principal.actor,
            action="run.backup",
            target=f"{run_id}",
            result=record["status"],
            request=request,
            details={
                "backup_id": record["backup_id"],
                "file_count": record["file_count"],
                "total_bytes": record["total_bytes"],
                "scope": record["scope"],
                "target_prefix": record["target_prefix"],
            },
        )
        self.idempotency.complete(key, record)
        return record

    def _create(
        self, run_id: str, preview: dict[str, Any], principal: Principal
    ) -> dict[str, Any]:
        backup_id = preview["backup_id"]
        backup_dir = self.backup_root / backup_id
        if backup_dir.exists():
            raise ConflictError(f"backup {backup_id!r} already exists")
        state_path = backup_dir / "state.json"
        atomic_write_json(
            state_path,
            {
                "schema_version": 1,
                "backup_id": backup_id,
                "run_id": run_id,
                "status": "REQUESTED",
                "actor": principal.actor,
                "updated_at": utc_now(),
            },
        )
        atomic_write_json(
            state_path,
            {
                "schema_version": 1,
                "backup_id": backup_id,
                "run_id": run_id,
                "status": "RUNNING",
                "actor": principal.actor,
                "updated_at": utc_now(),
            },
        )
        files, _ = self._scope_files(run_id, preview["scope"])
        records = []
        total_bytes = 0
        for source, relative in files:
            target = backup_dir / "files" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            size = target.stat().st_size
            total_bytes += size
            records.append(
                {
                    "schema_version": 1,
                    "path": relative,
                    "size_bytes": size,
                    "sha256": _sha256(target),
                    "source_uri": f"file://{source}",
                    "target_uri": f"{self.prefix}/{backup_id}/files/{relative}",
                }
            )
        manifest = {
            "schema_version": 1,
            "backup_id": backup_id,
            "run_id": run_id,
            "file_count": len(records),
            "total_bytes": total_bytes,
            "files": records,
        }
        atomic_write_json(backup_dir / "manifest.json", manifest)
        immutable = {
            "schema_version": 1,
            "backup_id": backup_id,
            "run_id": run_id,
            "created_by": principal.actor,
            "created_at": utc_now(),
            "scope": preview["scope"],
            "file_count": len(records),
            "total_bytes": total_bytes,
            "target_prefix": preview["target_prefix"],
            "retention": preview["retention"],
            "manifest_sha256": _sha256(backup_dir / "manifest.json"),
            "status": "VERIFIED",
        }
        self._write_once(backup_dir / "BACKUP.json", immutable)
        atomic_write_json(
            state_path,
            {
                "schema_version": 1,
                "backup_id": backup_id,
                "run_id": run_id,
                "status": "VERIFIED",
                "actor": principal.actor,
                "updated_at": utc_now(),
                "file_count": len(records),
                "total_bytes": total_bytes,
            },
        )
        return {**immutable, "path": str(backup_dir)}

    @staticmethod
    def _write_once(path: Path, value: Any) -> None:
        if path.exists():
            raise ConflictError(f"{path.name} is immutable and already exists")
        atomic_write_json(path, value)

    # ------------------------------------------------------------------- read

    def _load(self, backup_id: str) -> dict[str, Any]:
        directory = self.backup_root / safe_component(backup_id, "backup_id")
        immutable = read_json(directory / "BACKUP.json")
        if not isinstance(immutable, dict):
            raise NotFoundError(f"backup {backup_id!r} was not found")
        state = read_json(directory / "state.json")
        status = (
            str(state.get("status"))
            if isinstance(state, dict) and state.get("status")
            else str(immutable.get("status", "VERIFIED"))
        )
        expires_at = (immutable.get("retention") or {}).get("expires_at")
        if isinstance(expires_at, str):
            try:
                parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) > parsed and status == "VERIFIED":
                    status = "EXPIRED"
            except ValueError:
                pass
        return {
            **immutable,
            "status": status,
            "state": state,
            "path": str(directory),
            "manifest_path": str(directory / "manifest.json"),
        }

    def get(self, backup_id: str) -> dict[str, Any]:
        record = self._load(backup_id)
        manifest = read_json(Path(record["path"]) / "manifest.json")
        return {
            **record,
            "manifest": manifest if isinstance(manifest, dict) else None,
        }

    def list(self, run_id: str | None = None) -> dict[str, Any]:
        items = []
        directories = (
            sorted(self.backup_root.glob("*")) if self.backup_root.is_dir() else []
        )
        for directory in directories:
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            if not (directory / "BACKUP.json").is_file():
                continue
            try:
                record = self._load(directory.name)
            except (NotFoundError, ValueError):
                continue
            if run_id is not None and record.get("run_id") != run_id:
                continue
            items.append(record)
        items.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return {"items": items, "total": len(items)}

    # ---------------------------------------------------------------- restore

    def _verify(self, record: dict[str, Any]) -> dict[str, Any]:
        manifest = read_json(Path(record["path"]) / "manifest.json")
        if not isinstance(manifest, dict):
            raise NotFoundError("backup manifest is missing")
        missing = []
        size_mismatch = []
        checksum_mismatch = []
        files_dir = Path(record["path"]) / "files"
        for entry in manifest.get("files", []):
            target = files_dir / entry["path"]
            if not target.is_file():
                missing.append(entry["path"])
                continue
            if target.stat().st_size != entry["size_bytes"]:
                size_mismatch.append(entry["path"])
                continue
            if _sha256(target) != entry["sha256"]:
                checksum_mismatch.append(entry["path"])
        return {
            "file_count": manifest.get("file_count"),
            "total_bytes": manifest.get("total_bytes"),
            "missing": missing,
            "size_mismatch": size_mismatch,
            "checksum_mismatch": checksum_mismatch,
            "verified": not (missing or size_mismatch or checksum_mismatch),
        }

    def restore_preview(
        self, backup_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        record = self._load(backup_id)
        target_run_id = request.get("target_run_id") or (
            f"{record['run_id']}-restore-{timestamp_token()}"
        )
        safe_component(target_run_id, "target_run_id")
        target_dir = self.runs_root / target_run_id
        risks: list[dict[str, str]] = []
        if target_dir.exists():
            risks.append(
                {
                    "code": "TARGET_RUN_EXISTS",
                    "severity": "error",
                    "message": (
                        f"target run {target_run_id!r} already exists; restore "
                        "never overwrites an existing run"
                    ),
                }
            )
        if record["status"] == "EXPIRED":
            risks.append(
                {
                    "code": "BACKUP_EXPIRED",
                    "severity": "error",
                    "message": "backup retention has expired",
                }
            )
        verification = self._verify(record)
        if not verification["verified"]:
            risks.append(
                {
                    "code": "BACKUP_CORRUPT",
                    "severity": "error",
                    "message": (
                        "backup files failed verification: "
                        f"{len(verification['missing'])} missing, "
                        f"{len(verification['size_mismatch'])} size mismatches, "
                        f"{len(verification['checksum_mismatch'])} checksum mismatches"
                    ),
                }
            )
        errors = [risk for risk in risks if risk["severity"] == "error"]
        return {
            "backup_id": backup_id,
            "original_run_id": record["run_id"],
            "target_run_id": target_run_id,
            "ready": not errors,
            "verification": verification,
            "file_count": verification["file_count"],
            "total_bytes": verification["total_bytes"],
            "risks": risks,
            "next_steps": [
                "restore writes a new run_id and preserves the original "
                "fingerprints for reference",
                "validate, runtime-smoke, evaluation, and publish must be "
                "re-run before the restored artifact is released",
            ],
        }

    def restore(
        self, backup_id: str, request: dict[str, Any], principal: Principal
    ) -> dict[str, Any]:
        principal.require("restore")
        preview = self.restore_preview(backup_id, request)
        if not preview["ready"]:
            raise ConflictError(
                "restore preview is not ready: "
                + "; ".join(
                    risk["message"]
                    for risk in preview["risks"]
                    if risk["severity"] == "error"
                )
            )
        payload = {
            "backup_id": backup_id,
            "target_run_id": preview["target_run_id"],
        }
        explicit = request.get("idempotency_key")
        key = explicit or f"backup.restore:{preview['target_run_id']}"
        claim = self.idempotency.begin(key, payload)
        if claim["replayed"]:
            return claim["response"]
        try:
            record = self._load(backup_id)
            manifest = read_json(Path(record["path"]) / "manifest.json")
            if not isinstance(manifest, dict):
                raise NotFoundError("backup manifest is missing")
            target_dir = self.runs_root / preview["target_run_id"]
            if target_dir.exists():
                raise ConflictError(
                    f"target run {preview['target_run_id']!r} already exists"
                )
            files_dir = Path(record["path"]) / "files"
            for entry in manifest.get("files", []):
                source = _safe_manifest_join(files_dir, entry.get("path"))
                destination = _safe_manifest_join(target_dir, entry.get("path"))
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            artifact_entry = next(
                (
                    entry
                    for entry in manifest.get("files", [])
                    if str(entry.get("path", "")).endswith("artifact-manifest.json")
                ),
                None,
            )
            artifact_manifest = (
                read_json(target_dir / artifact_entry["path"])
                if isinstance(artifact_entry, dict)
                else None
            )
            restore_record = {
                "schema_version": 1,
                "restore_id": f"restore-{timestamp_token()}",
                "source_backup_id": backup_id,
                "original_run_id": record["run_id"],
                "target_run_id": preview["target_run_id"],
                "restored_by": principal.actor,
                "restored_at": utc_now(),
                "file_count": manifest.get("file_count"),
                "total_bytes": manifest.get("total_bytes"),
                "manifest_sha256": record.get("manifest_sha256"),
                "original_artifact_fingerprint": (
                    artifact_manifest.get("artifact_fingerprint")
                    if isinstance(artifact_manifest, dict)
                    else None
                ),
                "status": "VERIFIED",
                "released": False,
                "note": (
                    "the restored artifact is not a release candidate until "
                    "validate, runtime-smoke, evaluation, and publish pass again"
                ),
            }
            atomic_write_json(target_dir / "RESTORE.json", restore_record)
        except Exception as error:
            self.idempotency.fail(key, str(error))
            raise
        self.audit.record(
            actor=principal.actor,
            action="backup.restore",
            target=preview["target_run_id"],
            result="VERIFIED",
            request=request,
            details={
                "backup_id": backup_id,
                "original_run_id": record["run_id"],
                "file_count": restore_record["file_count"],
                "total_bytes": restore_record["total_bytes"],
            },
        )
        self.idempotency.complete(key, restore_record)
        return restore_record
