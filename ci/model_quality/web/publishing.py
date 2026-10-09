"""Publish control plane: preview, confirmation, and publish dispatch."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..config import ConfigError, load_model_config
from .audit import (
    AuditLog,
    ConflictError,
    IdempotencyStore,
    ValidationError,
    request_fingerprint,
)
from .executors import ExecutorAdapter, LaunchSpec
from .jobs import JobStore
from .plans import LaunchPolicy
from .security import Principal
from .store import NotFoundError, read_json, safe_component

# Files the publisher generates itself; they must not be part of the frozen
# upload manifest that they describe.
GENERATED_PUBLICATION_FILES = frozenset(
    {"reports/upload-manifest.json", "reports/SUCCESS.json", "SUCCESS.json"}
)
DEFAULT_PREVIEW_FILE_LIMIT = 200
# The preview digests files inside the synchronous HTTP request only to show an
# inventory before publishing. Quantized weight shards are tens of GB, so
# hashing them here would stall the request for minutes (a 32B W8A8 model is
# ~32GB across two safetensors files). The published artifact's integrity is
# already captured by ``artifact-manifest.json``'s content fingerprint, so files
# above this size are listed with their byte size and a null digest instead.
DEFAULT_PREVIEW_HASH_MAX_BYTES = 256 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PublishService:
    """Preview and request publication for a model artifact in a run."""

    def __init__(
        self,
        *,
        root: str | Path,
        config_path: str | Path | None,
        store,
        policy: LaunchPolicy,
        executor: ExecutorAdapter,
        audit: AuditLog | None = None,
        idempotency: IdempotencyStore | None = None,
        jobs: JobStore | None = None,
        file_limit: int = DEFAULT_PREVIEW_FILE_LIMIT,
        hash_max_bytes: int = DEFAULT_PREVIEW_HASH_MAX_BYTES,
    ) -> None:
        self.root = Path(root).expanduser()
        self.config_path = Path(config_path) if config_path else None
        self.store = store
        self.policy = policy
        self.executor = executor
        self.audit = audit or AuditLog(self.root)
        self.idempotency = idempotency or IdempotencyStore(self.root)
        self.jobs = jobs or JobStore(self.root)
        self.file_limit = int(file_limit)
        self.hash_max_bytes = int(hash_max_bytes)

    # ---------------------------------------------------------------- helpers

    def _model_dir(self, run_id: str, model_id: str) -> Path:
        path = (
            self.root
            / safe_component(run_id, "run_id")
            / safe_component(model_id, "model_id")
        )
        if not path.is_dir():
            raise NotFoundError(f"model {model_id!r} was not found in run {run_id!r}")
        return path

    def _definition(self, model_id: str, run_id: str) -> dict[str, Any]:
        """Return the model definition that owns this run's upload target.

        Manifest-defined models come from ``--config``. Script-first runs are
        not in that manifest, so their run-local ``model-config.yaml`` -- which
        carries the upload block the plan recorded -- is the authority.
        """

        candidates: list[Path] = []
        if self.config_path is not None:
            candidates.append(self.config_path)
        candidates.append(
            self.root / safe_component(run_id, "run_id") / "model-config.yaml"
        )
        for path in candidates:
            if not path.is_file():
                continue
            try:
                config = load_model_config(path)
            except ConfigError:
                continue
            for model in config["models"]:
                if model["id"] == model_id:
                    return model
        raise NotFoundError(
            f"model {model_id!r} is not defined in the manifest or the run config"
        )

    @staticmethod
    def _promoted_reports(model_dir: Path) -> tuple[Path, dict[str, Any] | None]:
        current = read_json(model_dir / "current-attempt.json")
        attempt_id = current.get("attempt_id") if isinstance(current, dict) else None
        if isinstance(attempt_id, str) and attempt_id:
            candidate = model_dir / "attempts" / attempt_id / "reports"
            if candidate.is_dir():
                return candidate, current
        return model_dir / "reports", current if isinstance(current, dict) else None

    def _collect_files(
        self, run_dir: Path, allowlist: list[str], reports_dir: Path
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        files: list[dict[str, Any]] = []
        risks: list[dict[str, str]] = []
        for relative in allowlist:
            if relative == "reports":
                candidate = reports_dir
            elif relative.startswith("reports/"):
                candidate = reports_dir / relative.removeprefix("reports/")
            else:
                candidate = run_dir / relative
            try:
                resolved = candidate.resolve()
                resolved.relative_to(run_dir.resolve())
            except (ValueError, OSError):
                risks.append(
                    {
                        "code": "ALLOWLIST_ESCAPES_RUN",
                        "severity": "error",
                        "message": (
                            f"allowlist entry escapes the run directory: {relative}"
                        ),
                    }
                )
                continue
            if not candidate.exists():
                risks.append(
                    {
                        "code": "ALLOWLIST_PATH_MISSING",
                        "severity": "error",
                        "message": f"allowlist path is missing: {relative}",
                    }
                )
                continue
            if candidate.is_file():
                candidates = [candidate]
            else:
                candidates = sorted(
                    path for path in candidate.rglob("*") if path.is_file()
                )
            for path in candidates:
                resolved_path = path.resolve()
                if resolved_path == run_dir.resolve():
                    continue
                relative_path = str(resolved_path.relative_to(run_dir.resolve()))
                if relative_path in GENERATED_PUBLICATION_FILES:
                    continue
                size_bytes = resolved_path.stat().st_size
                entry: dict[str, Any] = {
                    "path": relative_path,
                    "size_bytes": size_bytes,
                    "sha256": None,
                }
                # Only hash the first ``file_limit`` files, and never a file
                # larger than ``hash_max_bytes`` -- a multi-GB weight shard would
                # otherwise stall this synchronous request for minutes.
                if len(files) < self.file_limit and size_bytes <= self.hash_max_bytes:
                    entry["sha256"] = _sha256(resolved_path)
                files.append(entry)
        return files, risks

    # ---------------------------------------------------------------- preview

    def preview(self, run_id: str, model_id: str) -> dict[str, Any]:
        """Return the allowlist, file inventory, target, and publication risks."""

        model_dir = self._model_dir(run_id, model_id)
        definition = self._definition(model_id, run_id)
        upload = definition.get("upload", {})
        risks: list[dict[str, str]] = []

        remote_prefix = str(upload.get("remote_prefix", "")).rstrip("/")
        if not upload.get("enabled"):
            risks.append(
                {
                    "code": "UPLOAD_DISABLED",
                    "severity": "error",
                    "message": "upload.enabled is false for this model",
                }
            )
        if (
            not remote_prefix
            or remote_prefix in {"/", "bos:/"}
            or "replace-me" in remote_prefix
        ):
            risks.append(
                {
                    "code": "UNSAFE_REMOTE_PREFIX",
                    "severity": "error",
                    "message": "upload.remote_prefix is empty or a placeholder",
                }
            )
        remote_target = f"{remote_prefix}/{model_id}/runs/{run_id}"

        allowlist = upload.get("allowlist") or []
        if not isinstance(allowlist, list) or not allowlist:
            risks.append(
                {
                    "code": "EMPTY_ALLOWLIST",
                    "severity": "error",
                    "message": "upload.allowlist must name the published roots",
                }
            )
            allowlist = []

        reports_dir, current_attempt = self._promoted_reports(model_dir)
        files, file_risks = self._collect_files(Path(model_dir), allowlist, reports_dir)
        risks.extend(file_risks)

        manifest = read_json(model_dir / "artifact-manifest.json")
        summary = read_json(reports_dir / "summary.json")
        if not isinstance(manifest, dict):
            risks.append(
                {
                    "code": "MISSING_ARTIFACT_MANIFEST",
                    "severity": "error",
                    "message": "artifact-manifest.json is missing; validate first",
                }
            )
        if current_attempt is None:
            risks.append(
                {
                    "code": "MISSING_ATTEMPT_POINTER",
                    "severity": "error",
                    "message": "current-attempt.json is missing; report first",
                }
            )
        if not isinstance(summary, dict):
            risks.append(
                {
                    "code": "MISSING_PROMOTED_REPORT",
                    "severity": "error",
                    "message": "reports/summary.json is missing; report first",
                }
            )
        elif str(summary.get("status", "")).upper() == "FAIL":
            risks.append(
                {
                    "code": "REPORT_NOT_PASSING",
                    "severity": "error",
                    "message": "the promoted report status is FAIL",
                }
            )

        artifact_fingerprint = None
        content_fingerprint = None
        if isinstance(manifest, dict):
            artifact_fingerprint = manifest.get("artifact_fingerprint")
            content_fingerprint = manifest.get("artifact_content_fingerprint")
            if not content_fingerprint:
                risks.append(
                    {
                        "code": "MISSING_CONTENT_FINGERPRINT",
                        "severity": "error",
                        "message": "artifact-manifest.json has no content fingerprint",
                    }
                )
        if isinstance(current_attempt, dict) and isinstance(manifest, dict):
            if current_attempt.get("artifact_fingerprint") != artifact_fingerprint:
                risks.append(
                    {
                        "code": "FINGERPRINT_MISMATCH",
                        "severity": "error",
                        "message": (
                            "current-attempt.json and artifact-manifest.json "
                            "disagree on the artifact fingerprint"
                        ),
                    }
                )

        existing_publish = read_json(model_dir / "state" / "publish.json")
        already_success = (
            isinstance(existing_publish, dict)
            and str(existing_publish.get("status", "")).upper() in {"PASS", "SUCCESS"}
            and existing_publish.get("artifact_fingerprint") == artifact_fingerprint
        )
        if already_success:
            risks.append(
                {
                    "code": "ALREADY_PUBLISHED",
                    "severity": "warning",
                    "message": (
                        "this fingerprint was already published; re-publishing "
                        "is idempotent"
                    ),
                }
            )

        errors = [risk for risk in risks if risk["severity"] == "error"]
        return {
            "run_id": run_id,
            "model_id": model_id,
            "ready": not errors,
            "allowlist": list(allowlist),
            "file_count": len(files),
            "total_bytes": sum(int(entry["size_bytes"]) for entry in files),
            "files": files,
            "truncated": len(files) > self.file_limit,
            "remote_target": remote_target,
            "identity": {
                "artifact_fingerprint": artifact_fingerprint,
                "artifact_content_fingerprint": content_fingerprint,
                "evaluation_fingerprint": (
                    current_attempt.get("evaluation_fingerprint")
                    if isinstance(current_attempt, dict)
                    else None
                ),
                "attempt_id": (
                    current_attempt.get("attempt_id")
                    if isinstance(current_attempt, dict)
                    else None
                ),
            },
            "approvals_required": 2 if self.policy.require_dual_approval else 1,
            "risks": risks,
        }

    # ---------------------------------------------------------------- publish

    def publish(
        self,
        run_id: str,
        model_id: str,
        request: dict[str, Any],
        principal: Principal,
    ) -> dict[str, Any]:
        principal.require("publish")
        preview = self.preview(run_id, model_id)
        confirm = request.get("confirm")
        if confirm is not True:
            raise ValidationError(
                "publish requires confirm=true after reviewing the preview"
            )
        if self.policy.require_dual_approval:
            approvals = request.get("approvals")
            if not isinstance(approvals, list):
                raise ValidationError(
                    "this deployment requires two approvers; pass approvals=[a, b]"
                )
            distinct = {str(value) for value in approvals if str(value).strip()}
            if len(distinct) < 2:
                raise ValidationError("two distinct approvers are required")
        if not preview["ready"]:
            errors = [
                risk["message"]
                for risk in preview["risks"]
                if risk["severity"] == "error"
            ]
            raise ConflictError("publish preview is not ready: " + "; ".join(errors))

        payload = {"run_id": run_id, "model_id": model_id}
        explicit = request.get("idempotency_key")
        key = explicit or (
            "run.publish:"
            + request_fingerprint({**payload, "content": preview["identity"]})[:32]
        )
        claim = self.idempotency.begin(key, payload)
        if claim["replayed"]:
            return claim["response"]
        try:
            job = self.jobs.create(
                run_id=run_id,
                model_id=model_id,
                attempt_id=preview["identity"]["attempt_id"] or "default",
                kind="publish",
                executor=self.executor.name,
                command=[
                    "python3",
                    "-m",
                    "ci.model_quality.stage",
                    "--stage",
                    "publish",
                    "--run-mode",
                    "upload_only",
                ],
                queue="model-quality-gpu",
                gpu_count=0,
                gpu_hours=0.0,
                stages=["publish"],
                artifact_fingerprint=preview["identity"]["artifact_fingerprint"],
                evaluation_fingerprint=preview["identity"]["evaluation_fingerprint"],
                details={
                    "actor": principal.actor,
                    "remote_target": preview["remote_target"],
                },
            )
            dispatch = self.executor.dispatch(
                LaunchSpec(
                    run_id=run_id,
                    attempt_id=job["attempt_id"],
                    run_mode="upload_only",
                    git_sha=str(request.get("git_sha") or "local"),
                    plan={},
                ),
                [job],
            )
            response = {
                "run_id": run_id,
                "model_id": model_id,
                "job": job,
                "remote_target": preview["remote_target"],
                "allowlist": preview["allowlist"],
                "file_count": preview["file_count"],
                "total_bytes": preview["total_bytes"],
                "identity": preview["identity"],
                "executor": dispatch.as_dict(),
                "status": "QUEUED",
            }
        except Exception as error:
            self.idempotency.fail(key, str(error))
            raise
        self.audit.record(
            actor=principal.actor,
            action="run.publish",
            target=f"{run_id}/{model_id}",
            result="ACCEPTED",
            request=request,
            details={
                "job_id": response["job"]["job_id"],
                "remote_target": response["remote_target"],
                "file_count": response["file_count"],
                "total_bytes": response["total_bytes"],
                "identity": response["identity"],
                "executor": response["executor"],
            },
        )
        self.idempotency.complete(key, response)
        return response

    # ----------------------------------------------------------------- access

    def artifact_access(
        self, run_id: str, artifact_id: str, model_id: str, principal: Principal
    ) -> dict[str, Any]:
        """Return short-lived, read-only access details for an artifact."""

        principal.require("read")
        if artifact_id not in {"model", "artifact"}:
            raise NotFoundError(f"artifact {artifact_id!r} is not published")
        model_dir = self._model_dir(run_id, model_id)
        artifact_dir = model_dir / "model"
        if not artifact_dir.is_dir():
            raise NotFoundError(f"artifact {artifact_id!r} has no model directory")
        manifest = read_json(model_dir / "artifact-manifest.json")
        expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=self.policy.artifact_access_ttl_seconds
        )
        return {
            "mode": "local-path",
            "scope": "read-only",
            "path": str(artifact_dir),
            "expires_at": expires_at.isoformat(),
            "identity": {
                "artifact_fingerprint": (
                    manifest.get("artifact_fingerprint")
                    if isinstance(manifest, dict)
                    else None
                ),
                "artifact_content_fingerprint": (
                    manifest.get("artifact_content_fingerprint")
                    if isinstance(manifest, dict)
                    else None
                ),
            },
            "note": (
                "this deployment shares a filesystem; an object-store adapter "
                "replaces this with a scoped, expiring URL instead of a path"
            ),
        }
