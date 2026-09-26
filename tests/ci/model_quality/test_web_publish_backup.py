from __future__ import annotations

import json
from pathlib import Path

import pytest

from ci.model_quality.state import atomic_write_json
from ci.model_quality.web import (
    ConflictError,
    LaunchPolicy,
    PermissionDenied,
    Principal,
    PublishService,
    QueueExecutor,
    RunStore,
    ValidationError,
)
from ci.model_quality.web import backups as backups_module
from ci.model_quality.web.backups import BackupService

PUBLISHER = Principal(actor="carol", roles=frozenset({"publisher"}), authenticated=True)
ADMIN = Principal(actor="dave", roles=frozenset({"admin"}), authenticated=True)
VIEWER = Principal(actor="erin", roles=frozenset({"viewer"}), authenticated=True)

RUN_ID = "20260926T000000Z_publish"


def make_run(env, *, run_id: str = RUN_ID) -> str:
    run_dir = env.root / run_id
    model_dir = run_dir / env.model_id
    (model_dir / "model").mkdir(parents=True)
    (model_dir / "model" / "config.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "model" / "model.safetensors").write_bytes(b"weights")
    (model_dir / "reports").mkdir(parents=True)
    (model_dir / "reports" / "summary.json").write_text(
        json.dumps({"status": "PASS", "stages": {"validate": "PASS"}}),
        encoding="utf-8",
    )
    (model_dir / "reports" / "evaluation.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "metrics": [
                    {"name": "exact_match", "recovery": 0.98, "status": "PASS"}
                ],
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "logs" / "attempt-1").mkdir(parents=True)
    (model_dir / "logs" / "attempt-1" / "quantization.log").write_text(
        "quantized\n", encoding="utf-8"
    )
    atomic_write_json(
        run_dir / "execution-plan.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "attempt_id": "attempt-1",
            "run_mode": "quantize",
            "selected": [{"id": env.model_id}],
            "deferred": [],
            "budget": {"max_gpu_hours": 8, "selected_gpu_hours": 2},
        },
    )
    atomic_write_json(
        model_dir / "state" / "validate.json",
        {"status": "PASS", "attempt_id": "attempt-1"},
    )
    atomic_write_json(
        model_dir / "artifact-manifest.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "model_id": env.model_id,
            "attempt_id": "attempt-1",
            "artifact_fingerprint": "artifact-1",
            "artifact_content_fingerprint": "content-1",
            "files": [],
        },
    )
    atomic_write_json(
        model_dir / "current-attempt.json",
        {
            "schema_version": 1,
            "attempt_id": "attempt-1",
            "artifact_fingerprint": "artifact-1",
            "evaluation_fingerprint": "eval-1",
        },
    )
    atomic_write_json(
        model_dir / "input-manifest.json",
        {"schema_version": 1, "model": {"source": {"path": str(env.source)}}},
    )
    return run_id


def make_publisher(env, policy: LaunchPolicy | None = None) -> PublishService:
    policy = policy or LaunchPolicy()
    return PublishService(
        root=env.root,
        config_path=env.config,
        store=RunStore(env.root),
        policy=policy,
        executor=QueueExecutor(env.root),
    )


def test_publish_preview_lists_allowlisted_files(publish_env) -> None:
    run_id = make_run(publish_env)
    service = make_publisher(publish_env)
    preview = service.preview(run_id, publish_env.model_id)

    assert preview["ready"] is True
    assert preview["allowlist"] == ["model", "reports"]
    assert preview["file_count"] == 4
    assert {entry["path"] for entry in preview["files"]} == {
        "model/config.json",
        "model/model.safetensors",
        "reports/summary.json",
        "reports/evaluation.json",
    }
    assert preview["total_bytes"] == sum(
        entry["size_bytes"] for entry in preview["files"]
    )
    assert all(entry["sha256"] for entry in preview["files"])
    assert preview["remote_target"] == (
        f"bos:/model-quality-test/llm-compressor/{publish_env.model_id}/runs/{run_id}"
    )
    assert preview["identity"]["artifact_fingerprint"] == "artifact-1"
    assert preview["identity"]["artifact_content_fingerprint"] == "content-1"
    assert preview["approvals_required"] == 1


def test_publish_preview_blocks_disabled_upload(web_env) -> None:
    run_id = make_run(web_env)
    preview = make_publisher(web_env).preview(run_id, web_env.model_id)
    codes = {risk["code"] for risk in preview["risks"] if risk["severity"] == "error"}
    assert codes == {"UPLOAD_DISABLED", "UNSAFE_REMOTE_PREFIX", "EMPTY_ALLOWLIST"}
    assert preview["ready"] is False


def test_publish_preview_requires_identity_files(publish_env) -> None:
    run_id = make_run(publish_env)
    model_dir = publish_env.root / run_id / publish_env.model_id
    (model_dir / "current-attempt.json").unlink()
    preview = make_publisher(publish_env).preview(run_id, publish_env.model_id)
    codes = {risk["code"] for risk in preview["risks"]}
    assert "MISSING_ATTEMPT_POINTER" in codes
    assert preview["ready"] is False


def test_publish_requires_confirm_permission_and_is_idempotent(publish_env) -> None:
    run_id = make_run(publish_env)
    service = make_publisher(publish_env)

    with pytest.raises(PermissionDenied):
        service.publish(run_id, publish_env.model_id, {"confirm": True}, VIEWER)
    with pytest.raises(ValidationError, match="confirm=true"):
        service.publish(run_id, publish_env.model_id, {}, PUBLISHER)

    request = {"confirm": True, "idempotency_key": "publish-0001"}
    response = service.publish(run_id, publish_env.model_id, request, PUBLISHER)
    assert response["status"] == "QUEUED"
    assert response["job"]["kind"] == "publish"
    assert response["job"]["stages"] == ["publish"]
    assert response["file_count"] == 4
    assert response["remote_target"].endswith(f"{publish_env.model_id}/runs/{run_id}")

    replay = service.publish(run_id, publish_env.model_id, request, PUBLISHER)
    assert replay["job"]["job_id"] == response["job"]["job_id"]
    assert (
        service.jobs.list_jobs(run_id, kind="publish")[0]["job_id"]
        == response["job"]["job_id"]
    )
    audit = service.audit.read(action="run.publish")
    assert audit["total"] == 1
    assert audit["items"][0]["actor"] == "carol"
    assert audit["items"][0]["result"] == "ACCEPTED"


def test_publish_can_require_dual_approval(publish_env) -> None:
    run_id = make_run(publish_env)
    service = make_publisher(publish_env, LaunchPolicy(require_dual_approval=True))
    with pytest.raises(ValidationError, match="two approvers"):
        service.publish(run_id, publish_env.model_id, {"confirm": True}, PUBLISHER)
    with pytest.raises(ValidationError, match="two distinct approvers"):
        service.publish(
            run_id,
            publish_env.model_id,
            {"confirm": True, "approvals": ["carol"]},
            PUBLISHER,
        )
    with pytest.raises(ValidationError, match="two distinct approvers"):
        service.publish(
            run_id,
            publish_env.model_id,
            {"confirm": True, "approvals": ["carol", "carol"]},
            PUBLISHER,
        )
    response = service.publish(
        run_id,
        publish_env.model_id,
        {"confirm": True, "approvals": ["carol", "dave"]},
        PUBLISHER,
    )
    assert response["job"]["status"] == "QUEUED"
    assert service.preview(run_id, publish_env.model_id)["approvals_required"] == 2


def test_publish_refuses_when_preview_not_ready(publish_env) -> None:
    run_id = make_run(publish_env)
    model_dir = publish_env.root / run_id / publish_env.model_id
    (model_dir / "reports" / "summary.json").unlink()
    service = make_publisher(publish_env)
    with pytest.raises(ConflictError, match="not ready"):
        service.publish(run_id, publish_env.model_id, {"confirm": True}, PUBLISHER)


def test_artifact_access_is_scoped_and_expiring(publish_env) -> None:
    run_id = make_run(publish_env)
    service = make_publisher(publish_env)
    access = service.artifact_access(run_id, "model", publish_env.model_id, VIEWER)
    assert access["mode"] == "local-path"
    assert access["scope"] == "read-only"
    assert access["path"].endswith(f"{publish_env.model_id}/model")
    assert access["expires_at"]
    assert access["identity"]["artifact_content_fingerprint"] == "content-1"
    with pytest.raises(PermissionDenied):
        service.artifact_access(
            run_id, "model", publish_env.model_id, Principal("x", frozenset(), True)
        )


def test_backup_creates_verified_immutable_manifest(web_env) -> None:
    run_id = make_run(web_env)
    service = BackupService(runs_root=web_env.root, retention_days=7)
    preview = service.preview(
        run_id, {"metadata": True, "artifact": True, "logs": True}
    )
    assert preview["ready"] is True
    assert preview["file_count"] > 4
    assert preview["retention"]["days"] == 7
    assert preview["target_prefix"].endswith(preview["backup_id"])

    record = service.create(
        run_id,
        {"scope": {"metadata": True, "artifact": True, "logs": True}},
        ADMIN,
    )
    assert record["status"] == "VERIFIED"
    assert record["file_count"] == preview["file_count"]
    assert record["manifest_sha256"]

    manifest = json.loads(
        (Path(record["path"]) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["file_count"] == len(manifest["files"])
    for entry in manifest["files"]:
        assert entry["schema_version"] == 1
        assert entry["sha256"]
        assert entry["source_uri"].startswith("file://")
        assert entry["target_uri"].startswith("file://")

    backup_json = Path(record["path"]) / "BACKUP.json"
    with pytest.raises(ConflictError, match="immutable"):
        service._write_once(backup_json, {"tampered": True})

    listing = service.list(run_id)
    assert listing["total"] == 1
    assert listing["items"][0]["backup_id"] == record["backup_id"]
    loaded = service.get(record["backup_id"])
    assert loaded["status"] == "VERIFIED"
    assert loaded["manifest"]["file_count"] == manifest["file_count"]


def test_backup_requires_backup_permission(web_env) -> None:
    run_id = make_run(web_env)
    service = BackupService(runs_root=web_env.root)
    with pytest.raises(PermissionDenied):
        service.create(run_id, {}, VIEWER)


def test_source_checkpoint_needs_explicit_allowance(web_env) -> None:
    run_id = make_run(web_env)
    service = BackupService(runs_root=web_env.root)
    preview = service.preview(
        run_id, {"metadata": False, "artifact": False, "source_checkpoint": True}
    )
    codes = {risk["code"] for risk in preview["risks"]}
    assert "SOURCE_CHECKPOINT_NOT_ALLOWED" in codes
    with pytest.raises(ConflictError, match="not ready"):
        service.create(run_id, {"scope": {"source_checkpoint": True}}, ADMIN)

    allowed = BackupService(runs_root=web_env.root, max_source_bytes=10**9)
    preview = allowed.preview(
        run_id, {"metadata": False, "artifact": False, "source_checkpoint": True}
    )
    assert preview["ready"] is True
    assert any(
        entry["path"].startswith("source-checkpoint/") for entry in preview["files"]
    )


def test_backup_twice_keeps_each_manifest(web_env, monkeypatch) -> None:
    run_id = make_run(web_env)
    tokens = iter(["20260926T000000Z", "20260926T000001Z"])
    monkeypatch.setattr(backups_module, "timestamp_token", lambda: next(tokens))
    service = BackupService(runs_root=web_env.root)
    first = service.create(run_id, {}, ADMIN)
    second = service.create(run_id, {}, ADMIN)
    assert first["backup_id"] != second["backup_id"]
    assert (
        json.loads((Path(first["path"]) / "BACKUP.json").read_text(encoding="utf-8"))[
            "backup_id"
        ]
        == first["backup_id"]
    )
    assert service.list(run_id)["total"] == 2


def test_restore_preview_and_restore_write_a_new_run(web_env) -> None:
    run_id = make_run(web_env)
    service = BackupService(runs_root=web_env.root)
    backup = service.create(run_id, {}, ADMIN)

    preview = service.restore_preview(
        backup["backup_id"], {"target_run_id": "restored-run"}
    )
    assert preview["ready"] is True
    assert preview["verification"]["verified"] is True
    assert preview["verification"]["missing"] == []
    assert preview["original_run_id"] == run_id

    restored = service.restore(
        backup["backup_id"], {"target_run_id": "restored-run"}, ADMIN
    )
    assert restored["target_run_id"] == "restored-run"
    assert restored["original_run_id"] == run_id
    assert restored["released"] is False
    assert restored["original_artifact_fingerprint"] == "artifact-1"

    target = web_env.root / "restored-run"
    assert (target / "RESTORE.json").is_file()
    assert (target / web_env.model_id / "model" / "model.safetensors").is_file()
    assert (target / "execution-plan.json").is_file()
    assert (
        json.loads((target / "RESTORE.json").read_text())["source_backup_id"]
        == (backup["backup_id"])
    )

    with pytest.raises(PermissionDenied):
        service.restore(backup["backup_id"], {"target_run_id": "restored-2"}, PUBLISHER)


def test_restore_refuses_existing_target_and_corruption(web_env) -> None:
    run_id = make_run(web_env)
    service = BackupService(runs_root=web_env.root)
    backup = service.create(run_id, {}, ADMIN)

    existing = service.restore_preview(backup["backup_id"], {"target_run_id": run_id})
    assert existing["ready"] is False
    assert any(risk["code"] == "TARGET_RUN_EXISTS" for risk in existing["risks"])
    with pytest.raises(ConflictError, match="not ready"):
        service.restore(backup["backup_id"], {"target_run_id": run_id}, ADMIN)

    manifest = json.loads(
        (Path(backup["path"]) / "manifest.json").read_text(encoding="utf-8")
    )
    victim = Path(backup["path"]) / "files" / manifest["files"][0]["path"]
    victim.write_text("tampered", encoding="utf-8")
    corrupt = service.restore_preview(
        backup["backup_id"], {"target_run_id": "restored-corrupt"}
    )
    assert corrupt["ready"] is False
    assert (
        corrupt["verification"]["checksum_mismatch"]
        or corrupt["verification"]["size_mismatch"]
    )
    with pytest.raises(ConflictError, match="not ready"):
        service.restore(
            backup["backup_id"], {"target_run_id": "restored-corrupt"}, ADMIN
        )


def test_backup_preview_normalizes_scope(web_env) -> None:
    run_id = make_run(web_env)
    service = BackupService(runs_root=web_env.root)
    with pytest.raises(ValidationError, match="list of stages"):
        BackupService.normalize_scope({"logs": "everything"})
    with pytest.raises(ValidationError, match="unknown log stages"):
        BackupService.normalize_scope({"logs": ["quantize", "nope"]})
    preview = service.preview(run_id, {"logs": ["quantize"]})
    log_paths = [
        entry["path"] for entry in preview["files"] if "/logs/" in entry["path"]
    ]
    assert log_paths == [f"{web_env.model_id}/logs/attempt-1/quantization.log"]
