import json

import pytest
import yaml

from ci.model_quality.buildkite import render_pipeline
from ci.model_quality.config import evaluation_fingerprint, model_fingerprint
from ci.model_quality.executor import run_report
from ci.model_quality.planner import build_execution_plan
from ci.model_quality.state import (
    atomic_write_json,
    attempt_reports_dir,
    model_run_dir,
    write_stage_result,
)


def _model(*, enabled=True):
    return {
        "id": "model-a",
        "enabled": enabled,
        "priority": "P1",
        "source": {"path": "/models/a", "revision": "abc"},
        "resources": {
            "gpu_count": 1,
            "estimated_gpu_hours": 2.0,
            "estimated_eval_gpu_hours": 1.0,
        },
        "workflow": {
            "revision": "test-workflow",
            "quantize": ["python3", "quantize.py"],
        },
        "validation": {"profile": "causal_lm"},
        "runtime_smoke": {
            "enabled": True,
            "python": "python3",
            "runtime_revision": "test-runtime",
        },
        "evaluation": {"profile": "standard", "command": ["evaluate"]},
        "upload": {"enabled": True},
    }


@pytest.fixture
def runs_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path))
    return tmp_path


def test_eval_only_report_does_not_require_quantize(runs_root):
    model = _model()
    artifact = model_fingerprint(model)
    evaluation = evaluation_fingerprint(model, artifact)
    for stage in ("preflight", "validate", "runtime-smoke", "evaluate"):
        write_stage_result(
            "run",
            model["id"],
            stage,
            {"status": "PASS"},
            artifact_fingerprint=artifact,
            evaluation_fingerprint=evaluation,
        )
    report_path = model_run_dir("run", model["id"]) / "reports/evaluation.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "artifact_fingerprint": artifact,
                "evaluation_fingerprint": evaluation,
            }
        )
    )
    attempt_report = attempt_reports_dir("run", model["id"], "default")
    attempt_report.mkdir(parents=True)
    (attempt_report / "evaluation.json").write_text(report_path.read_text())

    result = run_report(
        model,
        "run",
        run_mode="eval_only",
        artifact_fingerprint=artifact,
        evaluation_fingerprint=evaluation,
    )

    assert result["status"] == "PASS"
    assert result["summary"]["quantization"] == "REUSED"


def test_quantize_report_writes_explicit_skipped_evaluation(runs_root):
    model = _model()
    for stage in ("preflight", "quantize", "validate", "runtime-smoke"):
        write_stage_result("run", model["id"], stage, {"status": "PASS"})

    run_report(model, "run", run_mode="quantize")

    report = json.loads(
        (model_run_dir("run", model["id"]) / "reports/evaluation.json").read_text()
    )
    assert report["status"] == "SKIPPED"
    assert report["reason_code"] == "EVALUATION_DISABLED"


def test_upload_only_report_does_not_create_fake_skipped_evaluation(runs_root):
    model = _model()
    for stage in ("preflight", "validate", "runtime-smoke"):
        write_stage_result("run", model["id"], stage, {"status": "PASS"})

    result = run_report(model, "run", run_mode="upload_only")

    evaluation_path = model_run_dir("run", model["id"]) / "reports/evaluation.json"
    assert not evaluation_path.exists()
    assert result["summary"]["evaluation_history"] == "MISSING_PRIOR_RESULT"


def test_report_promotes_only_the_selected_attempt(runs_root):
    model = _model()
    artifact = model_fingerprint(model)
    for attempt, status in (("attempt-old", "FAIL"), ("attempt-new", "PASS")):
        for stage in ("preflight", "quantize", "validate", "runtime-smoke"):
            write_stage_result(
                "run",
                model["id"],
                stage,
                {"status": status},
                attempt_id=attempt,
                artifact_fingerprint=artifact,
            )

    result = run_report(
        model,
        "run",
        run_mode="quantize",
        artifact_fingerprint=artifact,
        attempt_id="attempt-new",
    )

    assert result["status"] == "PASS"
    current = json.loads(
        (model_run_dir("run", model["id"]) / "current-attempt.json").read_text()
    )
    shared = json.loads(
        (model_run_dir("run", model["id"]) / "reports/summary.json").read_text()
    )
    old = json.loads(
        (
            model_run_dir("run", model["id"])
            / "attempts/attempt-old/state/validate.json"
        ).read_text()
    )
    assert current["attempt_id"] == "attempt-new"
    assert shared["attempt_id"] == "attempt-new"
    assert old["status"] == "FAIL"


def test_failed_preflight_does_not_create_input_manifest(runs_root, monkeypatch):
    from ci.model_quality.executor import run_preflight

    model = _model()
    model["source"]["path"] = str(runs_root / "missing-source")
    monkeypatch.setenv("VLLM_PYTHON_ENV", "/usr/bin/false")

    result = run_preflight(
        model,
        "run",
        run_mode="quantize",
        git_sha="abc",
        fingerprint=model_fingerprint(model),
        evaluation_fingerprint=evaluation_fingerprint(model, model_fingerprint(model)),
        attempt_id="attempt-failed",
    )

    assert result["status"] == "FAIL"
    assert not (model_run_dir("run", model["id"]) / "input-manifest.json").exists()


def test_eval_only_rejects_changed_source_provenance(runs_root, monkeypatch):
    from ci.model_quality.executor import run_preflight

    model = _model()
    model["runtime_smoke"] = {}
    model["upload"] = {"enabled": False}
    monkeypatch.setattr("torch.accelerator.is_available", lambda: True)
    monkeypatch.setattr("torch.accelerator.device_count", lambda: 1)
    model["workflow"]["quantize"] = ["/usr/bin/true"]
    source = runs_root / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    model["source"]["path"] = str(source)
    run_root = model_run_dir("run", model["id"])
    (run_root / "model").mkdir(parents=True)
    artifact = model_fingerprint(model)
    evaluation = evaluation_fingerprint(model, artifact)
    first = run_preflight(
        model,
        "run",
        run_mode="quantize",
        fingerprint=artifact,
        evaluation_fingerprint=evaluation,
    )
    assert first["status"] in {"PASS", "WARN"}
    (source / "config.json").write_text('{"changed": true}')

    reused = run_preflight(
        model,
        "run",
        run_mode="eval_only",
        fingerprint=artifact,
        evaluation_fingerprint=evaluation,
    )

    assert reused["status"] == "FAIL"
    assert any("provenance changed" in failure for failure in reused["failures"])


def test_eval_only_rejects_changed_artifact_contents(runs_root, monkeypatch):
    from ci.model_quality.config import finalized_artifact_fingerprint
    from ci.model_quality.executor import (
        _artifact_content_fingerprint,
        _artifact_file_records,
        _source_provenance,
        run_preflight,
    )

    model = _model()
    model["runtime_smoke"] = {}
    model["upload"] = {"enabled": False}
    model["workflow"]["quantize"] = ["/usr/bin/true"]
    monkeypatch.setattr("torch.accelerator.is_available", lambda: True)
    monkeypatch.setattr("torch.accelerator.device_count", lambda: 1)
    source = runs_root / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    model["source"]["path"] = str(source)
    run_root = model_run_dir("run", model["id"])
    output = run_root / "model"
    output.mkdir(parents=True)
    artifact_file = output / "weights.bin"
    artifact_file.write_bytes(b"original")
    request_fingerprint = model_fingerprint(model)
    source_provenance = _source_provenance(source)
    finalized = finalized_artifact_fingerprint(request_fingerprint, source_provenance)
    artifact_files = _artifact_file_records(output)
    atomic_write_json(
        run_root / "input-manifest.json",
        {
            "fingerprint": request_fingerprint,
            "finalized_artifact_fingerprint": finalized,
            "source_provenance": source_provenance,
        },
    )
    atomic_write_json(
        run_root / "artifact-manifest.json",
        {
            "artifact_fingerprint": finalized,
            "source_provenance": source_provenance,
            "artifact_content_fingerprint": _artifact_content_fingerprint(
                artifact_files
            ),
        },
    )

    first = run_preflight(
        model,
        "run",
        run_mode="eval_only",
        fingerprint=request_fingerprint,
        evaluation_fingerprint=evaluation_fingerprint(model, request_fingerprint),
    )
    assert first["status"] in {"PASS", "WARN"}

    artifact_file.write_bytes(b"changed")
    reused = run_preflight(
        model,
        "run",
        run_mode="eval_only",
        fingerprint=request_fingerprint,
        evaluation_fingerprint=evaluation_fingerprint(model, request_fingerprint),
    )
    assert reused["status"] == "FAIL"
    assert any("contents changed" in failure for failure in reused["failures"])


def test_upload_only_pipeline_revalidates_and_reconstructs_report():
    plan = build_execution_plan(
        {"models": [_model()]},
        git_sha="abc",
        max_gpu_hours=1,
        model_filter="model-a",
        run_mode="upload_only",
        run_id="existing",
    )

    pipeline = yaml.safe_load(render_pipeline(plan, config_path="models.yaml"))
    keys = [step["key"] for step in pipeline["steps"]]
    assert keys[:5] == [
        "model-a-preflight",
        "model-a-validate",
        "model-a-runtime-smoke",
        "model-a-report",
        "model-a-publish",
    ]


def test_disabled_model_can_only_be_explicitly_reused():
    config = {"models": [_model(enabled=False)]}
    implicit = build_execution_plan(
        config, git_sha="abc", max_gpu_hours=1, run_mode="upload_only", run_id="run"
    )
    explicit = build_execution_plan(
        config,
        git_sha="abc",
        max_gpu_hours=1,
        model_filter="model-a",
        run_mode="upload_only",
        run_id="run",
    )
    assert implicit["selected"] == []
    assert [job["id"] for job in explicit["selected"]] == ["model-a"]
