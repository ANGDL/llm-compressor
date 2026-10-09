import json
import sys
from pathlib import Path

import pytest

from ci.model_quality.config import evaluation_fingerprint, model_fingerprint
from ci.model_quality.executor import (
    StageError,
    _validate_argument_constraints,
    _artifact_content_fingerprint,
    _artifact_file_records,
    _source_provenance,
    run_evaluate,
    run_publish,
    run_quantize,
    run_report,
    run_validate,
)
from ci.model_quality.state import model_run_dir, read_stage_result, write_stage_result
from ci.model_quality import stage as stage_module


def _fake_model(tmp_path: Path) -> dict:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
    script = tmp_path / "quantize.py"
    script.write_text(
        """
import json
import sys
from pathlib import Path
import torch
from safetensors.torch import save_file

output = Path(sys.argv[1])
output.mkdir(parents=True)
(output / "config.json").write_text("{}")
save_file({"layer.weight": torch.ones(2, 2)}, output / "model.safetensors")
""",
        encoding="utf-8",
    )
    return {
        "id": "fake-model",
        "source": {"path": str(source)},
        "resources": {
            "gpu_count": 1,
            "estimated_gpu_hours": 1,
            "timeout_hours": 1,
        },
        "workflow": {
            "revision": "test-workflow",
            "quantize": [
                str(Path(__import__("sys").executable)),
                str(script),
                "{output_dir}",
            ],
        },
        "validation": {
            "profile": "causal_lm",
            "require_quantization_config": False,
            "required_files": ["config.json"],
        },
    }


def test_quantize_validate_report_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path / "runs"))
    model = _fake_model(tmp_path)
    run_id = "run-a"
    write_stage_result(run_id, model["id"], "preflight", {"status": "PASS"})

    quantize = run_quantize(model, run_id)
    write_stage_result(run_id, model["id"], "quantize", quantize)
    validate = run_validate(model, run_id)
    write_stage_result(run_id, model["id"], "validate", validate)
    write_stage_result(run_id, model["id"], "runtime-smoke", {"status": "PASS"})
    report = run_report(model, run_id)

    assert quantize["status"] == "PASS"
    assert validate["status"] == "WARN"
    assert report["status"] == "PASS"
    assert report["summary"]["warnings"] == ["validate completed with warnings"]
    assert read_stage_result(run_id, model["id"], "quantize")["status"] == "PASS"
    output = tmp_path / "runs/run-a/fake-model/model/model.safetensors"
    assert output.is_file()
    summary = json.loads(
        (tmp_path / "runs/run-a/fake-model/reports/summary.json").read_text()
    )
    assert summary["status"] == "PASS"


def test_report_carries_earlier_lanes_forward_on_partial_rerun(tmp_path, monkeypatch):
    """Re-running one node must not fail the report on the untouched lanes."""

    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path / "runs"))
    model = _fake_model(tmp_path)
    run_id = "run-a"
    fingerprint = model_fingerprint(model)

    # The original attempt ran the whole pipeline.
    for stage, status in (
        ("preflight", "PASS"),
        ("quantize", "PASS"),
        ("validate", "WARN"),
    ):
        write_stage_result(
            run_id,
            model["id"],
            stage,
            {"status": status},
            attempt_id="attempt-1",
            artifact_fingerprint=fingerprint,
        )

    # The re-run attempt owns only the stages it actually executed; evaluate was
    # explicitly skipped by the operator.
    write_stage_result(
        run_id,
        model["id"],
        "runtime-smoke",
        {"status": "PASS"},
        attempt_id="attempt-2",
        artifact_fingerprint=fingerprint,
    )
    write_stage_result(
        run_id,
        model["id"],
        "evaluate",
        {"status": "SKIPPED", "reason": "operator skip"},
        attempt_id="attempt-2",
    )

    result = run_report(
        model,
        run_id,
        run_mode="quantize_and_eval",
        artifact_fingerprint=fingerprint,
        evaluation_fingerprint="eval-fingerprint",
        attempt_id="attempt-2",
    )

    assert result["status"] == "PASS"
    assert result["summary"]["stages"] == {
        "preflight": "PASS",
        "quantize": "PASS",
        "validate": "WARN",
        "runtime-smoke": "PASS",
        "evaluate": "SKIPPED",
    }
    assert result["summary"]["stages_carried_from_prior_attempts"] == [
        "preflight",
        "quantize",
        "validate",
    ]
    assert result["summary"]["warnings"] == ["validate completed with warnings"]


def test_argument_constraints_reject_incompatible_flags(tmp_path):
    model = _fake_model(tmp_path)
    model["workflow"]["forbidden_argument_pairs"] = [
        ["--checkpoint-progress", "--async-save"]
    ]
    argv = ["python3", "job.py", "--checkpoint-progress", "--async-save"]

    failures = _validate_argument_constraints(model, argv)

    assert failures == [
        "arguments --checkpoint-progress and --async-save cannot be combined"
    ]


def test_evaluate_normalizes_generic_metric_result(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path / "runs"))
    model = _fake_model(tmp_path)
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_text(
        """
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "suite_id": "test-suite",
    "metrics": [{
        "name": "accuracy",
        "direction": "higher",
        "base_value": 0.8,
        "compressed_value": 0.78,
        "min_recovery": 0.95
    }]
}))
""",
        encoding="utf-8",
    )
    model["evaluation"] = {
        "command": [sys.executable, str(evaluator), "{reports_dir}/raw.json"],
        "result_file": "{reports_dir}/raw.json",
        "runtime_revision": "test-runtime",
    }

    artifact = model_fingerprint(model)
    result = run_evaluate(
        model,
        "run-a",
        artifact_fingerprint=artifact,
        evaluation_fingerprint=evaluation_fingerprint(model, artifact),
    )

    assert result["status"] == "PASS"
    normalized = json.loads(
        (
            tmp_path / "runs/run-a/fake-model/attempts/default/reports/evaluation.json"
        ).read_text()
    )
    assert normalized["status"] == "PASS"
    assert normalized["metrics"][0]["recovery"] == 0.975
    assert normalized["artifact_fingerprint"] == artifact
    assert normalized["evaluation_fingerprint"] == evaluation_fingerprint(
        model, artifact
    )


def test_evaluate_preserves_json_command_argument(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path / "runs"))
    model = _fake_model(tmp_path)
    evaluator = tmp_path / "evaluate-json.py"
    evaluator.write_text(
        """
import json
import sys
from pathlib import Path
assert json.loads(sys.argv[1])["name"] == "exact_match"
Path(sys.argv[2]).write_text(json.dumps({
    "schema_version": 1,
    "suite_id": "json-regression",
    "metrics": [{
        "name": "exact_match", "direction": "higher",
        "base_value": 1.0, "compressed_value": 1.0
    }]
}))
""",
        encoding="utf-8",
    )
    json_argument = '{"name":"exact_match","pattern":"a{2,4}"}'
    model["evaluation"] = {
        "command": [
            sys.executable,
            str(evaluator),
            json_argument,
            "{reports_dir}/raw.json",
        ],
        "result_file": "{reports_dir}/raw.json",
        "runtime_revision": "test-runtime",
    }
    artifact = model_fingerprint(model)

    result = run_evaluate(
        model,
        "run-a",
        artifact_fingerprint=artifact,
        evaluation_fingerprint=evaluation_fingerprint(model, artifact),
    )

    assert result["status"] == "PASS"
    assert result["argv"][2] == json_argument


def test_report_keeps_override_evaluation_fingerprint(tmp_path, monkeypatch):
    run_root = tmp_path / "runs"
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(run_root))
    monkeypatch.setenv("MODEL_QUALITY_EVALUATION_COMMAND_JSON", '["custom-eval"]')
    model = _fake_model(tmp_path)
    config = tmp_path / "models.yaml"
    config.write_text("unused")
    artifact = "final-artifact"
    override_evaluation = "override-evaluation"
    model_dir = model_run_dir("run-a", model["id"])
    model_dir.mkdir(parents=True)
    (model_dir / "input-manifest.json").write_text(
        json.dumps({"finalized_artifact_fingerprint": artifact})
    )
    monkeypatch.setattr(stage_module, "load_model_config", lambda _path: {"models": [model]})
    captured = {}
    monkeypatch.setattr(
        stage_module,
        "run_report",
        lambda *_args, **kwargs: captured.update(kwargs) or {"status": "PASS"},
    )
    monkeypatch.setattr(
        stage_module,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "config": config,
                "model": model["id"],
                "run_id": "run-a",
                "attempt_id": "attempt-1",
                "stage": "report",
                "models_json": "[]",
                "git_sha": "git",
                "fingerprint": "planned-artifact",
                "evaluation_fingerprint": override_evaluation,
                "run_mode": "eval_only",
                "dry_run": False,
            },
        )(),
    )

    assert stage_module.main() == 0
    assert captured["artifact_fingerprint"] == artifact
    assert captured["evaluation_fingerprint"] == override_evaluation


def test_eval_only_report_accepts_planned_runtime_smoke_fingerprint(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path / "runs"))
    model = _fake_model(tmp_path)
    model["runtime_smoke"] = {"enabled": True}
    run_id = "run-a"
    attempt_id = "attempt-a"
    planned = "planned-artifact"
    finalized = "finalized-artifact"
    evaluation = "override-evaluation"
    run_dir = model_run_dir(run_id, model["id"])
    run_dir.mkdir(parents=True)
    (run_dir / "input-manifest.json").write_text(
        json.dumps(
            {
                "fingerprint": planned,
                "finalized_artifact_fingerprint": finalized,
            }
        )
    )
    for stage, status, fingerprint in (
        ("preflight", "WARN", finalized),
        ("validate", "PASS", finalized),
        ("runtime-smoke", "PASS", planned),
        ("evaluate", "PASS", finalized),
    ):
        write_stage_result(
            run_id,
            model["id"],
            stage,
            {"status": status},
            attempt_id=attempt_id,
            artifact_fingerprint=fingerprint,
            evaluation_fingerprint=(evaluation if stage == "evaluate" else None),
        )
    reports = run_dir / "attempts" / attempt_id / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "evaluation.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "artifact_fingerprint": finalized,
                "evaluation_fingerprint": evaluation,
            }
        )
    )

    result = run_report(
        model,
        run_id,
        run_mode="eval_only",
        artifact_fingerprint=finalized,
        evaluation_fingerprint=evaluation,
        attempt_id=attempt_id,
    )

    assert result["status"] == "PASS"
    assert result["summary"]["stages"]["runtime-smoke"] == "PASS"


def test_evaluation_cache_requires_matching_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path / "runs"))
    model = _fake_model(tmp_path)
    evaluator = tmp_path / "counting-evaluator.py"
    counter = tmp_path / "counter.txt"
    evaluator.write_text(
        """
import json
import sys
from pathlib import Path
counter, output = map(Path, sys.argv[1:])
value = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(value))
output.write_text(json.dumps({
    "schema_version": 1, "suite_id": "cache-test",
    "metrics": [{"name": "score", "direction": "higher",
                 "base_value": 1.0, "compressed_value": 1.0}]
}))
""",
        encoding="utf-8",
    )
    model["evaluation"] = {
        "command": [
            sys.executable,
            str(evaluator),
            str(counter),
            "{reports_dir}/raw.json",
        ],
        "result_file": "{reports_dir}/raw.json",
        "runtime_revision": "test-runtime",
    }
    artifact = model_fingerprint(model)
    evaluation = evaluation_fingerprint(model, artifact)

    run_evaluate(
        model,
        "run-a",
        artifact_fingerprint=artifact,
        evaluation_fingerprint=evaluation,
    )
    cached = run_evaluate(
        model,
        "run-a",
        artifact_fingerprint=artifact,
        evaluation_fingerprint=evaluation,
    )
    run_evaluate(
        model,
        "run-a",
        artifact_fingerprint=artifact,
        evaluation_fingerprint="changed-evaluation",
    )

    assert cached["reason_code"] == "EVALUATION_CACHE_HIT"
    assert counter.read_text() == "2"


def test_evaluator_cannot_relabel_a_stale_raw_result(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path / "runs"))
    model = _fake_model(tmp_path)
    reports = tmp_path / "runs/run-a/fake-model/attempts/attempt-new/reports"
    reports.mkdir(parents=True)
    stale = reports / "raw.json"
    stale.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_id": "stale",
                "metrics": [
                    {
                        "name": "score",
                        "direction": "higher",
                        "base_value": 1.0,
                        "compressed_value": 1.0,
                    }
                ],
            }
        )
    )
    model["evaluation"] = {
        "command": ["/usr/bin/true"],
        "result_file": "{reports_dir}/raw.json",
        "runtime_revision": "test-runtime",
    }
    artifact = model_fingerprint(model)

    with pytest.raises(StageError, match="evaluation result is missing"):
        run_evaluate(
            model,
            "run-a",
            artifact_fingerprint=artifact,
            evaluation_fingerprint=evaluation_fingerprint(model, artifact),
            attempt_id="attempt-new",
        )
    assert not stale.exists()


def test_evaluate_requires_a_structured_result_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path / "runs"))
    model = _fake_model(tmp_path)
    model["evaluation"] = {"command": [sys.executable, "-c", "pass"]}
    artifact = model_fingerprint(model)

    with pytest.raises(StageError, match="evaluation.result_file is required"):
        run_evaluate(
            model,
            "run-a",
            artifact_fingerprint=artifact,
            evaluation_fingerprint=evaluation_fingerprint(model, artifact),
        )


def test_publish_executes_explicit_bcecmd_upload_and_verification(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path / "runs"))
    model = _fake_model(tmp_path)
    run_dir = tmp_path / "runs/run-a/fake-model"
    (run_dir / "model").mkdir(parents=True)
    (run_dir / "reports").mkdir()
    (run_dir / "artifact-manifest.json").write_text(
        json.dumps({"artifact_fingerprint": "artifact-a", "artifact_content_fingerprint": _artifact_content_fingerprint(_artifact_file_records(run_dir / "model"))})
    )
    (run_dir / "reports/summary.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "attempt_id": "attempt-a",
                "artifact_fingerprint": "artifact-a",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "current-attempt.json").write_text(
        json.dumps({"attempt_id": "attempt-a", "artifact_fingerprint": "artifact-a"})
    )
    bcecmd = tmp_path / "bcecmd"
    bcecmd.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    bcecmd.chmod(0o755)
    model["upload"] = {
        "enabled": True,
        "remote_prefix": "bos:/bucket/model-quality",
        "allowlist": ["model", "reports"],
        "commands": [
            [str(bcecmd), "upload", "{output_dir}", "{remote_run_prefix}/model"]
        ],
        "verify_commands": [[str(bcecmd), "verify", "{remote_run_prefix}"]],
        "success_commands": [
            [
                str(bcecmd),
                "upload",
                "{run_dir}/commit-markers/SUCCESS.json",
                "{remote_run_prefix}/SUCCESS.json",
            ]
        ],
    }

    result = run_publish(
        model,
        "run-a",
        artifact_fingerprint="artifact-a",
        evaluation_fingerprint="evaluation-a",
    )

    assert result["status"] == "PASS"
    assert result["remote_run_prefix"].endswith("/fake-model/runs/run-a")
    assert [command["kind"] for command in result["executed"]] == [
        "upload",
        "verify",
        "success",
    ]
    assert (run_dir / "commit-markers/SUCCESS.json").is_file()

    second = run_publish(
        model,
        "run-a",
        artifact_fingerprint="artifact-a",
        evaluation_fingerprint="evaluation-a",
    )
    frozen = json.loads((run_dir / "reports/upload-manifest.json").read_text())
    assert second["uploaded_file_count"] == result["uploaded_file_count"]
    assert all(
        item["path"] != "reports/upload-manifest.json" for item in frozen["files"]
    )
    assert all("SUCCESS.json" not in item["path"] for item in frozen["files"])
    actual_manifest = json.loads((run_dir / "reports/upload-manifest.json").read_text())
    assert frozen == actual_manifest


def test_source_provenance_is_stat_based_not_hashed(tmp_path):
    """Source provenance must never content-hash a (potentially multi-TB) source."""

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "model-00001.safetensors"}})
    )
    (source / "model-00001.safetensors").write_bytes(b"x" * 1024)

    provenance = _source_provenance(source)
    records = provenance["metadata_files"] + provenance["shards"]
    assert records, "expected provenance records"
    for record in records:
        assert "sha256" not in record  # no content hashing
        assert record["size_bytes"] >= 0
        assert "mtime_ns" in record
    # A size change is still visible in the record.
    (source / "model-00001.safetensors").write_bytes(b"x" * 2048)
    assert _source_provenance(source)["shards"][0]["size_bytes"] == 2048


def test_publish_fast_path_trusts_manifest_sizes_but_catches_size_change(
    tmp_path, monkeypatch
):
    """A large artifact must not be re-hashed when its paths+sizes are unchanged.

    The manifest's recorded file sizes are the fast-path check: matching sizes
    reuse the stored content fingerprint (no multi-hundred-GB re-hash), while a
    size change still fails the publish as a changed artifact.
    """

    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path / "runs"))
    model = _fake_model(tmp_path)
    run_dir = tmp_path / "runs/run-a/fake-model"
    (run_dir / "model").mkdir(parents=True)
    (run_dir / "reports").mkdir()
    (run_dir / "model" / "model.safetensors").write_bytes(b"weights-v1")
    (run_dir / "artifact-manifest.json").write_text(
        json.dumps(
            {
                "artifact_fingerprint": "artifact-a",
                "files": _artifact_file_records(run_dir / "model"),
                "artifact_content_fingerprint": _artifact_content_fingerprint(
                    _artifact_file_records(run_dir / "model")
                ),
            }
        )
    )
    (run_dir / "reports/summary.json").write_text(
        json.dumps(
            {"status": "PASS", "attempt_id": "attempt-a", "artifact_fingerprint": "artifact-a"}
        )
    )
    (run_dir / "current-attempt.json").write_text(
        json.dumps({"attempt_id": "attempt-a", "artifact_fingerprint": "artifact-a"})
    )
    bcecmd = tmp_path / "bcecmd"
    bcecmd.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    bcecmd.chmod(0o755)
    model["upload"] = {
        "enabled": True,
        "remote_prefix": "bos:/bucket/model-quality",
        "allowlist": ["model", "reports"],
        "commands": [[str(bcecmd), "upload", "{output_dir}", "{remote_run_prefix}/model"]],
        "verify_commands": [[str(bcecmd), "verify", "{remote_run_prefix}"]],
        "success_commands": [
            [str(bcecmd), "upload", "{run_dir}/commit-markers/SUCCESS.json", "{remote_run_prefix}/SUCCESS.json"]
        ],
    }

    # Same size (even if bytes differ) takes the fast path and publishes.
    (run_dir / "model" / "model.safetensors").write_bytes(b"weights-v2")
    assert run_publish(model, "run-a", artifact_fingerprint="artifact-a")["status"] == "PASS"

    # A size change is still caught as a changed artifact.
    (run_dir / "model" / "model.safetensors").write_bytes(b"weights-much-longer")
    with pytest.raises(StageError, match="changed since validation"):
        run_publish(model, "run-a", artifact_fingerprint="artifact-a")


def test_success_marker_is_created_only_after_remote_verify(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path / "runs"))
    model = _fake_model(tmp_path)
    run_dir = tmp_path / "runs/run-a/fake-model"
    (run_dir / "model").mkdir(parents=True)
    (run_dir / "reports").mkdir()
    (run_dir / "artifact-manifest.json").write_text(
        json.dumps({"artifact_fingerprint": "artifact-a", "artifact_content_fingerprint": _artifact_content_fingerprint(_artifact_file_records(run_dir / "model"))})
    )
    (run_dir / "reports/summary.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "attempt_id": "attempt-a",
                "artifact_fingerprint": "artifact-a",
            }
        )
    )
    (run_dir / "current-attempt.json").write_text(
        json.dumps({"attempt_id": "attempt-a", "artifact_fingerprint": "artifact-a"})
    )
    bcecmd = tmp_path / "bcecmd"
    bcecmd.write_text('#!/bin/sh\n[ "$1" != verify ]\n', encoding="utf-8")
    bcecmd.chmod(0o755)
    model["upload"] = {
        "enabled": True,
        "remote_prefix": "bos:/bucket/model-quality",
        "allowlist": ["model", "reports"],
        "commands": [[str(bcecmd), "upload", "{output_dir}"]],
        "verify_commands": [[str(bcecmd), "verify"]],
        "success_commands": [[str(bcecmd), "success"]],
    }

    with pytest.raises(StageError, match="verify command failed"):
        run_publish(
            model,
            "run-a",
            artifact_fingerprint="artifact-a",
            evaluation_fingerprint="evaluation-a",
        )

    assert not (run_dir / "commit-markers/SUCCESS.json").exists()


def test_publish_reads_only_promoted_attempt_reports(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path / "runs"))
    model = _fake_model(tmp_path)
    run_dir = tmp_path / "runs/run-a/fake-model"
    (run_dir / "model").mkdir(parents=True)
    (run_dir / "model/weights.bin").write_bytes(b"weights")
    (run_dir / "reports").mkdir()
    (run_dir / "artifact-manifest.json").write_text(
        json.dumps({"artifact_fingerprint": "artifact-a", "artifact_content_fingerprint": _artifact_content_fingerprint(_artifact_file_records(run_dir / "model"))})
    )
    old_reports = run_dir / "attempts/attempt-old/reports"
    new_reports = run_dir / "attempts/attempt-new/reports"
    old_reports.mkdir(parents=True)
    new_reports.mkdir(parents=True)
    for reports, attempt in (
        (old_reports, "attempt-old"),
        (new_reports, "attempt-new"),
    ):
        (reports / "summary.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "attempt_id": attempt,
                    "artifact_fingerprint": "artifact-a",
                }
            )
        )
    (run_dir / "reports/summary.json").write_text(
        (old_reports / "summary.json").read_text()
    )
    (run_dir / "current-attempt.json").write_text(
        json.dumps({"attempt_id": "attempt-new", "artifact_fingerprint": "artifact-a"})
    )
    bcecmd = tmp_path / "bcecmd"
    bcecmd.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    bcecmd.chmod(0o755)
    model["upload"] = {
        "enabled": True,
        "remote_prefix": "bos:/bucket/model-quality",
        "allowlist": ["model", "reports"],
        "commands": [[str(bcecmd), "upload", "{output_dir}"]],
        "verify_commands": [[str(bcecmd), "verify"]],
        "success_commands": [[str(bcecmd), "success"]],
    }

    result = run_publish(
        model,
        "run-a",
        artifact_fingerprint="artifact-a",
        evaluation_fingerprint="evaluation-a",
    )

    assert result["status"] == "PASS"
    assert (new_reports / "upload-manifest.json").is_file()
    assert not (old_reports / "upload-manifest.json").exists()
    frozen = json.loads((new_reports / "upload-manifest.json").read_text())
    assert all(not item["path"].startswith("attempts/") for item in frozen["files"])


def test_stage_result_identity_wins_over_payload_fingerprints(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path / "runs"))
    model = _fake_model(tmp_path)
    run_id = "run-a"

    write_stage_result(
        run_id,
        model["id"],
        "preflight",
        {
            "status": "PASS",
            "artifact_fingerprint": "planner-request-fingerprint",
            "evaluation_fingerprint": "planner-request-evaluation",
        },
        attempt_id="attempt-a",
        artifact_fingerprint="finalized-artifact-fingerprint",
        evaluation_fingerprint="finalized-evaluation-fingerprint",
    )

    stored = read_stage_result(
        run_id,
        model["id"],
        "preflight",
        attempt_id="attempt-a",
        artifact_fingerprint="finalized-artifact-fingerprint",
    )

    assert stored["artifact_fingerprint"] == "finalized-artifact-fingerprint"
    assert stored["evaluation_fingerprint"] == "finalized-evaluation-fingerprint"


def test_validate_uses_configured_quantization_scale_suffixes(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file

    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path / "runs"))
    model = _fake_model(tmp_path)
    model["validation"] = {
        "profile": "causal_lm",
        "require_quantization_config": True,
        "min_quantization_auxiliary_tensors": 1,
        "required_files": ["config.json", "model.safetensors.index.json"],
    }
    output = tmp_path / "runs/run-a/fake-model/model"
    output.mkdir(parents=True)
    (output / "config.json").write_text(
        json.dumps({"quantization_config": {"quantization_status": "compressed"}})
    )
    save_file(
        {"layer.weight": torch.ones(2, 2), "layer.scale": torch.ones(2)},
        output / "model.safetensors",
    )
    (output / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.weight": "model.safetensors",
                    "layer.scale": "model.safetensors",
                }
            }
        )
    )

    with pytest.raises(StageError) as error:
        run_validate(model, "run-a")
    assert "quantization auxiliary tensors" in str(error.value)

    model["validation"]["quantization_auxiliary_suffixes"] = [".weight_scale", ".scale"]
    model["validation"]["quantization_scale_suffixes"] = [".scale"]

    result = run_validate(model, "run-a")

    assert result["status"] in {"PASS", "WARN"}
    assert result["quantization_auxiliary_tensor_count"] == 1
    assert [record["name"] for record in result["quantization_tensor_health"]] == [
        "layer.scale"
    ]
