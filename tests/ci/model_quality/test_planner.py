import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from ci.model_quality.buildkite import render_pipeline
from ci.model_quality.config import (
    ConfigError,
    evaluation_fingerprint,
    finalized_artifact_fingerprint,
    load_model_config,
    model_fingerprint,
)
from ci.model_quality.executor import (
    StageError,
    run_aggregate,
    run_publish,
    run_report,
    run_validate,
)
from ci.model_quality.planner import build_execution_plan
from ci.model_quality.state import atomic_write_json, write_stage_result


def _model(model_id, priority, cost, enabled=True):
    return {
        "id": model_id,
        "enabled": enabled,
        "priority": priority,
        "max_result_age_hours": 24,
        "source": {"path": f"/models/{model_id}"},
        "resources": {"gpu_count": 1, "estimated_gpu_hours": cost},
        "workflow": {
            "revision": "test-workflow",
            "quantize": ["python", f"{model_id}.py"],
        },
    }


def test_priority_and_budget_are_deterministic():
    config = {
        "schema_version": 1,
        "models": [
            _model("large-p1", "P1", 7),
            _model("small-p2", "P2", 1),
            _model("small-p1", "P1", 2),
        ],
    }

    plan = build_execution_plan(
        config, git_sha="abc123", max_gpu_hours=3, run_id="test-run"
    )

    assert [job["id"] for job in plan["selected"]] == ["small-p1", "small-p2"]
    assert [job["id"] for job in plan["deferred"]] == ["large-p1"]
    assert plan["deferred"][0]["status"] == "DEFERRED_BUDGET"
    assert plan["budget"]["remaining_gpu_hours"] == 0
    assert plan["run_mode"] == "quantize"
    assert plan["attempt_id"].startswith("attempt-")


def test_filter_rejects_unknown_model():
    config = {"schema_version": 1, "models": [_model("known", "P1", 1)]}
    with pytest.raises(ConfigError, match="unknown models"):
        build_execution_plan(
            config,
            git_sha="abc123",
            max_gpu_hours=1,
            model_filter="missing",
        )


def test_fingerprint_changes_with_workflow():
    model = _model("model-a", "P1", 1)
    first = model_fingerprint(model, "abc123")
    model["workflow"]["quantize"].append("--changed")
    assert model_fingerprint(model, "abc123") != first


def test_fingerprint_ignores_scheduler_and_upload_changes():
    model = _model("model-a", "P1", 1)
    first = model_fingerprint(model, "abc123")
    model["priority"] = "P0"
    model["resources"]["estimated_gpu_hours"] = 99
    model["upload"] = {"enabled": True, "remote_prefix": "bos:/elsewhere"}
    model["runtime_smoke"] = {"python": "/new/vllm/python"}
    model["evaluation"] = {"command": ["new-evaluator"]}

    assert model_fingerprint(model, "abc123") == first
    assert model_fingerprint(model, "different-git-sha") == first


def test_evaluation_fingerprint_changes_without_invalidating_artifact():
    model = _model("model-a", "P1", 1)
    model["evaluation"] = {
        "command": ["evaluate", "--limit", "10"],
        "result_file": "result.json",
        "runtime_revision": "test-runtime",
    }
    artifact = model_fingerprint(model)
    first_evaluation = evaluation_fingerprint(model, artifact)
    model["evaluation"]["command"][-1] = "20"

    assert model_fingerprint(model) == artifact
    assert evaluation_fingerprint(model, artifact) != first_evaluation


def test_finalized_artifact_fingerprint_binds_source_provenance():
    request = "request-fingerprint"
    first = finalized_artifact_fingerprint(
        request, {"metadata_files": [{"sha256": "aaa"}]}
    )
    second = finalized_artifact_fingerprint(
        request, {"metadata_files": [{"sha256": "bbb"}]}
    )

    assert first != second


def test_runtime_plan_omits_model_configuration_but_keeps_upload_decision():
    model = _model("model-a", "P1", 1)
    model["upload"] = {
        "enabled": True,
        "remote_prefix": "bos:/bucket",
        "allowlist": ["model", "reports"],
        "commands": [["bcecmd", "upload", "{output_dir}"]],
        "verify_commands": [["bcecmd", "verify"]],
        "success_commands": [["bcecmd", "success"]],
    }
    config = {"schema_version": 1, "models": [model]}

    plan = build_execution_plan(
        config, git_sha="abc123", max_gpu_hours=1, run_id="run-a"
    )

    assert plan["selected"][0]["upload_enabled"] is True
    assert "model" not in plan["selected"][0]


def test_checked_in_manifest_is_safe_and_valid():
    root = Path(__file__).parents[3]
    config = load_model_config(root / "ci/model_quality/config/models.yaml")
    assert config["models"][0]["enabled"] is False


def test_deepseek_example_manifest_is_valid_and_complete():
    root = Path(__file__).parents[3]
    config = load_model_config(
        root / "ci/model_quality/config/models/deepseek_v4_wna8.example.yaml"
    )
    model = config["models"][0]
    argv = model["workflow"]["quantize"]

    assert model["enabled"] is False
    assert argv[argv.index("--quant-mode") + 1] == "wna8"
    assert "--checkpoint-progress" in argv
    assert "--no-async-save" in argv
    assert model["resources"]["runtime_gpu_count"] == 8


def test_manifest_supports_relative_includes(tmp_path):
    child = tmp_path / "models/child.yaml"
    child.parent.mkdir()
    child.write_text(
        """
schema_version: 1
models:
  - id: child-model
    source: {path: /models/child}
    resources: {gpu_count: 1, estimated_gpu_hours: 1}
    workflow: {revision: test-workflow, quantize: [python3, child.py]}
""",
        encoding="utf-8",
    )
    root = tmp_path / "models.yaml"
    root.write_text(
        "schema_version: 1\nincludes: [models/child.yaml]\nmodels: []\n",
        encoding="utf-8",
    )

    config = load_model_config(root)

    assert [model["id"] for model in config["models"]] == ["child-model"]


def test_manifest_rejects_enabled_upload_without_commands(tmp_path):
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        """
schema_version: 1
models:
  - id: model-a
    source: {path: /models/model-a}
    resources: {gpu_count: 1, estimated_gpu_hours: 1}
    workflow: {revision: test-workflow, quantize: [python3, model.py]}
    upload: {enabled: true, remote_prefix: 'bos:/bucket', allowlist: [model]}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="upload.commands"):
        load_model_config(manifest)


def test_manifest_rejects_upload_of_run_root(tmp_path):
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        """
schema_version: 1
models:
  - id: model-a
    source: {path: /models/model-a}
    resources: {gpu_count: 1, estimated_gpu_hours: 1}
    workflow: {revision: test-workflow, quantize: [python3, model.py]}
    upload:
      enabled: true
      remote_prefix: 'bos:/bucket'
      allowlist: [model, reports]
      commands: [[bcecmd, upload, '{run_dir}', remote]]
      verify_commands: [[bcecmd, verify, remote]]
      success_commands: [[bcecmd, upload, marker, remote]]
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="allowlisted model/report"):
        load_model_config(manifest)


def test_manifest_requires_evaluator_runtime_revision(tmp_path):
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        """
schema_version: 1
models:
  - id: model-a
    source: {path: /models/model-a}
    resources: {gpu_count: 1, estimated_gpu_hours: 1}
    workflow: {revision: test-workflow, quantize: [python3, model.py]}
    evaluation:
      command: [evaluate]
      result_file: result.json
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="runtime_revision"):
        load_model_config(manifest)


def test_manifest_rejects_path_traversal_model_id(tmp_path):
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        """
schema_version: 1
models:
  - id: ../escape
    source: {path: /models/model-a}
    resources: {gpu_count: 1, estimated_gpu_hours: 1}
    workflow: {revision: test-workflow, quantize: [python3, model.py]}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="only letters"):
        load_model_config(manifest)


def test_buildkite_pipeline_has_failure_tolerant_report():
    config = {"schema_version": 1, "models": [_model("model-a", "P1", 1)]}
    plan = build_execution_plan(
        config, git_sha="abc123", max_gpu_hours=1, run_id="test-run"
    )
    pipeline = render_pipeline(plan, config_path="models.yaml")
    assert "model-a-runtime-smoke" in pipeline
    assert "model-a-publish" not in pipeline
    assert "allow_dependency_failure: true" in pipeline
    assert "model-quality-summary" in pipeline
    assert "command: python3 -m ci.model_quality.stage" in pipeline
    assert "concurrency_group: llm-compressor/model-quality-host" in pipeline
    assert "--evaluation-fingerprint" in pipeline
    assert "--attempt-id" in pipeline


def test_buildkite_pipeline_adds_publish_when_enabled():
    model = _model("model-a", "P1", 1)
    model["upload"] = {"enabled": True}
    config = {"schema_version": 1, "models": [model]}
    plan = build_execution_plan(
        config, git_sha="abc123", max_gpu_hours=1, run_id="test-run"
    )

    pipeline = render_pipeline(plan, config_path="models.yaml")

    assert "model-a-publish" in pipeline


def test_eval_only_skips_quantization_and_adds_evaluation():
    model = _model("model-a", "P1", 1)
    model["resources"]["estimated_eval_gpu_hours"] = 0.25
    config = {"schema_version": 1, "models": [model]}
    plan = build_execution_plan(
        config,
        git_sha="abc123",
        max_gpu_hours=1,
        run_id="existing-run",
        run_mode="eval_only",
    )

    pipeline = render_pipeline(plan, config_path="models.yaml")

    assert "model-a-quantize" not in pipeline
    assert "model-a-evaluate" in pipeline
    assert "--run-mode eval_only" in pipeline
    assert plan["budget"]["selected_gpu_hours"] == 0.25


def test_upload_only_revalidates_reconstructs_report_and_publishes():
    config = {"schema_version": 1, "models": [_model("model-a", "P1", 1)]}
    plan = build_execution_plan(
        config,
        git_sha="abc123",
        max_gpu_hours=1,
        run_id="existing-run",
        run_mode="upload_only",
    )

    pipeline = render_pipeline(plan, config_path="models.yaml")

    assert "model-a-publish" in pipeline
    assert "model-a-preflight" in pipeline
    assert "model-a-validate" in pipeline
    assert "model-a-runtime-smoke" in pipeline
    assert "model-a-report" in pipeline
    assert plan["budget"]["selected_gpu_hours"] == 0


def test_eval_mode_requires_an_explicit_cost():
    config = {"schema_version": 1, "models": [_model("model-a", "P1", 1)]}

    with pytest.raises(ConfigError, match="estimated_eval_gpu_hours"):
        build_execution_plan(
            config,
            git_sha="abc123",
            max_gpu_hours=1,
            run_id="existing-run",
            run_mode="eval_only",
        )


def test_validate_checks_index_and_finite_tensors(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path))
    model = _model("model-a", "P1", 1)
    model["validation"] = {
        "profile": "causal_lm",
        "require_quantization_config": True,
        "min_quantization_auxiliary_tensors": 1,
        "required_files": ["config.json", "model.safetensors.index.json"],
    }
    output = tmp_path / "run-a" / "model-a" / "model"
    output.mkdir(parents=True)
    (output / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quantization_status": "compressed",
                    "format": "int-quantized",
                    "config_groups": {"group_0": {"targets": ["Linear"]}},
                }
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "layer.weight": torch.ones(2, 2),
            "layer.weight_scale": torch.ones(2),
        },
        output / "model-00001.safetensors",
    )
    (output / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.weight": "model-00001.safetensors",
                    "layer.weight_scale": "model-00001.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    result = run_validate(model, "run-a")

    assert result["status"] == "WARN"
    assert result["tensor_count"] == 2
    assert result["non_finite_tensor_count"] == 0
    assert result["quantization_auxiliary_tensor_count"] == 1
    assert result["quantization_tensor_health"][0]["min"] == 1.0
    checksums = (tmp_path / "run-a/model-a/reports/checksums.sha256").read_text()
    assert "model-00001.safetensors" in checksums
    assert "config.json" in checksums


def test_validate_rejects_non_positive_scales(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path))
    model = _model("model-a", "P1", 1)
    model["validation"] = {
        "profile": "causal_lm",
        "require_quantization_config": True,
        "required_files": ["config.json", "model.safetensors.index.json"],
    }
    output = tmp_path / "run-a/model-a/model"
    output.mkdir(parents=True)
    (output / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quantization_status": "compressed",
                    "config_groups": {"group_0": {"targets": ["Linear"]}},
                }
            }
        )
    )
    save_file(
        {"layer.weight_scale": torch.tensor([0.0, 1.0])},
        output / "model.safetensors",
    )
    (output / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight_scale": "model.safetensors"}})
    )

    with pytest.raises(StageError, match="non-positive weight scales"):
        run_validate(model, "run-a")


def test_report_marks_missing_required_stage_as_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path))
    model = _model("model-a", "P1", 1)
    write_stage_result("run-a", "model-a", "preflight", {"status": "PASS"})

    result = run_report(model, "run-a")

    assert result["status"] == "FAIL"
    summary = json.loads((tmp_path / "run-a/model-a/reports/summary.json").read_text())
    assert summary["status"] == "FAIL"


def test_aggregate_fails_when_expected_report_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path))
    config = {"schema_version": 1, "models": [_model("model-a", "P1", 1)]}

    result = run_aggregate(config, "run-a", expected_models=["model-a"])

    assert result["status"] == "FAIL"
    assert result["models"][0]["reason_code"] == "REPORT_MISSING"


def test_aggregate_includes_deferred_plan_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path))
    config = {"schema_version": 1, "models": []}
    atomic_write_json(
        tmp_path / "run-a/execution-plan.json",
        {
            "budget": {"max_gpu_hours": 1, "selected_gpu_hours": 0},
            "deferred": [{"id": "large-model", "status": "DEFERRED_BUDGET"}],
        },
    )

    result = run_aggregate(config, "run-a", expected_models=[])

    assert result["deferred"][0]["id"] == "large-model"
    assert result["budget"]["max_gpu_hours"] == 1


def test_buildkite_pipeline_reports_deferred_jobs():
    config = {"schema_version": 1, "models": [_model("model-a", "P1", 2)]}
    plan = build_execution_plan(
        config, git_sha="abc123", max_gpu_hours=1, run_id="test-run"
    )

    pipeline = render_pipeline(plan, config_path="models.yaml")

    assert "model-a-deferred" in pipeline
    assert "DEFERRED_BUDGET" in pipeline


def test_publish_is_safely_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path))
    model = _model("model-a", "P1", 1)
    model["upload"] = {"enabled": False}

    result = run_publish(model, "run-a")

    assert result == {"status": "SKIPPED", "reason_code": "UPLOAD_DISABLED"}


def test_explicit_upload_rejects_disabled_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path))
    model = _model("model-a", "P1", 1)
    model["upload"] = {"enabled": False}

    with pytest.raises(StageError, match="upload.enabled is false"):
        run_publish(model, "run-a", explicitly_requested=True)


def test_buildkite_requires_absolute_shared_runs_root(monkeypatch):
    from ci.model_quality.state import runs_root

    monkeypatch.setenv("BUILDKITE", "true")
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", "relative/runs")

    with pytest.raises(RuntimeError, match="must be absolute"):
        runs_root()


def test_run_directory_rejects_path_traversal(monkeypatch, tmp_path):
    from ci.model_quality.state import model_run_dir

    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="unsafe run_id"):
        model_run_dir("../escape", "model-a")


def test_stage_state_rejects_another_attempt(monkeypatch, tmp_path):
    from ci.model_quality.state import read_stage_result

    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path))
    write_stage_result(
        "run-a",
        "model-a",
        "evaluate",
        {"status": "PASS"},
        attempt_id="attempt-old",
        artifact_fingerprint="artifact-a",
    )

    with pytest.raises((ValueError, FileNotFoundError)):
        read_stage_result("run-a", "model-a", "evaluate", attempt_id="attempt-new")


def test_stage_state_is_preserved_per_attempt(monkeypatch, tmp_path):
    from ci.model_quality.state import read_stage_result

    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path))
    write_stage_result(
        "run-a",
        "model-a",
        "validate",
        {"status": "PASS"},
        attempt_id="attempt-one",
    )
    write_stage_result(
        "run-a",
        "model-a",
        "validate",
        {"status": "FAIL"},
        attempt_id="attempt-two",
    )

    first = read_stage_result("run-a", "model-a", "validate", attempt_id="attempt-one")
    second = read_stage_result("run-a", "model-a", "validate", attempt_id="attempt-two")
    assert first["status"] == "PASS"
    assert second["status"] == "FAIL"
