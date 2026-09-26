"""Fixtures for the model quality web control-plane tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


def _model(
    *,
    model_id: str,
    source: Path,
    upload: dict | None = None,
    workflow_revision: str = "git:abc1234",
    evaluation: bool = True,
) -> dict:
    model: dict = {
        "id": model_id,
        "enabled": True,
        "priority": "P1",
        "business_tier": "representative",
        "max_result_age_hours": 24,
        "source": {"path": str(source), "revision": "local-checksum"},
        "resources": {
            "gpu_count": 1,
            "gpu_memory_gib": 96,
            "host_memory_gib": 64,
            "io_weight": 1,
            "estimated_gpu_hours": 2,
            "estimated_eval_gpu_hours": 1,
            "timeout_hours": 3,
        },
        "workflow": {
            "revision": workflow_revision,
            "quantize": [
                "python3",
                "ci/model_quality/entrypoints/qwen3_dense_w4a8.py",
                "--model-id",
                "{source_path}",
                "--num-calibration-samples",
                "32",
                "--output-dir",
                "{output_dir}",
                "--work-dir",
                "{work_dir}",
            ],
        },
        "validation": {"profile": "causal_lm"},
        "runtime_smoke": {"enabled": False},
    }
    if evaluation:
        model["evaluation"] = {
            "enabled_by_default": True,
            "profile": "causal_lm",
            "suites": ["gsm8k"],
            "runtime_revision": "vllm:0.6.3",
            "command": [
                "python3",
                "evaluator.py",
                "--model",
                "{output_dir}",
                "--output",
                "{reports_dir}/evaluation-raw.json",
            ],
            "result_file": "{reports_dir}/evaluation-raw.json",
        }
    if upload is not None:
        model["upload"] = upload
    return model


def _write_manifest(tmp_path: Path, models: list[dict]) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump({"schema_version": 1, "models": models}, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source-model"
    source.mkdir(parents=True, exist_ok=True)
    (source / "config.json").write_text("{}\n", encoding="utf-8")
    return source


@pytest.fixture
def web_env(tmp_path: Path) -> SimpleNamespace:
    """A single enabled model without upload, plus an empty runs root."""

    root = tmp_path / "runs"
    source = _source(tmp_path)
    config = _write_manifest(tmp_path, [_model(model_id="demo-model", source=source)])
    return SimpleNamespace(
        tmp_path=tmp_path,
        root=root,
        config=config,
        source=source,
        model_id="demo-model",
    )


@pytest.fixture
def publish_env(tmp_path: Path) -> SimpleNamespace:
    """A single model with an enabled, reviewed BCE upload allowlist."""

    root = tmp_path / "runs"
    source = _source(tmp_path)
    upload = {
        "enabled": True,
        "remote_prefix": "bos:/model-quality-test/llm-compressor",
        "allowlist": ["model", "reports"],
        "commands": [
            [
                "bcecmd",
                "bos",
                "cp",
                "{output_dir}",
                "bos:/model-quality-test/llm-compressor/{run_id}/model",
            ]
        ],
        "verify_commands": [
            [
                "bcecmd",
                "bos",
                "ls",
                "bos:/model-quality-test/llm-compressor/{run_id}/model",
            ]
        ],
        "success_commands": [
            [
                "bcecmd",
                "bos",
                "cp",
                "{reports_dir}/SUCCESS.json",
                "bos:/model-quality-test/llm-compressor/{run_id}/SUCCESS.json",
            ]
        ],
    }
    config = _write_manifest(
        tmp_path,
        [_model(model_id="upload-model", source=source, upload=upload)],
    )
    return SimpleNamespace(
        tmp_path=tmp_path,
        root=root,
        config=config,
        source=source,
        model_id="upload-model",
    )
