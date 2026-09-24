"""Run directory and atomic state helpers."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def runs_root() -> Path:
    configured = os.getenv("MODEL_QUALITY_RUNS_ROOT")
    if configured is None:
        if os.getenv("BUILDKITE") == "true":
            raise RuntimeError(
                "MODEL_QUALITY_RUNS_ROOT is required in Buildkite so steps share state"
            )
        return Path(".model-quality/runs")
    root = Path(configured).expanduser()
    if os.getenv("BUILDKITE") == "true" and not root.is_absolute():
        raise RuntimeError(
            "MODEL_QUALITY_RUNS_ROOT must be absolute in Buildkite so steps share state"
        )
    return root


def model_run_dir(run_id: str, model_id: str) -> Path:
    for name, value in (("run_id", run_id), ("model_id", model_id)):
        if not _SAFE_PATH_COMPONENT.fullmatch(value):
            raise ValueError(f"unsafe {name}: {value!r}")
    return runs_root() / run_id / model_id


def attempt_state_dir(run_id: str, model_id: str, attempt_id: str) -> Path:
    if not _SAFE_PATH_COMPONENT.fullmatch(attempt_id):
        raise ValueError(f"unsafe attempt_id: {attempt_id!r}")
    return model_run_dir(run_id, model_id) / "attempts" / attempt_id / "state"


def attempt_reports_dir(run_id: str, model_id: str, attempt_id: str) -> Path:
    if not _SAFE_PATH_COMPONENT.fullmatch(attempt_id):
        raise ValueError(f"unsafe attempt_id: {attempt_id!r}")
    return model_run_dir(run_id, model_id) / "attempts" / attempt_id / "reports"


def evaluation_cache_path(
    run_id: str, model_id: str, evaluation_fingerprint: str
) -> Path:
    if not _SAFE_PATH_COMPONENT.fullmatch(evaluation_fingerprint):
        raise ValueError(f"unsafe evaluation_fingerprint: {evaluation_fingerprint!r}")
    return (
        model_run_dir(run_id, model_id)
        / "evaluation-cache"
        / f"{evaluation_fingerprint}.json"
    )


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_stage_result(
    run_id: str,
    model_id: str,
    stage: str,
    result: dict[str, Any],
    *,
    attempt_id: str = "default",
    artifact_fingerprint: str | None = None,
    evaluation_fingerprint: str | None = None,
) -> Path:
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "model_id": model_id,
        "stage": stage,
        "attempt_id": attempt_id,
        "artifact_fingerprint": artifact_fingerprint,
        "evaluation_fingerprint": evaluation_fingerprint,
        "recorded_at": utc_now(),
        **result,
    }
    attempt_path = attempt_state_dir(run_id, model_id, attempt_id) / f"{stage}.json"
    atomic_write_json(attempt_path, payload)
    latest_path = model_run_dir(run_id, model_id) / "state" / f"{stage}.json"
    atomic_write_json(latest_path, payload)
    return attempt_path


def read_stage_result(
    run_id: str,
    model_id: str,
    stage: str,
    *,
    attempt_id: str | None = None,
    artifact_fingerprint: str | None = None,
) -> dict[str, Any]:
    path = (
        attempt_state_dir(run_id, model_id, attempt_id) / f"{stage}.json"
        if attempt_id is not None
        else model_run_dir(run_id, model_id) / "state" / f"{stage}.json"
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    if attempt_id is not None and result.get("attempt_id") != attempt_id:
        raise ValueError(f"stale {stage} state belongs to another attempt")
    if (
        artifact_fingerprint is not None
        and result.get("artifact_fingerprint") != artifact_fingerprint
    ):
        raise ValueError(f"stale {stage} state has another artifact fingerprint")
    return result
