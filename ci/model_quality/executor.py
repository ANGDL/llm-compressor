"""P0 model quality stage executors."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from safetensors import safe_open

from .config import (
    ConfigError,
    finalized_artifact_fingerprint,
)
from .config import (
    evaluation_fingerprint as configured_evaluation_fingerprint,
)
from .metrics import compare_evaluation
from .placeholders import PlaceholderError, resolve_argument, resolve_argv
from .state import (
    atomic_write_json,
    attempt_reports_dir,
    attempt_state_dir,
    evaluation_cache_path,
    model_run_dir,
    read_stage_result,
    utc_now,
)


class StageError(RuntimeError):
    def __init__(self, message: str, *, reason_code: str = "STAGE_FAILED"):
        super().__init__(message)
        self.reason_code = reason_code


def find_model(config: dict[str, Any], model_id: str) -> dict[str, Any]:
    for model in config["models"]:
        if model["id"] == model_id:
            return model
    raise ConfigError(f"unknown model {model_id!r}")


def _paths(run_id: str, model_id: str) -> dict[str, Path]:
    root = model_run_dir(run_id, model_id)
    return {
        "run_dir": root,
        "output_dir": root / "model",
        "work_dir": root / "work",
        "logs_dir": root / "logs",
        "reports_dir": root / "reports",
    }


def _command_values(
    model: dict[str, Any], run_id: str, *, reports_dir: Path | None = None
) -> dict[str, str]:
    paths = _paths(run_id, model["id"])
    values = {
        "source_path": str(Path(model["source"]["path"]).expanduser()),
        **{key: str(path) for key, path in paths.items()},
    }
    if reports_dir is not None:
        values["reports_dir"] = str(reports_dir)
    return values


def _attempt_logs_dir(run_id: str, model_id: str, attempt_id: str) -> Path:
    return _paths(run_id, model_id)["logs_dir"] / attempt_id


def _resolved_argv(model: dict[str, Any], run_id: str) -> list[str]:
    values = _command_values(model, run_id)
    try:
        return resolve_argv(model["workflow"]["quantize"], values)
    except PlaceholderError as error:
        raise StageError(
            str(error),
            reason_code="CONFIG_ERROR",
        ) from error


def _validate_argument_constraints(model: dict[str, Any], argv: list[str]) -> list[str]:
    failures = []
    workflow = model["workflow"]
    for pair in workflow.get("forbidden_argument_pairs", []):
        if not isinstance(pair, list) or len(pair) != 2:
            failures.append("forbidden_argument_pairs entries must contain two flags")
        elif pair[0] in argv and pair[1] in argv:
            failures.append(f"arguments {pair[0]} and {pair[1]} cannot be combined")
    for group in workflow.get("exactly_one_argument_groups", []):
        if not isinstance(group, list) or len(group) < 2:
            failures.append(
                "exactly_one_argument_groups entries need at least two flags"
            )
            continue
        present = [flag for flag in group if flag in argv]
        if len(present) != 1:
            failures.append(
                f"exactly one of {', '.join(group)} is required; found {present}"
            )
    return failures


def _resolve_executable(value: str) -> str | None:
    if "/" in value:
        candidate = Path(value).expanduser()
        return str(candidate) if candidate.is_file() else None
    return shutil.which(value)


def _missing_index_shards(checkpoint: Path) -> list[str]:
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.is_file():
        return []
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise StageError(
            f"invalid source safetensors index: {error}",
            reason_code="PREFLIGHT_FAILED",
        ) from error
    if not isinstance(weight_map, dict) or not weight_map:
        raise StageError(
            "source safetensors weight_map is empty",
            reason_code="PREFLIGHT_FAILED",
        )
    return sorted(
        name for name in set(weight_map.values()) if not (checkpoint / name).is_file()
    )


def _run_logged(
    argv: list[str],
    log,
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    timeout_seconds: float | None = None,
) -> int:
    """Run a child process and forward termination to its process group."""

    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    previous_handlers = {}

    def forward(signum, _frame):
        if process.poll() is None:
            os.killpg(process.pid, signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, forward)
    try:
        try:
            return process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return 124
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def run_preflight(
    model: dict[str, Any],
    run_id: str,
    *,
    run_mode: str = "quantize",
    git_sha: str = "unknown",
    fingerprint: str | None = None,
    evaluation_fingerprint: str | None = None,
    attempt_id: str = "default",
) -> dict[str, Any]:
    source = Path(model["source"]["path"]).expanduser()
    argv = _resolved_argv(model, run_id)
    failures = []
    warnings = []
    failures.extend(_validate_argument_constraints(model, argv))

    if not source.is_dir():
        failures.append(f"source model directory does not exist: {source}")
    else:
        for required in ("config.json",):
            if not (source / required).is_file():
                failures.append(f"source model is missing {required}")
        try:
            missing_source_shards = _missing_index_shards(source)
        except StageError as error:
            failures.append(str(error))
        else:
            if missing_source_shards:
                failures.append(
                    "source model is missing "
                    f"{len(missing_source_shards)} indexed shards"
                )

    if run_mode in {"quantize", "quantize_and_eval"}:
        executable = _resolve_executable(argv[0])
        if executable is None:
            failures.append(f"quantization executable not found: {argv[0]}")
        if len(argv) > 1 and argv[1].endswith(".py") and not Path(argv[1]).is_file():
            failures.append(f"quantization script not found: {argv[1]}")
    elif not _paths(run_id, model["id"])["output_dir"].is_dir():
        failures.append(f"existing run has no model artifact for {run_mode}")

    required_gpus = model["resources"]["gpu_count"]
    try:
        import torch

        visible_gpus = torch.accelerator.device_count()
        accelerator_available = torch.accelerator.is_available()
    except Exception as error:  # pragma: no cover - environment dependent
        failures.append(f"unable to inspect torch accelerator: {error}")
        visible_gpus = 0
        accelerator_available = False
    if not accelerator_available or visible_gpus < required_gpus:
        failures.append(
            f"requires {required_gpus} GPU API compatible devices, found {visible_gpus}"
        )

    run_dir = _paths(run_id, model["id"])["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    current_source_provenance = _source_provenance(source) if source.is_dir() else None
    finalized_fingerprint = (
        finalized_artifact_fingerprint(fingerprint, current_source_provenance)
        if fingerprint is not None and current_source_provenance is not None
        else fingerprint
    )
    finalized_evaluation_fingerprint = (
        configured_evaluation_fingerprint(model, finalized_fingerprint)
        if finalized_fingerprint is not None
        else evaluation_fingerprint
    )
    manifest_path = run_dir / "input-manifest.json"
    existing_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else None
    )
    expected_evaluation_fingerprint = (
        configured_evaluation_fingerprint(model, fingerprint)
        if fingerprint is not None
        else None
    )
    if (
        run_mode in {"quantize", "quantize_and_eval"}
        and evaluation_fingerprint is not None
        and expected_evaluation_fingerprint != evaluation_fingerprint
    ):
        failures.append(
            "planner evaluation fingerprint does not match current configuration"
        )
    if run_mode in {"quantize", "quantize_and_eval"}:
        if not fingerprint:
            failures.append("planner fingerprint is required for quantization")
        elif (
            existing_manifest is not None
            and existing_manifest.get("fingerprint") != fingerprint
        ):
            failures.append(
                "run_id already has a different input fingerprint; use a new run_id"
            )
        elif (
            existing_manifest is not None
            and existing_manifest.get("source_provenance") != current_source_provenance
        ):
            failures.append(
                "source checkpoint changed since this run was created; use a new run_id"
            )
    elif existing_manifest is None:
        failures.append(f"existing run has no input manifest for {run_mode}")
    elif run_mode in {"eval_only", "upload_only"}:
        if existing_manifest.get("fingerprint") != fingerprint:
            failures.append(
                "current quantization inputs do not match the existing artifact "
                "fingerprint"
            )
        if (
            existing_manifest.get("finalized_artifact_fingerprint")
            != finalized_fingerprint
        ):
            failures.append(
                "current source provenance does not match the finalized artifact "
                "fingerprint"
            )
        if existing_manifest.get("source_provenance") != current_source_provenance:
            failures.append(
                "source checkpoint provenance changed since the artifact was created"
            )
        artifact_manifest_path = run_dir / "artifact-manifest.json"
        if not artifact_manifest_path.is_file():
            failures.append(
                "artifact-manifest.json is missing for an existing artifact"
            )
        else:
            try:
                artifact_manifest = json.loads(
                    artifact_manifest_path.read_text(encoding="utf-8")
                )
                artifact_files = _artifact_file_records(
                    _paths(run_id, model["id"])["output_dir"]
                )
                current_content_fingerprint = _artifact_content_fingerprint(
                    artifact_files
                )
            except (OSError, json.JSONDecodeError, ValueError) as error:
                failures.append(f"unable to inspect artifact manifest: {error}")
            else:
                if (
                    artifact_manifest.get("artifact_fingerprint")
                    != finalized_fingerprint
                ):
                    failures.append(
                        "artifact manifest has a stale artifact fingerprint"
                    )
                if (
                    artifact_manifest.get("source_provenance")
                    != current_source_provenance
                ):
                    failures.append(
                        "artifact source provenance does not match current source"
                    )
                if (
                    artifact_manifest.get("artifact_content_fingerprint")
                    != current_content_fingerprint
                ):
                    failures.append(
                        "quantized artifact contents changed since validation"
                    )
    free_bytes = shutil.disk_usage(run_dir).free
    minimum_free_gib = float(model["resources"].get("minimum_free_disk_gib", 1))
    if free_bytes < minimum_free_gib * 1024**3:
        failures.append(
            f"requires {minimum_free_gib:g} GiB free disk, found "
            f"{free_bytes / 1024**3:.2f} GiB"
        )

    if model.get("upload", {}).get("enabled") and not shutil.which("bcecmd"):
        failures.append("upload is enabled but bcecmd is not available")
    runtime = model.get("runtime_smoke") or {"enabled": False}
    runtime_python = os.getenv("VLLM_PYTHON_ENV") or runtime.get("python")
    if runtime.get("enabled", True):
        if not runtime_python or _resolve_executable(str(runtime_python)) is None:
            failures.append(
                "runtime_smoke.python or VLLM_PYTHON_ENV must name an "
                "existing executable"
            )
        else:
            runtime_check = subprocess.run(
                [str(runtime_python), "-c", "import vllm"],
                text=True,
                capture_output=True,
                check=False,
            )
            if runtime_check.returncode != 0:
                failures.append("configured runtime Python cannot import vllm")
    if not model.get("evaluation", {}).get("enabled_by_default", False):
        warnings.append("optional evaluation is disabled for this model")

    result = {
        "status": "FAIL" if failures else ("WARN" if warnings else "PASS"),
        "reason_code": "PREFLIGHT_FAILED" if failures else None,
        "source_path": str(source),
        "resolved_argv": argv,
        "visible_gpus": visible_gpus,
        "required_gpus": required_gpus,
        "run_mode": run_mode,
        "artifact_fingerprint": fingerprint,
        "finalized_artifact_fingerprint": finalized_fingerprint,
        "finalized_evaluation_fingerprint": finalized_evaluation_fingerprint,
        "evaluation_fingerprint": evaluation_fingerprint,
        "attempt_id": attempt_id,
        "free_disk_gib": round(free_bytes / 1024**3, 3),
        "failures": failures,
        "warnings": warnings,
        "source_provenance": current_source_provenance,
    }
    if (
        not failures
        and existing_manifest is None
        and run_mode in {"quantize", "quantize_and_eval"}
    ):
        atomic_write_json(
            manifest_path,
            {
                "schema_version": 1,
                "created_at": utc_now(),
                "git_sha": git_sha,
                "fingerprint": fingerprint,
                "finalized_artifact_fingerprint": finalized_fingerprint,
                "evaluation_fingerprint": evaluation_fingerprint,
                "finalized_evaluation_fingerprint": (finalized_evaluation_fingerprint),
                "model": model,
                "resolved_argv": argv,
                "source_provenance": result["source_provenance"],
            },
        )
    return result


def run_quantize(
    model: dict[str, Any],
    run_id: str,
    *,
    attempt_id: str = "default",
    artifact_fingerprint: str | None = None,
) -> dict[str, Any]:
    preflight = read_stage_result(
        run_id,
        model["id"],
        "preflight",
        attempt_id=attempt_id,
        artifact_fingerprint=artifact_fingerprint,
    )
    if preflight["status"] not in {"PASS", "WARN"}:
        raise StageError("preflight did not pass", reason_code="DEPENDENCY_FAILED")

    paths = _paths(run_id, model["id"])
    paths["logs_dir"].mkdir(parents=True, exist_ok=True)
    paths["work_dir"].mkdir(parents=True, exist_ok=True)
    argv = _resolved_argv(model, run_id)
    constraint_failures = _validate_argument_constraints(model, argv)
    if constraint_failures:
        raise StageError("; ".join(constraint_failures), reason_code="CONFIG_ERROR")
    log_path = _attempt_logs_dir(run_id, model["id"], attempt_id) / "quantization.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {key: str(value) for key, value in model["workflow"].get("env", {}).items()}
    )
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{utc_now()}] argv={json.dumps(argv)}\n")
        log.flush()
        timeout_seconds = float(model["resources"].get("timeout_hours", 24)) * 3600
        returncode = _run_logged(
            argv,
            log,
            cwd=model["workflow"].get("cwd"),
            env=env,
            timeout_seconds=timeout_seconds,
        )
    elapsed = time.monotonic() - started
    if returncode == 124:
        raise StageError(
            f"quantization exceeded {timeout_seconds / 3600:g} hours; see {log_path}",
            reason_code="QUANTIZATION_TIMEOUT",
        )
    if returncode != 0:
        raise StageError(
            f"quantization exited with {returncode}; see {log_path}",
            reason_code="QUANTIZATION_FAILED",
        )
    if not paths["output_dir"].is_dir():
        raise StageError(
            f"quantization succeeded but output is missing: {paths['output_dir']}",
            reason_code="OUTPUT_MISSING",
        )
    artifact_checksums = _artifact_file_records(paths["output_dir"])
    return {
        "status": "PASS",
        "argv": argv,
        "elapsed_seconds": round(elapsed, 3),
        "log_path": str(log_path),
        "output_dir": str(paths["output_dir"]),
        "work_dir": str(paths["work_dir"]),
        "artifact_files": artifact_checksums,
        "artifact_content_fingerprint": _artifact_content_fingerprint(
            artifact_checksums
        ),
    }


def _checkpoint_shards(output: Path) -> tuple[dict[str, str] | None, list[Path]]:
    index_path = output / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise StageError("invalid or empty safetensors weight_map")
        names = sorted(set(weight_map.values()))
        shards = [output / name for name in names]
        return weight_map, shards
    shards = sorted(output.glob("*.safetensors"))
    return None, shards


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_provenance(source: Path) -> dict[str, Any]:
    files = []
    for name in ("config.json", "model.safetensors.index.json"):
        path = source / name
        if path.is_file():
            files.append(
                {
                    "path": name,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    index_path = source / "model.safetensors.index.json"
    shard_metadata = []
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for name in sorted(set(index.get("weight_map", {}).values())):
            shard = source / name
            if shard.is_file():
                shard_metadata.append(
                    {
                        "path": name,
                        "size_bytes": shard.stat().st_size,
                        "sha256": _sha256(shard),
                    }
                )
    return {"metadata_files": files, "shards": shard_metadata}


def _artifact_file_records(output: Path) -> list[dict[str, Any]]:
    """Return a deterministic checksum inventory for the complete artifact."""

    return [
        {
            "path": str(path.relative_to(output)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(path for path in output.rglob("*") if path.is_file())
    ]


def _artifact_content_fingerprint(files: list[dict[str, Any]]) -> str:
    serialized = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _is_path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def run_validate(model: dict[str, Any], run_id: str) -> dict[str, Any]:
    output = _paths(run_id, model["id"])["output_dir"]
    failures = []
    warnings = []
    validation = model.get("validation", {})
    required = validation.get("required_files", ["config.json"])
    for relative in required:
        if not (output / relative).is_file():
            failures.append(f"missing required output file: {relative}")

    streaming = bool(validation.get("streaming", False))
    if (output / "FINALIZED").exists() is False:
        if streaming:
            failures.append("streaming output is missing required FINALIZED marker")
        else:
            warnings.append("FINALIZED marker is absent; non-streaming output")

    output_config = None
    quantization_config = None
    config_path = output / "config.json"
    if config_path.is_file():
        try:
            output_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"invalid output config.json: {error}")
        if isinstance(output_config, dict):
            quantization_config = output_config.get("quantization_config")
            if quantization_config is None and isinstance(
                output_config.get("text_config"), dict
            ):
                quantization_config = output_config["text_config"].get(
                    "quantization_config"
                )
    if validation.get("require_quantization_config", True):
        if not isinstance(quantization_config, dict) or not quantization_config:
            failures.append("output config has no quantization_config")
        else:
            status = quantization_config.get("quantization_status")
            if status is not None and status != "compressed":
                failures.append(
                    f"quantization_status is {status!r}, expected 'compressed'"
                )
            groups = quantization_config.get("config_groups")
            if groups is not None and (not isinstance(groups, dict) or not groups):
                failures.append("quantization_config.config_groups is empty or invalid")

    try:
        weight_map, shards = _checkpoint_shards(output)
    except (OSError, ValueError, json.JSONDecodeError, StageError) as error:
        failures.append(str(error))
        weight_map, shards = None, []
    if not shards:
        failures.append("no safetensors shards found")

    missing_shards = [str(path.name) for path in shards if not path.is_file()]
    if missing_shards:
        failures.append(f"missing shards: {', '.join(missing_shards)}")

    tensor_count = 0
    non_finite = []
    actual_names = set()
    tensor_health = []
    for shard in shards:
        if not shard.is_file():
            continue
        try:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    actual_names.add(name)
                    tensor = handle.get_tensor(name)
                    tensor_count += 1
                    finite = (
                        not tensor.is_floating_point() or tensor.isfinite().all().item()
                    )
                    if not finite:
                        non_finite.append(name)
                    if tensor.is_floating_point() and (
                        name.endswith(".weight_scale")
                        or name.endswith(".weight_zero_point")
                    ):
                        values = tensor.float()
                        tensor_health.append(
                            {
                                "name": name,
                                "dtype": str(tensor.dtype),
                                "shape": list(tensor.shape),
                                "min": values.min().item() if values.numel() else None,
                                "max": values.max().item() if values.numel() else None,
                                "mean": (
                                    values.mean().item() if values.numel() else None
                                ),
                                "zero_fraction": (
                                    (values == 0).float().mean().item()
                                    if values.numel()
                                    else None
                                ),
                            }
                        )
        except Exception as error:
            failures.append(f"unable to inspect {shard.name}: {error}")
    if non_finite:
        failures.append(
            f"non-finite tensors: {', '.join(non_finite[:20])}"
            + (" ..." if len(non_finite) > 20 else "")
        )
    if weight_map is not None:
        indexed = set(weight_map)
        missing_tensors = sorted(indexed - actual_names)
        extra_tensors = sorted(actual_names - indexed)
        if missing_tensors:
            failures.append(f"index references {len(missing_tensors)} missing tensors")
        if extra_tensors:
            failures.append(f"shards contain {len(extra_tensors)} unindexed tensors")

    scale_health = [
        record for record in tensor_health if record["name"].endswith(".weight_scale")
    ]
    non_positive_scales = [
        record["name"]
        for record in scale_health
        if record["min"] is not None and record["min"] <= 0
    ]
    if non_positive_scales:
        failures.append(
            f"non-positive weight scales: {', '.join(non_positive_scales[:20])}"
            + (" ..." if len(non_positive_scales) > 20 else "")
        )

    quantization_auxiliary_count = sum(
        name.endswith(
            (
                ".weight_scale",
                ".weight_zero_point",
                ".weight_packed",
                ".weight_compressed",
            )
        )
        for name in actual_names
    )
    minimum_auxiliary = int(validation.get("min_quantization_auxiliary_tensors", 0))
    if quantization_auxiliary_count < minimum_auxiliary:
        failures.append(
            f"found {quantization_auxiliary_count} quantization auxiliary tensors; "
            f"expected at least {minimum_auxiliary}"
        )

    if failures:
        raise StageError("; ".join(failures), reason_code="VALIDATION_FAILED")
    checksums = _artifact_file_records(output)
    reports_dir = _paths(run_id, model["id"])["reports_dir"]
    atomic_write_json(
        reports_dir / "checksums.json",
        {"schema_version": 1, "algorithm": "sha256", "files": checksums},
    )
    (reports_dir / "checksums.sha256").write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in checksums),
        encoding="utf-8",
    )
    result = {
        "status": "WARN" if warnings else "PASS",
        "output_dir": str(output),
        "shard_count": len(shards),
        "tensor_count": tensor_count,
        "non_finite_tensor_count": len(non_finite),
        "quantization_auxiliary_tensor_count": quantization_auxiliary_count,
        "quantization_format": (
            quantization_config.get("format")
            if isinstance(quantization_config, dict)
            else None
        ),
        "validation": model.get("validation", {}),
        "quantization_tensor_health": tensor_health,
        "warnings": warnings,
        "checksummed_file_count": len(checksums),
        "artifact_files": checksums,
        "artifact_content_fingerprint": _artifact_content_fingerprint(checksums),
    }
    atomic_write_json(
        reports_dir / "validation.json",
        {"schema_version": 1, **result},
    )
    return result


def run_runtime_smoke(model: dict[str, Any], run_id: str) -> dict[str, Any]:
    paths = _paths(run_id, model["id"])
    runtime = model.get("runtime_smoke", {})
    if runtime.get("enabled", True) is False:
        raise StageError(
            "runtime smoke is required for enabled production jobs",
            reason_code="CONFIG_ERROR",
        )
    profile = model.get("validation", {}).get("profile", "causal_lm")
    if profile != "causal_lm":
        raise StageError(
            f"runtime smoke adapter for profile {profile!r} is not implemented",
            reason_code="CONFIG_ERROR",
        )
    configured_python = runtime.get("python")
    python = os.getenv("VLLM_PYTHON_ENV") or configured_python
    if not python:
        raise StageError(
            "runtime_smoke.python or VLLM_PYTHON_ENV is required",
            reason_code="CONFIG_ERROR",
        )
    runtime_revision = runtime.get("runtime_revision")
    if not runtime_revision:
        raise StageError(
            "runtime_smoke.runtime_revision is required",
            reason_code="CONFIG_ERROR",
        )
    prompts = runtime.get("prompts", ["The capital of France is"])
    result_path = paths["reports_dir"] / "runtime-smoke-output.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        str(python),
        "-m",
        "ci.model_quality.vllm_smoke",
        "--model",
        str(paths["output_dir"]),
        "--tensor-parallel-size",
        str(
            runtime.get("tensor_parallel_size")
            or model["resources"].get("runtime_gpu_count")
            or model["resources"]["gpu_count"]
        ),
        "--prompts-json",
        json.dumps(prompts),
        "--output",
        str(result_path),
    ]
    env = os.environ.copy()
    repository_root = str(Path(__file__).parents[2])
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        repository_root
        if not existing_pythonpath
        else repository_root + os.pathsep + existing_pythonpath
    )
    completed = subprocess.run(
        argv, env=env, text=True, capture_output=True, check=False
    )
    log = paths["logs_dir"] / "runtime-smoke.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise StageError(
            f"vLLM runtime smoke failed with {completed.returncode}; see {log}",
            reason_code="RUNTIME_SMOKE_FAILED",
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "status": "PASS",
        "result": payload,
        "runtime_revision": runtime_revision,
        "log_path": str(log),
    }


def run_evaluate(
    model: dict[str, Any],
    run_id: str,
    *,
    artifact_fingerprint: str | None = None,
    evaluation_fingerprint: str | None = None,
    attempt_id: str = "default",
) -> dict[str, Any]:
    evaluation = model.get("evaluation", {})
    command = evaluation.get("command")
    if not command:
        raise StageError(
            "evaluation was requested but evaluation.command is not configured",
            reason_code="CONFIG_ERROR",
        )
    result_file = evaluation.get("result_file")
    if not isinstance(result_file, str) or not result_file:
        raise StageError(
            "evaluation.result_file is required for an auditable evaluation",
            reason_code="CONFIG_ERROR",
        )
    if not artifact_fingerprint or not evaluation_fingerprint:
        raise StageError(
            "artifact and evaluation fingerprints are required",
            reason_code="CONFIG_ERROR",
        )
    paths = _paths(run_id, model["id"])
    attempt_reports = attempt_reports_dir(run_id, model["id"], attempt_id)
    attempt_reports.mkdir(parents=True, exist_ok=True)
    values = _command_values(model, run_id, reports_dir=attempt_reports)
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(value, str) and value for value in command)
    ):
        raise StageError(
            "evaluation.command must be a non-empty argv list",
            reason_code="CONFIG_ERROR",
        )
    try:
        argv = resolve_argv(command, values)
    except PlaceholderError as error:
        raise StageError(
            str(error),
            reason_code="CONFIG_ERROR",
        ) from error
    log_path = _attempt_logs_dir(run_id, model["id"], attempt_id) / "evaluation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    normalized_path = attempt_reports / "evaluation.json"
    cache_path = evaluation_cache_path(run_id, model["id"], evaluation_fingerprint)
    if cache_path.is_file():
        prior = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            prior.get("artifact_fingerprint") == artifact_fingerprint
            and prior.get("evaluation_fingerprint") == evaluation_fingerprint
            and prior.get("status") in {"PASS", "FAIL"}
        ):
            atomic_write_json(normalized_path, prior)
            return {
                "status": prior["status"],
                "reason_code": "EVALUATION_CACHE_HIT",
                "result": prior,
                "log_path": str(log_path),
                "attempt_id": attempt_id,
            }
    result_path = Path(resolve_argument(result_file, values))
    if result_path.exists():
        if result_path.is_dir():
            raise StageError(
                f"evaluation result path is a directory: {result_path}",
                reason_code="CONFIG_ERROR",
            )
        result_path.unlink()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{utc_now()}] argv={json.dumps(argv)}\n")
        log.flush()
        returncode = _run_logged(argv, log, env=os.environ.copy())
    if returncode != 0:
        if result_path.is_file():
            raw_result = json.loads(result_path.read_text(encoding="utf-8"))
            if raw_result.get("status") == "FAIL":
                raw_result.update(
                    {
                        "artifact_fingerprint": artifact_fingerprint,
                        "evaluation_fingerprint": evaluation_fingerprint,
                        "attempt_id": attempt_id,
                    }
                )
                atomic_write_json(normalized_path, raw_result)
                atomic_write_json(cache_path, raw_result)
                raise StageError(
                    f"evaluation quality gate failed; see {normalized_path}",
                    reason_code="QUALITY_GATE_FAILED",
                )
        raise StageError(
            f"evaluation exited with {returncode}; see {log_path}",
            reason_code="EVALUATION_FAILED",
        )
    if not result_path.is_file():
        raise StageError(
            f"evaluation result is missing: {result_path}",
            reason_code="EVALUATION_RESULT_MISSING",
        )
    raw_result = json.loads(result_path.read_text(encoding="utf-8"))
    try:
        result = compare_evaluation(raw_result)
    except ConfigError as error:
        raise StageError(str(error), reason_code="CONFIG_ERROR") from error
    result.update(
        {
            "artifact_fingerprint": artifact_fingerprint,
            "evaluation_fingerprint": evaluation_fingerprint,
            "attempt_id": attempt_id,
        }
    )
    atomic_write_json(normalized_path, result)
    atomic_write_json(cache_path, result)
    if result["status"] == "FAIL":
        raise StageError(
            f"evaluation quality gate failed; see {normalized_path}",
            reason_code="QUALITY_GATE_FAILED",
        )
    return {"status": "PASS", "argv": argv, "result": result, "log_path": str(log_path)}


def run_report(
    model: dict[str, Any],
    run_id: str,
    *,
    run_mode: str = "quantize",
    artifact_fingerprint: str | None = None,
    evaluation_fingerprint: str | None = None,
    attempt_id: str = "default",
) -> dict[str, Any]:
    paths = _paths(run_id, model["id"])
    attempt_reports = attempt_reports_dir(run_id, model["id"], attempt_id)
    attempt_reports.mkdir(parents=True, exist_ok=True)
    state_dir = attempt_state_dir(run_id, model["id"], attempt_id)
    states = {}
    stale_stages = []
    for path in sorted(state_dir.glob("*.json")) if state_dir.exists() else []:
        candidate = json.loads(path.read_text(encoding="utf-8"))
        if candidate.get("attempt_id", "default") != attempt_id:
            stale_stages.append(path.stem)
            continue
        if (
            artifact_fingerprint is not None
            and candidate.get("artifact_fingerprint") != artifact_fingerprint
        ):
            stale_stages.append(path.stem)
            continue
        states[path.stem] = candidate
    preflight_state = states.get("preflight")
    validation_state = states.get("validate")
    if preflight_state is not None:
        atomic_write_json(
            attempt_reports / "input-provenance.json",
            {
                "schema_version": 1,
                "source_path": preflight_state.get("source_path"),
                "source_provenance": preflight_state.get("source_provenance"),
                "resolved_argv": preflight_state.get("resolved_argv"),
                "artifact_fingerprint": artifact_fingerprint,
                "evaluation_fingerprint": evaluation_fingerprint,
                "attempt_id": attempt_id,
            },
        )
    if validation_state is not None:
        atomic_write_json(
            attempt_reports / "validation.json",
            {"schema_version": 1, **validation_state},
        )
    required_by_mode = {
        "quantize": ["preflight", "quantize", "validate", "runtime-smoke"],
        "quantize_and_eval": [
            "preflight",
            "quantize",
            "validate",
            "runtime-smoke",
            "evaluate",
        ],
        "eval_only": ["preflight", "validate", "runtime-smoke", "evaluate"],
        "upload_only": ["preflight", "validate", "runtime-smoke"],
    }
    try:
        required = required_by_mode[run_mode]
    except KeyError as error:
        raise StageError(
            f"unknown run mode: {run_mode}", reason_code="CONFIG_ERROR"
        ) from error

    evaluation_path = paths["reports_dir"] / "evaluation.json"
    attempt_evaluation_path = attempt_reports / "evaluation.json"
    overall_evaluation = None
    if run_mode == "quantize":
        evaluation = {
            "schema_version": 1,
            "status": "SKIPPED",
            "reason_code": "EVALUATION_DISABLED",
            "requested_suite": model.get("evaluation", {}).get("profile"),
            "artifact_fingerprint": artifact_fingerprint,
            "evaluation_fingerprint": evaluation_fingerprint,
            "attempt_id": attempt_id,
        }
        atomic_write_json(attempt_evaluation_path, evaluation)
        atomic_write_json(evaluation_path, evaluation)
    elif run_mode == "upload_only":
        if not evaluation_path.is_file():
            overall_evaluation = "MISSING_PRIOR_RESULT"
        else:
            prior_evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            if (
                artifact_fingerprint is not None
                and prior_evaluation.get("artifact_fingerprint") != artifact_fingerprint
            ):
                overall_evaluation = "STALE_PRIOR_RESULT"
            else:
                overall_evaluation = "PRIOR_RESULT_PRESENT"
    else:
        overall_evaluation = None

    if run_mode in {"quantize_and_eval", "eval_only"}:
        if not attempt_evaluation_path.is_file():
            states.pop("evaluate", None)
        else:
            evaluation_result = json.loads(
                attempt_evaluation_path.read_text(encoding="utf-8")
            )
            matches = (
                evaluation_result.get("artifact_fingerprint") == artifact_fingerprint
                and evaluation_result.get("evaluation_fingerprint")
                == evaluation_fingerprint
            )
            if not matches:
                states.pop("evaluate", None)
                overall_evaluation = "STALE_PRIOR_RESULT"
    overall = "PASS"
    warn_stages = []
    for name in required:
        status = states.get(name, {}).get("status", "DEPENDENCY_FAILED")
        if status == "FAIL" or status == "DEPENDENCY_FAILED":
            overall = "FAIL"
            break
        if status == "WARN":
            warn_stages.append(name)

    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "model_id": model["id"],
        "run_mode": run_mode,
        "attempt_id": attempt_id,
        "artifact_fingerprint": artifact_fingerprint,
        "evaluation_fingerprint": evaluation_fingerprint,
        "evaluation_history": overall_evaluation,
        "quantization": "REUSED"
        if run_mode in {"eval_only", "upload_only"}
        else "CREATED",
        "status": overall,
        "warnings": [f"{name} completed with warnings" for name in warn_stages],
        "stale_stages_ignored": sorted(set(stale_stages)),
        "stages": {name: value.get("status") for name, value in states.items()},
    }
    atomic_write_json(attempt_reports / "summary.json", summary)
    quantize_state = states.get("quantize")
    atomic_write_json(
        attempt_reports / "quantization.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "model_id": model["id"],
            "attempt_id": attempt_id,
            "artifact_fingerprint": artifact_fingerprint,
            "status": (quantize_state.get("status") if quantize_state else "REUSED"),
            "elapsed_seconds": (
                quantize_state.get("elapsed_seconds") if quantize_state else None
            ),
            "log_path": quantize_state.get("log_path") if quantize_state else None,
        },
    )
    environment = {
        "schema_version": 1,
        "python": sys.version,
        "platform": platform.platform(),
        "git_sha": os.getenv("BUILDKITE_COMMIT"),
    }
    try:
        import torch

        environment["torch"] = torch.__version__
        environment["accelerator_device_count"] = torch.accelerator.device_count()
    except Exception as error:  # pragma: no cover - environment dependent
        environment["torch_probe_error"] = str(error)
    atomic_write_json(attempt_reports / "environment.json", environment)
    atomic_write_json(
        attempt_reports / "modifier-diagnostics.json",
        {
            "schema_version": 1,
            "status": "NOT_APPLICABLE",
            "reason_code": "NO_DIAGNOSTIC_ADAPTER_CONFIGURED",
        },
    )
    rows = "\n".join(
        f"| {name} | {states.get(name, {}).get('status', 'DEPENDENCY_FAILED')} |"
        for name in required
    )
    markdown = (
        f"# Model quality report: {model['id']}\n\n"
        f"Run: `{run_id}`  \nOverall: **{overall}**\n\n"
        "| Stage | Status |\n|---|---|\n" + rows + "\n"
    )
    (attempt_reports / "summary.md").write_text(markdown, encoding="utf-8")
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    for report in attempt_reports.iterdir():
        if not report.is_file():
            continue
        destination = paths["reports_dir"] / report.name
        if report.suffix == ".json":
            atomic_write_json(
                destination, json.loads(report.read_text(encoding="utf-8"))
            )
        else:
            temporary = destination.with_name(
                f".{destination.name}.{uuid.uuid4().hex}.tmp"
            )
            temporary.write_bytes(report.read_bytes())
            os.replace(temporary, destination)
    atomic_write_json(
        paths["run_dir"] / "current-attempt.json",
        {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "artifact_fingerprint": artifact_fingerprint,
            "evaluation_fingerprint": evaluation_fingerprint,
            "promoted_at": utc_now(),
        },
    )
    return {"status": overall, "summary": summary}


def _upload_values(
    model: dict[str, Any], run_id: str, *, reports_dir: Path | None = None
) -> dict[str, str]:
    paths = _paths(run_id, model["id"])
    remote_prefix = str(model.get("upload", {}).get("remote_prefix", "")).rstrip("/")
    return {
        "model_id": model["id"],
        "run_id": run_id,
        "run_dir": str(paths["run_dir"]),
        "output_dir": str(paths["output_dir"]),
        "reports_dir": str(reports_dir or paths["reports_dir"]),
        "remote_prefix": remote_prefix,
        "remote_run_prefix": f"{remote_prefix}/{model['id']}/runs/{run_id}",
    }


def _resolve_upload_commands(
    commands: Any, values: dict[str, str], field: str
) -> list[list[str]]:
    if not isinstance(commands, list) or not commands:
        raise StageError(
            f"{field} must be a non-empty list", reason_code="CONFIG_ERROR"
        )
    resolved = []
    for index, command in enumerate(commands):
        if not isinstance(command, list) or not command:
            raise StageError(
                f"{field}[{index}] must be a non-empty argv list",
                reason_code="CONFIG_ERROR",
            )
        if not all(isinstance(value, str) and value for value in command):
            raise StageError(
                f"{field}[{index}] entries must be strings",
                reason_code="CONFIG_ERROR",
            )
        try:
            resolved.append(resolve_argv(command, values))
        except PlaceholderError as error:
            raise StageError(
                str(error),
                reason_code="CONFIG_ERROR",
            ) from error
    return resolved


def run_publish(
    model: dict[str, Any],
    run_id: str,
    *,
    explicitly_requested: bool = False,
    artifact_fingerprint: str | None = None,
    evaluation_fingerprint: str | None = None,
) -> dict[str, Any]:
    upload = model.get("upload", {})
    if not upload.get("enabled", False):
        if explicitly_requested:
            raise StageError(
                "upload_only was requested but upload.enabled is false",
                reason_code="CONFIG_ERROR",
            )
        return {"status": "SKIPPED", "reason_code": "UPLOAD_DISABLED"}

    paths = _paths(run_id, model["id"])
    current_attempt_path = paths["run_dir"] / "current-attempt.json"
    if not current_attempt_path.is_file():
        raise StageError(
            "current-attempt.json is missing", reason_code="DEPENDENCY_FAILED"
        )
    current_attempt = json.loads(current_attempt_path.read_text(encoding="utf-8"))
    promoted_attempt_id = current_attempt.get("attempt_id")
    promoted_reports = (
        attempt_reports_dir(run_id, model["id"], promoted_attempt_id)
        if isinstance(promoted_attempt_id, str) and promoted_attempt_id
        else paths["reports_dir"]
    )
    if not promoted_reports.is_dir():
        promoted_reports = paths["reports_dir"]
    summary_path = promoted_reports / "summary.json"
    if not summary_path.is_file():
        raise StageError("model report is missing", reason_code="DEPENDENCY_FAILED")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") not in {"PASS", "WARN"}:
        raise StageError(
            "model did not pass required validation", reason_code="DEPENDENCY_FAILED"
        )
    if (
        summary.get("attempt_id") != current_attempt.get("attempt_id")
        or current_attempt.get("artifact_fingerprint") != artifact_fingerprint
    ):
        raise StageError(
            "shared report does not match the promoted current attempt",
            reason_code="DEPENDENCY_FAILED",
        )
    if (
        artifact_fingerprint is not None
        and summary.get("artifact_fingerprint") != artifact_fingerprint
    ):
        raise StageError(
            "model report has a stale artifact fingerprint",
            reason_code="DEPENDENCY_FAILED",
        )
    evaluation_path = promoted_reports / "evaluation.json"
    if evaluation_path.is_file():
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation.get("artifact_fingerprint") != artifact_fingerprint:
            raise StageError(
                "evaluation result has a stale artifact fingerprint",
                reason_code="DEPENDENCY_FAILED",
            )
        if (
            evaluation.get("status") in {"PASS", "FAIL"}
            and evaluation_fingerprint is not None
            and evaluation.get("evaluation_fingerprint") != evaluation_fingerprint
        ):
            raise StageError(
                "evaluation result has a stale evaluation fingerprint",
                reason_code="DEPENDENCY_FAILED",
            )
    if not paths["output_dir"].is_dir() or not promoted_reports.is_dir():
        raise StageError("publish inputs are missing", reason_code="DEPENDENCY_FAILED")
    artifact_manifest_path = paths["run_dir"] / "artifact-manifest.json"
    if not artifact_manifest_path.is_file():
        raise StageError(
            "artifact-manifest.json is missing", reason_code="DEPENDENCY_FAILED"
        )
    artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    current_artifact_files = _artifact_file_records(paths["output_dir"])
    current_content_fingerprint = _artifact_content_fingerprint(current_artifact_files)
    if artifact_manifest.get("artifact_fingerprint") != artifact_fingerprint:
        raise StageError(
            "artifact manifest has a stale artifact fingerprint",
            reason_code="DEPENDENCY_FAILED",
        )
    source = Path(model["source"]["path"]).expanduser()
    if (
        source.is_dir()
        and artifact_manifest.get("source_provenance") is not None
        and artifact_manifest.get("source_provenance") != _source_provenance(source)
    ):
        raise StageError(
            "artifact source provenance changed before publish",
            reason_code="DEPENDENCY_FAILED",
        )
    if (
        artifact_manifest.get("artifact_content_fingerprint")
        != current_content_fingerprint
    ):
        raise StageError(
            "quantized artifact contents changed since validation",
            reason_code="DEPENDENCY_FAILED",
        )

    values = _upload_values(model, run_id, reports_dir=promoted_reports)
    remote_prefix = values["remote_prefix"]
    if (
        not remote_prefix
        or remote_prefix in {"/", "bos:/"}
        or "replace-me" in remote_prefix
    ):
        raise StageError("unsafe upload remote_prefix", reason_code="CONFIG_ERROR")
    commands = _resolve_upload_commands(
        upload.get("commands"), values, "upload.commands"
    )
    verify_commands = _resolve_upload_commands(
        upload.get("verify_commands"), values, "upload.verify_commands"
    )
    if any(Path(command[0]).name != "bcecmd" for command in commands + verify_commands):
        raise StageError(
            "upload and verification commands must invoke bcecmd",
            reason_code="CONFIG_ERROR",
        )
    allowlist = upload.get("allowlist")
    if (
        not isinstance(allowlist, list)
        or not allowlist
        or not all(isinstance(value, str) and value for value in allowlist)
    ):
        raise StageError(
            "upload.allowlist must be a non-empty list",
            reason_code="CONFIG_ERROR",
        )
    run_root = paths["run_dir"].resolve()
    output_root = paths["output_dir"].resolve()
    reports_root = promoted_reports.resolve()
    allowed_roots = {
        "model": output_root,
        "reports": reports_root,
    }
    allowed_names = set(allowlist)
    for command in commands:
        # The first bare word is the bcecmd operation. All other positional
        # arguments are explicit absolute local paths or BOS destinations.
        for index, token in enumerate(command[1:]):
            if token.startswith("bos:/") or token.startswith("-"):
                continue
            if index == 0 and token in {"bos", "upload", "cp", "sync"}:
                continue
            if index == 1 and command[1] == "bos" and token in {"cp", "sync"}:
                continue
            candidate = Path(token).expanduser()
            if not candidate.is_absolute():
                raise StageError(
                    f"upload command uses an uncontrolled relative path: {token}",
                    reason_code="CONFIG_ERROR",
                )
            resolved = candidate.resolve()
            if not any(
                name in allowed_names and _is_path_within(resolved, allowed_root)
                for name, allowed_root in allowed_roots.items()
            ):
                raise StageError(
                    f"upload command references a path outside the allowlist: {token}",
                    reason_code="CONFIG_ERROR",
                )

    upload_manifest = []
    generated_publication_files = {
        "reports/upload-manifest.json",
        "reports/SUCCESS.json",
        "SUCCESS.json",
    }
    for relative in allowlist:
        if relative == "reports":
            candidate = promoted_reports.resolve()
        elif relative.startswith("reports/"):
            candidate = (promoted_reports / relative.removeprefix("reports/")).resolve()
        else:
            candidate = (paths["run_dir"] / relative).resolve()
        try:
            candidate.relative_to(paths["run_dir"].resolve())
        except ValueError as error:
            raise StageError(
                f"upload allowlist escapes run directory: {relative}",
                reason_code="CONFIG_ERROR",
            ) from error
        if not candidate.exists():
            raise StageError(
                f"upload allowlist path is missing: {relative}",
                reason_code="DEPENDENCY_FAILED",
            )
        if candidate.is_file():
            if relative in generated_publication_files:
                continue
            upload_manifest.append(
                {
                    "path": relative,
                    "size_bytes": candidate.stat().st_size,
                    "sha256": _sha256(candidate),
                }
            )
        else:
            for file in sorted(path for path in candidate.rglob("*") if path.is_file()):
                resolved_file = file.resolve()
                if _is_path_within(resolved_file, reports_root):
                    relative_file = "reports/" + str(
                        resolved_file.relative_to(reports_root)
                    )
                else:
                    relative_file = str(resolved_file.relative_to(run_root))
                if relative_file in generated_publication_files:
                    continue
                upload_manifest.append(
                    {
                        "path": relative_file,
                        "size_bytes": file.stat().st_size,
                        "sha256": _sha256(file),
                    }
                )
    upload_manifest_path = promoted_reports / "upload-manifest.json"
    upload_manifest_document = {
        "schema_version": 1,
        "artifact_fingerprint": artifact_fingerprint,
        "artifact_content_fingerprint": current_content_fingerprint,
        "evaluation_fingerprint": evaluation_fingerprint,
        "files": upload_manifest,
    }
    if upload_manifest_path.is_file():
        prior_upload_manifest = json.loads(
            upload_manifest_path.read_text(encoding="utf-8")
        )
        if prior_upload_manifest != upload_manifest_document:
            raise StageError(
                "publish retry does not match the frozen upload manifest",
                reason_code="DEPENDENCY_FAILED",
            )
    else:
        atomic_write_json(upload_manifest_path, upload_manifest_document)
    success_path = paths["run_dir"] / "commit-markers" / "SUCCESS.json"
    success_commands = _resolve_upload_commands(
        upload.get("success_commands"), values, "upload.success_commands"
    )
    if any(Path(command[0]).name != "bcecmd" for command in success_commands):
        raise StageError(
            "success commands must invoke bcecmd", reason_code="CONFIG_ERROR"
        )

    log_path = paths["logs_dir"] / "publish.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    executed = []
    with log_path.open("a", encoding="utf-8") as log:
        for kind, sequence in (("upload", commands), ("verify", verify_commands)):
            for command in sequence:
                log.write(f"[{utc_now()}] {kind} argv={json.dumps(command)}\n")
                log.flush()
                completed = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                executed.append({"kind": kind, "argv": command})
                if completed.returncode != 0:
                    message = (
                        f"{kind} command failed with {completed.returncode}; "
                        f"see {log_path}"
                    )
                    raise StageError(
                        message,
                        reason_code=(
                            "REMOTE_VERIFY_FAILED"
                            if kind == "verify"
                            else "UPLOAD_FAILED"
                        ),
                    )
        atomic_write_json(
            success_path,
            {
                "schema_version": 1,
                "run_id": run_id,
                "model_id": model["id"],
                "artifact_fingerprint": artifact_fingerprint,
                "artifact_content_fingerprint": current_content_fingerprint,
                "evaluation_fingerprint": evaluation_fingerprint,
                "file_count": len(upload_manifest),
                "files": upload_manifest,
                "created_at": utc_now(),
            },
        )
        for command in success_commands:
            log.write(f"[{utc_now()}] success argv={json.dumps(command)}\n")
            log.flush()
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            executed.append({"kind": "success", "argv": command})
            if completed.returncode != 0:
                message = (
                    f"success command failed with {completed.returncode}; "
                    f"see {log_path}"
                )
                raise StageError(
                    message,
                    reason_code="UPLOAD_FAILED",
                )

    return {
        "status": "PASS",
        "remote_run_prefix": values["remote_run_prefix"],
        "executed": executed,
        "log_path": str(log_path),
        "uploaded_file_count": len(upload_manifest),
        "success_manifest": str(success_path),
    }


def run_aggregate(
    config: dict[str, Any], run_id: str, *, expected_models: list[str]
) -> dict[str, Any]:
    root = model_run_dir(run_id, "aggregate").parent
    rows = []
    known = {model["id"] for model in config["models"]}
    unknown = set(expected_models) - known
    if unknown:
        raise ConfigError(f"unknown aggregate models: {', '.join(sorted(unknown))}")
    for model_id in expected_models:
        model_root = root / model_id
        current_attempt_path = model_root / "current-attempt.json"
        row = {
            "schema_version": 1,
            "run_id": run_id,
            "model_id": model_id,
            "status": "FAIL",
            "reason_code": "REPORT_MISSING",
        }
        try:
            current = json.loads(current_attempt_path.read_text(encoding="utf-8"))
            attempt_id = current["attempt_id"]
            report_path = attempt_reports_dir(run_id, model_id, attempt_id) / "summary.json"
            summary = json.loads(report_path.read_text(encoding="utf-8"))
            identity = ("attempt_id", "artifact_fingerprint", "evaluation_fingerprint")
            if (
                any(summary.get(key) != current.get(key) for key in identity)
                or summary.get("run_id") != run_id
                or summary.get("model_id") != model_id
            ):
                row["reason_code"] = "STALE_PROMOTED_REPORT"
            else:
                row = summary
                publish_path = attempt_state_dir(run_id, model_id, attempt_id) / "publish.json"
                if publish_path.is_file():
                    publish = json.loads(publish_path.read_text(encoding="utf-8"))
                    if any(publish.get(key) != current.get(key) for key in identity):
                        row.update(status="FAIL", reason_code="STALE_PUBLISH_STATE")
                    else:
                        row["publish_status"] = publish.get("status")
                        if publish.get("status") != "PASS":
                            row["status"] = "FAIL"
        except (OSError, ValueError, KeyError, TypeError):
            row.update(status="FAIL", reason_code="REPORT_MISSING")
        rows.append(row)
    execution_plan_path = root / "execution-plan.json"
    execution_plan = (
        json.loads(execution_plan_path.read_text(encoding="utf-8"))
        if execution_plan_path.is_file()
        else None
    )
    aggregate = {
        "schema_version": 1,
        "run_id": run_id,
        "models": rows,
        "deferred": execution_plan.get("deferred", []) if execution_plan else [],
        "budget": execution_plan.get("budget") if execution_plan else None,
        "status": ("FAIL" if any(row["status"] == "FAIL" for row in rows) else "PASS"),
    }
    atomic_write_json(root / "aggregate-report.json", aggregate)
    return aggregate
