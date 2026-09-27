"""Configuration loading and validation for model quality CI."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

PRIORITIES = ("P0", "P1", "P2", "P3")
RUN_MODES = ("quantize", "quantize_and_eval", "eval_only", "upload_only")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ConfigError(ValueError):
    """Raised when the model quality manifest is invalid."""


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    value = mapping.get(key)
    if value is None or value == "":
        raise ConfigError(f"{context}.{key} is required")
    return value


def _positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f"{field} must be a positive number")
    return float(value)


def _validate_argv(value: Any, field: str) -> None:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{field} must be a non-empty argv list")
    if not all(isinstance(item, str) and item for item in value):
        raise ConfigError(f"{field} entries must be strings")


def _validate_model(model: Any, index: int) -> dict[str, Any]:
    context = f"models[{index}]"
    if not isinstance(model, dict):
        raise ConfigError(f"{context} must be a mapping")

    model_id = _require(model, "id", context)
    if not isinstance(model_id, str):
        raise ConfigError(f"{context}.id must be a string")
    if not _SAFE_ID.fullmatch(model_id):
        raise ConfigError(
            f"{context}.id must contain only letters, numbers, dot, underscore, or dash"
        )

    priority = model.get("priority", "P2")
    if priority not in PRIORITIES:
        raise ConfigError(f"{context}.priority must be one of {PRIORITIES}")

    source = _require(model, "source", context)
    if not isinstance(source, dict):
        raise ConfigError(f"{context}.source must be a mapping")
    _require(source, "path", f"{context}.source")

    resources = _require(model, "resources", context)
    if not isinstance(resources, dict):
        raise ConfigError(f"{context}.resources must be a mapping")
    gpu_count = _positive_number(
        _require(resources, "gpu_count", f"{context}.resources"),
        f"{context}.resources.gpu_count",
    )
    if not gpu_count.is_integer():
        raise ConfigError(f"{context}.resources.gpu_count must be an integer")
    _positive_number(
        _require(resources, "estimated_gpu_hours", f"{context}.resources"),
        f"{context}.resources.estimated_gpu_hours",
    )

    workflow = _require(model, "workflow", context)
    if not isinstance(workflow, dict):
        raise ConfigError(f"{context}.workflow must be a mapping")
    quantize = _require(workflow, "quantize", f"{context}.workflow")
    _validate_argv(quantize, f"{context}.workflow.quantize")
    _require(workflow, "revision", f"{context}.workflow")

    validation = model.get("validation", {})
    if not isinstance(validation, dict):
        raise ConfigError(f"{context}.validation must be a mapping")
    profile = validation.get("profile", "causal_lm")
    if not isinstance(profile, str) or not profile:
        raise ConfigError(f"{context}.validation.profile must be a string")
    for suffixes_key in (
        "quantization_auxiliary_suffixes",
        "quantization_scale_suffixes",
    ):
        suffixes = validation.get(suffixes_key)
        if suffixes is None:
            continue
        if not isinstance(suffixes, list) or not all(
            isinstance(value, str) and value for value in suffixes
        ):
            raise ConfigError(
                f"{context}.validation.{suffixes_key} must be a list of "
                "non-empty strings"
            )

    runtime_smoke = model.get("runtime_smoke", {})
    if not isinstance(runtime_smoke, dict):
        raise ConfigError(f"{context}.runtime_smoke must be a mapping")
    if runtime_smoke.get("enabled", True) and profile != "causal_lm":
        raise ConfigError(
            f"{context}: runtime smoke adapter for profile {profile!r} "
            "is not implemented"
        )
    if runtime_smoke and runtime_smoke.get("enabled", True):
        _require(runtime_smoke, "runtime_revision", f"{context}.runtime_smoke")

    evaluation = model.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ConfigError(f"{context}.evaluation must be a mapping")
    if evaluation.get("command") is not None:
        _validate_argv(evaluation["command"], f"{context}.evaluation.command")
        _require(evaluation, "result_file", f"{context}.evaluation")
        _require(evaluation, "runtime_revision", f"{context}.evaluation")

    upload = model.get("upload", {})
    if not isinstance(upload, dict):
        raise ConfigError(f"{context}.upload must be a mapping")
    if upload.get("enabled", False):
        _require(upload, "remote_prefix", f"{context}.upload")
        allowlist = _require(upload, "allowlist", f"{context}.upload")
        if not isinstance(allowlist, list) or not allowlist:
            raise ConfigError(f"{context}.upload.allowlist must be non-empty")
        for field in ("commands", "verify_commands", "success_commands"):
            commands = _require(upload, field, f"{context}.upload")
            if not isinstance(commands, list) or not commands:
                raise ConfigError(f"{context}.upload.{field} must be non-empty")
            for command_index, command in enumerate(commands):
                _validate_argv(command, f"{context}.upload.{field}[{command_index}]")
        forbidden_upload_placeholders = ("{run_dir}", "{source_path}", "{work_dir}")
        for command in upload["commands"]:
            if any(
                placeholder in argument
                for argument in command
                for placeholder in forbidden_upload_placeholders
            ):
                raise ConfigError(
                    f"{context}.upload.commands may only upload allowlisted "
                    "model/report paths"
                )
            if not any(
                placeholder in argument
                for argument in command
                for placeholder in ("{output_dir}", "{reports_dir}")
            ):
                raise ConfigError(
                    f"{context}.upload.commands must use output_dir or reports_dir"
                )
            required_allowlist = {
                local_path
                for placeholder, local_path in (
                    ("{output_dir}", "model"),
                    ("{reports_dir}", "reports"),
                )
                if any(placeholder in argument for argument in command)
            }
            missing_allowlist = required_allowlist - set(allowlist)
            if missing_allowlist:
                raise ConfigError(
                    f"{context}.upload.commands references paths absent from "
                    f"allowlist: {', '.join(sorted(missing_allowlist))}"
                )

    normalized = dict(model)
    normalized.pop("fingerprint", None)
    normalized["enabled"] = bool(model.get("enabled", True))
    normalized["priority"] = priority
    normalized["max_result_age_hours"] = int(model.get("max_result_age_hours", 168))
    normalized["resources"] = dict(resources)
    normalized["resources"]["gpu_count"] = int(gpu_count)
    normalized["resources"]["estimated_gpu_hours"] = float(
        resources["estimated_gpu_hours"]
    )
    if resources.get("estimated_eval_gpu_hours") is not None:
        normalized["resources"]["estimated_eval_gpu_hours"] = _positive_number(
            resources["estimated_eval_gpu_hours"],
            f"{context}.resources.estimated_eval_gpu_hours",
        )
    for field in ("runtime_gpu_count", "evaluation_gpu_count"):
        if resources.get(field) is not None:
            value = _positive_number(resources[field], f"{context}.resources.{field}")
            if not value.is_integer():
                raise ConfigError(f"{context}.resources.{field} must be an integer")
            normalized["resources"][field] = int(value)
    return normalized


def _read_manifest(config_path: Path, stack: tuple[Path, ...]) -> list[Any]:
    resolved = config_path.resolve()
    if resolved in stack:
        chain = " -> ".join(str(path) for path in (*stack, resolved))
        raise ConfigError(f"recursive manifest include: {chain}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"Unable to read {config_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in {config_path}: {error}") from error

    if not isinstance(raw, dict):
        raise ConfigError("manifest must be a mapping")
    if raw.get("schema_version") != 1:
        raise ConfigError(f"{config_path}: schema_version must be 1")
    models = raw.get("models", [])
    if not isinstance(models, list):
        raise ConfigError(f"{config_path}: models must be a list")
    includes = raw.get("includes", [])
    if not isinstance(includes, list) or not all(
        isinstance(value, str) and value for value in includes
    ):
        raise ConfigError(f"{config_path}: includes must be a list of paths")
    combined = list(models)
    for include in includes:
        combined.extend(
            _read_manifest(config_path.parent / include, (*stack, resolved))
        )
    return combined


def load_model_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a versioned model quality manifest and includes."""

    models = _read_manifest(Path(path), ())

    normalized_models = [_validate_model(model, i) for i, model in enumerate(models)]
    ids = [model["id"] for model in normalized_models]
    duplicates = sorted({model_id for model_id in ids if ids.count(model_id) > 1})
    if duplicates:
        raise ConfigError(f"duplicate model ids: {', '.join(duplicates)}")

    return {"schema_version": 1, "models": normalized_models}


def model_execution_identity(model: dict[str, Any]) -> dict[str, Any]:
    """Return fields that define the compressed artifact identity."""

    normalized = dict(model)
    resources = dict(model["resources"])
    for field in ("estimated_gpu_hours", "estimated_eval_gpu_hours"):
        resources.pop(field, None)
    normalized["resources"] = resources
    normalized.pop("priority", None)
    normalized.pop("max_result_age_hours", None)
    normalized.pop("upload", None)
    normalized.pop("business_tier", None)
    normalized.pop("evaluation", None)
    normalized.pop("runtime_smoke", None)
    return normalized


def _fingerprint(identity: dict[str, Any]) -> str:
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def artifact_fingerprint(model: dict[str, Any]) -> str:
    """Return a stable identity for inputs that define the compressed artifact."""

    execution = model_execution_identity(model)
    return _fingerprint(
        {
            "id": execution["id"],
            "source": execution["source"],
            "workflow": execution["workflow"],
            "resources": execution["resources"],
            "validation": execution.get("validation"),
        }
    )


def artifact_request_fingerprint(model: dict[str, Any]) -> str:
    """Return artifact identity before probing mutable local checkpoint files."""

    return artifact_fingerprint(model)


def finalized_artifact_fingerprint(
    request_fingerprint: str, source_provenance: dict[str, Any]
) -> str:
    """Bind artifact identity to the source checkpoint observed by preflight."""

    return _fingerprint(
        {
            "request_fingerprint": request_fingerprint,
            "source_provenance": source_provenance,
        }
    )


def model_fingerprint(model: dict[str, Any], git_sha: str | None = None) -> str:
    """Backward-compatible alias for artifact identity.

    ``git_sha`` is intentionally excluded: evaluator/report code changes must not
    invalidate an existing compressed artifact. Quantization implementation identity
    belongs in an explicit workflow revision or pinned executable/container field.
    """

    return artifact_fingerprint(model)


def evaluation_fingerprint(model: dict[str, Any], artifact_fingerprint: str) -> str:
    """Return the cache identity for evaluation of an existing artifact."""

    return _fingerprint(
        {
            "artifact_fingerprint": artifact_fingerprint,
            "evaluation": model.get("evaluation", {}),
            "validation_profile": model.get("validation", {}).get("profile"),
        }
    )
