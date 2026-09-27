"""Plan preview, launch policy, and controlled command construction."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..config import RUN_MODES, ConfigError, load_model_config
from ..planner import build_execution_plan
from ..state import atomic_write_json, utc_now
from .audit import ConflictError, ValidationError, exclusive_lock
from .store import NotFoundError, read_json

# Only CI-owned placeholders may appear in an argument. Everything else is a
# literal token, so a request can never smuggle in a shell expansion.
PLACEHOLDERS = frozenset(
    {
        "{source_path}",
        "{output_dir}",
        "{work_dir}",
        "{reports_dir}",
        "{run_dir}",
        "{run_id}",
        "{model_id}",
    }
)

# Server addresses, ports, and arbitrary URLs are explicitly out of scope: the
# inference container host and lifecycle belong to the user or the platform
# team, not to the CI control plane.
FORBIDDEN_INFERENCE_FIELDS = (
    "server",
    "server_address",
    "host",
    "hostname",
    "port",
    "url",
    "endpoint",
    "base_url",
    "api_url",
    "ssh",
    "token",
    "password",
)

_SHELL_METACHARACTERS = ("|", "&", ";", "<", ">", "`", "$", "\n", "\r", "\\")
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/=:+-]*$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PLAN_HASH = re.compile(r"^[0-9a-f]{64}$")
_QUANT_FLAG_ALIASES: dict[str, tuple[str, str]] = {
    "--model-id": ("model_path", "value"),
    "--dataset-id": ("datasets", "list"),
    "--num-calibration-samples": ("calibration_samples", "int"),
    "--max-sequence-length": ("max_sequence_length", "int"),
    "--quant-mode": ("quant_mode", "value"),
    "--output-dir": ("output_dir", "value"),
    "--use-float32-scale-dtype": ("use_float32_scale_dtype", "flag"),
}


def _reject_shell_metacharacters(value: str, field_name: str) -> None:
    found = [token for token in _SHELL_METACHARACTERS if token in value]
    if found:
        raise ValidationError(
            f"{field_name} may not contain shell metacharacters: {''.join(found)!r}"
        )


def _check_token(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{field_name} must be a non-empty string")
    _reject_shell_metacharacters(value, field_name)
    if not _SAFE_TOKEN.fullmatch(value):
        raise ValidationError(f"{field_name} is not an allowed token: {value!r}")
    if ".." in value.split("/"):
        raise ValidationError(f"{field_name} may not traverse parent directories")
    return value


def _check_placed_token(value: str, field_name: str) -> str:
    """Allow a CI placeholder or a plain, shell-free token."""

    if value in PLACEHOLDERS:
        return value
    if value.startswith("{") and value.endswith("}"):
        raise ValidationError(f"{field_name} uses an unknown placeholder: {value!r}")
    return _check_token(value, field_name)


def _check_int(value: Any, field_name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValidationError(f"{field_name} must be an integer >= {minimum}")
    return value


def default_run_id(git_sha: str) -> str:
    """Build a run id that is always a safe single path component.

    The planner appends ``git_sha[:12]`` to the timestamp, which can contain
    characters such as ``:`` when the revision is a ref rather than a hex
    digest. Run ids become directory names, so the web layer sanitizes them.
    """

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = re.sub(r"[^A-Za-z0-9._-]", "", git_sha)[:12] or "local"
    return f"{stamp}_{token}"


def _within(candidate: Path, roots: tuple[Path, ...]) -> bool:
    for root in roots:
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        return True
    return False


@dataclass(frozen=True)
class LaunchPolicy:
    """Deployment policy for what a web request is allowed to request."""

    inference_containers: tuple[str, ...] = ()
    inference_script_roots: tuple[Path, ...] = ()
    evaluation_commands: tuple[str, ...] = ()
    quantization_executables: tuple[str, ...] = ("python3",)
    allowed_output_roots: tuple[Path, ...] = ()
    gpu_hour_capacity: float = 80.0
    require_dual_approval: bool = False
    artifact_access_ttl_seconds: int = 900
    backup_retention_days: int = 30
    backup_max_source_bytes: int = 0
    executor: str = "queue"
    quantization_flags: tuple[str, ...] = field(
        default_factory=lambda: tuple(sorted(_QUANT_FLAG_ALIASES))
    )

    @staticmethod
    def _split(value: str | None, separator: str) -> tuple[str, ...]:
        if not value:
            return ()
        return tuple(part.strip() for part in value.split(separator) if part.strip())

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "LaunchPolicy":
        env = env if env is not None else dict(os.environ)

        def flag(name: str, default: bool = False) -> bool:
            raw = env.get(name)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "on"}

        def number(name: str, default: float) -> float:
            raw = env.get(name)
            if raw is None or not raw.strip():
                return default
            try:
                return float(raw)
            except ValueError as error:
                raise ValidationError(f"{name} must be a number") from error

        executables = cls._split(
            env.get("MODEL_QUALITY_QUANTIZATION_EXECUTABLES"), ","
        ) or ("python3",)
        return cls(
            inference_containers=cls._split(
                env.get("MODEL_QUALITY_INFERENCE_CONTAINERS"), ","
            ),
            inference_script_roots=tuple(
                Path(part).expanduser()
                for part in cls._split(
                    env.get("MODEL_QUALITY_INFERENCE_SCRIPT_ROOTS"), os.pathsep
                )
            ),
            evaluation_commands=cls._split(
                env.get("MODEL_QUALITY_EVALUATION_COMMANDS"), os.pathsep
            ),
            quantization_executables=executables,
            allowed_output_roots=tuple(
                Path(part).expanduser()
                for part in cls._split(
                    env.get("MODEL_QUALITY_OUTPUT_ROOTS"), os.pathsep
                )
            ),
            gpu_hour_capacity=number("MODEL_QUALITY_GPU_HOUR_CAPACITY", 80.0),
            require_dual_approval=flag("MODEL_QUALITY_PUBLISH_DUAL_APPROVAL"),
            artifact_access_ttl_seconds=int(
                number("MODEL_QUALITY_ARTIFACT_ACCESS_TTL_SECONDS", 900)
            ),
            backup_retention_days=int(
                number("MODEL_QUALITY_BACKUP_RETENTION_DAYS", 30)
            ),
            backup_max_source_bytes=int(
                number("MODEL_QUALITY_BACKUP_MAX_SOURCE_BYTES", 0)
            ),
            executor=env.get("MODEL_QUALITY_WEB_EXECUTOR", "queue"),
        )


def validate_evaluation_command(
    value: Any, *, policy: LaunchPolicy
) -> list[str] | None:
    """Validate an optional evaluation argv override for the quant container."""

    if value is None:
        return None
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(token, str) and token for token in value)
    ):
        raise ValidationError("evaluation_command must be a non-empty argv list")
    command = list(value)
    if policy.evaluation_commands:
        executable = command[0]
        if executable not in policy.evaluation_commands:
            raise ValidationError(
                "evaluation command must start with one of "
                + ", ".join(policy.evaluation_commands)
            )
    for index, token in enumerate(command):
        literal = token
        for placeholder in PLACEHOLDERS:
            literal = literal.replace(placeholder, "placeholder")
        _reject_shell_metacharacters(literal, f"evaluation_command[{index}]")
        unknown = re.findall(r"\{[A-Za-z_][A-Za-z0-9_]*\}", literal)
        if unknown:
            raise ValidationError(
                f"evaluation_command[{index}] uses an unknown placeholder"
            )
    return command


def validate_inference_config(
    value: Any, *, policy: LaunchPolicy
) -> dict[str, Any] | None:
    """Validate a prestarted inference container and its script.

    The API accepts only a registered container name and an reviewed script
    path. Addresses, ports, URLs, and container lifecycle actions are rejected
    before they can become a command-injection or SSRF entry point.
    """

    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValidationError("inference must be a mapping")
    forbidden = sorted(set(value) & set(FORBIDDEN_INFERENCE_FIELDS))
    if forbidden:
        raise ValidationError(
            "inference must not configure server, port, or URL fields: "
            + ", ".join(forbidden)
        )

    container_name = value.get("container_name")
    if not isinstance(container_name, str) or not container_name.strip():
        raise ValidationError("inference.container_name is required")
    container_name = container_name.strip()
    if not policy.inference_containers:
        raise ValidationError(
            "no inference containers are registered; set "
            "MODEL_QUALITY_INFERENCE_CONTAINERS to a reviewed allowlist"
        )
    if container_name not in policy.inference_containers:
        raise ValidationError(
            f"inference container {container_name!r} is not registered"
        )

    script = value.get("script")
    if not isinstance(script, str) or not script.strip():
        raise ValidationError("inference.script is required")
    script = script.strip()
    _reject_shell_metacharacters(script, "inference.script")
    if not script.startswith("/"):
        raise ValidationError("inference.script must be an absolute container path")
    if ".." in Path(script).parts:
        raise ValidationError("inference.script may not traverse parent directories")
    if Path(script).suffix not in {".sh", ".bash", ".py"}:
        raise ValidationError("inference.script must be a .sh, .bash, or .py file")
    if not policy.inference_script_roots:
        raise ValidationError(
            "no inference script roots are registered; set "
            "MODEL_QUALITY_INFERENCE_SCRIPT_ROOTS to a reviewed allowlist"
        )
    if not _within(Path(script), tuple(policy.inference_script_roots)):
        raise ValidationError(
            f"inference.script is outside the registered script roots: {script}"
        )

    arguments = value.get("arguments", [])
    if not isinstance(arguments, list):
        raise ValidationError("inference.arguments must be a list")
    normalized_arguments = [
        _check_placed_token(argument, f"inference.arguments[{index}]")
        for index, argument in enumerate(arguments)
    ]

    result_file = value.get("result_file", "{reports_dir}/runtime-smoke-output.json")
    if not isinstance(result_file, str) or not result_file:
        raise ValidationError("inference.result_file must be a string")
    _reject_shell_metacharacters(result_file, "inference.result_file")
    if "{reports_dir}" not in result_file and not _within(
        Path(result_file), tuple(policy.inference_script_roots)
    ):
        raise ValidationError(
            "inference.result_file must be under {reports_dir} or a registered "
            "script root"
        )

    timeout_minutes = value.get("timeout_minutes", 60)
    timeout_minutes = _check_int(timeout_minutes, "inference.timeout_minutes")
    if timeout_minutes > 24 * 60:
        raise ValidationError("inference.timeout_minutes must be <= 1440")

    return {
        "container_name": container_name,
        "script": script,
        "arguments": normalized_arguments,
        "result_file": result_file,
        "timeout_minutes": timeout_minutes,
        "argv_preview": [script, *normalized_arguments],
    }


def build_quantization_argv(form: dict[str, Any], *, policy: LaunchPolicy) -> list[str]:
    """Turn the structured quantization form into a frozen argv array."""

    if not isinstance(form, dict):
        raise ValidationError("quantization must be a mapping")

    executable = form.get("executable", "python3")
    if (
        not isinstance(executable, str)
        or executable not in policy.quantization_executables
    ):
        raise ValidationError(
            "quantization.executable must be one of "
            + ", ".join(policy.quantization_executables)
        )

    entrypoint = form.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint:
        raise ValidationError("quantization.entrypoint is required")
    _reject_shell_metacharacters(entrypoint, "quantization.entrypoint")
    if entrypoint.startswith("/") or ".." in Path(entrypoint).parts:
        raise ValidationError(
            "quantization.entrypoint must be a repo-relative path without '..'"
        )
    if Path(entrypoint).suffix != ".py":
        raise ValidationError("quantization.entrypoint must be a .py file")

    argv = [executable, entrypoint]

    model_path = form.get("model_path", "{source_path}")
    argv += ["--model-id", _check_placed_token(model_path, "quantization.model_path")]

    datasets = form.get("datasets")
    if datasets is not None:
        if not isinstance(datasets, list) or not datasets:
            raise ValidationError("quantization.datasets must be a non-empty list")
        argv.append("--dataset-id")
        argv += [
            _check_placed_token(dataset, f"quantization.datasets[{index}]")
            for index, dataset in enumerate(datasets)
        ]

    for field_name, flag in (
        ("calibration_samples", "--num-calibration-samples"),
        ("max_sequence_length", "--max-sequence-length"),
    ):
        if form.get(field_name) is not None:
            argv += [
                flag,
                str(_check_int(form[field_name], f"quantization.{field_name}")),
            ]

    if form.get("quant_mode") is not None:
        argv += [
            "--quant-mode",
            _check_token(str(form["quant_mode"]), "quantization.quant_mode"),
        ]

    output_dir = form.get("output_dir", "{output_dir}")
    if output_dir != "{output_dir}":
        candidate = Path(str(output_dir)).expanduser()
        if not candidate.is_absolute() or not policy.allowed_output_roots:
            raise ValidationError(
                "quantization.output_dir must stay '{output_dir}' unless an "
                "allowed output root policy is configured"
            )
        if not _within(candidate, tuple(policy.allowed_output_roots)):
            raise ValidationError(
                f"quantization.output_dir is outside the allowed roots: {output_dir}"
            )
    argv += ["--output-dir", "{output_dir}"]

    if form.get("work_dir") is not None and form["work_dir"] != "{work_dir}":
        raise ValidationError("quantization.work_dir must stay '{work_dir}'")
    argv += ["--work-dir", "{work_dir}"]

    if form.get("use_float32_scale_dtype"):
        argv.append("--use-float32-scale-dtype")

    extra = form.get("extra_arguments", [])
    if extra:
        if not isinstance(extra, list):
            raise ValidationError("quantization.extra_arguments must be a list")
        index = 0
        while index < len(extra):
            token = extra[index]
            if not isinstance(token, str) or not token:
                raise ValidationError(
                    "quantization.extra_arguments entries must be strings"
                )
            if not token.startswith("--"):
                raise ValidationError(
                    f"quantization.extra_arguments must be flags: {token!r}"
                )
            if token not in policy.quantization_flags:
                raise ValidationError(f"unknown quantization flag: {token!r}")
            argv.append(token)
            index += 1
            if index < len(extra) and not str(extra[index]).startswith("--"):
                argv.append(
                    _check_token(str(extra[index]), "quantization.extra_arguments")
                )
                index += 1

    return argv


def import_quantization_command(
    command: str, *, policy: LaunchPolicy
) -> dict[str, Any]:
    """Import a pasted command after lexing and allowlist validation."""

    if not isinstance(command, str) or not command.strip():
        raise ValidationError("command must be a non-empty string")
    _reject_shell_metacharacters(command, "command")
    if command.count("\n") > 0:
        raise ValidationError("command must be a single line")

    try:
        tokens = shlex.split(command, comments=False, posix=True)
    except ValueError as error:
        raise ValidationError(f"command could not be parsed: {error}") from error
    if not tokens:
        raise ValidationError("command is empty")
    if tokens[0] not in policy.quantization_executables:
        raise ValidationError(
            "command executable must be one of "
            + ", ".join(policy.quantization_executables)
        )
    if len(tokens) < 2 or tokens[1].startswith("-"):
        raise ValidationError("command must name a quantization entrypoint")

    structured: dict[str, Any] = {
        "executable": tokens[0],
        "entrypoint": tokens[1],
        "use_float32_scale_dtype": False,
    }
    index = 2
    while index < len(tokens):
        flag = tokens[index]
        if not flag.startswith("--"):
            raise ValidationError(f"unexpected positional argument: {flag!r}")
        if flag not in _QUANT_FLAG_ALIASES:
            raise ValidationError(f"unknown quantization flag: {flag!r}")
        field_name, kind = _QUANT_FLAG_ALIASES[flag]
        index += 1
        if kind == "flag":
            structured[field_name] = True
            continue
        values = []
        while index < len(tokens) and not tokens[index].startswith("--"):
            values.append(tokens[index])
            index += 1
        if not values:
            raise ValidationError(f"{flag} requires a value")
        if kind == "list":
            structured[field_name] = values
        elif kind == "int":
            try:
                structured[field_name] = int(values[0])
            except ValueError as error:
                raise ValidationError(f"{flag} must be an integer") from error
            if len(values) > 1:
                raise ValidationError(f"{flag} accepts a single value")
        else:
            structured[field_name] = values[0]
            if len(values) > 1:
                raise ValidationError(f"{flag} accepts a single value")

    argv = build_quantization_argv(structured, policy=policy)
    return {"argv": argv, "structured": structured}


class PlanRegistry:
    """Store confirmed plan previews so a launch can only use a reviewed plan."""

    def __init__(self, root: str | Path, *, ttl_seconds: int = 3600) -> None:
        self.root = Path(root).expanduser()
        self.ttl_seconds = int(ttl_seconds)

    @property
    def directory(self) -> Path:
        return self.root / "_audit" / "plans"

    def _path(self, plan_hash: str) -> Path:
        if not isinstance(plan_hash, str) or not _PLAN_HASH.fullmatch(plan_hash):
            raise ValidationError("plan_hash must be a 64 character hex digest")
        return self.directory / f"{plan_hash}.json"

    @staticmethod
    def _expired(record: dict[str, Any]) -> bool:
        expires_at = record.get("expires_at")
        if not isinstance(expires_at, str):
            return False
        try:
            parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        except ValueError:
            return False
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) > parsed

    def save(self, plan_hash: str, record: dict[str, Any]) -> None:
        path = self._path(plan_hash)
        atomic_write_json(
            path,
            {
                "schema_version": 1,
                "plan_hash": plan_hash,
                "created_at": utc_now(),
                "expires_at": (
                    datetime.now(timezone.utc) + timedelta(seconds=self.ttl_seconds)
                ).isoformat(),
                "consumed_by": None,
                "consumed_at": None,
                **record,
            },
        )

    def load(self, plan_hash: str) -> dict[str, Any]:
        value = read_json(self._path(plan_hash))
        if not isinstance(value, dict):
            raise NotFoundError(f"plan {plan_hash!r} was not found")
        return value

    def consume(self, plan_hash: str) -> dict[str, Any]:
        """Mark a preview as launched; a preview can start exactly one run."""

        path = self._path(plan_hash)
        with exclusive_lock(path.with_suffix(".lock")):
            record = self.load(plan_hash)
            if self._expired(record):
                raise ConflictError("plan preview expired; preview it again")
            if record.get("consumed_by"):
                raise ConflictError(
                    f"plan was already launched by run {record['consumed_by']!r}"
                )
            record.update(
                consumed_by=record.get("planned_run_id"), consumed_at=utc_now()
            )
            atomic_write_json(path, record)
        return record


class PlanService:
    """Build reviewed execution plans and the launch spec derived from them."""

    def __init__(
        self,
        *,
        config_path: str | Path,
        registry: PlanRegistry,
        policy: LaunchPolicy,
        git_sha: str = "local",
    ) -> None:
        self.config_path = Path(config_path).expanduser()
        self.registry = registry
        self.policy = policy
        self.git_sha = git_sha

    def config_snapshot(self) -> dict[str, Any]:
        config = load_model_config(self.config_path)
        digest = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "path": str(self.config_path),
            "schema_version": config["schema_version"],
            "fingerprint": digest,
            "model_count": len(config["models"]),
        }

    def preview(self, request: dict[str, Any]) -> dict[str, Any]:
        """Build an execution plan without consuming any GPU capacity."""

        if not isinstance(request, dict):
            raise ValidationError("request body must be a JSON object")
        run_mode = request.get("run_mode", "quantize")
        if run_mode not in RUN_MODES:
            raise ValidationError(f"run_mode must be one of {RUN_MODES}")
        run_id = request.get("run_id")
        if run_mode in {"eval_only", "upload_only"}:
            if not isinstance(run_id, str) or not _SAFE_ID.fullmatch(run_id):
                raise ValidationError(f"run_id is required for run_mode={run_mode}")
        elif run_id is not None:
            if not isinstance(run_id, str) or not _SAFE_ID.fullmatch(run_id):
                raise ValidationError("run_id contains unsupported characters")
        else:
            run_id = default_run_id(self.git_sha)

        max_gpu_hours = request.get("max_gpu_hours", 80.0)
        if isinstance(max_gpu_hours, bool) or not isinstance(
            max_gpu_hours, (int, float)
        ):
            raise ValidationError("max_gpu_hours must be a number")
        if max_gpu_hours <= 0:
            raise ValidationError("max_gpu_hours must be positive")

        inference = validate_inference_config(
            request.get("inference"), policy=self.policy
        )
        evaluation_command = validate_evaluation_command(
            request.get("evaluation_command"), policy=self.policy
        )
        quantization_argv = None
        if request.get("quantization") is not None:
            quantization_argv = build_quantization_argv(
                request["quantization"], policy=self.policy
            )
        elif request.get("command") is not None:
            imported = import_quantization_command(
                request["command"], policy=self.policy
            )
            quantization_argv = imported["argv"]

        config = load_model_config(self.config_path)
        try:
            plan = build_execution_plan(
                config,
                git_sha=self.git_sha,
                max_gpu_hours=float(max_gpu_hours),
                model_filter=request.get("models", "all"),
                priority_override=request.get("priority_override", "none"),
                run_mode=run_mode,
                run_id=run_id,
            )
        except ConfigError as error:
            raise ValidationError(str(error)) from error

        models = {model["id"]: model for model in config["models"]}
        for job in plan["selected"]:
            if quantization_argv is not None:
                job["quantization_argv"] = list(quantization_argv)
        review = self._review(
            plan,
            request,
            models,
            inference,
            evaluation_command,
            quantization_argv,
        )
        payload = {
            "plan": plan,
            "review": review,
            "inference": inference,
            "evaluation_command": evaluation_command,
            "quantization_argv": quantization_argv,
        }
        plan_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.registry.save(plan_hash, {"planned_run_id": plan["run_id"], **payload})
        return {
            "plan_hash": plan_hash,
            "generated_at": utc_now(),
            **payload,
        }

    def _review(
        self,
        plan: dict[str, Any],
        request: dict[str, Any],
        models: dict[str, dict[str, Any]],
        inference: dict[str, Any] | None,
        evaluation_command: list[str] | None,
        quantization_argv: list[str] | None,
    ) -> dict[str, Any]:
        selected_ids = [job["id"] for job in plan["selected"]]
        risks: list[dict[str, str]] = []
        upload_auto_publish = any(job.get("upload_enabled") for job in plan["selected"])
        if upload_auto_publish:
            risks.append(
                {
                    "code": "UPLOAD_AUTO_PUBLISH",
                    "severity": "warning",
                    "message": (
                        "upload.enabled is true, so the pipeline adds a publish "
                        "stage that uploads to the remote prefix"
                    ),
                }
            )
        if plan["deferred"]:
            risks.append(
                {
                    "code": "BUDGET_DEFERRED",
                    "severity": "warning",
                    "message": (
                        f"{len(plan['deferred'])} model(s) are deferred by the "
                        "GPU-hour budget"
                    ),
                }
            )
        if not plan["selected"]:
            risks.append(
                {
                    "code": "NOTHING_SELECTED",
                    "severity": "error",
                    "message": "no model matched the filter and budget",
                }
            )
        if self.git_sha in {"local", "unknown", ""}:
            risks.append(
                {
                    "code": "UNPINNED_GIT_SHA",
                    "severity": "warning",
                    "message": "git_sha is not pinned; the plan is not reproducible",
                }
            )
        for model_id in selected_ids:
            model = models[model_id]
            if not model["enabled"]:
                risks.append(
                    {
                        "code": "DISABLED_MODEL_SELECTED",
                        "severity": "warning",
                        "message": f"{model_id} is disabled in the manifest",
                    }
                )
            if model["workflow"]["revision"].startswith("replace-with"):
                risks.append(
                    {
                        "code": "UNPINNED_WORKFLOW_REVISION",
                        "severity": "error",
                        "message": (
                            f"{model_id}.workflow.revision is still the placeholder"
                        ),
                    }
                )
            upload = model.get("upload", {})
            if upload.get("enabled") and (
                "replace-me" in str(upload.get("remote_prefix", ""))
            ):
                risks.append(
                    {
                        "code": "PLACEHOLDER_REMOTE_PREFIX",
                        "severity": "error",
                        "message": f"{model_id}.upload.remote_prefix is a placeholder",
                    }
                )
        if inference is None:
            risks.append(
                {
                    "code": "NO_INFERENCE_CONTAINER",
                    "severity": "warning",
                    "message": (
                        "no prestarted inference container was confirmed; the "
                        "runtime smoke stage will run inside the CI executor"
                    ),
                }
            )

        estimated_max_hours = max(
            (
                float(models[model_id]["resources"].get("timeout_hours", 0))
                for model_id in selected_ids
            ),
            default=0.0,
        )
        environments = {
            "quantization": {
                "queue": "model-quality-gpu",
                "executables": list(self.policy.quantization_executables),
                "argv_override": quantization_argv,
                "models": [
                    {
                        "model_id": model_id,
                        "gpu_count": models[model_id]["resources"]["gpu_count"],
                        "workflow_revision": models[model_id]["workflow"]["revision"],
                        "quantize_argv": (
                            quantization_argv
                            or models[model_id]["workflow"]["quantize"]
                        ),
                    }
                    for model_id in selected_ids
                ],
            },
            "inference": {
                "container": inference,
                "registered_containers": list(self.policy.inference_containers),
                "script_roots": [
                    str(root) for root in self.policy.inference_script_roots
                ],
            },
            "evaluation": {
                "argv_override": evaluation_command,
                "models": [
                    {
                        "model_id": model_id,
                        "enabled": bool(
                            models[model_id]
                            .get("evaluation", {})
                            .get("enabled_by_default")
                        )
                        or plan["run_mode"] in {"quantize_and_eval", "eval_only"},
                        "suites": models[model_id]
                        .get("evaluation", {})
                        .get("suites", []),
                        "runtime_revision": models[model_id]
                        .get("evaluation", {})
                        .get("runtime_revision"),
                    }
                    for model_id in selected_ids
                ],
            },
        }
        return {
            "config": self.config_snapshot(),
            "budget": plan["budget"],
            "selected": plan["selected"],
            "deferred": plan["deferred"],
            "upload_auto_publish": upload_auto_publish,
            "estimated_max_hours": estimated_max_hours,
            "environments": environments,
            "requested_models": request.get("models", "all"),
            "run_mode": plan["run_mode"],
            "risks": risks,
            "notices": [
                "preview only: this request does not consume GPU capacity",
                (
                    "launching will consume GPU capacity and may trigger a "
                    "remote publish"
                    if upload_auto_publish
                    else "launching will consume GPU capacity"
                ),
            ],
        }

    def launch_spec(self, plan_hash: str) -> dict[str, Any]:
        """Load a preview and return the reviewed plan plus its launch details."""

        record = self.registry.load(plan_hash)
        if record.get("consumed_by"):
            raise ConflictError(
                f"plan was already launched by run {record['consumed_by']!r}"
            )
        return record

    def fingerprint_payload(self, plan_hash: str) -> dict[str, Any]:
        record = self.registry.load(plan_hash)
        return {
            "plan_hash": plan_hash,
            "plan": record.get("plan"),
            "inference": record.get("inference"),
            "evaluation_command": record.get("evaluation_command"),
            "quantization_argv": record.get("quantization_argv"),
        }
