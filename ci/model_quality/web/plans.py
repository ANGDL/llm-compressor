"""Plan preview, launch policy, and controlled command construction."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..config import RUN_MODES, ConfigError, _validate_model, load_model_config
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
# team, not to the CI control plane. A local service ``port`` is allowed (it is
# only ever probed on 127.0.0.1 by the smoke harness), but hosts/URLs are not.
FORBIDDEN_INFERENCE_FIELDS = (
    "server",
    "server_address",
    "host",
    "hostname",
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
# Reviewed default destination for manual publication. Deployments widen or
# narrow the accepted set with MODEL_QUALITY_UPLOAD_PREFIXES.
DEFAULT_UPLOAD_REMOTE_PREFIX = "bos:/your-bucket/model-quality"
_UPLOAD_PREFIX = re.compile(r"^bos:/[A-Za-z0-9][A-Za-z0-9._/-]*$")
# The repository root (``.../ci/model_quality/web/plans.py`` → four parents up).
# Script introspection resolves repo-relative entrypoints against this root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
# Only these stages may be skipped; preflight/quantize/report are mandatory and
# dropping them would make a run meaningless. ``runtime-smoke`` joins the set so
# the script-first flow can de-select the inference smoke stage.
SKIPPABLE_STAGES: frozenset[str] = frozenset(
    {"validate", "runtime-smoke", "evaluate", "publish"}
)
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
    # BOS prefixes a plan may publish under. The operator still triggers the
    # upload by hand from the run page; this only says where it may go.
    upload_remote_prefixes: tuple[str, ...] = (DEFAULT_UPLOAD_REMOTE_PREFIX,)
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
    # Python interpreter that can import each eval tool. The informational
    # harness runs as ``<python> -m ci.model_quality.evaluators.informational``,
    # so deployments whose tools live in dedicated conda envs point these at
    # e.g. /root/miniconda/envs/model_quality_evalscope/bin/python.
    eval_tool_executables: dict[str, str] = field(
        default_factory=lambda: {"lm_eval": "lm_eval", "evalscope": "evalscope"}
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
            upload_remote_prefixes=cls._split(
                env.get("MODEL_QUALITY_UPLOAD_PREFIXES"), ","
            )
            or (DEFAULT_UPLOAD_REMOTE_PREFIX,),
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
            eval_tool_executables={
                "lm_eval": env.get("MODEL_QUALITY_EVAL_LM_EVAL", "lm_eval"),
                "evalscope": env.get("MODEL_QUALITY_EVAL_EVALSCOPE", "evalscope"),
            },
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


def validate_upload_config(
    value: Any, *, policy: LaunchPolicy
) -> dict[str, Any] | None:
    """Validate the optional manual-publish target for a script-first plan.

    Only the BOS prefix is operator-controlled; the allowlist and the bcecmd
    argv are generated here, so a plan can never name a local path outside the
    published roots or a different executable. The upload itself stays manual:
    this only records where an operator may publish from the run page.
    """

    if value is None:
        return None
    if isinstance(value, str):
        value = {"remote_prefix": value}
    if not isinstance(value, dict):
        raise ValidationError("upload must be a mapping or a remote prefix string")
    unknown = sorted(set(value) - {"remote_prefix"})
    if unknown:
        raise ValidationError(
            "upload accepts only remote_prefix: " + ", ".join(unknown)
        )

    remote_prefix = str(value.get("remote_prefix") or "").strip().rstrip("/")
    if not remote_prefix:
        raise ValidationError("upload.remote_prefix is required")
    if remote_prefix in {"/", "bos:/"} or "replace-me" in remote_prefix:
        raise ValidationError("upload.remote_prefix is a placeholder")
    _reject_shell_metacharacters(remote_prefix, "upload.remote_prefix")
    if not _UPLOAD_PREFIX.fullmatch(remote_prefix):
        raise ValidationError(
            "upload.remote_prefix must look like bos:/bucket/prefix"
        )
    if remote_prefix not in policy.upload_remote_prefixes:
        raise ValidationError(
            "upload.remote_prefix must be one of "
            + ", ".join(policy.upload_remote_prefixes)
        )

    destination = "{remote_run_prefix}"
    return {
        "enabled": True,
        "remote_prefix": remote_prefix,
        "allowlist": ["model", "reports"],
        "commands": [
            [
                "bcecmd",
                "bos",
                "cp",
                "-r",
                "-y",
                "--quiet",
                "--disable-bar",
                "{output_dir}",
                f"{destination}/model",
            ],
            [
                "bcecmd",
                "bos",
                "cp",
                "-r",
                "-y",
                "--quiet",
                "--disable-bar",
                "{reports_dir}",
                f"{destination}/reports",
            ],
        ],
        "verify_commands": [
            [
                "bcecmd",
                "bos",
                "cp",
                "-y",
                "--quiet",
                "--disable-bar",
                f"{destination}/model/config.json",
                "/tmp/model-quality-publish-verify/config.json",
            ],
            [
                "bcecmd",
                "bos",
                "cp",
                "-y",
                "--quiet",
                "--disable-bar",
                f"{destination}/reports/summary.json",
                "/tmp/model-quality-publish-verify/summary.json",
            ],
        ],
        "success_commands": [
            [
                "bcecmd",
                "bos",
                "cp",
                "-y",
                "--quiet",
                "--disable-bar",
                "{run_dir}/commit-markers/SUCCESS.json",
                f"{destination}/SUCCESS.json",
            ],
        ],
    }


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

    # A local service port the smoke harness probes on 127.0.0.1. Not an SSRF
    # vector (no host is accepted), so it is allowed unlike other address fields.
    port = value.get("port", 8025)
    port = _check_int(port, "inference.port")
    if port > 65535:
        raise ValidationError("inference.port must be between 1 and 65535")

    return {
        "container_name": container_name,
        "script": script,
        "arguments": normalized_arguments,
        "result_file": result_file,
        "timeout_minutes": timeout_minutes,
        "port": port,
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


def _validate_script_path(path: Any, field_name: str = "script") -> str:
    """Reuse the entrypoint rule: repo-relative ``.py`` with no traversal."""

    if not isinstance(path, str) or not path.strip():
        raise ValidationError(f"{field_name} is required")
    path = path.strip()
    _reject_shell_metacharacters(path, field_name)
    if path.startswith("/") or ".." in Path(path).parts:
        raise ValidationError(
            f"{field_name} must be a repo-relative path without '..'"
        )
    if Path(path).suffix != ".py":
        raise ValidationError(f"{field_name} must be a .py file")
    return path


def _ast_literal(node: ast.AST | None) -> tuple[bool, Any]:
    """Best-effort literal evaluation; never executes arbitrary code."""

    if node is None:
        return False, None
    try:
        return True, ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return False, None


_AST_TYPE_NAMES = {
    "int": "int",
    "float": "float",
    "str": "str",
    "Path": "str",
    "bool": "bool",
}


def _introspect_argument(
    options: list[str], kwargs: dict[str, ast.AST]
) -> dict[str, Any]:
    """Turn one ``add_argument`` call into an editable parameter descriptor."""

    flags = [opt for opt in options if opt.startswith("-")]
    positional = [opt for opt in options if not opt.startswith("-")]
    primary = flags[-1] if flags else positional[0]

    dest_ok, dest_val = _ast_literal(kwargs.get("dest"))
    if dest_ok and isinstance(dest_val, str):
        dest = dest_val
    else:
        dest = primary.lstrip("-").replace("-", "_")

    action_ok, action_val = _ast_literal(kwargs.get("action"))
    action = action_val if action_ok and isinstance(action_val, str) else None
    if action is None:
        # ``action=argparse.BooleanOptionalAction`` (and a bare
        # ``BooleanOptionalAction``) is a reference, not a string literal, so
        # ``_ast_literal`` can't resolve it. Recognize it by name so the flag is
        # treated as a valueless boolean (--flag / --no-flag) instead of a
        # value-taking option — otherwise the generated argv passes a stray
        # ``true``/``false`` token the script rejects.
        action_node = kwargs.get("action")
        if isinstance(action_node, ast.Attribute):
            action_name = action_node.attr
        elif isinstance(action_node, ast.Name):
            action_name = action_node.id
        else:
            action_name = None
        if action_name == "BooleanOptionalAction":
            action = "BooleanOptionalAction"
    is_bool = action in {"store_true", "store_false", "BooleanOptionalAction"}

    type_name: str | None = None
    type_node = kwargs.get("type")
    if isinstance(type_node, ast.Name):
        type_name = _AST_TYPE_NAMES.get(type_node.id)
    elif isinstance(type_node, ast.Attribute):
        type_name = _AST_TYPE_NAMES.get(type_node.attr)
    if is_bool:
        type_name = "bool"

    default: Any = None
    editable = True
    if "default" in kwargs:
        ok, default = _ast_literal(kwargs["default"])
        if not ok:
            editable = False
            default = None
    elif is_bool:
        default = action == "store_false"

    req_ok, req_val = _ast_literal(kwargs.get("required"))
    required = bool(req_val) if req_ok else False

    choices = None
    if "choices" in kwargs:
        ok, choice_val = _ast_literal(kwargs["choices"])
        if ok and isinstance(choice_val, (list, tuple)):
            choices = [str(choice) for choice in choice_val]

    nargs = None
    if "nargs" in kwargs:
        ok, nargs_val = _ast_literal(kwargs["nargs"])
        if ok:
            nargs = nargs_val

    help_text = None
    if "help" in kwargs:
        ok, help_val = _ast_literal(kwargs["help"])
        if ok and isinstance(help_val, str):
            help_text = help_val

    return {
        "flag": primary,
        "dest": dest,
        "type": type_name or "str",
        "action": action,
        "default": default,
        "editable": editable,
        "required": required,
        "choices": choices,
        "nargs": nargs,
        "help": help_text,
        "positional": not bool(flags),
    }


def introspect_script(path: Any, *, policy: LaunchPolicy) -> dict[str, Any]:
    """Statically parse an argparse entrypoint's ``add_argument`` calls.

    The file is read and AST-parsed only; it is never imported or executed, so
    pointing at a reviewed repo script cannot run arbitrary code. Returns the
    declared parameters with their defaults for the UI to render as an editable
    form. Scripts without literal argparse yield ``parseable: False`` so the UI
    falls back to a manual argv entry.
    """

    entrypoint = _validate_script_path(path)
    resolved = (_REPO_ROOT / entrypoint).resolve()
    try:
        resolved.relative_to(_REPO_ROOT)
    except ValueError as error:
        raise ValidationError(
            "script must resolve inside the repository"
        ) from error
    if not resolved.is_file():
        raise ValidationError(f"script not found: {entrypoint}")

    try:
        tree = ast.parse(resolved.read_text(encoding="utf-8"))
    except SyntaxError as error:
        raise ValidationError(f"script is not valid Python: {error}") from error

    parameters: list[dict[str, Any]] = []
    seen: set[str] = set()
    parseable = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        options = [
            arg.value
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        ]
        if not options:
            continue
        parseable = True
        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        param = _introspect_argument(options, kwargs)
        if param["flag"] in seen:
            continue
        seen.add(param["flag"])
        parameters.append(param)

    return {
        "entrypoint": entrypoint,
        "parameters": parameters,
        "parseable": parseable,
    }


# Argparse destinations that must map onto run-scoped CI placeholders so the
# downstream validate/report stages find the artifact the quantize stage wrote.
_OUTPUT_DESTS = frozenset(
    {"output_dir", "output", "save_dir", "save_path", "save_directory", "out_dir"}
)
_WORK_DESTS = frozenset({"work_dir", "workdir", "cache_dir", "tmp_dir", "scratch_dir"})
_MODEL_DESTS = frozenset(
    {
        "model_id",
        "model",
        "model_path",
        "model_name_or_path",
        "source",
        "source_path",
        "model_name",
        "pretrained",
    }
)
_VALUE_TOKEN = re.compile(r"^[A-Za-z0-9._/@:=+,-]+$")


def _check_value_token(value: str, field_name: str) -> str:
    """Validate a free-form argv value: no shell metacharacters, no traversal.

    More permissive than ``_check_token`` (allows a leading ``/`` so absolute
    model paths pass) but still single-argv-safe because the queue worker never
    runs the argv through a shell.
    """

    if value in PLACEHOLDERS:
        return value
    if value.startswith("{") and value.endswith("}"):
        raise ValidationError(f"{field_name} uses an unknown placeholder: {value!r}")
    _reject_shell_metacharacters(value, field_name)
    if ".." in value.split("/"):
        raise ValidationError(f"{field_name} may not traverse parent directories")
    if not _VALUE_TOKEN.fullmatch(value):
        raise ValidationError(f"{field_name} is not an allowed value: {value!r}")
    return value


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def build_script_quantization_argv(
    script: str,
    parameters: Any,
    introspected: dict[str, Any],
    *,
    policy: LaunchPolicy,
) -> tuple[list[str], str | None]:
    """Build the quantize argv from an introspected script and edited params.

    The declared argparse flags are the per-script allowlist: a request can only
    pass flags the reviewed script actually defines. Output/work directories are
    forced onto CI placeholders so the run stays self-contained. Returns the argv
    and the detected source/model path (for the synthesized ``source.path``).
    """

    if not isinstance(parameters, list):
        raise ValidationError("parameters must be a list of {flag, value} entries")
    declared = {param["flag"]: param for param in introspected["parameters"]}

    executable = "python3"
    if executable not in policy.quantization_executables:
        raise ValidationError(
            "quantization executable must be one of "
            + ", ".join(policy.quantization_executables)
        )

    missing = object()
    submitted: dict[str, Any] = {}
    for item in parameters:
        if not isinstance(item, dict):
            raise ValidationError("each parameter must be a {flag, value} mapping")
        flag = item.get("flag")
        if flag not in declared:
            raise ValidationError(f"unknown script parameter: {flag!r}")
        submitted[flag] = item.get("value", missing)

    argv = [executable, script]
    source_path: str | None = None

    # Iterate the declared order so required flags and CI-managed directories
    # are always present even if the request omitted them.
    for flag, spec in declared.items():
        dest = spec["dest"]
        has_value = flag in submitted and submitted[flag] is not missing
        value = submitted[flag] if has_value else spec.get("default")

        if spec["action"] in {"store_true", "store_false"}:
            desired = _truthy(value)
            # store_true: present ⇒ True; store_false: present ⇒ False.
            if (spec["action"] == "store_true" and desired) or (
                spec["action"] == "store_false" and not desired
            ):
                argv.append(flag)
            continue

        if spec["action"] == "BooleanOptionalAction":
            # argparse.BooleanOptionalAction takes no value: emit --flag when
            # true and --no-<flag> when false (never a bare true/false token).
            desired = _truthy(value)
            argv.append(flag if desired else f"--no-{flag[2:]}" if flag.startswith("--") else flag)
            continue

        if dest in _OUTPUT_DESTS:
            token = "{output_dir}"
        elif dest in _WORK_DESTS:
            token = "{work_dir}"
        else:
            if value is None or value == "" or value == []:
                if spec["required"]:
                    raise ValidationError(f"script parameter {flag} is required")
                continue
            # ``nargs``-style parameters carry a list (e.g. a list default like
            # ``--dataset_id ["a", "b"]``); emit one validated token per element
            # instead of stringifying the whole list into a single bad token.
            if isinstance(value, (list, tuple)):
                tokens = [
                    _check_value_token(str(item), f"parameter {flag}")
                    for item in value
                ]
                if dest in _MODEL_DESTS and source_path is None and tokens:
                    source_path = str(value[0])
                if spec["positional"]:
                    argv += tokens
                else:
                    argv += [flag, *tokens]
                continue
            token = _check_value_token(str(value), f"parameter {flag}")
            if dest in _MODEL_DESTS and source_path is None:
                source_path = str(value)

        if spec["positional"]:
            argv.append(token)
        else:
            argv += [flag, token]

    return argv, source_path


def build_manual_quantization_argv(
    script: str, tokens: Any, *, policy: LaunchPolicy
) -> list[str]:
    """Build a quantize argv from a raw token list (non-argparse fallback).

    Each token is validated as a flag or a placed value token; the queue worker
    never runs the argv through a shell, so a single validated token is safe.
    """

    if not isinstance(tokens, list) or not tokens:
        raise ValidationError("manual_argv must be a non-empty argv list")
    executable = "python3"
    if executable not in policy.quantization_executables:
        raise ValidationError(
            "quantization executable must be one of "
            + ", ".join(policy.quantization_executables)
        )
    argv = [executable, script]
    for index, token in enumerate(tokens):
        if not isinstance(token, str) or not token:
            raise ValidationError(f"manual_argv[{index}] must be a non-empty string")
        if token.startswith("-"):
            _reject_shell_metacharacters(token, f"manual_argv[{index}]")
            argv.append(token)
        else:
            argv.append(_check_value_token(token, f"manual_argv[{index}]"))
    return argv
# runtime-smoke container (no user-supplied URL), so the base endpoint is a
# container-local default; only the dataset/task tokens vary per preset.
# Each preset only names the dataset (and optional few-shot) the informational
# harness feeds to the tool; the served model id is discovered at run time via
# GET /v1/models, so there is no per-preset URL or model token here.
EVAL_DATASET_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "lm_eval": {
        "gsm8k": {"dataset": "gsm8k", "num_fewshot": 5},
        "mmlu": {"dataset": "mmlu", "num_fewshot": 5},
    },
    "evalscope": {
        "gsm8k": {"dataset": "gsm8k"},
        "ceval": {"dataset": "ceval"},
    },
}


def normalize_cuda_visible_devices(value: Any) -> str | None:
    """Validate an optional ``CUDA_VISIBLE_DEVICES`` request value.

    Accepts a comma-separated string (``"0,1,3"``), a single int, or a list of
    ints/str indices, and returns a normalized comma-separated string. Entries
    must be non-negative integers — the value is injected into the quantize
    subprocess environment, so anything else is rejected rather than trusted.
    Returns ``None`` when nothing is configured.
    """

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValidationError("cuda_visible_devices must be GPU indices")
    if isinstance(value, int):
        tokens = [str(value)]
    elif isinstance(value, (list, tuple)):
        tokens = [str(item).strip() for item in value]
    elif isinstance(value, str):
        tokens = [token.strip() for token in value.split(",")]
    else:
        raise ValidationError(
            "cuda_visible_devices must be a string or a list of GPU indices"
        )
    tokens = [token for token in tokens if token]
    if not tokens:
        return None
    for token in tokens:
        if not re.fullmatch(r"\d+", token):
            raise ValidationError(
                "cuda_visible_devices entries must be non-negative integers, "
                f"got {token!r}"
            )
    return ",".join(tokens)


def build_eval_command(
    request: dict[str, Any], *, policy: LaunchPolicy, port: int
) -> list[str]:
    """Resolve the evaluation argv from an override or a tool+dataset preset.

    The preset drives the *informational* harness
    (``ci.model_quality.evaluators.informational``): it discovers the one served
    model via ``GET /v1/models`` on ``127.0.0.1:<port>`` (the runtime-smoke
    container's local endpoint), runs the selected tool, and records every score
    with reference value 1 / ``min_recovery`` 0 so the stage always PASSES and a
    human gates the numbers. No service URL is ever accepted from the request.
    """

    override = request.get("evaluation_command")
    if override is not None:
        command = validate_evaluation_command(override, policy=policy)
        if command is None:
            raise ValidationError("evaluation_command must be a non-empty argv list")
        return command

    tool = request.get("evaluation_tool", "lm_eval")
    presets = EVAL_DATASET_PRESETS.get(tool)
    if presets is None:
        raise ValidationError(
            "evaluation_tool must be one of " + ", ".join(sorted(EVAL_DATASET_PRESETS))
        )
    dataset = request.get("evaluation_dataset")
    if dataset not in presets:
        raise ValidationError(
            f"evaluation_dataset for {tool} must be one of "
            + ", ".join(sorted(presets))
        )
    preset = presets[dataset]
    # The interpreter that can import the tool (deployments point this at e.g.
    # /root/miniconda/envs/model_quality_evalscope/bin/python).
    python = policy.eval_tool_executables.get(tool)
    if not python:
        raise ValidationError(f"no eval interpreter configured for tool {tool!r}")
    command = [
        python,
        "-m",
        "ci.model_quality.evaluators.informational",
        "--tool",
        tool,
        "--api-url",
        f"http://127.0.0.1:{int(port)}",
        "--dataset",
        preset["dataset"],
        "--work-dir",
        "{reports_dir}/eval-work",
        "--output",
        "{reports_dir}/evaluation.json",
    ]
    if preset.get("num_fewshot") is not None:
        command += ["--num-fewshot", str(preset["num_fewshot"])]
    return validate_evaluation_command(command, policy=policy)


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

    @staticmethod
    def _validate_skip_stages(value: Any) -> list[str]:
        """Validate an optional ``skip_stages`` list from the plan request."""

        if value is None:
            return []
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValidationError("skip_stages must be a list of stage names")
        unknown = [stage for stage in value if stage not in SKIPPABLE_STAGES]
        if unknown:
            raise ValidationError(
                "skip_stages may only contain "
                + ", ".join(sorted(SKIPPABLE_STAGES))
                + f"; got {', '.join(unknown)}"
            )
        # De-duplicate while preserving a stable order for a deterministic hash.
        return [stage for stage in sorted(SKIPPABLE_STAGES) if stage in set(value)]

    _STAGE_CHOICES = frozenset(
        {"quantize", "validate", "inference", "evaluate", "publish"}
    )

    def _preview_script_first(self, request: dict[str, Any]) -> dict[str, Any]:
        """Preview a plan built from a script path + edited params + stages."""

        script = _validate_script_path(request.get("script"))
        introspected = introspect_script(script, policy=self.policy)
        resolved = (_REPO_ROOT / script).resolve()
        script_hash = hashlib.sha256(resolved.read_bytes()).hexdigest()
        revision = f"sha256:{script_hash}"

        # Normalize the stage checkbox set; quantize is always the base.
        raw_stages = request.get("stages")
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValidationError("stages must be a non-empty list")
        stages = set()
        for stage in raw_stages:
            if stage not in self._STAGE_CHOICES:
                raise ValidationError(
                    "stages may only contain "
                    + ", ".join(sorted(self._STAGE_CHOICES))
                )
            stages.add(stage)
        stages.add("quantize")
        if "publish" in stages:
            raise ValidationError(
                "publish is never scheduled by a plan; configure the upload "
                "target and publish from the run page"
            )
        if "evaluate" in stages and "inference" not in stages:
            raise ValidationError(
                "evaluate requires inference: the evaluation runs against the "
                "runtime-smoke container"
            )

        argv, source_path = build_script_quantization_argv(
            script, request.get("parameters", []), introspected, policy=self.policy
        )
        if request.get("manual_argv"):
            # Non-argparse scripts fall back to a raw token list the user typed.
            argv = build_manual_quantization_argv(
                script, request["manual_argv"], policy=self.policy
            )
            source_path = source_path or script

        run_mode = "quantize_and_eval" if "evaluate" in stages else "quantize"
        skip = set()
        if "validate" not in stages:
            skip.add("validate")
        if "inference" not in stages:
            skip.add("runtime-smoke")
        if "evaluate" not in stages:
            skip.add("evaluate")
        skip_stages = [stage for stage in sorted(SKIPPABLE_STAGES) if stage in skip]

        upload = validate_upload_config(request.get("upload"), policy=self.policy)
        model, inference, evaluation_command = self._synthesize_model(
            request, script, revision, source_path, argv, stages, upload
        )
        try:
            normalized = _validate_model(model, 0)
        except ConfigError as error:
            raise ValidationError(str(error)) from error
        config = {"schema_version": 1, "models": [normalized]}

        run_id = request.get("run_id")
        if run_id is not None:
            if not isinstance(run_id, str) or not _SAFE_ID.fullmatch(run_id):
                raise ValidationError("run_id contains unsupported characters")
        else:
            run_id = default_run_id(self.git_sha)

        est = float(normalized["resources"]["estimated_gpu_hours"])
        default_budget = est * (3 if "evaluate" in stages else 2) + 2.0
        max_gpu_hours = request.get("max_gpu_hours") or default_budget
        if isinstance(max_gpu_hours, bool) or not isinstance(
            max_gpu_hours, (int, float)
        ):
            raise ValidationError("max_gpu_hours must be a number")

        try:
            plan = build_execution_plan(
                config,
                git_sha=self.git_sha,
                max_gpu_hours=float(max_gpu_hours),
                model_filter="all",
                priority_override="none",
                run_mode=run_mode,
                run_id=run_id,
            )
        except ConfigError as error:
            raise ValidationError(str(error)) from error

        for job in plan["selected"]:
            job["quantization_argv"] = list(argv)
        if skip_stages:
            plan["skip_stages"] = {
                job["id"]: list(skip_stages) for job in plan["selected"]
            }
        # Publishing is always an explicit operator action here: the plan only
        # records the reviewed upload target.
        plan["publish_trigger"] = "manual"
        plan["generated_config"] = config
        # Persist the original script-first inputs so a failed run can be cloned
        # into a prefilled New plan form (edit a few fields, re-run) instead of
        # retyping everything.
        plan["source_request"] = {
            "flow": "script_first",
            "script": script,
            "parameters": request.get("parameters", []),
            "stages": sorted(stages),
            "resources": request.get("resources", {}),
            "max_gpu_hours": request.get("max_gpu_hours"),
            "inference": request.get("inference"),
            "evaluation_tool": request.get("evaluation_tool"),
            "evaluation_dataset": request.get("evaluation_dataset"),
            "evaluation_command": request.get("evaluation_command"),
            "upload": request.get("upload"),
        }

        models = {normalized["id"]: normalized}
        review = self._review(
            plan, request, models, inference, evaluation_command, argv
        )
        payload = {
            "plan": plan,
            "review": review,
            "inference": inference,
            "evaluation_command": evaluation_command,
            "quantization_argv": argv,
        }
        plan_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.registry.save(
            plan_hash, {"planned_run_id": plan["run_id"], **payload}
        )
        return {"plan_hash": plan_hash, "generated_at": utc_now(), **payload}

    def _synthesize_model(
        self,
        request: dict[str, Any],
        script: str,
        revision: str,
        source_path: str | None,
        argv: list[str],
        stages: set[str],
        upload: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, list[str] | None]:
        """Build the one-model manifest entry for a script-first run."""

        res_req = request.get("resources") or {}
        if not isinstance(res_req, dict):
            raise ValidationError("resources must be a mapping")
        resources: dict[str, Any] = {
            "gpu_count": res_req.get("gpu_count", 1),
            "estimated_gpu_hours": res_req.get("estimated_gpu_hours", 1.0),
        }
        if res_req.get("estimated_eval_gpu_hours") is not None:
            resources["estimated_eval_gpu_hours"] = res_req[
                "estimated_eval_gpu_hours"
            ]

        stem = re.sub(r"[^A-Za-z0-9._-]", "-", Path(script).stem).strip("-")
        model_id = f"{stem or 'model'}-{revision.split(':')[1][:8]}"
        if not _SAFE_ID.fullmatch(model_id):
            model_id = f"m-{revision.split(':')[1][:8]}"

        model: dict[str, Any] = {
            "id": model_id,
            "enabled": True,
            "priority": "P2",
            "source": {"path": source_path or script},
            "resources": resources,
            "workflow": {"revision": revision, "quantize": list(argv)},
            "validation": {"profile": "causal_lm"},
        }

        # Pin the quantize process to specific GPUs when requested. run_quantize
        # applies ``workflow.env`` on top of the inherited environment before
        # spawning the quant script, so the child sees CUDA_VISIBLE_DEVICES
        # before it initializes CUDA.
        cuda_visible_devices = normalize_cuda_visible_devices(
            res_req.get("cuda_visible_devices")
        )
        if cuda_visible_devices is not None:
            model["workflow"]["env"] = {
                "CUDA_VISIBLE_DEVICES": cuda_visible_devices
            }

        inference = None
        # The runtime-smoke here is driven by the prestarted container + the
        # built-in harness (the ``inference`` payload and the runtime-smoke job),
        # NOT the in-executor runtime_smoke path. So the model's runtime_smoke
        # block stays disabled — otherwise preflight would demand an in-executor
        # ``runtime_smoke.python``/VLLM_PYTHON_ENV that this flow never uses.
        model["runtime_smoke"] = {"enabled": False}
        if "inference" in stages:
            inference = validate_inference_config(
                request.get("inference"), policy=self.policy
            )
            if inference is None:
                raise ValidationError(
                    "inference container config is required when the inference "
                    "stage is selected"
                )

        evaluation_command = None
        if "evaluate" in stages:
            if inference is None:
                # Defended earlier in preview(), but keep the invariant local:
                # the eval harness targets the runtime-smoke container's local
                # endpoint, so there is no port to hit without inference.
                raise ValidationError("evaluate requires inference")
            evaluation_command = build_eval_command(
                request, policy=self.policy, port=int(inference["port"])
            )
            model["evaluation"] = {
                "command": evaluation_command,
                "result_file": "{reports_dir}/evaluation.json",
                "runtime_revision": revision,
            }

        if upload is not None:
            # Recorded so an operator can publish from the run page. The lane
            # itself stays out of the pipeline; see the ``publish_trigger``
            # marker the preview sets on the plan.
            model["upload"] = upload

        return model, inference, evaluation_command

    def preview(self, request: dict[str, Any]) -> dict[str, Any]:
        """Build an execution plan without consuming any GPU capacity."""

        if not isinstance(request, dict):
            raise ValidationError("request body must be a JSON object")
        if request.get("script") is not None:
            return self._preview_script_first(request)
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

        skip_stages = self._validate_skip_stages(request.get("skip_stages"))

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
        # Persist the operator's skip choice per selected model so later launches
        # (start/retry/rerun) all read the same source of truth.
        if skip_stages:
            plan["skip_stages"] = {
                job["id"]: list(skip_stages) for job in plan["selected"]
            }
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
                        "runtime smoke stage will be skipped"
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
