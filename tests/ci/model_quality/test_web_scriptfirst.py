from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ci.model_quality.web import (
    ControlPlane,
    DEFAULT_UPLOAD_REMOTE_PREFIX,
    DryRunExecutor,
    LaunchPolicy,
    PlanRegistry,
    PlanService,
    Principal,
    RunStore,
    ValidationError,
    build_script_quantization_argv,
    introspect_script,
)
from ci.model_quality.web.control import _publish_is_scheduled
from ci.model_quality.web.worker import execution_argv

OPERATOR = Principal(actor="alice", roles=frozenset({"operator"}), authenticated=True)
STARTER = Principal(
    actor="carol",
    roles=frozenset({"operator", "publisher", "admin"}),
    authenticated=True,
)
SCRIPT = "ci/model_quality/entrypoints/qwen3_dense_w4a8.py"
INFERENCE_POLICY = LaunchPolicy(
    inference_containers=("demo-container",),
    inference_script_roots=(Path("/workspace"),),
)


def _service(tmp_path: Path, policy: LaunchPolicy | None = None) -> PlanService:
    return PlanService(
        config_path="ci/model_quality/config/models.yaml",
        registry=PlanRegistry(root=tmp_path / "plans"),
        policy=policy or LaunchPolicy(),
        git_sha="git:abc1234",
    )


# --------------------------------------------------------------- introspection


def test_introspect_parses_argparse_entrypoint() -> None:
    result = introspect_script(SCRIPT, policy=LaunchPolicy())
    assert result["parseable"] is True
    params = {param["flag"]: param for param in result["parameters"]}
    assert params["--model-id"]["required"] is True
    assert params["--num-calibration-samples"]["type"] == "int"
    assert params["--num-calibration-samples"]["default"] == 16
    assert params["--checkpoint-progress"]["type"] == "bool"
    assert params["--checkpoint-progress"]["action"] == "store_true"


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../escape.py", "ci/model_quality/config/models.yaml"],
)
def test_introspect_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        introspect_script(path, policy=LaunchPolicy())


def test_introspect_marks_non_argparse_unparseable(tmp_path: Path) -> None:
    script = Path("ci/model_quality/__init__.py")
    result = introspect_script(str(script), policy=LaunchPolicy())
    assert result["parseable"] is False
    assert result["parameters"] == []


def test_build_script_argv_forces_managed_dirs_and_fills_defaults() -> None:
    introspected = introspect_script(SCRIPT, policy=LaunchPolicy())
    argv, source = build_script_quantization_argv(
        SCRIPT,
        [{"flag": "--model-id", "value": "Qwen/Qwen3-8B"}],
        introspected,
        policy=LaunchPolicy(),
    )
    assert argv[:2] == ["python3", SCRIPT]
    # CI-managed directories are forced to run-scoped placeholders.
    assert "{output_dir}" in argv and "{work_dir}" in argv
    # Omitted non-required params fall back to their argparse defaults.
    assert "--num-calibration-samples" in argv
    assert source == "Qwen/Qwen3-8B"


def test_build_script_argv_rejects_undeclared_flag() -> None:
    introspected = introspect_script(SCRIPT, policy=LaunchPolicy())
    with pytest.raises(ValidationError, match="unknown script parameter"):
        build_script_quantization_argv(
            SCRIPT,
            [{"flag": "--rm", "value": "-rf"}],
            introspected,
            policy=LaunchPolicy(),
        )


def test_boolean_optional_action_emits_valueless_flags() -> None:
    """``argparse.BooleanOptionalAction`` flags must render as --flag/--no-flag.

    Regression: they were parsed as value-taking options, so the argv carried a
    stray ``true``/``false`` token the script rejected with "unrecognized
    arguments".
    """

    script = "examples/quantizing_moe/glm5_w8a8.py"
    introspected = introspect_script(script, policy=LaunchPolicy())
    params = {param["flag"]: param for param in introspected["parameters"]}
    assert params["--skip_restore_from_accelerate"]["action"] == "BooleanOptionalAction"
    assert params["--skip_restore_from_accelerate"]["type"] == "bool"
    argv, _ = build_script_quantization_argv(
        script,
        [
            {"flag": "--model_id", "value": "/models/GLM"},
            {"flag": "--skip_restore_from_accelerate", "value": "true"},
            {"flag": "--print_autosmooth_matches", "value": "false"},
            {"flag": "--moe-calibrate-all-experts", "value": "false"},
        ],
        introspected,
        policy=LaunchPolicy(),
    )
    assert "--skip_restore_from_accelerate" in argv
    assert "--no-print_autosmooth_matches" in argv
    assert "--no-moe-calibrate-all-experts" in argv
    assert "true" not in argv and "false" not in argv
    # A nargs list default (``--dataset_id`` is nargs="+") must be emitted as a
    # flag followed by each element, not a stringified Python list.
    assert argv[argv.index("--dataset_id") + 1] == "HuggingFaceH4/ultrachat_200k"


# ------------------------------------------------------------------- preview


def test_preview_script_first_derives_mode_and_skips(tmp_path: Path) -> None:
    service = _service(tmp_path, INFERENCE_POLICY)
    preview = service.preview(
        {
            "script": SCRIPT,
            "parameters": [{"flag": "--model-id", "value": "Qwen/Qwen3-8B"}],
            "stages": ["quantize", "validate", "inference", "evaluate"],
            "resources": {"gpu_count": 1, "estimated_gpu_hours": 2},
            "inference": {
                "container_name": "demo-container",
                "script": "/workspace/smoke.sh",
            },
            "evaluation_tool": "lm_eval",
            "evaluation_dataset": "gsm8k",
        }
    )
    plan = preview["plan"]
    model = plan["generated_config"]["models"][0]
    assert plan["run_mode"] == "quantize_and_eval"
    assert model["workflow"]["revision"].startswith("sha256:")
    assert model["workflow"]["quantize"][:2] == ["python3", SCRIPT]
    assert model["evaluation"]["command"][0] == "lm_eval"
    # container-path smoke ⇒ model runtime_smoke stays disabled (no in-executor python)
    assert model["runtime_smoke"]["enabled"] is False
    # inference + evaluate selected ⇒ nothing skipped.
    assert "skip_stages" not in plan or not plan["skip_stages"]
    assert not [
        risk for risk in preview["review"]["risks"] if risk["severity"] == "error"
    ]


def test_preview_script_first_skips_unselected_inference(tmp_path: Path) -> None:
    service = _service(tmp_path)
    preview = service.preview(
        {
            "script": SCRIPT,
            "parameters": [{"flag": "--model-id", "value": "Qwen/Qwen3-8B"}],
            "stages": ["quantize", "validate"],
            "resources": {"gpu_count": 1, "estimated_gpu_hours": 2},
        }
    )
    plan = preview["plan"]
    model_id = plan["generated_config"]["models"][0]["id"]
    assert plan["run_mode"] == "quantize"
    assert set(plan["skip_stages"][model_id]) == {"runtime-smoke", "evaluate"}


def test_preview_stores_source_request_for_cloning(tmp_path: Path) -> None:
    service = _service(tmp_path, INFERENCE_POLICY)
    preview = service.preview(
        {
            "script": SCRIPT,
            "parameters": [{"flag": "--model-id", "value": "Qwen/Qwen3-8B"}],
            "stages": ["quantize", "validate", "inference"],
            "resources": {"gpu_count": 1, "estimated_gpu_hours": 2},
            "inference": {
                "container_name": "demo-container",
                "script": "/workspace/smoke.sh",
                "port": 30000,
            },
        }
    )
    src = preview["plan"]["source_request"]
    assert src["flow"] == "script_first"
    assert src["script"] == SCRIPT
    assert src["parameters"] == [{"flag": "--model-id", "value": "Qwen/Qwen3-8B"}]
    assert "inference" in src["stages"]
    assert src["inference"]["container_name"] == "demo-container"
    assert src["inference"]["port"] == 30000


def test_preview_records_upload_target_for_manual_publish(tmp_path: Path) -> None:
    """The plan records where publishing may go, but never schedules it."""

    service = _service(tmp_path)
    preview = service.preview(
        {
            "script": SCRIPT,
            "parameters": [{"flag": "--model-id", "value": "Qwen/Qwen3-8B"}],
            "stages": ["quantize", "validate"],
            "resources": {"gpu_count": 1, "estimated_gpu_hours": 2},
            "upload": {"remote_prefix": DEFAULT_UPLOAD_REMOTE_PREFIX},
        }
    )
    plan = preview["plan"]
    model = plan["generated_config"]["models"][0]
    upload = model["upload"]

    assert upload["enabled"] is True
    assert upload["remote_prefix"] == DEFAULT_UPLOAD_REMOTE_PREFIX
    assert upload["allowlist"] == ["model", "reports"]
    assert all(command[0] == "bcecmd" for command in upload["commands"])
    assert all(command[0] == "bcecmd" for command in upload["verify_commands"])
    assert plan["publish_trigger"] == "manual"
    # The generated plan knows the model can be published ...
    assert plan["selected"][0]["upload_enabled"] is True
    # ... but the pipeline never runs the lane for it.
    assert _publish_is_scheduled(plan, plan["selected"][0], "quantize") is False
    assert plan["source_request"]["upload"] == {
        "remote_prefix": DEFAULT_UPLOAD_REMOTE_PREFIX
    }


def test_preview_rejects_upload_prefix_outside_the_reviewed_set(
    tmp_path: Path,
) -> None:
    service = _service(
        tmp_path, LaunchPolicy(upload_remote_prefixes=("bos:/reviewed/only",))
    )
    with pytest.raises(ValidationError, match="remote_prefix must be one of"):
        service.preview(
            {
                "script": SCRIPT,
                "parameters": [{"flag": "--model-id", "value": "Qwen/Qwen3-8B"}],
                "stages": ["quantize"],
                "resources": {"gpu_count": 1, "estimated_gpu_hours": 2},
                "upload": {"remote_prefix": "bos:/someone-else/bucket"},
            }
        )


@pytest.mark.parametrize(
    "upload, message",
    [
        ({"remote_prefix": "bos:/replace-me/x"}, "placeholder"),
        ({"remote_prefix": "s3:/nope"}, r"must look like bos:/"),
        ({"remote_prefix": ""}, "remote_prefix is required"),
        ({"bucket": "x"}, "accepts only remote_prefix"),
    ],
)
def test_preview_rejects_malformed_upload_target(
    tmp_path: Path, upload: dict, message: str
) -> None:
    service = _service(tmp_path)
    with pytest.raises(ValidationError, match=message):
        service.preview(
            {
                "script": SCRIPT,
                "parameters": [{"flag": "--model-id", "value": "Qwen/Qwen3-8B"}],
                "stages": ["quantize"],
                "resources": {"gpu_count": 1, "estimated_gpu_hours": 2},
                "upload": upload,
            }
        )


def test_preview_rejects_publish_as_a_scheduled_stage(tmp_path: Path) -> None:
    service = _service(tmp_path)
    with pytest.raises(ValidationError, match="publish is never scheduled"):
        service.preview(
            {
                "script": SCRIPT,
                "parameters": [{"flag": "--model-id", "value": "Qwen/Qwen3-8B"}],
                "stages": ["quantize", "publish"],
                "resources": {"gpu_count": 1, "estimated_gpu_hours": 2},
            }
        )


def test_preview_rejects_evaluate_without_inference(tmp_path: Path) -> None:
    service = _service(tmp_path)
    with pytest.raises(ValidationError, match="evaluate requires inference"):
        service.preview(
            {
                "script": SCRIPT,
                "parameters": [{"flag": "--model-id", "value": "Qwen/Qwen3-8B"}],
                "stages": ["quantize", "evaluate"],
                "resources": {"gpu_count": 1, "estimated_gpu_hours": 2},
            }
        )


def test_evaluate_builds_informational_harness_command(tmp_path: Path) -> None:
    service = _service(tmp_path, INFERENCE_POLICY)
    preview = service.preview(
        {
            "script": SCRIPT,
            "parameters": [{"flag": "--model-id", "value": "Qwen/Qwen3-8B"}],
            "stages": ["quantize", "validate", "inference", "evaluate"],
            "resources": {"gpu_count": 1, "estimated_gpu_hours": 2},
            "inference": {
                "container_name": "demo-container",
                "script": "/workspace/smoke.sh",
                "port": 30000,
            },
            "evaluation_tool": "evalscope",
            "evaluation_dataset": "ceval",
        }
    )
    model = preview["plan"]["generated_config"]["models"][0]
    command = model["evaluation"]["command"]
    assert "ci.model_quality.evaluators.informational" in command
    assert command[command.index("--tool") + 1] == "evalscope"
    assert command[command.index("--dataset") + 1] == "ceval"
    # The eval harness targets the runtime-smoke container's local port — never
    # a user-supplied URL.
    assert command[command.index("--api-url") + 1] == "http://127.0.0.1:30000"
    assert "{reports_dir}/evaluation.json" in command
    # evaluate selected ⇒ runtime-smoke runs (not skipped).
    skip = preview["plan"].get("skip_stages", {})
    assert all("runtime-smoke" not in stages for stages in skip.values())


def _script_first_inference(tmp_path: Path, arguments: list[str]) -> dict:
    service = _service(tmp_path, INFERENCE_POLICY)
    return service.preview(
        {
            "script": SCRIPT,
            "parameters": [{"flag": "--model-id", "value": "Qwen/Qwen3-8B"}],
            "stages": ["quantize", "validate", "inference"],
            "resources": {"gpu_count": 1, "estimated_gpu_hours": 2},
            "inference": {
                "container_name": "demo-container",
                "script": "/workspace/smoke.sh",
                "arguments": arguments,
            },
        }
    )


def test_inference_defaults_port_and_no_contract_warning(tmp_path: Path) -> None:
    preview = _script_first_inference(tmp_path, [])
    model = preview["plan"]["generated_config"]["models"][0]
    assert model["runtime_smoke"]["enabled"] is False
    assert preview["inference"]["port"] == 8025
    codes = {r["code"] for r in preview["review"]["risks"]}
    # The env-injection harness replaced the old argv-placeholder contract warning.
    assert "INFERENCE_SCRIPT_CONTRACT" not in codes


def test_runtime_smoke_execution_argv_wraps_serve_script_with_harness(
    tmp_path: Path,
) -> None:
    job = {
        "run_id": "run-1",
        "model_id": "m",
        "attempt_id": "a",
        "container_name": "serving-container",
        "script": "/workspace/run_server.sh",
        "details": {"inference_arguments": [], "inference_port": 9100},
    }
    (tmp_path / "run-1" / "m").mkdir(parents=True)
    argv = execution_argv(
        job,
        "runtime-smoke",
        root=tmp_path,
        config_path="global.yaml",
        argv_prefix=("python3",),
        container_runtime=("direct",),
    )
    assert argv[0] == "bash"
    assert argv[1].endswith("entrypoints/runtime_smoke_harness.sh")
    # harness args: serve_script, port, reports_dir, output_dir
    assert argv[2] == "/workspace/run_server.sh"
    assert argv[3] == "9100"
    assert argv[4].endswith("/run-1/m/attempts/a/reports")
    assert argv[5].endswith("/run-1/m/model")


# -------------------------------------------------------------------- launch


def _control(env: SimpleNamespace, policy: LaunchPolicy | None = None) -> ControlPlane:
    return ControlPlane(
        root=env.root,
        store=RunStore(env.root),
        config_path=env.config,
        policy=policy or LaunchPolicy(),
        git_sha="git:abc1234",
        executor=DryRunExecutor(env.root),
    )


def test_launch_writes_run_local_config(web_env) -> None:
    control = _control(web_env)
    preview = control.preview_plan(
        {
            "script": SCRIPT,
            "parameters": [{"flag": "--model-id", "value": "Qwen/Qwen3-8B"}],
            "stages": ["quantize", "validate"],
            "resources": {"gpu_count": 1, "estimated_gpu_hours": 2},
        },
        OPERATOR,
    )
    run = control.start_run({"plan_hash": preview["plan_hash"]}, STARTER)
    run_dir = web_env.root / run["run_id"]
    config_file = run_dir / "model-config.yaml"
    assert config_file.is_file()
    manifest = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["models"][0]["workflow"]["quantize"][:2] == ["python3", SCRIPT]


def test_launch_without_inference_skips_runtime_smoke(web_env) -> None:
    """A container-registering deployment still launches a quantize-only run.

    Regression: ``start_run`` used to 400 with "inference.container_name ... are
    required" whenever the deployment registered inference containers, so the UI
    "Confirm and launch" button silently failed. The run now launches and the
    runtime-smoke stage is recorded SKIPPED instead of blocking.
    """

    scripts = web_env.tmp_path / "scripts"
    scripts.mkdir(exist_ok=True)
    policy = LaunchPolicy(
        gpu_hour_capacity=80.0,
        inference_containers=("user-runtime",),
        inference_script_roots=(scripts,),
    )
    control = _control(web_env, policy=policy)
    preview = control.preview_plan(
        {
            "script": SCRIPT,
            "parameters": [{"flag": "--model-id", "value": "Qwen/Qwen3-8B"}],
            "stages": ["quantize", "validate"],
            "resources": {"gpu_count": 1, "estimated_gpu_hours": 2},
        },
        OPERATOR,
    )
    # Missing inference is only a warning, so the launch is not blocked.
    assert not [
        risk
        for risk in preview["review"]["risks"]
        if risk["severity"] == "error"
    ]
    run = control.start_run({"plan_hash": preview["plan_hash"]}, STARTER)
    # No inference/runtime-smoke job is scheduled ...
    assert all(job["kind"] != "inference" for job in run["jobs"])
    # ... and runtime-smoke is recorded SKIPPED for the model.
    model_id = preview["plan"]["generated_config"]["models"][0]["id"]
    state = web_env.root / run["run_id"] / model_id / "state" / "runtime-smoke.json"
    assert yaml.safe_load(state.read_text(encoding="utf-8"))["status"] == "SKIPPED"


# -------------------------------------------------------------- worker seam


def test_execution_argv_prefers_run_local_config(tmp_path: Path) -> None:
    job = {"run_id": "run-1", "model_id": "m", "attempt_id": "a", "details": {}}
    run_dir = tmp_path / "run-1"
    run_dir.mkdir(parents=True)

    # No run-local config yet → the global config is used.
    argv = execution_argv(
        job,
        "quantize",
        root=tmp_path,
        config_path="global.yaml",
        argv_prefix=("python3", "-m", "ci.model_quality.stage"),
        container_runtime=("direct",),
    )
    assert "global.yaml" in argv

    (run_dir / "model-config.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    argv = execution_argv(
        job,
        "quantize",
        root=tmp_path,
        config_path="global.yaml",
        argv_prefix=("python3", "-m", "ci.model_quality.stage"),
        container_runtime=("direct",),
    )
    assert str(run_dir / "model-config.yaml") in argv
    assert "global.yaml" not in argv
