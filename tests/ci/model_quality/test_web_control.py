from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ci.model_quality.state import atomic_write_json
from ci.model_quality.web import (
    ConflictError,
    ControlPlane,
    DryRunExecutor,
    JobStore,
    LaunchPolicy,
    NotFoundError,
    PermissionDenied,
    PlanRegistry,
    PlanService,
    Principal,
    ReservationLedger,
    RunStore,
    ValidationError,
    build_quantization_argv,
    create_app,
    import_quantization_command,
    validate_evaluation_command,
    validate_inference_config,
)

OPERATOR = Principal(actor="alice", roles=frozenset({"operator"}), authenticated=True)
VIEWER = Principal(actor="bob", roles=frozenset({"viewer"}), authenticated=True)


def make_control(env, *, capacity: float = 80.0, policy: LaunchPolicy | None = None):
    policy = policy or LaunchPolicy(gpu_hour_capacity=capacity)
    return ControlPlane(
        root=env.root,
        store=RunStore(env.root),
        config_path=env.config,
        policy=policy,
        git_sha="git:abc1234",
        executor=DryRunExecutor(env.root),
    )


def call(app, method: str, path: str, body=None, headers=None, query: str = ""):
    payload = json.dumps(body).encode("utf-8") if body is not None else b""
    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "QUERY_STRING": query,
        "CONTENT_LENGTH": str(len(payload)),
        "wsgi.input": io.BytesIO(payload),
    }
    for key, value in (headers or {}).items():
        environ[f"HTTP_{key.upper().replace('-', '_')}"] = value
    captured: dict = {}

    def start(status, response_headers):
        captured.update(status=status, headers=dict(response_headers))

    output = b"".join(app(environ, start))
    return captured, (json.loads(output) if output else None)


def test_preview_is_read_only_and_returns_review(web_env) -> None:
    control = make_control(web_env)
    preview = control.preview_plan(
        {"run_mode": "quantize_and_eval", "models": "all", "max_gpu_hours": 10},
        OPERATOR,
    )

    assert preview["plan_hash"]
    assert [job["id"] for job in preview["plan"]["selected"]] == [web_env.model_id]
    assert preview["plan"]["budget"]["selected_gpu_hours"] == 3.0
    review = preview["review"]
    assert review["config"]["fingerprint"]
    assert review["config"]["path"] == str(web_env.config)
    assert review["estimated_max_hours"] == 3.0
    assert review["upload_auto_publish"] is False
    assert any("does not consume GPU" in notice for notice in review["notices"])
    assert {risk["code"] for risk in review["risks"]} == {"NO_INFERENCE_CONTAINER"}

    assert control.ledger.summary()["reserved_gpu_hours"] == 0.0
    assert control.jobs.list_jobs(preview["plan"]["run_id"]) == []
    record = control.plan_service.registry.load(preview["plan_hash"])
    assert record["consumed_by"] is None
    assert control.audit.read(action="plan.preview")["total"] == 1


def test_preview_and_start_require_permissions(web_env) -> None:
    control = make_control(web_env)
    with pytest.raises(PermissionDenied):
        control.preview_plan({"run_mode": "quantize"}, VIEWER)
    preview = control.preview_plan({"run_mode": "quantize"}, OPERATOR)
    with pytest.raises(PermissionDenied):
        control.start_run({"plan_hash": preview["plan_hash"]}, VIEWER)


def test_start_run_reserves_capacity_and_is_idempotent(web_env) -> None:
    control = make_control(web_env, capacity=8.0)
    preview = control.preview_plan(
        {"run_mode": "quantize_and_eval", "max_gpu_hours": 8}, OPERATOR
    )
    request = {
        "plan_hash": preview["plan_hash"],
        "idempotency_key": "start-demo-0001",
    }
    response = control.start_run(request, OPERATOR)

    assert response["status"] == "QUEUED"
    assert response["executor"]["mode"] == "dry-run"
    kinds = sorted(job["kind"] for job in response["jobs"])
    assert kinds == ["evaluate", "inference", "quantize", "report"]
    assert response["budget"]["reserved_gpu_hours"] == 3.0
    assert response["budget"]["available_gpu_hours"] == 5.0

    quantize = next(job for job in response["jobs"] if job["kind"] == "quantize")
    assert quantize["stages"] == ["preflight", "quantize", "validate"]
    inference = next(job for job in response["jobs"] if job["kind"] == "inference")
    assert inference["stages"] == ["runtime-smoke"]
    report = next(job for job in response["jobs"] if job["kind"] == "report")
    assert report["stages"] == ["report"]
    assert quantize["command"][0] == "python3"
    assert quantize["gpu_hours"] == 3.0
    assert quantize["status"] == "QUEUED"

    replay = control.start_run(request, OPERATOR)
    assert replay["jobs"][0]["job_id"] == response["jobs"][0]["job_id"]
    assert control.ledger.summary()["reserved_gpu_hours"] == 3.0
    assert len(control.jobs.list_jobs(response["run_id"])) == 4

    run_dir = web_env.root / response["run_id"]
    plan = json.loads((run_dir / "execution-plan.json").read_text())
    assert plan["run_id"] == response["run_id"]
    run_record = json.loads((run_dir / "run.json").read_text())
    assert run_record["created_by"] == "alice"
    assert run_record["attempts"][0]["reason"] == "start"
    assert control.audit.read(action="run.start")["total"] == 1

    with pytest.raises(ConflictError, match="already launched"):
        control.start_run(
            {"plan_hash": preview["plan_hash"], "idempotency_key": "start-demo-0002"},
            OPERATOR,
        )


def test_start_run_blocks_on_error_risks(tmp_path: Path) -> None:
    import yaml

    from tests.ci.model_quality.conftest import _model, _source, _write_manifest

    source = _source(tmp_path)
    config = _write_manifest(
        tmp_path,
        [
            _model(
                model_id="unreviewed",
                source=source,
                workflow_revision="replace-with-entrypoint-commit",
            )
        ],
    )
    env = type("Env", (), {"root": tmp_path / "runs", "config": config})()
    control = make_control(env)
    preview = control.preview_plan({"run_mode": "quantize"}, OPERATOR)
    codes = {
        risk["code"]
        for risk in preview["review"]["risks"]
        if risk["severity"] == "error"
    }
    assert "UNPINNED_WORKFLOW_REVISION" in codes
    with pytest.raises(ConflictError, match="blocking risks"):
        control.start_run({"plan_hash": preview["plan_hash"]}, OPERATOR)
    assert yaml is not None


def test_admission_rejects_when_capacity_is_exhausted(web_env) -> None:
    control = make_control(web_env, capacity=1.0)
    preview = control.preview_plan({"run_mode": "quantize_and_eval"}, OPERATOR)
    with pytest.raises(ConflictError, match="admission rejected"):
        control.start_run({"plan_hash": preview["plan_hash"]}, OPERATOR)
    assert control.ledger.summary()["reserved_gpu_hours"] == 0.0
    assert control.jobs.list_jobs(preview["plan"]["run_id"]) == []


def test_retry_creates_new_attempt_from_failed_stage(web_env) -> None:
    control = make_control(web_env)
    preview = control.preview_plan({"run_mode": "quantize_and_eval"}, OPERATOR)
    started = control.start_run({"plan_hash": preview["plan_hash"]}, OPERATOR)
    model_dir = web_env.root / started["run_id"] / web_env.model_id
    atomic_write_json(
        model_dir / "state" / "validate.json",
        {"status": "FAIL", "reason_code": "VALIDATION_FAILED"},
    )

    retried = control.retry(started["run_id"], {"from_stage": "validate"}, OPERATOR)
    assert retried["reason"] == "retry"
    assert retried["attempt_id"] != started["attempt_id"]
    stages = {stage for job in retried["jobs"] for stage in job["stages"]}
    assert stages == {"validate", "runtime-smoke", "evaluate", "report"}
    attempts = json.loads((web_env.root / started["run_id"] / "run.json").read_text())[
        "attempts"
    ]
    assert [attempt["reason"] for attempt in attempts] == ["start", "retry"]
    assert retried["reservations"][0]["gpu_hours"] == 3.0


def test_retry_without_failed_stage_conflicts(web_env) -> None:
    control = make_control(web_env)
    preview = control.preview_plan({"run_mode": "quantize"}, OPERATOR)
    started = control.start_run({"plan_hash": preview["plan_hash"]}, OPERATOR)
    with pytest.raises(ConflictError, match="no failed stage"):
        control.retry(started["run_id"], {}, OPERATOR)


def test_retry_restores_stored_inference(web_env) -> None:
    scripts = web_env.tmp_path / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "serve.sh").write_text("#!/bin/bash\n")
    policy = LaunchPolicy(
        gpu_hour_capacity=80.0,
        inference_containers=("user-runtime",),
        inference_script_roots=(scripts,),
    )
    control = make_control(web_env, policy=policy)
    inference = {
        "container_name": "user-runtime",
        "script": str(scripts / "serve.sh"),
        "port": 30000,
    }
    preview = control.preview_plan(
        {"run_mode": "quantize", "inference": inference}, OPERATOR
    )
    started = control.start_run({"plan_hash": preview["plan_hash"]}, OPERATOR)
    model_dir = web_env.root / started["run_id"] / web_env.model_id
    atomic_write_json(
        model_dir / "state" / "runtime-smoke.json",
        {"status": "FAIL", "reason_code": "SCRIPT_FAILED"},
    )
    # A plain retry (empty body) must restore the stored container config.
    retried = control.retry(started["run_id"], {}, OPERATOR)
    smoke = next(job for job in retried["jobs"] if job["kind"] == "inference")
    assert smoke["container_name"] == "user-runtime"
    assert smoke["details"]["inference_port"] == 30000


def test_evaluate_requires_passing_validate(web_env) -> None:
    control = make_control(web_env)
    preview = control.preview_plan({"run_mode": "quantize"}, OPERATOR)
    started = control.start_run({"plan_hash": preview["plan_hash"]}, OPERATOR)
    run_id = started["run_id"]

    with pytest.raises(ConflictError, match="passing validate"):
        control.evaluate(run_id, {}, OPERATOR)

    model_dir = web_env.root / run_id / web_env.model_id
    atomic_write_json(model_dir / "state" / "validate.json", {"status": "PASS"})
    evaluated = control.evaluate(run_id, {}, OPERATOR)
    assert evaluated["run_mode"] == "eval_only"
    stages = {stage for job in evaluated["jobs"] for stage in job["stages"]}
    assert "evaluate" in stages
    assert "quantize" not in stages
    assert "publish" not in stages
    assert evaluated["reservations"][0]["gpu_hours"] == 1.0


def test_cancel_marks_jobs_and_releases_reservations(web_env) -> None:
    control = make_control(web_env)
    preview = control.preview_plan({"run_mode": "quantize_and_eval"}, OPERATOR)
    started = control.start_run({"plan_hash": preview["plan_hash"]}, OPERATOR)

    result = control.cancel(
        started["run_id"], {"reason": "operator stopped it"}, OPERATOR
    )
    assert result["status"] == "CANCELED"
    assert len(result["canceled_jobs"]) == 4
    assert control.ledger.summary()["reserved_gpu_hours"] == 0.0
    assert control.ledger.summary()["released_gpu_hours"] == 3.0
    marker = json.loads((web_env.root / started["run_id"] / "cancel.json").read_text())
    assert marker["canceled_by"] == "alice"
    assert marker["reason"] == "operator stopped it"
    assert control.audit.read(action="run.cancel")["total"] == 1


def test_cancel_before_first_stage_reports_canceled(web_env) -> None:
    """A run canceled before any stage records must not keep reporting PENDING."""

    control = make_control(web_env)
    preview = control.preview_plan({"run_mode": "quantize_and_eval"}, OPERATOR)
    started = control.start_run({"plan_hash": preview["plan_hash"]}, OPERATOR)
    run_id = started["run_id"]

    store = RunStore(web_env.root)
    assert store.get_run(run_id)["status"] == "PLANNED"
    assert store.list_runs()["items"][0]["run_id"] == run_id

    control.cancel(run_id, {"reason": "operator stopped it"}, OPERATOR)

    assert store.get_run(run_id)["status"] == "CANCELED"
    assert store.get_model(run_id, web_env.model_id)["status"] == "CANCELED"
    listed = store.list_runs()["items"][0]
    assert listed["run_id"] == run_id
    assert listed["status"] == "CANCELED"
    assert listed["updated_at"] is not None


def test_retry_job_requires_retryable_failure(web_env) -> None:
    control = make_control(web_env)
    preview = control.preview_plan({"run_mode": "quantize"}, OPERATOR)
    started = control.start_run({"plan_hash": preview["plan_hash"]}, OPERATOR)
    job = started["jobs"][0]

    with pytest.raises(ConflictError, match="cannot be retried"):
        control.retry_job(
            started["run_id"], web_env.model_id, job["job_id"], {}, OPERATOR
        )

    control.jobs.update(
        job, status="FAILED", failure_class="RESOURCE_TIMEOUT", exit_code=124
    )
    retried = control.retry_job(
        started["run_id"], web_env.model_id, job["job_id"], {}, OPERATOR
    )
    assert retried["job"]["job_id"] != job["job_id"]
    assert retried["job"]["status"] == "QUEUED"
    assert retried["job"]["details"]["retried_from"] == job["job_id"]
    assert (
        control.jobs.get(started["run_id"], web_env.model_id, job["job_id"])["status"]
        == "FAILED"
    )

    original = control.jobs.get(started["run_id"], web_env.model_id, job["job_id"])
    control.jobs.update(original, status="FAILED", failure_class="QUALITY_GATE_FAILED")
    with pytest.raises(ConflictError, match="not retryable"):
        control.retry_job(
            started["run_id"], web_env.model_id, job["job_id"], {}, OPERATOR
        )


def test_cancel_job_does_not_touch_the_container(web_env) -> None:
    control = make_control(web_env)
    preview = control.preview_plan({"run_mode": "quantize"}, OPERATOR)
    started = control.start_run({"plan_hash": preview["plan_hash"]}, OPERATOR)
    inference = next(job for job in started["jobs"] if job["kind"] == "inference")

    result = control.cancel_job(
        started["run_id"], web_env.model_id, inference["job_id"], {}, OPERATOR
    )
    assert result["job"]["status"] == "CANCELED"
    assert result["inference_container_action"] == "none"
    assert control.ledger.summary()["reserved_gpu_hours"] == 2.0
    assert control.audit.read(action="job.cancel")["total"] == 1


def test_inference_config_is_allowlisted(tmp_path: Path) -> None:
    roots = tmp_path / "scripts"
    roots.mkdir()
    policy = LaunchPolicy(
        inference_containers=("user-runtime",),
        inference_script_roots=(roots,),
    )
    valid = validate_inference_config(
        {
            "container_name": "user-runtime",
            "script": str(roots / "run_inference_smoke.sh"),
            "arguments": ["{output_dir}", "{reports_dir}", "{run_id}"],
            "result_file": "{reports_dir}/runtime-smoke-output.json",
        },
        policy=policy,
    )
    assert valid["argv_preview"][0].endswith("run_inference_smoke.sh")
    assert valid["timeout_minutes"] == 60
    assert valid["port"] == 8025

    # A local service port is now allowed (probed only on 127.0.0.1).
    with_port = validate_inference_config(
        {
            "container_name": "user-runtime",
            "script": str(roots / "run_inference_smoke.sh"),
            "port": 9001,
        },
        policy=policy,
    )
    assert with_port["port"] == 9001
    with pytest.raises(ValidationError, match="between 1 and 65535"):
        validate_inference_config(
            {
                "container_name": "user-runtime",
                "script": str(roots / "x.sh"),
                "port": 70000,
            },
            policy=policy,
        )

    # Hosts/URLs remain forbidden (SSRF), even though port is allowed.
    with pytest.raises(ValidationError, match="must not configure"):
        validate_inference_config(
            {
                "container_name": "user-runtime",
                "script": str(roots / "x.sh"),
                "url": "http://evil.example/v1",
            },
            policy=policy,
        )
    with pytest.raises(ValidationError, match="not registered"):
        validate_inference_config(
            {"container_name": "other", "script": str(roots / "x.sh")}, policy=policy
        )
    with pytest.raises(ValidationError, match="absolute"):
        validate_inference_config(
            {"container_name": "user-runtime", "script": "x.sh"}, policy=policy
        )
    with pytest.raises(ValidationError, match="outside the registered"):
        validate_inference_config(
            {"container_name": "user-runtime", "script": "/opt/other/x.sh"},
            policy=policy,
        )
    with pytest.raises(ValidationError, match="unknown placeholder"):
        validate_inference_config(
            {
                "container_name": "user-runtime",
                "script": str(roots / "x.sh"),
                "arguments": ["{unknown_thing}"],
            },
            policy=policy,
        )


def test_evaluation_command_is_allowlisted() -> None:
    policy = LaunchPolicy(
        evaluation_commands=("/root/miniconda/envs/model_quality_lm_eval/bin/python",)
    )
    command = validate_evaluation_command(
        [
            "/root/miniconda/envs/model_quality_lm_eval/bin/python",
            "-m",
            "ci.model_quality.evaluators.lm_eval_api_pair",
            "--output",
            "{reports_dir}/evaluation-raw.json",
        ],
        policy=policy,
    )
    assert command[-1] == "{reports_dir}/evaluation-raw.json"

    with pytest.raises(ValidationError, match="start with"):
        validate_evaluation_command(["python", "-V"], policy=policy)
    with pytest.raises(ValidationError, match="metacharacters"):
        validate_evaluation_command(
            [policy.evaluation_commands[0], "$(whoami)"], policy=policy
        )


def test_quantization_form_and_command_import(tmp_path: Path) -> None:
    policy = LaunchPolicy()
    argv = build_quantization_argv(
        {
            "executable": "python3",
            "entrypoint": "examples/streaming_oneshot/deepseek_v4_wNa8.py",
            "model_path": "{source_path}",
            "datasets": ["calibration.jsonl", "ultrachat_200k"],
            "calibration_samples": 32,
            "max_sequence_length": 8192,
            "quant_mode": "wna8",
            "use_float32_scale_dtype": True,
        },
        policy=policy,
    )
    assert argv[:2] == ["python3", "examples/streaming_oneshot/deepseek_v4_wNa8.py"]
    assert argv[argv.index("--dataset-id") + 1 : argv.index("--dataset-id") + 3] == [
        "calibration.jsonl",
        "ultrachat_200k",
    ]
    assert argv[argv.index("--output-dir") + 1] == "{output_dir}"
    assert argv[argv.index("--work-dir") + 1] == "{work_dir}"
    assert "--use-float32-scale-dtype" in argv

    imported = import_quantization_command(
        "python3 examples/streaming_oneshot/deepseek_v4_wNa8.py "
        "--model-id {source_path} --dataset-id calibration.jsonl ultrachat_200k "
        "--num-calibration-samples 32 --max-sequence-length 8192 "
        "--quant-mode wna8 --output-dir {output_dir} --use-float32-scale-dtype",
        policy=policy,
    )
    assert imported["structured"]["calibration_samples"] == 32
    assert imported["structured"]["datasets"] == ["calibration.jsonl", "ultrachat_200k"]
    assert imported["argv"] == argv

    for bad in (
        "python3 x.py --quant-mode wna8 | tee log",
        "python3 x.py --quant-mode $(whoami)",
        "python3 x.py --quant-mode `id`",
        "python3 x.py --unknown-flag 1",
        "sh x.py --quant-mode wna8",
        "python3 --quant-mode wna8",
    ):
        with pytest.raises(ValidationError):
            import_quantization_command(bad, policy=policy)

    with pytest.raises(ValidationError, match="repo-relative"):
        build_quantization_argv({"entrypoint": "../escape.py"}, policy=policy)
    with pytest.raises(ValidationError, match="output root policy"):
        build_quantization_argv(
            {
                "entrypoint": "x.py",
                "output_dir": "/tmp/somewhere",
            },
            policy=policy,
        )


def test_plan_registry_is_single_use(web_env) -> None:
    registry = PlanRegistry(web_env.root, ttl_seconds=60)
    service = PlanService(
        config_path=web_env.config,
        registry=registry,
        policy=LaunchPolicy(),
        git_sha="git:abc1234",
    )
    preview = service.preview({"run_mode": "quantize"})
    service.registry.consume(preview["plan_hash"])
    with pytest.raises(ConflictError, match="already launched"):
        service.registry.consume(preview["plan_hash"])
    with pytest.raises(ValidationError, match="64 character hex"):
        service.registry.load("not-a-hash")


def test_wsgi_write_routes_enforce_roles(web_env) -> None:
    app = create_app(
        RunStore(web_env.root), config_path=web_env.config, git_sha="git:abc1234"
    )
    body = {"run_mode": "quantize", "max_gpu_hours": 4}

    status, payload = call(app, "POST", "/api/plans/preview", body)
    assert status["status"] == "403 Forbidden"
    assert "plan" in payload["error"]

    status, payload = call(
        app,
        "POST",
        "/api/plans/preview",
        body,
        headers={"X-Model-Quality-Actor": "bob"},
    )
    assert status["status"] == "403 Forbidden"

    headers = {"X-Model-Quality-Actor": "alice", "X-Model-Quality-Roles": "operator"}
    status, payload = call(app, "POST", "/api/plans/preview", body, headers=headers)
    assert status["status"] == "200 OK"
    assert payload["plan_hash"]

    status, payload = call(
        app, "POST", "/api/runs", {"plan_hash": payload["plan_hash"]}, headers=headers
    )
    assert status["status"] == "200 OK"
    assert payload["status"] == "QUEUED"

    status, payload = call(app, "GET", f"/api/runs/{payload['run_id']}/jobs")
    assert status["status"] == "200 OK"
    assert payload["total"] == 3

    status, payload = call(app, "POST", "/api/whoami", {}, headers=headers)
    assert status["status"] != "200 OK"


def test_wsgi_write_routes_reject_bad_bodies(web_env) -> None:
    app = create_app(RunStore(web_env.root), config_path=web_env.config)
    headers = {"X-Model-Quality-Actor": "alice", "X-Model-Quality-Roles": "admin"}

    captured: dict = {}

    def start(status, response_headers):
        captured.update(status=status, headers=dict(response_headers))

    environ = {
        "REQUEST_METHOD": "POST",
        "PATH_INFO": "/api/plans/preview",
        "QUERY_STRING": "",
        "CONTENT_LENGTH": "7",
        "wsgi.input": io.BytesIO(b"notjson"),
    }
    environ.update({"HTTP_X_MODEL_QUALITY_ACTOR": headers["X-Model-Quality-Actor"]})
    environ.update({"HTTP_X_MODEL_QUALITY_ROLES": headers["X-Model-Quality-Roles"]})
    output = b"".join(app(environ, start))
    assert captured["status"] == "400 Bad Request"
    assert "must be JSON" in json.loads(output)["error"]


class _FailingExecutor(DryRunExecutor):
    """Dispatch always raises, to exercise the reservation-release path."""

    def dispatch(self, spec, jobs):
        raise RuntimeError("dispatch boom")


def test_retry_only_relaunches_failed_models(tmp_path: Path) -> None:
    from tests.ci.model_quality.conftest import _model, _source, _write_manifest

    source = _source(tmp_path)
    config = _write_manifest(
        tmp_path,
        [
            _model(model_id="model-a", source=source),
            _model(model_id="model-b", source=source),
        ],
    )
    env = type("Env", (), {"root": tmp_path / "runs", "config": config})()
    control = make_control(env, capacity=80.0)

    preview = control.preview_plan({"run_mode": "quantize_and_eval"}, OPERATOR)
    started = control.start_run({"plan_hash": preview["plan_hash"]}, OPERATOR)
    assert {job["model_id"] for job in started["jobs"]} == {"model-a", "model-b"}

    # Only model-a's validate stage failed; model-b passed and must not be
    # relaunched (nor re-reserved) by the retry.
    atomic_write_json(
        env.root / started["run_id"] / "model-a" / "state" / "validate.json",
        {"status": "FAIL", "reason_code": "VALIDATION_FAILED"},
    )

    retried = control.retry(started["run_id"], {"from_stage": "validate"}, OPERATOR)
    assert {job["model_id"] for job in retried["jobs"]} == {"model-a"}
    assert [record["model_id"] for record in retried["reservations"]] == ["model-a"]


def test_retry_job_releases_reservation_when_dispatch_fails(web_env) -> None:
    control = make_control(web_env, capacity=8.0)
    preview = control.preview_plan({"run_mode": "quantize"}, OPERATOR)
    started = control.start_run({"plan_hash": preview["plan_hash"]}, OPERATOR)
    job = started["jobs"][0]
    control.jobs.update(
        job, status="FAILED", failure_class="RESOURCE_TIMEOUT", exit_code=124
    )

    before = control.ledger.summary()["reserved_gpu_hours"]
    control.executor = _FailingExecutor(web_env.root)
    with pytest.raises(RuntimeError, match="dispatch boom"):
        control.retry_job(
            started["run_id"], web_env.model_id, job["job_id"], {}, OPERATOR
        )

    # The freshly reserved GPU-hours for the aborted retry must be released,
    # not leaked, so the ledger returns to its pre-retry level.
    assert control.ledger.summary()["reserved_gpu_hours"] == before


def test_job_store_and_ledger_reject_unsafe_ids(web_env) -> None:
    jobs = JobStore(web_env.root)
    ledger = ReservationLedger(web_env.root, capacity_gpu_hours=4.0)
    with pytest.raises(ValueError, match="unsafe run_id"):
        jobs.list_jobs("../escape")
    with pytest.raises(ValidationError, match="positive"):
        ReservationLedger(web_env.root, capacity_gpu_hours=0)
    with pytest.raises(ValueError, match="unsafe attempt_id"):
        ledger.reserve(
            run_id="run-a",
            model_id="model-a",
            attempt_id="../bad",
            gpu_hours=1.0,
        )


def test_rerun_stage_relaunches_a_passed_node(web_env) -> None:
    control = make_control(web_env)
    preview = control.preview_plan({"run_mode": "quantize_and_eval"}, OPERATOR)
    started = control.start_run({"plan_hash": preview["plan_hash"]}, OPERATOR)
    run_id = started["run_id"]
    # The node is reported PASS; rerun must still relaunch it (unlike retry,
    # which refuses when nothing failed).
    atomic_write_json(
        web_env.root / run_id / web_env.model_id / "state" / "validate.json",
        {"status": "PASS"},
    )

    rerun = control.rerun_stage(run_id, web_env.model_id, "validate", {}, OPERATOR)
    assert rerun["reason"] == "rerun"
    assert rerun["attempt_id"] != started["attempt_id"]
    stages = {stage for job in rerun["jobs"] for stage in job["stages"]}
    assert stages == {"validate", "runtime-smoke", "evaluate", "report"}

    attempts = json.loads((web_env.root / run_id / "run.json").read_text())["attempts"]
    assert [attempt["reason"] for attempt in attempts] == ["start", "rerun"]

    # A byte-identical request derives the same idempotency key and replays
    # rather than reserving a second attempt.
    replay = control.rerun_stage(run_id, web_env.model_id, "validate", {}, OPERATOR)
    assert replay["attempt_id"] == rerun["attempt_id"]


def test_rerun_stage_is_scoped_to_one_model(tmp_path: Path) -> None:
    from tests.ci.model_quality.conftest import _model, _source, _write_manifest

    source = _source(tmp_path)
    config = _write_manifest(
        tmp_path,
        [
            _model(model_id="model-a", source=source),
            _model(model_id="model-b", source=source),
        ],
    )
    env = type("Env", (), {"root": tmp_path / "runs", "config": config})()
    control = make_control(env, capacity=80.0)
    preview = control.preview_plan({"run_mode": "quantize_and_eval"}, OPERATOR)
    started = control.start_run({"plan_hash": preview["plan_hash"]}, OPERATOR)

    rerun = control.rerun_stage(started["run_id"], "model-a", "quantize", {}, OPERATOR)
    assert {job["model_id"] for job in rerun["jobs"]} == {"model-a"}
    assert [record["model_id"] for record in rerun["reservations"]] == ["model-a"]


def test_rerun_stage_rejects_viewer_and_unknown_stage(web_env) -> None:
    control = make_control(web_env)
    preview = control.preview_plan({"run_mode": "quantize"}, OPERATOR)
    started = control.start_run({"plan_hash": preview["plan_hash"]}, OPERATOR)
    run_id = started["run_id"]

    with pytest.raises(PermissionDenied):
        control.rerun_stage(run_id, web_env.model_id, "quantize", {}, VIEWER)
    with pytest.raises(ValidationError, match="stage must be one of"):
        control.rerun_stage(run_id, web_env.model_id, "nope", {}, OPERATOR)
    # ``evaluate`` is not part of a quantize-only run order.
    with pytest.raises(ValidationError, match="stage must be one of"):
        control.rerun_stage(run_id, web_env.model_id, "evaluate", {}, OPERATOR)
    with pytest.raises(NotFoundError, match="not part of run"):
        control.rerun_stage(run_id, "ghost-model", "quantize", {}, OPERATOR)


def test_wsgi_rerun_stage_route_enforces_roles(web_env) -> None:
    control = make_control(web_env)
    preview = control.preview_plan({"run_mode": "quantize"}, OPERATOR)
    started = control.start_run({"plan_hash": preview["plan_hash"]}, OPERATOR)
    run_id = started["run_id"]
    app = create_app(
        RunStore(web_env.root), config_path=web_env.config, git_sha="git:abc1234"
    )
    path = f"/api/runs/{run_id}/models/{web_env.model_id}/stages/quantize/rerun"

    status, _ = call(
        app, "POST", path, {}, headers={"X-Model-Quality-Actor": "bob"}
    )
    assert status["status"] == "403 Forbidden"

    status, payload = call(
        app,
        "POST",
        path,
        {"idempotency_key": "rerun-ui-1"},
        headers={
            "X-Model-Quality-Actor": "alice",
            "X-Model-Quality-Roles": "operator",
        },
    )
    assert status["status"] == "200 OK"
    assert payload["reason"] == "rerun"
    assert {job["model_id"] for job in payload["jobs"]} == {web_env.model_id}


def test_preview_persists_skip_stages_and_rejects_non_optional(web_env) -> None:
    control = make_control(web_env)
    preview = control.preview_plan(
        {"run_mode": "quantize_and_eval", "skip_stages": ["evaluate", "validate"]},
        OPERATOR,
    )
    plan = preview["plan"]
    assert plan["skip_stages"] == {web_env.model_id: ["evaluate", "validate"]}

    # A mandatory stage is not skippable and must be rejected at preview time.
    with pytest.raises(ValidationError, match="skip_stages may only contain"):
        control.preview_plan(
            {"run_mode": "quantize_and_eval", "skip_stages": ["quantize"]}, OPERATOR
        )
    # Non-list skip payloads are rejected too.
    with pytest.raises(ValidationError, match="skip_stages must be a list"):
        control.preview_plan(
            {"run_mode": "quantize_and_eval", "skip_stages": "evaluate"}, OPERATOR
        )


def test_launch_with_skipped_stage_omits_job_and_records_skipped(web_env) -> None:
    control = make_control(web_env)
    preview = control.preview_plan(
        {"run_mode": "quantize_and_eval", "skip_stages": ["evaluate"]}, OPERATOR
    )
    started = control.start_run({"plan_hash": preview["plan_hash"]}, OPERATOR)
    run_id = started["run_id"]

    # No evaluate job is created when the stage is skipped at plan time.
    kinds = sorted(job["kind"] for job in started["jobs"])
    assert "evaluate" not in kinds
    assert all(
        "evaluate" not in (job.get("stages") or [])
        for job in control.jobs.list_jobs(run_id)
    )

    # A SKIPPED record is persisted so the pipeline shows the node as skipped.
    record = json.loads(
        (web_env.root / run_id / web_env.model_id / "state" / "evaluate.json").read_text()
    )
    assert record["status"] == "SKIPPED"
    assert record["reason"] == "operator skip"

    # The plan carries the skip so retries/re-runs stay consistent.
    plan = json.loads((web_env.root / run_id / "execution-plan.json").read_text())
    assert plan["skip_stages"] == {web_env.model_id: ["evaluate"]}

    # The run still rolls up to PASS once every other stage passes.
    model_dir = web_env.root / run_id / web_env.model_id
    for stage in ("preflight", "quantize", "validate", "runtime-smoke", "report"):
        atomic_write_json(model_dir / "state" / f"{stage}.json", {"status": "PASS"})
    summary = RunStore(web_env.root).get_run(run_id)
    model = next(m for m in summary["models"] if m["model_id"] == web_env.model_id)
    assert model["stages"]["evaluate"] == "SKIPPED"
    assert model["status"] == "PASS"


def test_skip_stage_cancels_job_and_persists(web_env) -> None:
    control = make_control(web_env)
    preview = control.preview_plan({"run_mode": "quantize_and_eval"}, OPERATOR)
    started = control.start_run({"plan_hash": preview["plan_hash"]}, OPERATOR)
    run_id = started["run_id"]
    evaluate_job = next(job for job in started["jobs"] if job["kind"] == "evaluate")

    result = control.skip_stage(run_id, web_env.model_id, "evaluate", {}, OPERATOR)
    assert result["skipped"] == "evaluate"
    assert result["canceled_jobs"] == [evaluate_job["job_id"]]

    # The queued evaluate job is canceled and its reservation released.
    canceled = next(
        job
        for job in control.jobs.list_jobs(run_id)
        if job["job_id"] == evaluate_job["job_id"]
    )
    assert canceled["status"] == "CANCELED"

    # SKIPPED record written and the plan updated for future re-runs.
    record = json.loads(
        (web_env.root / run_id / web_env.model_id / "state" / "evaluate.json").read_text()
    )
    assert record["status"] == "SKIPPED"
    plan = json.loads((web_env.root / run_id / "execution-plan.json").read_text())
    assert plan["skip_stages"] == {web_env.model_id: ["evaluate"]}

    # Idempotent: a byte-identical request replays rather than re-canceling.
    replay = control.skip_stage(run_id, web_env.model_id, "evaluate", {}, OPERATOR)
    assert replay == result


def test_skip_stage_rejects_validate_and_viewer(web_env) -> None:
    control = make_control(web_env)
    preview = control.preview_plan({"run_mode": "quantize_and_eval"}, OPERATOR)
    started = control.start_run({"plan_hash": preview["plan_hash"]}, OPERATOR)
    run_id = started["run_id"]

    with pytest.raises(PermissionDenied):
        control.skip_stage(run_id, web_env.model_id, "evaluate", {}, VIEWER)
    # ``validate`` is bundled with quantize; it can only be skipped at plan time.
    with pytest.raises(ValidationError, match="only be skipped"):
        control.skip_stage(run_id, web_env.model_id, "validate", {}, OPERATOR)
    # Mandatory stages are never skippable.
    with pytest.raises(ValidationError, match="stage must be one of"):
        control.skip_stage(run_id, web_env.model_id, "quantize", {}, OPERATOR)


def test_rerun_stage_clears_skip(web_env) -> None:
    control = make_control(web_env)
    preview = control.preview_plan(
        {"run_mode": "quantize_and_eval", "skip_stages": ["evaluate"]}, OPERATOR
    )
    started = control.start_run({"plan_hash": preview["plan_hash"]}, OPERATOR)
    run_id = started["run_id"]
    plan = json.loads((web_env.root / run_id / "execution-plan.json").read_text())
    assert plan["skip_stages"] == {web_env.model_id: ["evaluate"]}

    # Re-running from a skipped stage un-skips it so the relaunch actually runs.
    rerun = control.rerun_stage(run_id, web_env.model_id, "evaluate", {}, OPERATOR)
    stages = {stage for job in rerun["jobs"] for stage in job["stages"]}
    assert "evaluate" in stages
    plan = json.loads((web_env.root / run_id / "execution-plan.json").read_text())
    assert web_env.model_id not in plan.get("skip_stages", {})


def test_wsgi_skip_stage_route_enforces_roles(web_env) -> None:
    control = make_control(web_env)
    preview = control.preview_plan({"run_mode": "quantize_and_eval"}, OPERATOR)
    started = control.start_run({"plan_hash": preview["plan_hash"]}, OPERATOR)
    run_id = started["run_id"]
    app = create_app(
        RunStore(web_env.root), config_path=web_env.config, git_sha="git:abc1234"
    )
    path = f"/api/runs/{run_id}/models/{web_env.model_id}/stages/evaluate/skip"

    status, _ = call(app, "POST", path, {}, headers={"X-Model-Quality-Actor": "bob"})
    assert status["status"] == "403 Forbidden"

    status, payload = call(
        app,
        "POST",
        path,
        {"idempotency_key": "skip-ui-1"},
        headers={
            "X-Model-Quality-Actor": "alice",
            "X-Model-Quality-Roles": "operator",
        },
    )
    assert status["status"] == "200 OK"
    assert payload["skipped"] == "evaluate"
    assert payload["model_id"] == web_env.model_id
