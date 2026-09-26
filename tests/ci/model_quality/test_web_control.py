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

    with pytest.raises(ValidationError, match="must not configure"):
        validate_inference_config(
            {
                "container_name": "user-runtime",
                "script": str(roots / "x.sh"),
                "port": 8000,
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
