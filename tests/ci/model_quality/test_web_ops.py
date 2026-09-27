from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ci.model_quality.state import atomic_write_json
from ci.model_quality.web import (
    ControlPlane,
    DryRunExecutor,
    JobStore,
    LaunchPolicy,
    NotificationService,
    Principal,
    QueueExecutor,
    ReservationLedger,
    RunStore,
    ScheduleStore,
    TrendService,
    ValidationError,
    capabilities,
    capacity_forecast,
    next_run,
    scheduler,
)
from ci.model_quality.web.ops import cron_matches, parse_cron, run_due_schedules
from ci.model_quality.web.worker import execute_job, execution_argv, run_once

OPERATOR = Principal(actor="alice", roles=frozenset({"operator"}), authenticated=True)
ADMIN = Principal(actor="dave", roles=frozenset({"admin"}), authenticated=True)

STUB_STAGE = """
import json
import pathlib
import sys
import time

control = pathlib.Path(__file__).with_suffix(".json")
config = json.loads(control.read_text()) if control.is_file() else {}
args = sys.argv[1:]
stage = args[args.index("--stage") + 1] if "--stage" in args else "unknown"
with pathlib.Path(__file__).with_suffix(".argv").open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(args) + "\\n")
if stage in config.get("sleep_stages", []):
    time.sleep(config.get("sleep_seconds", 5))
if stage in config.get("fail_stages", []):
    raise SystemExit(2)
print("ok", stage)
"""


def make_control(env, *, capacity: float = 80.0, executor=None) -> ControlPlane:
    return ControlPlane(
        root=env.root,
        store=RunStore(env.root),
        config_path=env.config,
        policy=LaunchPolicy(gpu_hour_capacity=capacity),
        git_sha="git:abc1234",
        executor=executor or DryRunExecutor(env.root),
    )


def make_trend_run(
    root: Path,
    *,
    run_id: str,
    model_id: str,
    created_at: str,
    recovery: float,
    elapsed_seconds: float,
    deferred: int = 0,
) -> None:
    run_dir = root / run_id
    model_dir = run_dir / model_id
    (model_dir / "reports").mkdir(parents=True)
    atomic_write_json(
        run_dir / "execution-plan.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "attempt_id": "attempt-1",
            "run_mode": "quantize_and_eval",
            "created_at": created_at,
            "selected": [{"id": model_id}],
            "deferred": [{"id": f"deferred-{index}"} for index in range(deferred)],
            "budget": {"max_gpu_hours": 8, "selected_gpu_hours": 3},
        },
    )
    atomic_write_json(
        model_dir / "state" / "quantize.json",
        {
            "status": "PASS",
            "elapsed_seconds": elapsed_seconds,
            "recorded_at": created_at,
        },
    )
    atomic_write_json(
        model_dir / "reports" / "evaluation.json",
        {
            "status": "PASS",
            "metrics": [
                {"name": "exact_match", "recovery": recovery, "status": "PASS"}
            ],
        },
    )


def test_cron_helpers() -> None:
    moment = datetime(2026, 9, 26, 10, 7, tzinfo=timezone.utc)
    assert next_run("*/15 * * * *", moment) == "2026-09-26T10:15:00+00:00"
    assert next_run("0 3 * * *", moment) == "2026-09-27T03:00:00+00:00"
    parsed = parse_cron("0 3 * * *")
    assert cron_matches(parsed, datetime(2026, 9, 27, 3, 0, tzinfo=timezone.utc))
    assert not cron_matches(parsed, datetime(2026, 9, 27, 4, 0, tzinfo=timezone.utc))

    both = parse_cron("0 0 1 * 1")
    assert cron_matches(both, datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc))
    assert cron_matches(both, datetime(2026, 9, 7, 0, 0, tzinfo=timezone.utc))
    assert not cron_matches(both, datetime(2026, 9, 8, 0, 0, tzinfo=timezone.utc))

    for bad in ("", "* * * *", "60 * * * *", "*/0 * * * *", "a * * * *", "5-1 * * * *"):
        with pytest.raises(ValidationError):
            parse_cron(bad)


def test_schedule_store_crud_and_due(tmp_path: Path) -> None:
    schedules = ScheduleStore(tmp_path / "runs")
    with pytest.raises(ValidationError, match="plan_request"):
        schedules.create(cron="0 3 * * *", plan_request={}, actor="alice")
    with pytest.raises(ValidationError, match="run_mode is required"):
        schedules.create(
            cron="0 3 * * *", plan_request={"models": "all"}, actor="alice"
        )

    schedule = schedules.create(
        cron="0 3 * * *",
        plan_request={"run_mode": "quantize_and_eval", "models": "all"},
        actor="alice",
    )
    assert schedule["next_run_at"].endswith("03:00:00+00:00")
    assert schedules.list()["total"] == 1
    assert schedules.get(schedule["schedule_id"])["cron"] == "0 3 * * *"

    disabled = schedules.set_enabled(schedule["schedule_id"], False)
    assert disabled["enabled"] is False
    assert disabled["next_run_at"] is None
    assert schedules.due(datetime.now(timezone.utc)) == []

    enabled = schedules.set_enabled(schedule["schedule_id"], True)
    assert enabled["next_run_at"] is not None
    forced = schedules.update(
        schedule["schedule_id"], next_run_at="2026-01-01T00:00:00+00:00"
    )
    due = schedules.due(datetime(2026, 6, 1, tzinfo=timezone.utc))
    assert [item["schedule_id"] for item in due] == [forced["schedule_id"]]

    assert schedules.delete(schedule["schedule_id"])["deleted"] is True
    with pytest.raises(LookupError):
        schedules.get(schedule["schedule_id"])


def test_run_due_schedules_starts_runs_once_per_occurrence(web_env) -> None:
    control = make_control(web_env, executor=QueueExecutor(web_env.root))
    schedules = ScheduleStore(web_env.root)
    notifications = NotificationService(web_env.root)
    schedule = schedules.create(
        cron="*/15 * * * *",
        plan_request={"run_mode": "quantize_and_eval", "models": "all"},
        actor="alice",
    )
    schedules.update(schedule["schedule_id"], next_run_at="2026-01-01T00:00:00+00:00")

    result = run_due_schedules(
        schedules=schedules, control=control, notifications=notifications
    )
    assert result["total"] == 1
    assert result["items"][0]["status"] == "STARTED"
    run_id = result["items"][0]["run_id"]
    jobs = control.jobs.list_jobs(run_id)
    assert len(jobs) == 4
    assert control.ledger.summary()["reserved_gpu_hours"] == 3.0

    again = run_due_schedules(
        schedules=schedules, control=control, notifications=notifications
    )
    assert again["total"] == 0
    assert len(control.jobs.list_jobs(run_id)) == 4

    schedules.update(schedule["schedule_id"], next_run_at="2026-01-01T00:00:00+00:00")
    replayed = run_due_schedules(
        schedules=schedules, control=control, notifications=notifications
    )
    assert replayed["items"][0]["status"] == "REPLAYED"
    assert replayed["items"][0]["run_id"] == run_id
    assert len(control.jobs.list_jobs(run_id)) == 4

    updated = schedules.get(schedule["schedule_id"])
    assert updated["last_run_id"] == run_id
    assert updated["next_run_at"] != "2026-01-01T00:00:00+00:00"


def test_run_due_schedules_records_failures(web_env) -> None:
    control = make_control(web_env, capacity=0.5, executor=QueueExecutor(web_env.root))
    schedules = ScheduleStore(web_env.root)
    notifications = NotificationService(web_env.root)
    schedule = schedules.create(
        cron="0 3 * * *",
        plan_request={"run_mode": "quantize_and_eval", "models": "all"},
        actor="alice",
    )
    schedules.update(schedule["schedule_id"], next_run_at="2026-01-01T00:00:00+00:00")
    result = run_due_schedules(
        schedules=schedules, control=control, notifications=notifications
    )
    assert result["items"][0]["status"] == "FAILED"
    assert "admission rejected" in result["items"][0]["error"]
    log = notifications.log()
    assert log["items"][0]["event"] == "schedule.failed"


def test_scheduler_tick_starts_due_runs_and_cli_reports(
    web_env, capsys, monkeypatch
) -> None:
    schedules = ScheduleStore(web_env.root)
    schedule = schedules.create(
        cron="*/30 * * * *",
        plan_request={"run_mode": "quantize", "models": "all"},
        actor="alice",
    )
    schedules.update(schedule["schedule_id"], next_run_at="2026-01-01T00:00:00+00:00")

    result = scheduler.tick(
        root=web_env.root, config_path=web_env.config, git_sha="git:abc1234"
    )
    assert result["total"] == 1
    assert result["items"][0]["status"] == "STARTED"
    run_id = result["items"][0]["run_id"]

    schedules.update(schedule["schedule_id"], next_run_at="2026-01-01T00:00:00+00:00")
    replayed = scheduler.tick(
        root=web_env.root, config_path=web_env.config, git_sha="git:abc1234"
    )
    assert replayed["items"][0]["run_id"] == run_id
    assert replayed["items"][0]["status"] == "REPLAYED"

    exit_code = scheduler.main(
        [
            "--runs-root",
            str(web_env.root),
            "--config",
            str(web_env.config),
            "--git-sha",
            "git:abc1234",
        ]
    )
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["total"] == 0
    monkeypatch.delenv("MODEL_QUALITY_RUNS_ROOT", raising=False)
    assert scheduler.main(["--config", str(web_env.config)]) == 2


def test_notifications_are_recorded_and_suppressed(tmp_path: Path) -> None:
    service = NotificationService(tmp_path / "runs")
    record = service.notify("run.failed", {"run_id": "run-1"})
    assert record["delivery"] == "disabled"

    atomic_write_json(
        service.config_path,
        {
            "schema_version": 1,
            "enabled": True,
            "webhook_url": "https://example.invalid/hook",
            "events": ["run.failed"],
        },
    )
    suppressed = service.notify("run.failed", {"run_id": "run-1"})
    assert suppressed["delivery"] == "suppressed"
    assert "ALLOW_NETWORK" in suppressed["message"]

    unrouted = service.notify("run.started", {"run_id": "run-1"})
    assert unrouted["delivery"] == "disabled"

    log = service.log()
    assert log["total"] == 3
    assert log["items"][0]["event"] == "run.started"


def test_trends_report_quality_and_cost(web_env) -> None:
    make_trend_run(
        web_env.root,
        run_id="20260926T000000Z_first",
        model_id=web_env.model_id,
        created_at="2026-09-26T00:00:00+00:00",
        recovery=0.9,
        elapsed_seconds=7200,
    )
    make_trend_run(
        web_env.root,
        run_id="20260927T000000Z_second",
        model_id=web_env.model_id,
        created_at="2026-09-27T00:00:00+00:00",
        recovery=0.97,
        elapsed_seconds=3600,
        deferred=1,
    )
    trends = TrendService(RunStore(web_env.root), web_env.root)

    quality = trends.quality(model_id=web_env.model_id)
    assert quality["total"] == 2
    assert quality["metric_names"] == ["exact_match"]
    assert [point["metrics"][0]["recovery"] for point in quality["points"]] == [
        0.9,
        0.97,
    ]
    assert quality["points"][0]["run_id"].endswith("first")

    cost = trends.cost()
    assert cost["total"] == 2
    assert cost["estimated_gpu_hours"] == 6.0
    assert cost["actual_gpu_hours"] == 3.0
    assert cost["points"][-1]["deferred_models"] == 1
    assert cost["points"][-1]["run_status"] == "PASS"

    with pytest.raises(ValidationError, match="between 1 and 500"):
        trends.cost(limit=0)


def test_capabilities_reports_capacity(web_env) -> None:
    ledger = ReservationLedger(web_env.root, capacity_gpu_hours=10.0)
    ledger.reserve(
        run_id="run-a", model_id=web_env.model_id, attempt_id="attempt-1", gpu_hours=4
    )
    snapshot = capabilities(
        config_path=web_env.config,
        ledger=ledger,
        policy=LaunchPolicy(gpu_hour_capacity=10.0),
        jobs=JobStore(web_env.root),
        env={"MODEL_QUALITY_GPU_DEVICES": "8"},
    )
    assert snapshot["capacity"]["reserved_gpu_hours"] == 4.0
    assert snapshot["capacity"]["available_gpu_hours"] == 6.0
    assert snapshot["models"] == {"total": 1, "enabled": 1, "disabled": 0}
    assert snapshot["queues"][0]["gpu_devices"] == 8
    assert snapshot["hosts"][0]["gpu_devices"] == 8
    assert len(snapshot["reservations"]) == 1
    forecast = snapshot["forecast"]
    assert forecast["pending_gpu_hours"] == 4.0
    assert forecast["available_gpu_hours"] == 6.0
    assert forecast["confidence"] == "none"
    assert forecast["eta_hours"] is None


def test_capacity_forecast_projects_drain_from_observed_throughput(web_env) -> None:
    ledger = ReservationLedger(web_env.root, capacity_gpu_hours=20.0)
    done = ledger.reserve(
        run_id="run-done",
        model_id=web_env.model_id,
        attempt_id="attempt-1",
        gpu_hours=6,
    )
    ledger.release(done["reservation_id"], reason="completed")
    ledger.reserve(
        run_id="run-pending",
        model_id=web_env.model_id,
        attempt_id="attempt-2",
        gpu_hours=4,
    )

    forecast = capacity_forecast(ledger=ledger, window_hours=2.0)
    assert forecast["pending_gpu_hours"] == 4.0
    assert forecast["observed_releases"] == 1
    assert forecast["observed_gpu_hours"] == 6.0
    assert forecast["throughput_gpu_hours_per_hour"] == 3.0
    assert forecast["eta_hours"] == pytest.approx(4 / 3, rel=1e-3)
    assert forecast["estimated_drain_at"] is not None
    assert forecast["confidence"] == "low"

    with pytest.raises(ValidationError, match="window_hours"):
        capacity_forecast(ledger=ledger, window_hours=0)


def _stub(tmp_path: Path) -> Path:
    stub = tmp_path / "stub_stage.py"
    stub.write_text(STUB_STAGE, encoding="utf-8")
    return stub


def _queued_job(web_env, *, run_mode: str = "quantize") -> dict:
    control = make_control(web_env, executor=QueueExecutor(web_env.root))
    preview = control.preview_plan({"run_mode": run_mode}, OPERATOR)
    started = control.start_run({"plan_hash": preview["plan_hash"]}, OPERATOR)
    quantize = next(job for job in started["jobs"] if job["kind"] == "quantize")
    return {"control": control, "run_id": started["run_id"], "job": quantize}


def test_worker_executes_queued_job_and_releases_reservation(web_env) -> None:
    stub = _stub(web_env.tmp_path)
    context = _queued_job(web_env)
    control, job = context["control"], context["job"]

    result = run_once(
        root=web_env.root,
        config_path=web_env.config,
        jobs=control.jobs,
        ledger=control.ledger,
        notifications=NotificationService(web_env.root),
        argv_prefix=(sys.executable, str(stub)),
    )
    assert result is not None
    assert result["job_id"] == job["job_id"]
    assert result["status"] == "SUCCEEDED"
    assert result["exit_code"] == 0
    # The whole model/attempt group shares one reservation, so it stays held
    # until every lane is terminal.
    assert control.ledger.summary()["reserved_gpu_hours"] == 2.0
    assert {record["state"] for record in control.ledger.summary()["reservations"]} == {
        "QUEUED"
    }

    while run_once(
        root=web_env.root,
        config_path=web_env.config,
        jobs=control.jobs,
        ledger=control.ledger,
        notifications=NotificationService(web_env.root),
        argv_prefix=(sys.executable, str(stub)),
    ):
        pass
    assert control.ledger.summary()["reserved_gpu_hours"] == 0.0
    assert control.ledger.summary()["released_gpu_hours"] == 2.0
    assert {record["state"] for record in control.ledger.summary()["reservations"]} == {
        "RELEASED"
    }

    recorded = [
        json.loads(line)
        for line in stub.with_suffix(".argv").read_text(encoding="utf-8").splitlines()
    ]
    stages = [entry[entry.index("--stage") + 1] for entry in recorded]
    assert stages == ["preflight", "quantize", "validate", "runtime-smoke", "report"]
    assert all(
        entry[entry.index("--run-id") + 1] == context["run_id"] for entry in recorded
    )
    assert all(entry[entry.index("--fingerprint") + 1] for entry in recorded)
    log = (
        web_env.root
        / context["run_id"]
        / web_env.model_id
        / "logs"
        / job["attempt_id"]
        / "quantization.log"
    )
    assert "ok quantize" in log.read_text(encoding="utf-8")


def test_worker_passes_evaluation_override_to_stage(web_env) -> None:
    captured = {}

    def runner(argv, **kwargs):
        captured.update(argv=argv, env=kwargs["env"])
        return type("Completed", (), {"returncode": 0})()

    jobs = JobStore(web_env.root)
    job = jobs.create(
        run_id="run-a",
        model_id=web_env.model_id,
        attempt_id="attempt-1",
        kind="evaluate",
        executor="queue",
        stages=["evaluate"],
        details={
            "run_mode": "eval_only",
            "evaluation_command": [
                "/root/miniconda/envs/model_quality_lm_eval/bin/python",
                "eval.py",
                "--output",
                "{reports_dir}/evaluation-raw.json",
            ],
        },
    )
    execute_job(
        job,
        root=web_env.root,
        config_path=web_env.config,
        jobs=jobs,
        ledger=ReservationLedger(web_env.root, capacity_gpu_hours=10),
        argv_prefix=(sys.executable, "stage.py"),
        runner=runner,
    )
    assert captured["argv"][captured["argv"].index("--stage") + 1] == "evaluate"
    assert captured["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(
        Path(__file__).parents[3]
    )
    assert json.loads(captured["env"]["MODEL_QUALITY_EVALUATION_COMMAND_JSON"])[
        0
    ].endswith("model_quality_lm_eval/bin/python")


def test_worker_builds_inference_container_command(web_env) -> None:
    job = {
        "run_id": "run-a",
        "model_id": web_env.model_id,
        "attempt_id": "attempt-1",
        "kind": "inference",
        "container_name": "zhuang_xsgl_0923",
        "script": "/workspace/smoke.sh",
        "details": {
            "source_path": str(web_env.source),
            "inference_arguments": ["{output_dir}", "{reports_dir}"],
        },
    }
    argv = execution_argv(
        job,
        "runtime-smoke",
        root=web_env.root,
        config_path=web_env.config,
        argv_prefix=(sys.executable,),
        container_runtime=("docker",),
    )
    assert argv[:4] == ["docker", "exec", "zhuang_xsgl_0923", "/workspace/smoke.sh"]
    assert argv[-2].endswith(f"run-a/{web_env.model_id}/model")
    assert argv[-1].endswith("attempts/attempt-1/reports")
    direct = execution_argv(
        job,
        "runtime-smoke",
        root=web_env.root,
        config_path=web_env.config,
        argv_prefix=(sys.executable,),
        container_runtime=("direct",),
    )
    assert direct[0] == "/workspace/smoke.sh"


def test_worker_classifies_script_failure(web_env) -> None:
    stub = _stub(web_env.tmp_path)
    context = _queued_job(web_env)
    control, job = context["control"], context["job"]
    stub.with_suffix(".json").write_text(
        json.dumps({"fail_stages": ["quantize"]}), encoding="utf-8"
    )

    result = execute_job(
        job,
        root=web_env.root,
        config_path=web_env.config,
        jobs=control.jobs,
        ledger=control.ledger,
        notifications=NotificationService(web_env.root),
        argv_prefix=(sys.executable, str(stub)),
    )
    assert result["status"] == "FAILED"
    assert result["failure_class"] == "SCRIPT_FAILED"
    assert result["exit_code"] == 2
    assert result["details"]["failed_stage"] == "quantize"
    assert control.ledger.summary()["reserved_gpu_hours"] == 0.0
    assert control.audit.read(action="job.failed")["total"] == 1
    assert NotificationService(web_env.root).log()["items"][0]["event"] == "run.failed"


def test_worker_classifies_timeout_and_quality_failure(web_env) -> None:
    stub = _stub(web_env.tmp_path)
    context = _queued_job(web_env, run_mode="quantize_and_eval")
    control = context["control"]
    evaluate_job = next(
        item
        for item in control.jobs.list_jobs(context["run_id"])
        if item["kind"] == "evaluate"
    )
    stub.with_suffix(".json").write_text(
        json.dumps({"sleep_stages": ["evaluate"], "sleep_seconds": 5}),
        encoding="utf-8",
    )
    result = execute_job(
        evaluate_job,
        root=web_env.root,
        config_path=web_env.config,
        jobs=control.jobs,
        ledger=control.ledger,
        argv_prefix=(sys.executable, str(stub)),
        timeout_seconds=0.05,
    )
    assert result["status"] == "FAILED"
    assert result["failure_class"] == "RESOURCE_TIMEOUT"
    assert result["exit_code"] == 124

    stub.with_suffix(".json").write_text(
        json.dumps({"fail_stages": ["quantize"]}), encoding="utf-8"
    )
    quantize_job = next(
        item
        for item in control.jobs.list_jobs(context["run_id"])
        if item["kind"] == "quantize"
    )
    script_failed = execute_job(
        quantize_job,
        root=web_env.root,
        config_path=web_env.config,
        jobs=control.jobs,
        ledger=control.ledger,
        argv_prefix=(sys.executable, str(stub)),
    )
    assert script_failed["failure_class"] == "SCRIPT_FAILED"


def test_worker_records_missing_executable_as_resource_failure(web_env) -> None:
    context = _queued_job(web_env)
    control, job = context["control"], context["job"]
    result = execute_job(
        job,
        root=web_env.root,
        config_path=web_env.config,
        jobs=control.jobs,
        ledger=control.ledger,
        argv_prefix=(str(web_env.tmp_path / "missing-python"),),
    )
    assert result["status"] == "FAILED"
    assert result["failure_class"] == "RESOURCE_UNAVAILABLE"
    assert result["exit_code"] == 127
    assert control.ledger.summary()["reserved_gpu_hours"] == 0.0

    # A retryable infrastructure failure re-admits the GPU-hours instead of
    # running the retried job for free.
    retried = control.retry_job(
        context["run_id"], web_env.model_id, job["job_id"], {}, OPERATOR
    )
    assert retried["reservation"]["gpu_hours"] == 2.0
    assert retried["job"]["status"] == "QUEUED"
    assert control.ledger.summary()["reserved_gpu_hours"] == 2.0
    assert {record["state"] for record in control.ledger.summary()["reservations"]} == {
        "RESERVED",
        "RELEASED",
    }


def test_worker_returns_none_when_queue_is_empty(web_env) -> None:
    assert (
        run_once(
            root=web_env.root, config_path=web_env.config, jobs=JobStore(web_env.root)
        )
        is None
    )


def test_worker_filters_queue_by_job_kind(web_env) -> None:
    jobs = JobStore(web_env.root)
    jobs.create(
        run_id="run-a",
        model_id=web_env.model_id,
        attempt_id="attempt-1",
        kind="inference",
        executor="queue",
        stages=["runtime-smoke"],
    )
    assert (
        run_once(
            root=web_env.root,
            config_path=web_env.config,
            jobs=jobs,
            kinds=frozenset({"evaluate"}),
        )
        is None
    )


def test_worker_rejects_jobs_without_stages(web_env) -> None:
    jobs = JobStore(web_env.root)
    job = jobs.create(
        run_id="run-a",
        model_id=web_env.model_id,
        attempt_id="attempt-1",
        kind="quantize",
        executor="queue",
        stages=[],
    )
    with pytest.raises(ValidationError, match="no stages"):
        execute_job(
            job,
            root=web_env.root,
            config_path=web_env.config,
            jobs=jobs,
            ledger=ReservationLedger(web_env.root, capacity_gpu_hours=8.0),
        )
