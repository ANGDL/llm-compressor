"""Controlled execution: preview, launch, retry, evaluate, cancel."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..config import load_model_config
from ..state import atomic_write_json, utc_now
from .audit import (
    AuditLog,
    ConflictError,
    IdempotencyStore,
    ValidationError,
    request_fingerprint,
    timestamp_token,
)
from .executors import DispatchResult, ExecutorAdapter, LaunchSpec, create_executor
from .jobs import (
    ACTIVE_JOB_STATUSES,
    RETRYABLE_FAILURE_CLASSES,
    JobStore,
    ReservationLedger,
    settle_model_group,
)
from .plans import LaunchPolicy, PlanService
from .security import Principal
from .store import NotFoundError, RunStore, read_json, safe_component

# Stage order per run mode, mirroring ``ci.model_quality.buildkite``.
STAGES_BY_MODE: dict[str, tuple[str, ...]] = {
    "quantize": ("preflight", "quantize", "validate", "runtime-smoke", "report"),
    "quantize_and_eval": (
        "preflight",
        "quantize",
        "validate",
        "runtime-smoke",
        "evaluate",
        "report",
    ),
    "eval_only": ("preflight", "validate", "runtime-smoke", "evaluate", "report"),
    "upload_only": ("preflight", "validate", "runtime-smoke", "report"),
}

_FAILED_STATUSES = {"FAIL", "FAILED"}


def _new_attempt_id() -> str:
    return f"attempt-{timestamp_token()}-{uuid4().hex[:8]}"


def _reservation_hours(definition: dict[str, Any], run_mode: str) -> float:
    """Mirror ``planner._job_cost`` so admission matches the plan budget."""

    resources = definition["resources"]
    quantize = float(resources.get("estimated_gpu_hours", 0.0))
    evaluate = float(resources.get("estimated_eval_gpu_hours", 0.0))
    if run_mode == "eval_only":
        return evaluate
    if run_mode == "upload_only":
        return 0.0
    if run_mode == "quantize_and_eval":
        return quantize + evaluate
    return quantize


def _timestamp(value: Any) -> float:
    if not isinstance(value, str):
        return float("-inf")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return float("-inf")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


class ControlPlane:
    """Write-side API for model quality runs.

    Every mutating call takes an idempotency key, records an audit entry, and
    only ever asks a registered executor adapter to run reviewed CLI commands.
    """

    def __init__(
        self,
        *,
        root: str | Path,
        store: RunStore,
        config_path: str | Path | None,
        policy: LaunchPolicy | None = None,
        plan_service: PlanService | None = None,
        executor: ExecutorAdapter | None = None,
        ledger: ReservationLedger | None = None,
        jobs: JobStore | None = None,
        audit: AuditLog | None = None,
        idempotency: IdempotencyStore | None = None,
        git_sha: str = "local",
        executor_upload: bool = False,
    ) -> None:
        self.root = Path(root).expanduser()
        self.store = store
        self.config_path = Path(config_path) if config_path else None
        self.policy = policy or LaunchPolicy.from_env()
        self.audit = audit or AuditLog(self.root)
        self.idempotency = idempotency or IdempotencyStore(self.root)
        self.jobs = jobs or JobStore(self.root)
        self.ledger = ledger or ReservationLedger(
            self.root, capacity_gpu_hours=self.policy.gpu_hour_capacity
        )
        self.plan_service = plan_service or self._build_plan_service(git_sha)
        self.executor = executor or create_executor(
            self.policy.executor,
            root=self.root,
            config_path=self.config_path,
            upload=executor_upload,
        )

    def _build_plan_service(self, git_sha: str) -> PlanService:
        from .plans import PlanRegistry

        if self.config_path is None:
            raise ValidationError(
                "a model quality --config path is required for control operations"
            )
        return PlanService(
            config_path=self.config_path,
            registry=PlanRegistry(self.root),
            policy=self.policy,
            git_sha=git_sha,
        )

    # ------------------------------------------------------------------ plans

    def preview_plan(
        self, request: dict[str, Any], principal: Principal
    ) -> dict[str, Any]:
        """Build and persist a reviewed plan without consuming capacity."""

        principal.require("plan")
        preview = self.plan_service.preview(request)
        self.audit.record(
            actor=principal.actor,
            action="plan.preview",
            target=preview["plan"]["run_id"],
            result="ACCEPTED",
            request=request,
            details={
                "plan_hash": preview["plan_hash"],
                "selected": [job["id"] for job in preview["plan"]["selected"]],
                "deferred": [job["id"] for job in preview["plan"]["deferred"]],
                "run_mode": preview["plan"]["run_mode"],
            },
        )
        return preview

    def get_plan(self, plan_hash: str, principal: Principal) -> dict[str, Any]:
        principal.require("read")
        return self.plan_service.launch_spec(plan_hash)

    # ------------------------------------------------------------------- runs

    def start_run(
        self, request: dict[str, Any], principal: Principal
    ) -> dict[str, Any]:
        """Launch a reviewed plan, reserving GPU-hours in the same operation."""

        principal.require("start")
        plan_hash = request.get("plan_hash")
        if not isinstance(plan_hash, str) or not plan_hash:
            raise ValidationError("plan_hash is required; preview a plan first")
        key = self._resolve_key(request, action="run.start", target=plan_hash)
        claim = self.idempotency.begin(
            key, {"plan_hash": plan_hash, "actor": principal.actor}
        )
        if claim["replayed"]:
            return claim["response"]
        try:
            response = self._start_run(plan_hash, principal, request)
        except Exception as error:
            self.idempotency.fail(key, str(error))
            raise
        self.idempotency.complete(key, response)
        return response

    def _start_run(
        self, plan_hash: str, principal: Principal, request: dict[str, Any]
    ) -> dict[str, Any]:
        record = self.plan_service.registry.consume(plan_hash)
        plan = record["plan"]
        review = record.get("review", {})
        blocking = [
            risk for risk in review.get("risks", []) if risk.get("severity") == "error"
        ]
        if blocking:
            raise ConflictError(
                "plan has blocking risks: "
                + "; ".join(risk["message"] for risk in blocking)
            )
        if self.policy.inference_containers and not record.get("inference"):
            raise ValidationError(
                "inference.container_name and inference.script are required "
                "because this deployment registers inference containers"
            )
        if self.plan_service.git_sha not in {"local", "unknown", ""}:
            if plan.get("git_sha") != self.plan_service.git_sha:
                raise ConflictError(
                    "plan git_sha does not match the deployed revision; preview again"
                )
        return self._launch(
            plan=plan,
            plan_hash=plan_hash,
            attempt_id=plan["attempt_id"],
            run_mode=plan["run_mode"],
            git_sha=plan["git_sha"],
            inference=record.get("inference"),
            principal=principal,
            reason="start",
            extra_request=request,
        )

    def retry(
        self, run_id: str, request: dict[str, Any], principal: Principal
    ) -> dict[str, Any]:
        """Create a new attempt for the failed stages of an existing run."""

        principal.require("retry")
        key = self._resolve_key(request, action="run.retry", target=run_id)
        claim = self.idempotency.begin(key, {"run_id": run_id, **request})
        if claim["replayed"]:
            return claim["response"]
        try:
            response = self._retry(run_id, request, principal)
        except Exception as error:
            self.idempotency.fail(key, str(error))
            raise
        self.idempotency.complete(key, response)
        return response

    def _retry(
        self, run_id: str, request: dict[str, Any], principal: Principal
    ) -> dict[str, Any]:
        run_dir = self._require_run(run_id)
        plan = read_json(run_dir / "execution-plan.json")
        if not isinstance(plan, dict):
            raise NotFoundError(f"run {run_id!r} has no execution plan")
        run_mode = plan["run_mode"]
        order = STAGES_BY_MODE[run_mode]
        model_id = request.get("model_id")
        models = self.store.get_run(run_id)["models"]
        targets = []
        for model in models:
            if model_id and model["model_id"] != model_id:
                continue
            failed = [
                stage
                for stage in order
                if str(model["stages"].get(stage, "PENDING")) in _FAILED_STATUSES
            ]
            if failed:
                targets.append((model["model_id"], failed))
        if not targets:
            raise ConflictError("no failed stage was found to retry")
        requested_stage = request.get("from_stage")
        attempt_id = request.get("attempt_id") or _new_attempt_id()
        safe_component(attempt_id, "attempt_id")
        retry_jobs = []
        for model_id_value, failed in targets:
            start = 0
            if requested_stage:
                if requested_stage not in order:
                    raise ValidationError(
                        f"from_stage must be one of {', '.join(order)}"
                    )
                start = order.index(requested_stage)
            first_failed = order.index(failed[0])
            stages = list(order[min(start, first_failed) :])
            retry_jobs.append((model_id_value, stages))
        return self._launch(
            plan=plan,
            plan_hash=plan.get("plan_hash"),
            attempt_id=attempt_id,
            run_mode=run_mode,
            git_sha=plan.get("git_sha", "local"),
            inference=request.get("inference"),
            principal=principal,
            reason="retry",
            extra_request=request,
            job_overrides=dict(retry_jobs),
        )

    def evaluate(
        self, run_id: str, request: dict[str, Any], principal: Principal
    ) -> dict[str, Any]:
        """Create an eval_only attempt against an already-passing artifact."""

        principal.require("evaluate")
        key = self._resolve_key(request, action="run.evaluate", target=run_id)
        claim = self.idempotency.begin(key, {"run_id": run_id, **request})
        if claim["replayed"]:
            return claim["response"]
        try:
            response = self._evaluate(run_id, request, principal)
        except Exception as error:
            self.idempotency.fail(key, str(error))
            raise
        self.idempotency.complete(key, response)
        return response

    def _evaluate(
        self, run_id: str, request: dict[str, Any], principal: Principal
    ) -> dict[str, Any]:
        run_dir = self._require_run(run_id)
        plan = read_json(run_dir / "execution-plan.json")
        if not isinstance(plan, dict):
            raise NotFoundError(f"run {run_id!r} has no execution plan")
        if self.config_path is None:
            raise ValidationError("a model quality --config path is required")
        config = load_model_config(self.config_path)
        configured = {model["id"]: model for model in config["models"]}

        run = self.store.get_run(run_id)
        requested = request.get("model_id")
        targets = []
        for model in run["models"]:
            if requested and model["model_id"] != requested:
                continue
            validate_status = str(model["stages"].get("validate", "PENDING"))
            if validate_status not in {"PASS", "WARN", "PASSED"}:
                continue
            definition = configured.get(model["model_id"])
            if definition is None:
                continue
            if not definition.get("evaluation", {}).get("command"):
                raise ValidationError(
                    f"{model['model_id']} has no evaluation command configured"
                )
            targets.append(model["model_id"])
        if not targets:
            raise ConflictError(
                "no model with a passing validate stage was found for evaluation"
            )
        attempt_id = request.get("attempt_id") or _new_attempt_id()
        safe_component(attempt_id, "attempt_id")
        return self._launch(
            plan={
                **plan,
                "run_mode": "eval_only",
                "budget": plan.get("budget", {}),
            },
            plan_hash=plan.get("plan_hash"),
            attempt_id=attempt_id,
            run_mode="eval_only",
            git_sha=plan.get("git_sha", "local"),
            inference=request.get("inference"),
            principal=principal,
            reason="evaluate",
            extra_request=request,
            job_overrides={
                model_id: list(STAGES_BY_MODE["eval_only"]) for model_id in targets
            },
        )

    def cancel(
        self, run_id: str, request: dict[str, Any], principal: Principal
    ) -> dict[str, Any]:
        principal.require("cancel")
        key = self._resolve_key(request, action="run.cancel", target=run_id)
        claim = self.idempotency.begin(key, {"run_id": run_id, **request})
        if claim["replayed"]:
            return claim["response"]
        try:
            run_dir = self._require_run(run_id)
            reason = str(request.get("reason") or "canceled from the control plane")
            canceled_jobs = []
            for job in self.jobs.list_jobs(run_id):
                if job.get("status") in ACTIVE_JOB_STATUSES:
                    canceled_jobs.append(self.jobs.cancel(job, reason)["job_id"])
            released = self.ledger.release_for_run(run_id, reason=f"cancel: {reason}")
            marker = {
                "schema_version": 1,
                "run_id": run_id,
                "canceled_by": principal.actor,
                "reason": reason,
                "recorded_at": utc_now(),
                "canceled_jobs": canceled_jobs,
                "released_reservations": released,
            }
            atomic_write_json(run_dir / "cancel.json", marker)
            dispatch = self.executor.cancel(run_id=run_id, reason=reason)
            response = {
                "run_id": run_id,
                "status": "CANCELED",
                "canceled_jobs": canceled_jobs,
                "released_reservations": released,
                "executor": dispatch.as_dict(),
            }
        except Exception as error:
            self.idempotency.fail(key, str(error))
            raise
        self.audit.record(
            actor=principal.actor,
            action="run.cancel",
            target=run_id,
            result="ACCEPTED",
            request=request,
            details=response,
        )
        self.idempotency.complete(key, response)
        return response

    # ------------------------------------------------------------------- jobs

    def list_jobs(
        self,
        run_id: str,
        *,
        model_id: str | None = None,
        attempt_id: str | None = None,
        principal: Principal | None = None,
    ) -> dict[str, Any]:
        if principal is not None:
            principal.require("read")
        self._require_run(run_id)
        jobs = self.jobs.list_jobs(run_id, model_id=model_id, attempt_id=attempt_id)
        return {"items": jobs, "total": len(jobs)}

    def cancel_job(
        self,
        run_id: str,
        model_id: str,
        job_id: str,
        request: dict[str, Any],
        principal: Principal,
    ) -> dict[str, Any]:
        """Cancel one execution job; the inference container is never touched.

        A job-level cancel is surgical: sibling stages stay queued and the model
        reservation is released only once the whole group settles. Use the
        run-level cancel to stop every job of a run at once.
        """

        principal.require("cancel")
        key = self._resolve_key(
            request, action="job.cancel", target=f"{run_id}/{model_id}/{job_id}"
        )
        claim = self.idempotency.begin(key, {"job_id": job_id, **request})
        if claim["replayed"]:
            return claim["response"]
        try:
            job = self.jobs.get(run_id, model_id, job_id)
            reason = str(request.get("reason") or "job canceled from the control plane")
            job = self.jobs.cancel(job, reason)
            released = self._release_if_settled(job)
            response = {
                "job": job,
                "released_reservations": [
                    record["reservation_id"] for record in released
                ],
                "inference_container_action": "none",
            }
        except Exception as error:
            self.idempotency.fail(key, str(error))
            raise
        self.audit.record(
            actor=principal.actor,
            action="job.cancel",
            target=f"{run_id}/{model_id}/{job_id}",
            result="ACCEPTED",
            request=request,
            details={"reason": request.get("reason")},
        )
        self.idempotency.complete(key, response)
        return response

    def retry_job(
        self,
        run_id: str,
        model_id: str,
        job_id: str,
        request: dict[str, Any],
        principal: Principal,
    ) -> dict[str, Any]:
        """Re-queue a resource failure as a new job; old jobs stay immutable."""

        principal.require("retry")
        try:
            current = self.jobs.get(run_id, model_id, job_id)
            state = {
                "job_status": current.get("status"),
                "failure_class": current.get("failure_class"),
                "updated_at": current.get("updated_at"),
            }
        except NotFoundError:
            state = {}
        key = self._resolve_key(
            request,
            action="job.retry",
            target=f"{run_id}/{model_id}/{job_id}",
            state=state,
        )
        claim = self.idempotency.begin(key, {"job_id": job_id, **request})
        if claim["replayed"]:
            return claim["response"]
        try:
            job = self.jobs.get(run_id, model_id, job_id)
            if job.get("status") not in {"FAILED", "EXPIRED", "CANCELED"}:
                raise ConflictError(
                    f"job {job_id!r} is {job.get('status')} and cannot be retried"
                )
            failure_class = job.get("failure_class")
            if failure_class not in RETRYABLE_FAILURE_CLASSES:
                raise ConflictError(
                    f"job failure class {failure_class!r} is not retryable; "
                    "fix the cause and create a new run instead"
                )
            attempt_id = request.get("attempt_id") or _new_attempt_id()
            safe_component(attempt_id, "attempt_id")
            # The failed group already released its reservation, so a retried
            # job needs its own admission or the GPU-hours would be free.
            hours = float(job.get("gpu_hours") or 0.0)
            reservation = None
            if hours > 0:
                reservation = self.ledger.reserve(
                    run_id=run_id,
                    model_id=model_id,
                    attempt_id=attempt_id,
                    gpu_hours=hours,
                    gpu_count=job.get("gpu_count"),
                )
            retried = self.jobs.create(
                run_id=run_id,
                model_id=model_id,
                attempt_id=attempt_id,
                kind=job["kind"],
                executor=job["executor"],
                command=list(job.get("command", [])),
                queue=job.get("queue"),
                gpu_count=job.get("gpu_count"),
                gpu_hours=job.get("gpu_hours"),
                stages=list(job.get("stages", [])),
                container_name=job.get("container_name"),
                script=job.get("script"),
                result_file=job.get("result_file"),
                artifact_fingerprint=job.get("artifact_fingerprint"),
                evaluation_fingerprint=job.get("evaluation_fingerprint"),
                reservation_id=(reservation["reservation_id"] if reservation else None),
                details={**job.get("details", {}), "retried_from": job_id},
            )
            dispatch = self.executor.dispatch(
                LaunchSpec(
                    run_id=run_id,
                    attempt_id=attempt_id,
                    run_mode=job.get("details", {}).get("run_mode", "quantize"),
                    git_sha=job.get("details", {}).get("git_sha", "local"),
                    plan={},
                ),
                [retried],
            )
            response = {
                "job": retried,
                "executor": dispatch.as_dict(),
                "reservation": reservation,
            }
        except Exception as error:
            self.idempotency.fail(key, str(error))
            raise
        self.audit.record(
            actor=principal.actor,
            action="job.retry",
            target=f"{run_id}/{model_id}/{job_id}",
            result="ACCEPTED",
            request=request,
            details={"new_job_id": response["job"]["job_id"]},
        )
        self.idempotency.complete(key, response)
        return response

    # ------------------------------------------------------------- internals

    def _resolve_key(
        self,
        request: dict[str, Any],
        *,
        action: str,
        target: str,
        state: dict[str, Any] | None = None,
    ) -> str:
        explicit = request.get("idempotency_key")
        if isinstance(explicit, str) and explicit:
            return explicit
        payload = {
            key: value for key, value in request.items() if key != "idempotency_key"
        }
        digest = request_fingerprint(
            {"target": target, "state": state or {}, **payload}
        )[:32]
        return f"{action}:{digest}"

    def _require_run(self, run_id: str) -> Path:
        run_dir = self.root / safe_component(run_id, "run_id")
        if not run_dir.is_dir():
            raise NotFoundError(f"run {run_id!r} was not found")
        return run_dir

    def _launch(
        self,
        *,
        plan: dict[str, Any],
        plan_hash: str | None,
        attempt_id: str,
        run_mode: str,
        git_sha: str,
        inference: dict[str, Any] | None,
        principal: Principal,
        reason: str,
        extra_request: dict[str, Any],
        job_overrides: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        run_id = plan["run_id"]
        run_dir = self.root / safe_component(run_id, "run_id")
        if self.config_path is None:
            raise ValidationError("a model quality --config path is required")
        config = load_model_config(self.config_path)
        definitions = {model["id"]: model for model in config["models"]}

        planned = {
            job["id"]: job for job in plan.get("selected", []) if isinstance(job, dict)
        }
        if job_overrides:
            for model_id in job_overrides:
                if model_id not in definitions:
                    raise ValidationError(f"unknown model: {model_id!r}")
                planned.setdefault(
                    model_id, {"id": model_id, "estimated_gpu_hours": 0.0}
                )
        if not planned:
            raise ConflictError("the plan has no selected model to launch")

        reservations: list[dict[str, Any]] = []
        created_jobs: list[dict[str, Any]] = []
        try:
            for model_id, job in planned.items():
                definition = definitions[model_id]
                stages = (
                    job_overrides[model_id]
                    if job_overrides and model_id in job_overrides
                    else list(STAGES_BY_MODE[run_mode])
                )
                if job.get("upload_enabled") or run_mode == "upload_only":
                    stages = [*stages, "publish"]
                reservation = self.ledger.reserve(
                    run_id=run_id,
                    model_id=model_id,
                    attempt_id=attempt_id,
                    gpu_hours=_reservation_hours(definition, run_mode),
                    gpu_count=definition["resources"]["gpu_count"],
                )
                reservations.append(reservation)
                created_jobs.extend(
                    self._create_jobs(
                        run_id=run_id,
                        attempt_id=attempt_id,
                        run_mode=run_mode,
                        git_sha=git_sha,
                        definition=definition,
                        planned=job,
                        stages=stages,
                        reservation=reservation,
                        inference=inference,
                        plan_hash=plan_hash,
                    )
                )
        except Exception:
            for reservation in reservations:
                try:
                    self.ledger.release(
                        reservation["reservation_id"], reason="launch aborted"
                    )
                except NotFoundError:
                    pass
            raise

        run_dir.mkdir(parents=True, exist_ok=True)
        if not (run_dir / "execution-plan.json").is_file():
            atomic_write_json(run_dir / "execution-plan.json", plan)
        if attempt_id != plan.get("attempt_id"):
            atomic_write_json(
                run_dir / "attempt-plans" / f"{attempt_id}.json",
                {**plan, "attempt_id": attempt_id, "run_mode": run_mode},
            )
        marker = read_json(run_dir / "run.json")
        run_record = (
            dict(marker)
            if isinstance(marker, dict)
            else {
                "schema_version": 1,
                "run_id": run_id,
                "created_at": utc_now(),
            }
        )
        run_record.update(
            {
                "run_id": run_id,
                "run_mode": run_mode,
                "git_sha": git_sha,
                "plan_hash": plan_hash,
                "updated_at": utc_now(),
                "last_actor": principal.actor,
                "inference": inference,
                "canceled": False,
            }
        )
        run_record.setdefault("created_by", principal.actor)
        run_record.setdefault("created_at", utc_now())
        attempts = [
            item for item in run_record.get("attempts", []) if isinstance(item, dict)
        ]
        attempts.append(
            {
                "attempt_id": attempt_id,
                "run_mode": run_mode,
                "reason": reason,
                "actor": principal.actor,
                "created_at": utc_now(),
                "plan_hash": plan_hash,
            }
        )
        run_record["attempts"] = attempts
        atomic_write_json(run_dir / "run.json", run_record)

        spec = LaunchSpec(
            run_id=run_id,
            attempt_id=attempt_id,
            run_mode=run_mode,
            git_sha=git_sha,
            plan=plan,
            inference=inference,
            quantization_argv=plan.get("quantization_argv"),
        )
        dispatch: DispatchResult = self.executor.dispatch(spec, created_jobs)
        response = {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "run_mode": run_mode,
            "plan_hash": plan_hash,
            "reason": reason,
            "jobs": created_jobs,
            "reservations": reservations,
            "budget": self.ledger.summary(),
            "executor": dispatch.as_dict(),
            "status": "QUEUED" if not dispatch.dispatched else "DISPATCHED",
        }
        self.audit.record(
            actor=principal.actor,
            action=f"run.{reason}",
            target=run_id,
            result="ACCEPTED",
            request=extra_request,
            details={
                "attempt_id": attempt_id,
                "run_mode": run_mode,
                "plan_hash": plan_hash,
                "jobs": [job["job_id"] for job in created_jobs],
                "reservations": [
                    reservation["reservation_id"] for reservation in reservations
                ],
                "executor": dispatch.as_dict(),
            },
        )
        return response

    def _create_jobs(
        self,
        *,
        run_id: str,
        attempt_id: str,
        run_mode: str,
        git_sha: str,
        definition: dict[str, Any],
        planned: dict[str, Any],
        stages: list[str],
        reservation: dict[str, Any],
        inference: dict[str, Any] | None,
        plan_hash: str | None,
    ) -> list[dict[str, Any]]:
        model_id = definition["id"]
        resources = definition["resources"]
        details = {
            "run_mode": run_mode,
            "git_sha": git_sha,
            "plan_hash": plan_hash,
        }
        artifact_fingerprint = planned.get("fingerprint")
        evaluation_fingerprint = planned.get("evaluation_fingerprint")
        # Each job owns exactly one environment lane so no stage is executed
        # twice: quantization works on the artifact, the inference lane owns
        # the runtime smoke, evaluation/publish run on their own, and the
        # run-level report is built last so it can read every other lane.
        quantize_stages = [
            stage for stage in stages if stage in {"preflight", "quantize", "validate"}
        ]
        report_stages = [stage for stage in stages if stage == "report"]
        evaluate_stages = [stage for stage in stages if stage == "evaluate"]
        publish_stages = [stage for stage in stages if stage == "publish"]

        jobs = []
        if quantize_stages:
            argv = (
                planned.get("quantization_argv") or definition["workflow"]["quantize"]
            )
            jobs.append(
                self.jobs.create(
                    run_id=run_id,
                    model_id=model_id,
                    attempt_id=attempt_id,
                    kind="quantize",
                    executor=self.executor.name,
                    command=list(argv),
                    queue="model-quality-gpu",
                    gpu_count=resources["gpu_count"],
                    gpu_hours=planned.get("estimated_gpu_hours"),
                    stages=quantize_stages,
                    artifact_fingerprint=artifact_fingerprint,
                    evaluation_fingerprint=evaluation_fingerprint,
                    reservation_id=reservation["reservation_id"],
                    details=details,
                )
            )
        if "runtime-smoke" in stages:
            jobs.append(
                self.jobs.create(
                    run_id=run_id,
                    model_id=model_id,
                    attempt_id=attempt_id,
                    kind="inference",
                    executor=self.executor.name,
                    queue="model-quality-gpu",
                    gpu_count=resources.get(
                        "runtime_gpu_count", resources["gpu_count"]
                    ),
                    gpu_hours=0.0,
                    stages=["runtime-smoke"],
                    container_name=(inference["container_name"] if inference else None),
                    script=inference["script"] if inference else None,
                    result_file=inference["result_file"] if inference else None,
                    artifact_fingerprint=artifact_fingerprint,
                    reservation_id=reservation["reservation_id"],
                    details={
                        **details,
                        "inference_arguments": (
                            inference["arguments"] if inference else []
                        ),
                        "inference_source": (
                            "prestarted-container" if inference else "ci-executor"
                        ),
                        "inference_container_action": "none",
                    },
                )
            )
        if evaluate_stages:
            jobs.append(
                self.jobs.create(
                    run_id=run_id,
                    model_id=model_id,
                    attempt_id=attempt_id,
                    kind="evaluate",
                    executor=self.executor.name,
                    queue="model-quality-gpu",
                    gpu_count=resources.get(
                        "evaluation_gpu_count", resources["gpu_count"]
                    ),
                    gpu_hours=resources.get("estimated_eval_gpu_hours"),
                    stages=evaluate_stages,
                    artifact_fingerprint=artifact_fingerprint,
                    evaluation_fingerprint=evaluation_fingerprint,
                    reservation_id=reservation["reservation_id"],
                    details={
                        **details,
                        "suites": definition.get("evaluation", {}).get("suites", []),
                    },
                )
            )
        if report_stages:
            jobs.append(
                self.jobs.create(
                    run_id=run_id,
                    model_id=model_id,
                    attempt_id=attempt_id,
                    kind="report",
                    executor=self.executor.name,
                    queue="model-quality-gpu",
                    gpu_count=0,
                    gpu_hours=0.0,
                    stages=report_stages,
                    artifact_fingerprint=artifact_fingerprint,
                    evaluation_fingerprint=evaluation_fingerprint,
                    reservation_id=reservation["reservation_id"],
                    details=details,
                )
            )
        if publish_stages:
            jobs.append(
                self.jobs.create(
                    run_id=run_id,
                    model_id=model_id,
                    attempt_id=attempt_id,
                    kind="publish",
                    executor=self.executor.name,
                    queue="model-quality-gpu",
                    gpu_count=0,
                    gpu_hours=0.0,
                    stages=publish_stages,
                    artifact_fingerprint=artifact_fingerprint,
                    evaluation_fingerprint=evaluation_fingerprint,
                    reservation_id=reservation["reservation_id"],
                    details={
                        **details,
                        "upload_enabled": bool(
                            definition.get("upload", {}).get("enabled")
                        ),
                    },
                )
            )
        return jobs

    def _release_if_settled(self, job: dict[str, Any]) -> list[dict[str, Any]]:
        return settle_model_group(
            self.jobs,
            self.ledger,
            job,
            reason="canceled",
            cancel_active=False,
        )
