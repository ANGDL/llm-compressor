"""Deterministic priority and GPU-hour planning."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable
from uuid import uuid4

from .config import (
    PRIORITIES,
    RUN_MODES,
    ConfigError,
    evaluation_fingerprint,
    model_fingerprint,
)

_PRIORITY_RANK = {priority: rank for rank, priority in enumerate(PRIORITIES)}


def _parse_filter(model_filter: str | Iterable[str] | None) -> set[str] | None:
    if model_filter is None or model_filter == "all":
        return None
    if isinstance(model_filter, str):
        values = model_filter.split(",")
    else:
        values = model_filter
    selected = {value.strip() for value in values if value.strip()}
    return selected or None


def _job_cost(model: dict[str, Any], run_mode: str) -> float:
    resources = model["resources"]
    quantize = float(resources["estimated_gpu_hours"])
    evaluate = float(resources.get("estimated_eval_gpu_hours", 0))
    if run_mode == "quantize":
        return quantize
    if run_mode == "quantize_and_eval":
        return quantize + evaluate
    if run_mode == "eval_only":
        return evaluate
    return 0.0


def build_execution_plan(
    config: dict[str, Any],
    *,
    git_sha: str,
    max_gpu_hours: float,
    model_filter: str | Iterable[str] | None = None,
    priority_override: str | None = None,
    run_mode: str = "quantize",
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    """Select enabled jobs deterministically within a GPU-hour budget."""

    if max_gpu_hours <= 0:
        raise ConfigError("max_gpu_hours must be positive")
    if run_mode not in RUN_MODES:
        raise ConfigError(f"run_mode must be one of {RUN_MODES}")
    if priority_override in (None, "none"):
        priority_override = None
    elif priority_override not in PRIORITIES:
        raise ConfigError(f"priority_override must be one of {PRIORITIES} or none")

    requested = _parse_filter(model_filter)
    known = {model["id"] for model in config["models"]}
    unknown = requested - known if requested is not None else set()
    if unknown:
        raise ConfigError(f"unknown models in filter: {', '.join(sorted(unknown))}")

    candidates = []
    for model in config["models"]:
        if not model["enabled"] and not (
            requested is not None
            and model["id"] in requested
            and run_mode in {"eval_only", "upload_only"}
        ):
            continue
        if requested is not None and model["id"] not in requested:
            continue
        job = dict(model)
        if priority_override is not None:
            job["priority"] = priority_override
        job["fingerprint"] = model_fingerprint(job)
        job["evaluation_fingerprint"] = evaluation_fingerprint(job, job["fingerprint"])
        candidates.append(job)

    candidates.sort(
        key=lambda job: (
            _PRIORITY_RANK[job["priority"]],
            job["resources"]["estimated_gpu_hours"],
            job["id"],
        )
    )

    selected = []
    deferred = []
    consumed = 0.0
    for job in candidates:
        cost = _job_cost(job, run_mode)
        if run_mode in {"quantize_and_eval", "eval_only"} and cost == 0:
            raise ConfigError(
                f"model {job['id']!r} requires resources.estimated_eval_gpu_hours "
                f"for run_mode={run_mode}"
            )
        summary = {
            "id": job["id"],
            "priority": job["priority"],
            "gpu_count": job["resources"]["gpu_count"],
            "estimated_gpu_hours": cost,
            "run_mode": run_mode,
            "fingerprint": job["fingerprint"],
            "evaluation_fingerprint": job["evaluation_fingerprint"],
            "upload_enabled": bool(job.get("upload", {}).get("enabled", False)),
        }
        if consumed + cost <= max_gpu_hours:
            summary["status"] = "SELECTED"
            selected.append(summary)
            consumed += cost
        else:
            summary["status"] = "DEFERRED_BUDGET"
            summary["reason"] = (
                f"requires {cost:g} GPU-hours with "
                f"{max_gpu_hours - consumed:g} remaining"
            )
            deferred.append(summary)

    if run_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{timestamp}_{git_sha[:12]}"
    if attempt_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        attempt_id = f"attempt-{timestamp}-{uuid4().hex[:8]}"

    return {
        "schema_version": 1,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "git_sha": git_sha,
        "run_mode": run_mode,
        "budget": {
            "max_gpu_hours": float(max_gpu_hours),
            "selected_gpu_hours": consumed,
            "remaining_gpu_hours": float(max_gpu_hours) - consumed,
        },
        "selected": selected,
        "deferred": deferred,
    }
