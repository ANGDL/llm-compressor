"""Run due model quality schedules; the Phase-4 timer entry point.

A cron unit or systemd timer calls this once per minute. One occurrence always
maps to one run: replaying the same occurrence returns the run that was already
recorded instead of charging GPU-hours twice.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from .app import ModelQualityAPI
from .audit import ValidationError
from .ops import NotificationService, ScheduleStore, run_due_schedules
from .store import RunStore


def tick(
    *,
    root: str | Path,
    config_path: str | Path,
    git_sha: str | None = None,
    actor: str = "scheduler",
    now: datetime | None = None,
) -> dict:
    """Start every due schedule once and return the per-schedule results."""

    api = ModelQualityAPI(RunStore(root), config_path=str(config_path), git_sha=git_sha)
    if api.control is None:
        raise ValidationError("a model quality --config path is required")
    return run_due_schedules(
        schedules=ScheduleStore(root),
        control=api.control,
        notifications=NotificationService(root),
        actor=actor,
        now=now,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Start every due model quality schedule once"
    )
    parser.add_argument("--runs-root", default=os.getenv("MODEL_QUALITY_RUNS_ROOT"))
    parser.add_argument("--config", default=os.getenv("MODEL_QUALITY_CONFIG"))
    parser.add_argument("--git-sha", default=os.getenv("MODEL_QUALITY_GIT_SHA"))
    parser.add_argument("--actor", default="scheduler")
    args = parser.parse_args(argv)
    if not args.runs_root:
        print("--runs-root or MODEL_QUALITY_RUNS_ROOT is required", file=sys.stderr)
        return 2
    if not args.config:
        print("--config or MODEL_QUALITY_CONFIG is required", file=sys.stderr)
        return 2
    result = tick(
        root=args.runs_root,
        config_path=args.config,
        git_sha=args.git_sha,
        actor=args.actor,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
