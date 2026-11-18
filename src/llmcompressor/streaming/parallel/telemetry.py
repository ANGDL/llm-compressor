"""Phase-local EP workload reporting without device synchronization."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from time import perf_counter

import torch

from .._logging import streaming_logger


@dataclass
class _OwnerWork:
    tokens: int = 0
    dispatches: int = 0
    calls: int = 0
    max_tokens: int = 0
    experts: set[str] = field(default_factory=set)
    queued: int = 0
    pending: int = 0
    modules: int = 0
    host_seconds: float = 0.0
    started_at: float | None = None
    state: str = "waiting"


def _visible_device(device: torch.device) -> str:
    if device.type != "cuda" or device.index is None:
        return "-"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        visible = os.environ.get("XPU_VISIBLE_DEVICES")
    if visible is None:
        return str(device.index)
    identifiers = visible.split(",")
    return identifiers[device.index].strip() if device.index < len(identifiers) else "?"


class EPPhaseLogger:
    """Report host progress and routed token load separately for each EP phase."""

    def __init__(
        self,
        devices: Sequence[torch.device],
        *,
        label: str,
        phase: str,
        token_capacity: Mapping[torch.device, int] | None = None,
        total_batches: int | None = None,
        update_modules: Mapping[torch.device, int] | None = None,
        interval: float = 30.0,
    ) -> None:
        if interval <= 0:
            raise ValueError("EP reporting interval must be positive")
        self.devices = tuple(devices)
        self.label = label
        self.phase = phase
        self.token_capacity = dict(token_capacity or {})
        self.total_batches = total_batches
        self.update_modules = (
            dict(update_modules) if update_modules is not None else None
        )
        self.interval = interval
        self._owners = {device: _OwnerWork() for device in self.devices}
        self._lock = Lock()
        self._stop = Event()
        self._thread: Thread | None = None
        self._started_at = perf_counter()
        self._batches = 0
        self._host_times = {
            "load": 0.0,
            "fx": 0.0,
            "pack": 0.0,
            "wait": 0.0,
            "boundary": 0.0,
        }
        self._activity = "starting"

    def __enter__(self):
        self._started_at = perf_counter()
        self.report("start")
        self._thread = Thread(target=self._heartbeat, name="ep-progress", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.report("failed" if exc_type is not None else "complete")

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.interval):
            self.report("progress")

    def advance(self, batches: int) -> None:
        with self._lock:
            self._batches += batches

    def activity(self, name: str) -> None:
        with self._lock:
            self._activity = name

    def host_time(self, name: str, seconds: float) -> None:
        with self._lock:
            self._host_times[name] += seconds

    @contextmanager
    def primary_work(self, name: str, activity: str):
        self.activity(activity)
        started_at = perf_counter()
        try:
            yield
        finally:
            self.host_time(name, perf_counter() - started_at)

    def queue(self, device: torch.device, dispatches: int) -> None:
        with self._lock:
            self._owners[device].queued = dispatches

    def dispatch(self, device: torch.device, tokens: int, names: Sequence[str]) -> None:
        with self._lock:
            owner = self._owners[device]
            owner.tokens += tokens
            owner.dispatches += 1
            owner.calls += len(names)
            owner.max_tokens = max(owner.max_tokens, tokens)
            owner.experts.update(names)
            owner.pending += 1
            owner.state = "queued"

    def joined(self, device: torch.device) -> None:
        with self._lock:
            owner = self._owners[device]
            owner.pending -= 1
            if owner.state != "failed":
                owner.state = "waiting"

    @contextmanager
    def owner_work(self, device: torch.device, *, modules: int = 0):
        started_at = perf_counter()
        with self._lock:
            owner = self._owners[device]
            owner.started_at = started_at
            owner.state = "running"
        try:
            yield
        except BaseException:
            with self._lock:
                owner.state = "failed"
            raise
        else:
            with self._lock:
                owner.modules += modules
                owner.state = "returned"
        finally:
            with self._lock:
                owner.host_seconds += perf_counter() - started_at
                owner.started_at = None

    def format(self, status: str) -> str:
        now = perf_counter()
        with self._lock:
            elapsed = now - self._started_at
            total_tokens = sum(owner.tokens for owner in self._owners.values())
            mean_tokens = total_tokens / len(self.devices) if self.devices else 0
            imbalance = (
                max(owner.tokens for owner in self._owners.values()) / mean_tokens
                if mean_tokens
                else 0
            )
            progress = (
                f" | batches={self._batches}/{self.total_batches}"
                if self.total_batches is not None
                else ""
            )
            lines = [
                f"{self.label} | EP {self.phase} | {status} | "
                f"elapsed={elapsed:.1f}s{progress} | primary={self._activity}",
            ]
            if self.update_modules is None:
                lines.extend(
                    [
                        f"  routed_tokens={total_tokens:,} | "
                        f"token_max/mean={imbalance:.2f} | host_s: "
                        + " ".join(
                            f"{name}={seconds:.2f}"
                            for name, seconds in self._host_times.items()
                        ),
                        "  rank device   visible  routed_tokens  share  experts/calls "
                        "dispatches  avg/max_tokens  capacity  peak/cap  "
                        "queue/pending  host_s  host_state",
                    ]
                )
            else:
                lines.append(
                    "  rank device   visible  modules_done/planned  host_s  host_state"
                )
            for rank, device in enumerate(self.devices):
                owner = self._owners[device]
                host_seconds = owner.host_seconds + (
                    now - owner.started_at if owner.started_at is not None else 0
                )
                prefix = f"  {rank:>4} {str(device):<8} {_visible_device(device):>7}"
                if self.update_modules is not None:
                    lines.append(
                        f"{prefix}  "
                        f"{owner.modules:>7}/{self.update_modules.get(device, 0):<7} "
                        f"{host_seconds:>7.2f}  {owner.state}"
                    )
                    continue
                share = owner.tokens / total_tokens if total_tokens else 0
                average = owner.tokens / owner.dispatches if owner.dispatches else 0
                capacity = self.token_capacity.get(device)
                fill = f"{owner.max_tokens / capacity:.1%}" if capacity else "-"
                lines.append(
                    f"{prefix}  {owner.tokens:>13,}  {share:>5.1%}  "
                    f"{len(owner.experts):>3}/{owner.calls:<6} "
                    f"{owner.dispatches:>10}  "
                    f"{average:>7.0f}/{owner.max_tokens:<7} "
                    f"{str(capacity) if capacity else '-':>8}  {fill:>8}  "
                    f"{owner.queued:>5}/{owner.pending:<7} "
                    f"{host_seconds:>7.2f}  {owner.state}"
                )
            lines.append(
                "  host_s/state describe CPU work and submission; "
                "returned/pending do not measure device completion or utilization."
            )
            return "\n".join(lines)

    def report(self, status: str) -> None:
        streaming_logger.info(self.format(status))
