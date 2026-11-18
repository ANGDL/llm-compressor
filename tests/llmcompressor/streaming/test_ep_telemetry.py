from __future__ import annotations

import threading

import pytest
import torch
from torch import nn

from llmcompressor.streaming.parallel.modifier_update import (
    OwnerModuleGroup,
    OwnerUpdateScheduler,
)
from llmcompressor.streaming.parallel.telemetry import EPPhaseLogger
from llmcompressor.streaming.pipeline import _target_label


def test_phase_reports_routed_tokens_capacity_and_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    devices = (torch.device("cuda:0"), torch.device("cuda:1"))
    metrics = EPPhaseLogger(
        devices,
        label="[002/248]",
        phase="calibration",
        total_batches=4,
        token_capacity={devices[0]: 100, devices[1]: 200},
    )
    metrics.dispatch(devices[0], 20, ("experts.0", "experts.2"))
    metrics.dispatch(devices[0], 30, ("experts.0",))
    metrics.dispatch(devices[1], 50, ("experts.1",))
    metrics.advance(2)
    text = metrics.format("progress")

    assert "batches=2/4" in text
    assert "routed_tokens=100" in text
    assert "token_max/mean=1.00" in text
    assert "cuda:0" in text and "cuda:1" in text
    assert "30.0%" in text and "25.0%" in text
    rows = [line.split() for line in text.splitlines() if "cuda:" in line]
    assert [row[2] for row in rows] == ["4", "5"]
    assert rows[0][5] == "2/3"
    assert "do not measure device completion or utilization" in text


def test_observation_heartbeat_never_synchronizes_devices(monkeypatch):
    reported = []
    progress_reported = threading.Event()

    def report(self, status):
        reported.append(self.format(status))
        if status == "progress":
            progress_reported.set()

    def unexpected_device_access(*args, **kwargs):
        raise AssertionError("observation must not access or synchronize devices")

    monkeypatch.setattr(EPPhaseLogger, "report", report)
    monkeypatch.setattr(torch.cuda, "synchronize", unexpected_device_access)
    monkeypatch.setattr(torch.cuda, "Event", unexpected_device_access)
    monkeypatch.setattr(torch.cuda, "current_device", unexpected_device_access)
    with EPPhaseLogger(
        (torch.device("cuda:0"),),
        label="group",
        phase="calibration",
        interval=0.01,
    ):
        assert progress_reported.wait(timeout=5)
    assert "| start |" in reported[0]
    assert "| complete |" in reported[-1]


def test_owner_update_telemetry_preserves_concurrency_and_counts():
    devices = tuple(torch.device(f"cpu:{index}") for index in range(4))
    groups = tuple(OwnerModuleGroup(device, (nn.Linear(2, 2),)) for device in devices)
    metrics = EPPhaseLogger(
        devices,
        label="group",
        phase="modifier update",
        update_modules={device: 1 for device in devices},
    )
    barrier = threading.Barrier(4)

    def update(owner, modules):
        barrier.wait(timeout=5)
        return len(modules)

    results = OwnerUpdateScheduler().run(groups, update=update, telemetry=metrics)
    assert results == tuple((device, 1) for device in devices)
    rows = [
        line.split()
        for line in metrics.format("complete").splitlines()
        if "cpu:" in line
    ]
    assert len(rows) == 4
    assert all(row[3] == "1/1" for row in rows)
    assert all(row[-1] == "returned" for row in rows)


def test_owner_failure_is_visible_without_hiding_exception():
    device = torch.device("cpu:0")
    metrics = EPPhaseLogger(
        (device,),
        label="group",
        phase="modifier update",
        update_modules={device: 1},
    )

    def fail(owner, modules):
        raise RuntimeError("update failed")

    with pytest.raises(RuntimeError, match="update failed"):
        OwnerUpdateScheduler().run(
            (OwnerModuleGroup(device, (nn.Linear(2, 2),)),),
            update=fail,
            telemetry=metrics,
        )
    assert "failed" in metrics.format("failed")
    assert "0/1" in metrics.format("failed")


def test_target_progress_abbreviates_groups_without_changing_group_count():
    label = _target_label(1, 248, " + ".join(f"experts.{index}" for index in range(64)))
    assert label == "[002/248] experts.0 ... experts.63 (64 targets)"
