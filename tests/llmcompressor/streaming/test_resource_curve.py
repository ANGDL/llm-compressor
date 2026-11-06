from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.measure_streaming_resource_curve import (
    build_report,
    parse_case_spec,
    parse_probe_output,
    run_case,
    run_recovery_case,
)


def test_parse_case_spec():
    case = parse_case_spec("depth8=/tmp/checkpoint=with-equals")

    assert case.label == "depth8"
    assert case.checkpoint == Path("/tmp/checkpoint=with-equals")


def test_parse_case_spec_rejects_missing_parts():
    with pytest.raises(ValueError, match="LABEL=CHECKPOINT_DIRECTORY"):
        parse_case_spec("depth8")


def test_parse_probe_output_uses_last_valid_json_line():
    output = "\n".join(
        [
            "streaming log",
            json.dumps({"unrelated": True}),
            json.dumps(
                {
                    "checkpoint_bytes": 32,
                    "incremental_peak_bytes": 12,
                    "output": "/tmp/output",
                }
            ),
        ]
    )

    assert parse_probe_output(output)["incremental_peak_bytes"] == 12


def test_build_report_preserves_case_order_and_curve_points():
    report = build_report(
        [
            {
                "label": "depth8",
                "returncode": 0,
                "checkpoint_bytes": 8,
                "incremental_peak_bytes": 3,
            },
            {
                "label": "depth32",
                "returncode": 1,
                "error": "probe failed",
            },
            {
                "label": "depth128",
                "returncode": 0,
                "checkpoint_bytes": 128,
                "incremental_peak_bytes": 15,
            },
        ]
    )

    assert [item["label"] for item in report["cases"]] == [
        "depth8",
        "depth32",
        "depth128",
    ]
    assert report["curve"] == {
        "checkpoint_bytes": [8, 128],
        "incremental_peak_bytes": [3, 15],
    }


@pytest.mark.parametrize("spec", ["../depth8=/tmp/model", "/tmp/x=/tmp/model", "=x"])
def test_case_labels_cannot_escape_work_directory(spec):
    with pytest.raises(ValueError):
        parse_case_spec(spec)


def test_failed_probe_is_reported_without_curve_point(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tools.measure_streaming_resource_curve.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=9, stdout="partial log", stderr="out of memory"
        ),
    )
    measurement = run_case(
        parse_case_spec("depth8=/checkpoint"),
        probe=Path("probe.py"), work_root=tmp_path,
        output_root=tmp_path / "output", sequence_length=8,
    )
    assert measurement["returncode"] == 9
    assert measurement["stderr"] == "out of memory"
    assert build_report([measurement])["curve"]["checkpoint_bytes"] == []


def test_recovery_preparation_and_measurement_are_separate_processes(
    tmp_path, monkeypatch
):
    modes = []

    def run(command, **kwargs):
        modes.append(command[2])
        assert kwargs["capture_output"]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"depth": 8, "peak_live_batches": 1}),
            stderr="",
        )

    monkeypatch.setattr("tools.measure_streaming_resource_curve.subprocess.run", run)
    result = run_recovery_case(
        8, probe=Path("probe.py"), work_root=tmp_path,
        batch_count=4, width=1024,
    )
    assert modes == ["prepare", "measure"]
    assert result["peak_live_batches"] == 1
