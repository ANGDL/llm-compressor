from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_incremental_peak_stays_below_source_checkpoint(tmp_path):
    checkpoint_value = os.environ.get(
        "LLMCOMPRESSOR_STREAMING_RESOURCE_CHECKPOINT"
    )
    if not checkpoint_value:
        pytest.skip(
            "set LLMCOMPRESSOR_STREAMING_RESOURCE_CHECKPOINT to a local "
            "safetensors checkpoint"
        )
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_dir():
        pytest.fail(f"checkpoint does not exist: {checkpoint}")

    probe = Path(__file__).with_name("resource_probe.py")
    result = subprocess.run(
        [
            sys.executable,
            str(probe),
            str(checkpoint),
            str(tmp_path / "work"),
            str(tmp_path / "output"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    measurements = json.loads(result.stdout.strip().splitlines()[-1])

    assert Path(measurements["output"], "FINALIZED").is_file()
    assert (
        measurements["incremental_peak_bytes"]
        < measurements["checkpoint_bytes"]
    ), measurements
