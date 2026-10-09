from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ci.model_quality.evaluators import informational


def test_informational_writes_passing_scores(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(informational, "_discover_model", lambda url, key: "served-model")
    monkeypatch.setattr(
        informational,
        "_scores",
        lambda tool, args, model: {"gsm8k/acc": 0.73, "gsm8k/acc_stderr": 0.01},
    )
    output = tmp_path / "evaluation.json"
    argv = [
        "informational",
        "--tool",
        "lm_eval",
        "--api-url",
        "http://127.0.0.1:30000",
        "--dataset",
        "gsm8k",
        "--work-dir",
        str(tmp_path / "work"),
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert informational.main() == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    # Reference value 1 + min_recovery 0 ⇒ the stage records scores and PASSES.
    assert result["status"] == "PASS"
    by_name = {metric["name"]: metric for metric in result["metrics"]}
    assert by_name["gsm8k/acc"]["compressed_value"] == pytest.approx(0.73)
    assert by_name["gsm8k/acc"]["base_value"] == 1.0
    assert by_name["gsm8k/acc"]["min_recovery"] == 0.0


def test_informational_errors_when_no_scores(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(informational, "_discover_model", lambda url, key: "served-model")
    monkeypatch.setattr(informational, "_scores", lambda tool, args, model: {})
    argv = [
        "informational",
        "--tool",
        "evalscope",
        "--api-url",
        "http://127.0.0.1:30000",
        "--dataset",
        "ceval",
        "--work-dir",
        str(tmp_path / "work"),
        "--output",
        str(tmp_path / "evaluation.json"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        informational.main()
