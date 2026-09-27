from __future__ import annotations

import importlib.metadata
import json
from argparse import Namespace

from ci.model_quality.evaluators import evalscope_api


def test_load_report_metrics_reads_evalscope_v2_report(tmp_path) -> None:
    report = tmp_path / "reports" / "demo" / "gsm8k.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "name": "demo@gsm8k",
                "metrics": [
                    {
                        "identity": {"key": "accuracy"},
                        "score": 0.75,
                    }
                ],
            }
        )
    )
    assert evalscope_api.load_report_metrics(tmp_path, "demo", "gsm8k") == {
        "accuracy": 0.75
    }


def test_load_report_metrics_reads_evalscope_1_12_identity(tmp_path) -> None:
    report = tmp_path / "reports" / "demo" / "gsm8k.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "name": "demo@gsm8k",
                "metrics": [
                    {
                        "identity": {"name": "accuracy", "aggregation": "mean"},
                        "score": 0.75,
                    }
                ],
            }
        )
    )
    assert evalscope_api.load_report_metrics(tmp_path, "demo", "gsm8k") == {
        "accuracy": 0.75
    }


def test_evalscope_adapter_accepts_reference_baseline(tmp_path, monkeypatch) -> None:
    output = tmp_path / "evaluation.json"
    args = Namespace(
        api_url="http://compressed:30000",
        model="compressed",
        dataset="gsm8k",
        metrics_json='[{"name":"accuracy","direction":"higher","min_recovery":0.95}]',
        limit=10,
        seed=42,
        api_key="EMPTY",
        eval_batch_size=8,
        generation_config="{}",
        dataset_args="{}",
        work_dir=tmp_path / "work",
        output=output,
        reference_values_json='{"accuracy":0.8}',
        reference_id="gsm8k-evalscope-v1",
        base_url=None,
        base_model=None,
    )
    monkeypatch.setattr(evalscope_api, "parse_args", lambda: args)
    monkeypatch.setattr(
        evalscope_api,
        "run_evalscope",
        lambda *_args, **_kwargs: {"accuracy": 0.78},
    )
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: [])

    assert evalscope_api.main() == 0
    payload = json.loads(output.read_text())
    assert payload["evaluator"] == "evalscope/openai_api"
    assert payload["metrics"][0]["recovery"] == 0.975
    assert payload["base_metadata"]["reference_id"] == "gsm8k-evalscope-v1"
