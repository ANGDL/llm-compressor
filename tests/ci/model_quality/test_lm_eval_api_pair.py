from __future__ import annotations

import json
from argparse import Namespace

from ci.model_quality.evaluators import lm_eval_api_pair


def test_metric_specs_validate_shape() -> None:
    specs = lm_eval_api_pair.metric_specs(
        '[{"name":"exact_match","direction":"higher","min_recovery":0.95}]'
    )
    assert specs[0]["name"] == "exact_match"


def test_api_pair_compares_endpoint_results(tmp_path, monkeypatch) -> None:
    output = tmp_path / "evaluation.json"
    work = tmp_path / "work"
    args = Namespace(
        base_url="http://base:30000",
        reference_values_json=None,
        reference_id=None,
        compressed_url="http://compressed:30000",
        base_model="base",
        compressed_model="compressed",
        tokenizer=None,
        task="gsm8k",
        metrics_json='[{"name":"exact_match","direction":"higher","min_recovery":0.95}]',
        num_fewshot=5,
        limit=10,
        batch_size=1,
        max_gen_toks=128,
        seed=42,
        api_key="EMPTY",
        work_dir=work,
        output=output,
    )
    monkeypatch.setattr(lm_eval_api_pair, "parse_args", lambda: args)
    monkeypatch.setattr(
        lm_eval_api_pair,
        "run_endpoint",
        lambda _args, *, base_url, model_name: {
            "metrics": {"exact_match": 1.0 if model_name == "base" else 0.96},
            "versions": {"gsm8k": 1},
        },
    )
    monkeypatch.setattr(lm_eval_api_pair, "package_versions", lambda: {})

    assert lm_eval_api_pair.main() == 0
    payload = json.loads(output.read_text())
    assert payload["status"] == "PASS"
    assert payload["metrics"][0]["recovery"] == 0.96
    assert (work / "base.json").is_file()
    assert (work / "compressed.json").is_file()


def test_api_pair_accepts_versioned_reference_values(tmp_path, monkeypatch) -> None:
    output = tmp_path / "evaluation.json"
    args = Namespace(
        base_url=None,
        reference_values_json='{"exact_match":0.8}',
        reference_id="gsm8k-5shot-seed42-v1",
        compressed_url="http://compressed:30000",
        base_model="unused",
        compressed_model="compressed",
        tokenizer=None,
        task="gsm8k",
        metrics_json='[{"name":"exact_match","direction":"higher","min_recovery":0.95}]',
        num_fewshot=5,
        limit=10,
        batch_size=1,
        max_gen_toks=128,
        seed=42,
        api_key="EMPTY",
        work_dir=tmp_path / "work",
        output=output,
    )
    monkeypatch.setattr(lm_eval_api_pair, "parse_args", lambda: args)
    monkeypatch.setattr(
        lm_eval_api_pair,
        "run_endpoint",
        lambda _args, *, base_url, model_name: {
            "metrics": {"exact_match": 0.78},
            "versions": {"gsm8k": 1},
        },
    )
    monkeypatch.setattr(lm_eval_api_pair, "package_versions", lambda: {})

    assert lm_eval_api_pair.main() == 0
    payload = json.loads(output.read_text())
    assert payload["metrics"][0]["base_value"] == 0.8
    assert payload["metrics"][0]["recovery"] == 0.975
    assert payload["base_metadata"]["source"] == "reference-values"
    assert payload["base_metadata"]["reference_id"] == "gsm8k-5shot-seed42-v1"
