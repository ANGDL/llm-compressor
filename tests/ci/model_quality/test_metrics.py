import pytest

from ci.model_quality.config import ConfigError
from ci.model_quality.metrics import compare_evaluation, compare_metric


def test_higher_metric_reports_unrounded_recovery():
    result = compare_metric(
        {
            "name": "exact_match",
            "direction": "higher",
            "base_value": 0.75,
            "compressed_value": 0.71,
            "min_recovery": 0.95,
            "absolute_floor": 0.70,
        }
    )

    assert result["status"] == "FAIL"
    assert result["recovery"] == pytest.approx(0.9466666667)


def test_lower_metric_reports_relative_increase_and_delta_nll():
    result = compare_metric(
        {
            "name": "word_perplexity",
            "direction": "lower",
            "base_value": 6.24,
            "compressed_value": 6.85,
            "max_relative_increase": 0.10,
        }
    )

    assert result["status"] == "PASS"
    assert result["relative_degradation"] == pytest.approx(0.0977564103)
    assert result["recovery"] == pytest.approx(0.9109489051)
    assert result["delta_nll"] == pytest.approx(0.0932684699)


def test_zero_baseline_rejects_ratio_gate():
    result = compare_metric(
        {
            "name": "score",
            "direction": "higher",
            "base_value": 0.0,
            "compressed_value": 0.0,
            "min_recovery": 0.95,
        }
    )

    assert result["status"] == "FAIL"
    assert result["recovery"] is None


def test_zero_compressed_lower_metric_is_json_safe():
    result = compare_metric(
        {
            "name": "error",
            "direction": "lower",
            "base_value": 0.5,
            "compressed_value": 0.0,
            "min_recovery": 0.95,
        }
    )

    assert result["status"] == "PASS"
    assert result["recovery"] is None


def test_evaluation_fails_when_any_metric_fails():
    result = compare_evaluation(
        {
            "suite_id": "smoke-v1",
            "metrics": [
                {
                    "name": "accuracy",
                    "direction": "higher",
                    "base_value": 0.8,
                    "compressed_value": 0.6,
                    "min_recovery": 0.95,
                }
            ],
        }
    )

    assert result["status"] == "FAIL"


def test_invalid_direction_is_configuration_error():
    with pytest.raises(ConfigError, match="direction"):
        compare_metric(
            {
                "name": "accuracy",
                "direction": "sideways",
                "base_value": 1.0,
                "compressed_value": 1.0,
            }
        )
