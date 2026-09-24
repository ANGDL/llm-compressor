"""Generic quality metric comparison independent of compression modifiers."""

from __future__ import annotations

import math
from typing import Any

from .config import ConfigError


def _finite_number(record: dict[str, Any], field: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"metric {record.get('name')!r}: {field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"metric {record.get('name')!r}: {field} must be finite")
    return result


def compare_metric(record: dict[str, Any]) -> dict[str, Any]:
    """Compare a base/compressed metric without rounding gate values."""

    name = record.get("name")
    if not isinstance(name, str) or not name:
        raise ConfigError("metric.name must be a non-empty string")
    direction = record.get("direction")
    if direction not in {"higher", "lower"}:
        raise ConfigError(f"metric {name!r}: direction must be higher or lower")
    base = _finite_number(record, "base_value")
    compressed = _finite_number(record, "compressed_value")

    recovery = None
    relative_degradation = None
    if direction == "higher" and base > 0:
        recovery = compressed / base
        relative_degradation = (base - compressed) / base
    elif direction == "lower" and base > 0:
        recovery = base / compressed if compressed > 0 else None
        relative_degradation = (compressed - base) / base

    failures = []
    if record.get("min_recovery") is not None:
        minimum = _finite_number(record, "min_recovery")
        if direction == "lower" and base > 0 and compressed <= 0:
            pass
        elif recovery is None:
            failures.append("recovery is undefined because baseline is not positive")
        elif recovery < minimum:
            failures.append(f"recovery {recovery:.8g} is below {minimum:.8g}")

    if record.get("absolute_floor") is not None:
        floor = _finite_number(record, "absolute_floor")
        if direction != "higher":
            raise ConfigError(
                f"metric {name!r}: absolute_floor requires direction=higher"
            )
        if compressed < floor:
            failures.append(f"compressed value {compressed:.8g} is below {floor:.8g}")

    if record.get("max_relative_increase") is not None:
        maximum = _finite_number(record, "max_relative_increase")
        if direction != "lower":
            raise ConfigError(
                f"metric {name!r}: max_relative_increase requires direction=lower"
            )
        if relative_degradation is None:
            failures.append(
                "relative increase is undefined because baseline is not positive"
            )
        elif relative_degradation > maximum:
            failures.append(
                f"relative increase {relative_degradation:.8g} exceeds {maximum:.8g}"
            )

    result = dict(record)
    result.update(
        {
            "base_value": base,
            "compressed_value": compressed,
            "absolute_change": compressed - base,
            "recovery": recovery,
            "relative_degradation": relative_degradation,
            "status": "FAIL" if failures else "PASS",
            "failures": failures,
        }
    )
    if "perplexity" in name and base > 0 and compressed > 0:
        result["delta_nll"] = math.log(compressed) - math.log(base)
    return result


def compare_evaluation(document: dict[str, Any]) -> dict[str, Any]:
    """Normalize and gate a generic evaluator result document."""

    if not isinstance(document, dict):
        raise ConfigError("evaluation result must be an object")
    metrics = document.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ConfigError("evaluation result must contain a non-empty metrics list")
    compared = [compare_metric(metric) for metric in metrics]
    return {
        **document,
        "schema_version": 1,
        "status": (
            "FAIL" if any(metric["status"] == "FAIL" for metric in compared) else "PASS"
        ),
        "metrics": compared,
    }
