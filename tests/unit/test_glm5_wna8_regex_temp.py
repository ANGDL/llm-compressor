import json
import re
from pathlib import Path

import pytest


INDEX_PATH = Path(
    "/Users/ang/Downloads/shijinxiang/models/GLM-5-w4a8/model.safetensors.index.json"
)

# Keep these patterns in sync with examples/quantizing_moe/glm5_wNa8.py
EXPERTS_PATTERNS = [
    r"re:.*mlp\.experts\..*\.(gate_proj|up_proj|down_proj)$",
    r"re:.*mlp\.shared_experts(?:\..+)?\.(gate_proj|up_proj|down_proj)$",
]

OTHER_LINEAR_PATTERNS = [
    r"re:.*self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)$",
    r"re:.*mlp\.(gate_proj|up_proj|down_proj)$",
]

IGNORE_PATTERNS = [
    r"re:.*mlp\.gate$",
    r"re:.*embed_tokens$",
    r"re:.*indexer.*",
    "lm_head",
]


def _matches(pattern: str, module_name: str) -> bool:
    if pattern.startswith("re:"):
        return re.fullmatch(pattern[3:], module_name) is not None
    return module_name == pattern


def _matches_any(patterns: list[str], module_name: str) -> bool:
    return any(_matches(pattern, module_name) for pattern in patterns)


def _preview(values: list[str], limit: int = 8) -> list[str]:
    return values[:limit]


@pytest.mark.unit
def test_glm5_wna8_regex_matches_weight_scale_entries():
    if not INDEX_PATH.exists():
        pytest.skip(f"External index file not found: {INDEX_PATH}")

    with INDEX_PATH.open("r", encoding="utf-8") as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    weight_modules = sorted(
        key[: -len(".weight")] for key in weight_map if key.endswith(".weight")
    )
    assert weight_modules, "No module .weight entries found in safetensors index"

    experts_modules = sorted(
        module
        for module in weight_modules
        if _matches_any(EXPERTS_PATTERNS, module)
    )
    other_linear_modules = sorted(
        module
        for module in weight_modules
        if _matches_any(OTHER_LINEAR_PATTERNS, module)
    )

    assert experts_modules, "No module matched experts/shared_experts regex"
    assert other_linear_modules, "No module matched other linear regex"
    assert any(".mlp.shared_experts" in name for name in experts_modules), (
        "shared_experts regex did not match any module"
    )

    missing_experts_scales = sorted(
        f"{module}.weight_scale"
        for module in experts_modules
        if f"{module}.weight_scale" not in weight_map
    )
    missing_other_scales = sorted(
        f"{module}.weight_scale"
        for module in other_linear_modules
        if f"{module}.weight_scale" not in weight_map
    )

    assert not missing_experts_scales, (
        "Some experts/shared_experts modules are missing weight_scale entries: "
        f"{_preview(missing_experts_scales)}"
    )
    assert not missing_other_scales, (
        "Some other linear modules are missing weight_scale entries: "
        f"{_preview(missing_other_scales)}"
    )

    ignored_and_matched = sorted(
        module
        for module in weight_modules
        if _matches_any(IGNORE_PATTERNS, module)
        and (
            _matches_any(EXPERTS_PATTERNS, module)
            or _matches_any(OTHER_LINEAR_PATTERNS, module)
        )
    )
    assert not ignored_and_matched, (
        "Ignored modules should not be captured by quantization regex: "
        f"{_preview(ignored_and_matched)}"
    )

    experts_unexpected = sorted(
        module
        for module in experts_modules
        if ".mlp.experts." not in module and ".mlp.shared_experts" not in module
    )
    other_unexpected = sorted(
        module
        for module in other_linear_modules
        if ".mlp.experts." in module or ".mlp.shared_experts" in module
    )
    assert not experts_unexpected, (
        "Experts regex matched unexpected modules: "
        f"{_preview(experts_unexpected)}"
    )
    assert not other_unexpected, (
        "Other-linear regex should not match experts/shared_experts modules: "
        f"{_preview(other_unexpected)}"
    )
