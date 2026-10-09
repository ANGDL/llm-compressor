import ast
import re
from pathlib import Path

import pytest
import torch
from compressed_tensors.utils.match import match_name, match_named_modules


SCRIPT_PATH = Path(__file__).parents[2] / "examples/multimodal_vision/glm5_next_wNa8.py"

# Keep in sync with the example's own constants via the AST loader below.
WANTED_NAMES = {
    "_config_value",
    "_glm5_next_runtime_name_aliases",
    "_runtime_portable_pattern",
    "build_glm5_next_w8a8_ignores",
    "build_glm5_next_w8a8_targets",
    "collect_glm5_next_ignored_checkpoint_names",
    "GLM5_NEXT_HF_TEXT_ROOT",
    "GLM5_NEXT_MODEL_ROOT",
    "GLM5_NEXT_VLLM_FUSED_IGNORES",
    "GLM5_NEXT_W4A8_EXPERT_TARGETS",
}


class TinyGlm5LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "mlp": torch.nn.ModuleDict(
                            {
                                "gate_proj": torch.nn.Linear(4, 4),
                                "experts": torch.nn.ModuleList(
                                    [
                                        torch.nn.ModuleDict(
                                            {
                                                "gate_proj": torch.nn.Linear(4, 4),
                                                "up_proj": torch.nn.Linear(4, 4),
                                                "down_proj": torch.nn.Linear(4, 4),
                                            }
                                        )
                                    ]
                                ),
                            }
                        ),
                        "self_attn": torch.nn.ModuleDict(
                            {"q_proj": torch.nn.Linear(4, 4)}
                        ),
                    }
                )
            ]
        )


def _load_script_namespace() -> dict:
    """Load the example's helpers without executing its argparse script body."""
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in WANTED_NAMES:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in WANTED_NAMES
            for target in node.targets
        ):
            body.append(node)

    namespace = {"re": re, "torch": torch, "match_name": match_name}
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(SCRIPT_PATH), "exec"),
        namespace,
    )
    return namespace


@pytest.mark.unit
def test_w8a8_targets_are_the_non_expert_difference_set():
    namespace = _load_script_namespace()
    model = TinyGlm5LinearModel()
    ignores = [r"re:^model\.language_model\.layers\.0\.self_attn\.q_proj$"]
    w8_targets = namespace["build_glm5_next_w8a8_targets"](model, ignores)
    w4_targets = namespace["GLM5_NEXT_W4A8_EXPERT_TARGETS"]

    w4_modules = dict(match_named_modules(model, w4_targets, ignores))
    w8_modules = dict(match_named_modules(model, w8_targets, ignores))

    assert set(w4_modules) == {
        "model.language_model.layers.0.mlp.experts.0.gate_proj",
        "model.language_model.layers.0.mlp.experts.0.up_proj",
        "model.language_model.layers.0.mlp.experts.0.down_proj",
    }
    assert set(w8_modules) == {"model.language_model.layers.0.mlp.gate_proj"}
    assert not set(w4_modules) & set(w8_modules)
    assert "model.language_model.layers.0.self_attn.q_proj" not in w8_modules
    assert w8_targets[0].startswith("re:^")
    assert w8_targets[0].endswith("$")


@pytest.mark.unit
def test_w8a8_targets_match_runtime_namespaces():
    namespace = _load_script_namespace()
    model = TinyGlm5LinearModel()
    w8_targets = namespace["build_glm5_next_w8a8_targets"](model, [])
    target = w8_targets[0]

    # HF checkpoint namespace, vLLM-style flattened root, and wrapper prefixes
    # must all resolve to the same quantized layer.
    assert match_name("model.language_model.layers.0.mlp.gate_proj", target)
    assert match_name("model.layers.0.mlp.gate_proj", target)
    assert match_name("language_model.model.layers.0.mlp.gate_proj", target)

    # Routed experts belong to the W4A8 group only.
    assert not match_name("model.layers.0.mlp.experts.0.gate_proj", target)
    assert not match_name("model.language_model.layers.0.mlp.experts.0.up_proj", target)


@pytest.mark.unit
def test_w8a8_targets_group_repeated_suffixes_compactly():
    namespace = _load_script_namespace()
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.language_model = torch.nn.Module()
    model.model.language_model.layers = torch.nn.ModuleList()
    for _ in range(12):
        model.model.language_model.layers.append(
            torch.nn.ModuleDict(
                {
                    "mlp": torch.nn.ModuleDict(
                        {
                            "shared_experts": torch.nn.ModuleDict(
                                {"up_proj": torch.nn.Linear(4, 4)}
                            )
                        }
                    )
                }
            )
        )

    target = namespace["build_glm5_next_w8a8_targets"](model, [])[0]
    assert len(target) < 300
    assert match_name("model.language_model.layers.0.mlp.shared_experts.up_proj", target)
    assert match_name("language_model.model.layers.11.mlp.shared_experts.up_proj", target)


@pytest.mark.unit
def test_ignore_inference_skips_packed_auxiliary_tensors():
    namespace = _load_script_namespace()
    tensor_names = [
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight_scale",
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight_packed",
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight_shape",
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight_scale",
        "model.language_model.layers.0.self_attn.b_proj.weight",
    ]
    config = {"text_config": {"num_hidden_layers": 4, "num_nextn_predict_layers": 1}}

    ignores = namespace["collect_glm5_next_ignored_checkpoint_names"](
        tensor_names, config
    )

    assert "model.language_model.layers.0.self_attn.b_proj" in ignores
    assert not [
        name
        for name in ignores
        if name.endswith((".weight_packed", ".weight_shape", ".weight_scale"))
    ], ignores
    assert "model.language_model.layers.0.mlp.gate_proj" not in ignores
    assert "model.language_model.layers.3.mlp.experts.0.gate_proj" not in ignores
    assert not [
        name
        for name in ignores
        if ".mlp.experts." in name or ".mlp.gate_proj" in name
    ], ignores
