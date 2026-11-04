import ast
import re
from pathlib import Path

import pytest
import torch
from compressed_tensors.utils.match import match_name, match_named_modules
from transformers import PretrainedConfig

SCRIPT_PATH = Path(__file__).parents[2] / "examples/multimodal_vision/glm5_next_wNa8.py"
STREAMING_SCRIPT_PATH = (
    Path(__file__).parents[2] / "examples/streaming_oneshot/glm5_next_wNa8.py"
)

# Keep in sync with the example's own constants via the AST loader below.
WANTED_NAMES = {
    "_config_value",
    "build_ignores",
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


def _load_script_namespace(script_path: Path = SCRIPT_PATH) -> dict:
    """Load the example's helpers without executing its argparse script body."""
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
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
        compile(ast.Module(body=body, type_ignores=[]), str(script_path), "exec"),
        namespace,
    )
    return namespace


@pytest.mark.unit
@pytest.mark.parametrize("root_as_dict", [False, True])
@pytest.mark.parametrize("text_as_dict", [False, True])
@pytest.mark.parametrize("linear_attn_as_dict", [False, True])
@pytest.mark.parametrize("layer_type", [list, tuple])
def test_streaming_kda_ignores_match_multimodal_layout(
    root_as_dict, text_as_dict, linear_attn_as_dict, layer_type
):
    kda_layers = [index for index in range(48) if index % 4 != 3]
    linear_attn_config = {"kda_layers": layer_type(kda_layers[::-1] + [1])}
    if not linear_attn_as_dict:
        linear_attn_config = PretrainedConfig(**linear_attn_config)
    text_config = {
        "num_hidden_layers": 48,
        "linear_attn_config": linear_attn_config,
    }
    if not text_as_dict:
        text_config = PretrainedConfig(**text_config)
    config = {"text_config": text_config}
    if not root_as_dict:
        config = PretrainedConfig(**config)

    streaming_ignores = _load_script_namespace(STREAMING_SCRIPT_PATH)["build_ignores"](
        config
    )
    multimodal_ignores = _load_script_namespace()["build_glm5_next_w8a8_ignores"](
        config
    )
    for root in ("model", "model.language_model", "language_model.model"):
        for index in range(48):
            for suffix in ("", ".o_proj", ".o_proj.weight_scale", ".q_a_proj"):
                name = f"{root}.layers.{index}.self_attn{suffix}"
                expected = index in kda_layers
                assert (
                    any(match_name(name, rule) for rule in streaming_ignores)
                    == expected
                )
                assert (
                    any(match_name(name, rule) for rule in multimodal_ignores)
                    == expected
                )
            name = f"{root}.layers.{index}.mlp.gate_proj"
            assert not any(match_name(name, rule) for rule in streaming_ignores)


@pytest.mark.unit
@pytest.mark.parametrize("as_dict", [False, True])
def test_streaming_kda_ignores_accept_unwrapped_text_config(as_dict):
    config = {"num_hidden_layers": 4, "linear_attn_config": {"kda_layers": [2, 0, 2]}}
    if not as_dict:
        config = PretrainedConfig(**config)
    ignores = _load_script_namespace(STREAMING_SCRIPT_PATH)["build_ignores"](config)
    assert ignores[0] == r"re:.*\.layers\.(?:0|2)\.self_attn(?:\..*)?$"


@pytest.mark.unit
def test_streaming_w8a8_targets_exclude_kda_but_keep_dsa_and_mlp():
    tree = ast.parse(STREAMING_SCRIPT_PATH.read_text(encoding="utf-8"))
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    targets = next(
        ast.literal_eval(node.value)
        for node in main.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "other_targets"
            for target in node.targets
        )
    )
    model = TinyGlm5LinearModel()
    model.model.language_model.layers.extend(
        [TinyGlm5LinearModel().model.language_model.layers[0] for _ in range(3)]
    )
    for layer in model.model.language_model.layers:
        layer["self_attn"] = torch.nn.ModuleDict(
            {
                name: torch.nn.Linear(4, 4)
                for name in ("o_proj", "q_a_proj", "kv_b_proj")
            }
        )
    ignores = _load_script_namespace(STREAMING_SCRIPT_PATH)["build_ignores"](
        {
            "text_config": {
                "num_hidden_layers": 4,
                "linear_attn_config": {"kda_layers": [0, 1, 2]},
            }
        }
    )
    matched = dict(match_named_modules(model, targets, ignores))
    assert set(matched) == {
        *(f"model.language_model.layers.{index}.mlp.gate_proj" for index in range(4)),
        "model.language_model.layers.3.self_attn.o_proj",
        "model.language_model.layers.3.self_attn.q_a_proj",
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "text_config",
    [
        {},
        {"num_hidden_layers": 4},
        {"num_hidden_layers": 4, "linear_attn_config": {}},
        {"num_hidden_layers": 4, "linear_attn_config": {"kda_layers": None}},
        {"num_hidden_layers": 4, "linear_attn_config": {"kda_layers": []}},
        {"num_hidden_layers": 4, "linear_attn_config": {"kda_layers": "0,1"}},
        {"num_hidden_layers": 4, "linear_attn_config": {"kda_layers": [0, -1]}},
        {"num_hidden_layers": 4, "linear_attn_config": {"kda_layers": [0, 4]}},
        {"num_hidden_layers": 4, "linear_attn_config": {"kda_layers": [0, "1"]}},
        {"num_hidden_layers": 4, "linear_attn_config": {"kda_layers": [0, 1.5]}},
        {"linear_attn_config": {"kda_layers": [0]}},
        {"num_hidden_layers": "4", "linear_attn_config": {"kda_layers": [0]}},
    ],
)
def test_streaming_kda_ignores_reject_invalid_layout(text_config):
    build_ignores = _load_script_namespace(STREAMING_SCRIPT_PATH)["build_ignores"]
    with pytest.raises(ValueError, match="GLM-5.3 text config"):
        build_ignores(PretrainedConfig(text_config=PretrainedConfig(**text_config)))


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
    assert match_name(
        "model.language_model.layers.0.mlp.shared_experts.up_proj", target
    )
    assert match_name(
        "language_model.model.layers.11.mlp.shared_experts.up_proj", target
    )


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
        name for name in ignores if ".mlp.experts." in name or ".mlp.gate_proj" in name
    ], ignores
