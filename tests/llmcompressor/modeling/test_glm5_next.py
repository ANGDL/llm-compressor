import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llmcompressor.modeling.glm5_next import (
    CalibrationGlm5NextTextMoE,
    Glm5NextMTPLayer,
    SequentialGlm5NextExperts,
    _extended_text_config,
    _fuse_mtp_experts,
)
from llmcompressor.modeling.moe_context import moe_calibration_context


def _load_example_assignment(name):
    example_path = (
        Path(__file__).parents[3] / "examples" / "multimodal_vision" / "glm5_next_w8a8.py"
    )
    tree = ast.parse(example_path.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    )
    return ast.literal_eval(assignment.value)


def _load_example_function(name):
    example_path = (
        Path(__file__).parents[3] / "examples" / "multimodal_vision" / "glm5_next_w8a8.py"
    )
    tree = ast.parse(example_path.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_config_value",
            "_glm5_next_runtime_name_aliases",
            "build_glm5_next_w8a8_ignores",
            "collect_glm5_next_ignored_checkpoint_names",
            "replace_glm5_next_saved_ignores",
        }
    }
    namespace = {
        "re": re,
        "GLM5_NEXT_VLLM_FUSED_IGNORES": _load_example_assignment(
            "GLM5_NEXT_VLLM_FUSED_IGNORES"
        ),
    }
    exec(
        compile(
            ast.Module(body=list(functions.values()), type_ignores=[]),
            str(example_path),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


GLM5_NEXT_W8A8_TARGETS = _load_example_assignment("GLM5_NEXT_W8A8_TARGETS")
GLM5_NEXT_VLLM_FUSED_IGNORES = _load_example_assignment("GLM5_NEXT_VLLM_FUSED_IGNORES")
BUILD_GLM5_NEXT_W8A8_IGNORES = _load_example_function("build_glm5_next_w8a8_ignores")
REPLACE_GLM5_NEXT_SAVED_IGNORES = _load_example_function(
    "replace_glm5_next_saved_ignores"
)
COLLECT_GLM5_NEXT_IGNORED_CHECKPOINT_NAMES = _load_example_function(
    "collect_glm5_next_ignored_checkpoint_names"
)
GLM5_NEXT_W8A8_IGNORES = BUILD_GLM5_NEXT_W8A8_IGNORES(
    SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=4,
            linear_attn_config={"kda_layers": [0, 2]},
        )
    )
)


def _ignore_matches(name: str) -> bool:
    return any(
        re.match(pattern[3:], name) if pattern.startswith("re:") else pattern == name
        for pattern in GLM5_NEXT_W8A8_IGNORES
    )


def test_glm5_next_calibration_moe_uses_routed_expert_config_field():
    config = SimpleNamespace(n_routed_experts=287)
    replacement = object.__new__(CalibrationGlm5NextTextMoE)
    assert replacement._get_num_experts(config) == 287


def test_glm5_next_calibration_moe_unpacks_packed_experts():
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

    config = Glm5NextTextConfig(
        hidden_size=4,
        intermediate_size=8,
        moe_intermediate_size=3,
        n_routed_experts=2,
        num_experts_per_tok=1,
    )
    original = SimpleNamespace(
        gate_up_proj=torch.arange(48, dtype=torch.float32).reshape(2, 6, 4),
        down_proj=torch.arange(24, dtype=torch.float32).reshape(2, 4, 3),
    )

    experts = CalibrationGlm5NextTextMoE._make_experts(object(), config, original)

    assert isinstance(experts, SequentialGlm5NextExperts)
    assert len(experts) == 2
    torch.testing.assert_close(
        experts[0].gate_proj.weight,
        original.gate_up_proj[0, :3],
    )
    torch.testing.assert_close(
        experts[0].up_proj.weight,
        original.gate_up_proj[0, 3:],
    )
    torch.testing.assert_close(experts[0].down_proj.weight, original.down_proj[0])


def test_glm5_next_routed_experts_use_silu_like_packed_experts():
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMoE

    config = Glm5NextTextConfig(
        hidden_size=4,
        intermediate_size=8,
        moe_intermediate_size=3,
        n_routed_experts=2,
        num_experts_per_tok=1,
        hidden_act="gelu",
    )
    original = Glm5NextTextMoE(config)
    experts = SequentialGlm5NextExperts(config, original.experts)
    hidden_states = torch.randn(5, config.hidden_size)

    packed_gate_up = torch.nn.functional.linear(
        hidden_states, original.experts.gate_up_proj[0]
    )
    gate, up = packed_gate_up.chunk(2, dim=-1)
    gate = gate.clamp(max=config.swiglu_limit)
    up = up.clamp(min=-config.swiglu_limit, max=config.swiglu_limit)
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(gate) * up,
        original.experts.down_proj[0],
    )

    torch.testing.assert_close(experts[0](hidden_states), expected)


def test_glm5_next_uses_moe_calibration_context_for_packed_moe():
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMoE

    config = Glm5NextTextConfig(
        hidden_size=4,
        intermediate_size=8,
        moe_intermediate_size=3,
        n_routed_experts=2,
        num_experts_per_tok=1,
    )

    model = nn.Module()
    model.config = config
    model.mlp = Glm5NextTextMoE(config)

    assert isinstance(model.mlp, Glm5NextTextMoE)
    with moe_calibration_context(model, calibrate_all_experts=False):
        assert isinstance(model.mlp, CalibrationGlm5NextTextMoE)
        output = model.mlp(torch.randn(1, 2, config.hidden_size))
        assert output.shape == (1, 2, config.hidden_size)
        assert torch.isfinite(output).all()


@pytest.mark.parametrize("calibrate_all_experts", [False, True])
def test_glm5_next_calibration_forward_matches_transformers_bfloat16(
    calibrate_all_experts,
):
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMoE

    torch.manual_seed(7)
    config = Glm5NextTextConfig(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        n_routed_experts=8,
        num_experts_per_tok=4,
        n_group=1,
        topk_group=1,
    )
    original = Glm5NextTextMoE(config).to(dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in original.parameters():
            parameter.uniform_(-0.05, 0.05)

    model = nn.Module()
    model.config = config
    model.mlp = original
    hidden_states = torch.randn(4, 5, config.hidden_size, dtype=torch.bfloat16)
    expected = original(hidden_states)

    with moe_calibration_context(model, calibrate_all_experts=calibrate_all_experts):
        actual = model.mlp(hidden_states)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_glm5_next_w8a8_targets_all_linear_modules():
    assert GLM5_NEXT_W8A8_TARGETS == ["Linear"]


def test_glm5_next_w8a8_ignores_multimodal_vision_modules():
    assert _ignore_matches("lm_head")
    assert _ignore_matches("language_model.lm_head")
    assert _ignore_matches("model.visual.patch_embed.proj")
    assert _ignore_matches("visual.patch_embed.proj")
    assert not _ignore_matches("model.language_model.layers.0.mlp.gate_proj")
    assert not _ignore_matches("model.audio_tower.proj")


def test_glm5_next_w8a8_ignores_follow_configured_kda_layers():
    ignores = BUILD_GLM5_NEXT_W8A8_IGNORES(
        SimpleNamespace(
            text_config=SimpleNamespace(
                num_hidden_layers=49,
                linear_attn_config={"kda_layers": [0, 46, 48]},
            )
        )
    )

    def is_ignored(name):
        return any(re.match(pattern.removeprefix("re:"), name) for pattern in ignores)

    for prefix in (
        "model.language_model.layers",
        "model.layers",
        "language_model.model.layers",
    ):
        assert is_ignored(f"{prefix}.48.self_attn.o_proj")
        assert not is_ignored(f"{prefix}.47.self_attn.o_proj")
        assert is_ignored(f"{prefix}.47.self_attn.kv_b_proj")
        assert is_ignored(f"{prefix}.47.self_attn.indexer.index_kpool_compress_gate")

    for prefix in ("model.layers", "mtp.model.layers"):
        assert is_ignored(f"{prefix}.49.eh_proj")
        assert is_ignored(f"{prefix}.49.mtp_block.self_attn.kv_b_proj")
        assert is_ignored(f"{prefix}.49.mtp_block.mlp.gate")
        assert not is_ignored(f"{prefix}.49.mtp_block.self_attn.o_proj")


def test_glm5_next_collects_ignored_checkpoint_aliases_and_parameters():
    prefix = "model.language_model.layers.3.self_attn"
    ignored = COLLECT_GLM5_NEXT_IGNORED_CHECKPOINT_NAMES(
        [
            f"{prefix}.f_a_proj.weight",
            f"{prefix}.A_log",
            f"{prefix}.indexer.k_norm.weight",
            f"{prefix}.indexer.k_norm.bias",
            f"{prefix}.indexer.index_kpool_compress_ape",
            f"{prefix}.indexer.index_kpool_compress_gate",
            "model.language_model.layers.3.mlp.experts.0.gate_proj.weight",
            "model.language_model.layers.3.mlp.experts.0.gate_proj.weight_scale",
        ],
        {
            "text_config": {
                "num_hidden_layers": 45,
                "num_nextn_predict_layers": 1,
            }
        },
    )

    assert f"{prefix}.f_a_proj" in ignored
    assert f"{prefix}.forget_gate.f_a_proj" in ignored
    assert f"{prefix}.A_log" in ignored
    assert f"{prefix}.indexer.k_norm" in ignored
    assert f"{prefix}.indexer.k_norm.bias" in ignored
    assert f"{prefix}.indexer.index_kpool_compress_ape" in ignored
    assert f"{prefix}.indexer.index_kpool_compress_gate" in ignored
    assert not any("experts.0.gate_proj" in name for name in ignored)
    assert not any(name.startswith("language_model.model.") for name in ignored)
    assert not any(name.startswith("model.layers.") for name in ignored)
    assert set(GLM5_NEXT_VLLM_FUSED_IGNORES).issubset(ignored)


@pytest.mark.parametrize(
    "name",
    [
        "language_model.model.layers.0.self_attn.in_proj_qkvgfab",
        "model.layers.44.self_attn.in_proj_qkvgfab",
        "mtp.model.layers.45.mtp_block.self_attn.in_proj_qkvgfab",
    ],
)
def test_glm5_next_saved_ignores_match_vllm_kda_fused_projection(name):
    assert any(
        re.match(pattern.removeprefix("re:"), name)
        for pattern in GLM5_NEXT_VLLM_FUSED_IGNORES
    )


@pytest.mark.parametrize(
    "name",
    [
        "language_model.model.layers.0.self_attn.fused_qkv_a_proj",
        "language_model.model.layers.0.mlp.gate_up_proj",
        "language_model.model.layers.0.self_attn.conv1d",
    ],
)
def test_glm5_next_saved_ignores_do_not_match_other_vllm_fusions(name):
    assert not any(
        re.match(pattern.removeprefix("re:"), name)
        for pattern in GLM5_NEXT_VLLM_FUSED_IGNORES
    )


def test_glm5_next_checkpoint_ignores_include_vllm_mtp_aliases():
    checkpoint_name = (
        "model.language_model.layers.45.self_attn.indexer.index_kpool_compress_gate"
    )
    ignored = COLLECT_GLM5_NEXT_IGNORED_CHECKPOINT_NAMES(
        [checkpoint_name],
        {
            "text_config": {
                "num_hidden_layers": 45,
                "num_nextn_predict_layers": 1,
            }
        },
    )

    assert checkpoint_name in ignored
    assert (
        "model.layers.45.mtp_block.self_attn.indexer.index_kpool_compress_gate"
    ) in ignored
    assert (
        "mtp.model.layers.45.mtp_block.self_attn.indexer.index_kpool_compress_gate"
    ) in ignored


def test_glm5_next_saved_ignores_replace_stale_names():
    ignored_name = (
        "model.language_model.layers.45.self_attn.indexer.index_kpool_compress_gate"
    )
    config = {"quantization_config": {"ignore": ["stale.ignore.from.previous.config"]}}

    updated = REPLACE_GLM5_NEXT_SAVED_IGNORES(
        config,
        [ignored_name],
    )

    assert updated["quantization_config"]["ignore"] == [ignored_name]


def test_glm5_next_index_ignore_update_is_enabled_by_default():
    example_path = (
        Path(__file__).parents[3] / "examples" / "multimodal_vision" / "glm5_next_w8a8.py"
    )
    tree = ast.parse(example_path.read_text(encoding="utf-8"))
    argument = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and ast.literal_eval(node.args[0]) == "--update-ignore-from-index"
    )
    keywords = {keyword.arg: keyword.value for keyword in argument.keywords}

    assert ast.literal_eval(keywords["default"]) is True
    assert isinstance(keywords["action"], ast.Attribute)
    assert keywords["action"].attr == "BooleanOptionalAction"

    guarded_update = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and node.test.attr == "update_ignore_from_index"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "update_glm5_next_saved_ignores"
        for node in ast.walk(guarded_update)
    )


def test_glm5_next_w8a8_example_uses_processor_for_multimodal_calibration():
    example_path = (
        Path(__file__).parents[3] / "examples" / "multimodal_vision" / "glm5_next_w8a8.py"
    )
    source = example_path.read_text(encoding="utf-8")

    assert 'default="lmms-lab/flickr30k"' in source
    assert '"--text_dataset_id"' in source
    assert '"--text_dataset_split"' in source
    assert '"--text_calibration_samples"' in source
    assert "text_examples = list(text_dataset)" in source
    assert "with_indices=True" in source
    assert "_format_text_example" in source
    assert "_build_calibration_messages" in source
    assert "AutoProcessor.from_pretrained" in source
    assert '"pixel_values"' in source
    assert "tokenize=True" in source
    assert "return_dict=True" in source
    assert "processor=processor" in source
    assert "processor.save_pretrained(save_path)" in source


def test_fuse_mtp_experts_reconstructs_packed_transformers_layout():
    raw = {
        "mlp.experts.0.gate_proj.weight": torch.full((2, 3), 1.0),
        "mlp.experts.0.up_proj.weight": torch.full((2, 3), 2.0),
        "mlp.experts.0.down_proj.weight": torch.full((3, 2), 3.0),
        "mlp.experts.1.gate_proj.weight": torch.full((2, 3), 4.0),
        "mlp.experts.1.up_proj.weight": torch.full((2, 3), 5.0),
        "mlp.experts.1.down_proj.weight": torch.full((3, 2), 6.0),
        "mlp.gate.weight": torch.ones(2, 3),
    }

    fused = _fuse_mtp_experts(raw)
    assert set(fused) == {
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
        "mlp.gate.weight",
    }
    torch.testing.assert_close(
        fused["mlp.experts.gate_up_proj"],
        torch.tensor(
            [
                [[1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2]],
                [[4, 4, 4], [4, 4, 4], [5, 5, 5], [5, 5, 5]],
            ],
            dtype=torch.float32,
        ),
    )


@pytest.mark.parametrize(
    "name",
    [
        "model.language_model.layers.0.mlp.gate_proj",
        "model.language_model.layers.45.mlp.experts.287.up_proj",
        "model.language_model.layers.45.self_attn.q_a_proj",
        "model.language_model.layers.45.self_attn.kv_a_proj_with_mqa",
    ],
)
def test_glm5_next_ignores_keep_main_and_mtp_w8a8_modules_quantized(name):
    assert not _ignore_matches(name)


@pytest.mark.parametrize(
    "name",
    [
        "model.language_model.layers.0.self_attn.f_a_proj",
        "model.language_model.layers.0.self_attn.indexer.wk",
        "model.language_model.layers.45.eh_proj",
        "model.language_model.layers.45.self_attn.kv_b_proj",
        "lm_head",
    ],
)
def test_glm5_next_ignores_keep_non_fp8_modules_unquantized(name):
    assert _ignore_matches(name)


def test_glm5_next_mtp_skips_empty_continuation():
    layer = object.__new__(Glm5NextMTPLayer)
    torch.nn.Module.__init__(layer)
    previous = torch.empty(2, 0, 8)
    embeds = torch.empty(2, 0, 8)

    result = layer(
        previous_hidden_states=previous,
        input_ids=None,
        input_embeds=embeds,
        embed_tokens=None,
    )

    assert result is previous


def test_glm5_next_mtp_config_extends_with_runtime_mtp_topology():
    text_config = SimpleNamespace(
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "sparse"],
        indexer_types=["full", "full"],
    )
    config = SimpleNamespace(get_text_config=lambda: text_config)

    extended = _extended_text_config(config, layer_idx=2)

    assert extended is not text_config
    assert extended.layer_types == [
        "linear_attention",
        "deepseek_sparse_attention",
        "deepseek_sparse_attention",
    ]
    assert extended.mlp_layer_types == ["dense", "sparse", "sparse"]
    assert extended.indexer_types == ["full", "full", "full"]
    assert text_config.layer_types == [
        "linear_attention",
        "deepseek_sparse_attention",
    ]


class _Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, value):
        return value * self.scale


class _CaptureAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.kwargs = None

    def forward(self, **kwargs):
        self.kwargs = kwargs
        return kwargs["hidden_states"] * 2, None, None


def test_glm5_next_mtp_forward_matches_non_mhc_runtime_path():
    layer = object.__new__(Glm5NextMTPLayer)
    nn.Module.__init__(layer)
    layer.enorm = _Scale(2)
    layer.hnorm = _Scale(3)
    layer.eh_proj = nn.Linear(4, 2, bias=False)
    layer.eh_proj.weight.data.copy_(
        torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    )
    layer.input_layernorm = nn.Identity()
    layer.self_attn = _CaptureAttention()
    layer.post_attention_layernorm = nn.Identity()
    layer.mlp = _Scale(4)
    layer.shared_head = nn.Module()
    layer.shared_head.norm = nn.Identity()

    embeds = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    previous = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])
    mask = torch.tensor([[True, True]])
    positions = torch.tensor([[1, 2]])

    result = layer(
        previous_hidden_states=previous,
        input_ids=None,
        input_embeds=embeds,
        embed_tokens=None,
        attention_mask=mask,
        position_ids=positions,
    )

    # eh_proj([2 * embed, 3 * previous]) followed by the two ordinary
    # residual additions: x + 2x, then residual + 4 * residual.
    projected = 2 * embeds + 3 * previous
    torch.testing.assert_close(result, projected * 15)
    assert layer.self_attn.kwargs["attention_mask"] is mask
    assert layer.self_attn.kwargs["position_ids"] is positions
    assert layer.self_attn.kwargs["past_key_values"] is None
    assert layer.self_attn.kwargs["use_cache"] is False
    assert layer.self_attn.kwargs["prev_topk_indices"] is None
