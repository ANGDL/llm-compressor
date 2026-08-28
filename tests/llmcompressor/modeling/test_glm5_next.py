from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llmcompressor.modeling.glm5_next import (
    GLM5_NEXT_W8A8_TARGETS,
    CalibrationGlm5NextTextMoE,
    Glm5NextMTPLayer,
    SequentialGlm5NextExperts,
    _extended_text_config,
    _fuse_mtp_experts,
    _target_matches,
    load_fp8_quantized_weight_names,
)
from llmcompressor.modeling.moe_context import moe_calibration_context


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


def test_glm5_next_fp8_target_patterns_cover_reference_checkpoint():
    reference = "/Users/ang/models/GLM-5.3-Flash"
    try:
        names = load_fp8_quantized_weight_names(reference)
    except FileNotFoundError:
        pytest.skip("GLM-5.3-Flash reference checkpoint is not installed")

    assert len(names) == 37338
    assert all(_target_matches(name) for name in names)
    assert not any("self_attn.q_proj" in name for name in names)
    assert not any("self_attn.indexer" in name for name in names)
    assert len(GLM5_NEXT_W8A8_TARGETS) == 4


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
def test_glm5_next_target_patterns_include_main_and_mtp_w8a8_modules(name):
    assert _target_matches(name)


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
def test_glm5_next_target_patterns_keep_non_fp8_modules_unquantized(name):
    assert not _target_matches(name)


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
