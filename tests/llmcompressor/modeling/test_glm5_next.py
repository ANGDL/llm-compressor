import torch
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextExperts

import llmcompressor.modeling.glm5_next as glm5_next
from llmcompressor.modeling.moe.context import moe_calibration_context
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
from llmcompressor.modeling.moe.linearize import linearize_moe


class _Model(torch.nn.Module):
    def __init__(self, config, experts):
        super().__init__()
        self.config = config
        self.experts = experts


def test_linearize_glm5_meta_experts():
    config = Glm5NextTextConfig(
        hidden_size=16,
        moe_intermediate_size=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        linear_num_heads=2,
        linear_head_dim=4,
    )
    with torch.device("meta"):
        model = _Model(config, Glm5NextTextExperts(config))

    linearize_moe(model)

    assert isinstance(model.experts, LinearExperts2D)
    assert model.experts.num_experts == config.n_routed_experts
    assert all(parameter.is_meta for parameter in model.experts.parameters())


def test_linearize_glm5_preserves_expert_outputs():
    config = Glm5NextTextConfig(
        hidden_size=16,
        moe_intermediate_size=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        linear_num_heads=2,
        linear_head_dim=4,
    )
    experts = Glm5NextTextExperts(config)
    model = _Model(config, experts)
    torch.nn.init.normal_(experts.gate_up_proj)
    torch.nn.init.normal_(experts.down_proj)
    hidden_states = torch.randn(6, config.hidden_size)
    top_k_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]])
    top_k_weights = torch.rand(6, config.num_experts_per_tok)
    expected = experts(hidden_states, top_k_index, top_k_weights)

    linearize_moe(model)
    actual = model.experts(hidden_states, top_k_index, top_k_weights)
    with moe_calibration_context():
        calibrated = model.experts(hidden_states, top_k_index, top_k_weights)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(calibrated, expected)


def test_kda_backend_choice_honors_env(monkeypatch):
    monkeypatch.setenv("LLMCOMPRESSOR_DISABLE_TRITON", "0")
    monkeypatch.setenv("LLMCOMPRESSOR_KDA_BACKEND", "torch")
    assert glm5_next._kda_backend_choice() == "torch"
    monkeypatch.setenv("LLMCOMPRESSOR_KDA_BACKEND", "triton")
    assert glm5_next._kda_backend_choice() == "fla"


def test_global_triton_switch_forces_fla_reference_kda(monkeypatch):
    monkeypatch.setenv("LLMCOMPRESSOR_DISABLE_TRITON", "1")
    monkeypatch.setenv("LLMCOMPRESSOR_KDA_BACKEND", "triton")
    import pytest

    with pytest.raises(RuntimeError, match="backend 'fla'.*Triton is disabled"):
        glm5_next._kda_backend_choice()


def test_kda_backend_choice_auto_follows_xpu(monkeypatch):
    import importlib.util

    monkeypatch.setenv("LLMCOMPRESSOR_DISABLE_TRITON", "0")
    monkeypatch.delenv("LLMCOMPRESSOR_KDA_BACKEND", raising=False)
    importlib.util.find_spec("torch_xmlir")
    assert glm5_next._kda_backend_choice() == "default"


def test_kda_backend_choice_auto_falls_back_to_torch_when_triton_disabled(monkeypatch):
    monkeypatch.setenv("LLMCOMPRESSOR_DISABLE_TRITON", "1")
    monkeypatch.delenv("LLMCOMPRESSOR_KDA_BACKEND", raising=False)
    assert glm5_next._kda_backend_choice() == "torch"


def test_force_fla_reference_kda_backend_replaces_fla_entrypoints(monkeypatch):
    import sys
    from types import ModuleType

    kda = ModuleType("fla.ops.kda")
    kda.chunk_kda = lambda *a, **k: None
    kda.fused_recurrent_kda = lambda *a, **k: None
    fla = ModuleType("fla")
    ops = ModuleType("fla.ops")
    monkeypatch.setitem(sys.modules, "fla", fla)
    monkeypatch.setitem(sys.modules, "fla.ops", ops)
    monkeypatch.setitem(sys.modules, "fla.ops.kda", kda)

    glm5_next._force_fla_reference_kda_backend()

    assert callable(kda.chunk_kda)
    assert callable(kda.fused_recurrent_kda)
