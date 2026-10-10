from types import SimpleNamespace

import torch
from torch import nn

import llmcompressor.modeling.glm5_next as glm5_next


class _MetaRoutedExpertMLP(nn.Module):
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            intermediate_size,
            bias=False,
            device="meta",
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            intermediate_size,
            bias=False,
            device="meta",
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            device="meta",
        )


def test_sequential_glm5_experts_materialize_meta_weights(monkeypatch):
    monkeypatch.setattr(
        glm5_next, "Glm5NextTextRoutedExpertMLP", _MetaRoutedExpertMLP
    )
    monkeypatch.setattr(glm5_next, "_require_transformers_glm5", lambda: None)
    config = SimpleNamespace(hidden_size=3, moe_intermediate_size=2)
    gate_up_proj = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    down_proj = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    original = SimpleNamespace(
        gate_up_proj=gate_up_proj,
        down_proj=down_proj,
    )

    experts = glm5_next.SequentialGlm5NextExperts(config, original)

    assert len(experts) == 2
    for index, expert in enumerate(experts):
        gate, up = gate_up_proj[index].chunk(2, dim=0)
        assert expert.gate_proj.weight.device.type == "cpu"
        assert isinstance(expert.gate_proj.weight, nn.Parameter)
        assert expert.gate_proj.weight.requires_grad
        torch.testing.assert_close(expert.gate_proj.weight, gate)
        torch.testing.assert_close(expert.up_proj.weight, up)
        torch.testing.assert_close(expert.down_proj.weight, down_proj[index])


def test_kda_backend_choice_honors_env(monkeypatch):
    monkeypatch.setenv("LLMCOMPRESSOR_KDA_BACKEND", "torch")
    assert glm5_next._kda_backend_choice() == "torch"
    monkeypatch.setenv("LLMCOMPRESSOR_KDA_BACKEND", "triton")
    assert glm5_next._kda_backend_choice() == "triton"


def test_kda_backend_choice_auto_follows_xpu(monkeypatch):
    import importlib.util

    monkeypatch.delenv("LLMCOMPRESSOR_KDA_BACKEND", raising=False)
    on_xpu = importlib.util.find_spec("torch_xmlir") is not None
    assert glm5_next._kda_backend_choice() == ("torch" if on_xpu else "triton")


def test_force_torch_kda_backend_nulls_fla_entrypoints(monkeypatch):
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

    glm5_next._force_torch_kda_backend()

    assert kda.chunk_kda is None
    assert kda.fused_recurrent_kda is None
