import sys
from types import ModuleType

from llmcompressor.utils.triton_backend import is_triton_disabled


def test_global_triton_switch(monkeypatch):
    monkeypatch.setenv("LLMCOMPRESSOR_DISABLE_TRITON", "1")
    assert is_triton_disabled()

    monkeypatch.setenv("LLMCOMPRESSOR_DISABLE_TRITON", "0")
    assert not is_triton_disabled()


def test_triton_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LLMCOMPRESSOR_DISABLE_TRITON", raising=False)
    assert is_triton_disabled()


def test_enable_fla_reference_kda_entrypoints(monkeypatch):
    from llmcompressor.utils.triton_backend import enable_fla_reference_kda

    fla = ModuleType("fla")
    ops = ModuleType("fla.ops")
    kda = ModuleType("fla.ops.kda")
    gate = ModuleType("fla.ops.kda.gate")
    naive = ModuleType("fla.ops.kda.naive")
    naive.naive_chunk_kda = lambda **kwargs: (kwargs["q"], None)
    naive.naive_recurrent_kda = lambda **kwargs: (kwargs["q"], None)
    gate.naive_kda_gate = lambda g, *args, **kwargs: g
    gate.naive_kda_lowerbound_gate = lambda g, *args, **kwargs: g
    monkeypatch.setitem(sys.modules, "fla", fla)
    monkeypatch.setitem(sys.modules, "fla.ops", ops)
    monkeypatch.setitem(sys.modules, "fla.ops.kda", kda)
    monkeypatch.setitem(sys.modules, "fla.ops.kda.gate", gate)
    monkeypatch.setitem(sys.modules, "fla.ops.kda.naive", naive)

    enable_fla_reference_kda()

    assert callable(kda.chunk_kda)
    assert callable(kda.fused_recurrent_kda)
    assert fla.chunk_kda is kda.chunk_kda
    assert fla.fused_recurrent_kda is kda.fused_recurrent_kda
