import os
import subprocess
import sys
from types import ModuleType

from llmcompressor._triton_env import configure_compressed_tensors_triton
from llmcompressor.utils.triton_backend import is_triton_disabled


def test_global_triton_switch(monkeypatch):
    monkeypatch.setenv("LLMCOMPRESSOR_DISABLE_TRITON", "1")
    assert is_triton_disabled()

    monkeypatch.setenv("LLMCOMPRESSOR_DISABLE_TRITON", "0")
    assert not is_triton_disabled()


def test_triton_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LLMCOMPRESSOR_DISABLE_TRITON", raising=False)
    assert is_triton_disabled()


def test_compressed_tensors_eager_follows_triton_switch(monkeypatch):
    monkeypatch.setenv("LLMCOMPRESSOR_DISABLE_TRITON", "1")
    monkeypatch.setenv("CT_ENFORCE_EAGER", "")

    configure_compressed_tensors_triton()

    assert os.environ["CT_ENFORCE_EAGER"] == "1"


def test_compressed_tensors_triton_allowed_when_enabled(monkeypatch):
    monkeypatch.setenv("LLMCOMPRESSOR_DISABLE_TRITON", "0")
    monkeypatch.delenv("CT_ENFORCE_EAGER", raising=False)

    configure_compressed_tensors_triton()

    assert "CT_ENFORCE_EAGER" not in os.environ


def test_import_llmcompressor_configures_compressed_tensors_eager():
    script = """
import os
import llmcompressor
from compressed_tensors.utils import impl_backend

assert os.environ["CT_ENFORCE_EAGER"] == "1"
assert impl_backend.ENFORCE_EAGER
"""
    environment = os.environ.copy()
    environment.pop("LLMCOMPRESSOR_DISABLE_TRITON", None)
    environment.pop("CT_ENFORCE_EAGER", None)

    subprocess.run([sys.executable, "-c", script], check=True, env=environment)


def test_import_llmcompressor_updates_loaded_compressed_tensors_backend():
    script = """
import os
import compressed_tensors
from compressed_tensors.utils import impl_backend

assert not impl_backend.ENFORCE_EAGER
import llmcompressor
assert os.environ["CT_ENFORCE_EAGER"] == "1"
assert impl_backend.ENFORCE_EAGER
"""
    environment = os.environ.copy()
    environment.pop("LLMCOMPRESSOR_DISABLE_TRITON", None)
    environment.pop("CT_ENFORCE_EAGER", None)

    subprocess.run([sys.executable, "-c", script], check=True, env=environment)


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
