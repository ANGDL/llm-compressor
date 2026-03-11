import re
from types import SimpleNamespace
from typing import cast

import pytest

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers.awq import AWQMapping
from llmcompressor.modifiers.smooth_gptq.base import (
    SmoothGPTQModifier,
    _backup_ignore,
)
import llmcompressor.modifiers.smooth_gptq.base as smooth_gptq_base


class _FakeModel:
    def __init__(self):
        self.applied = []

    def apply(self, fn):
        self.applied.append(fn)
        return self


def _is_ignored(ignore_patterns: list[str], module_name: str) -> bool:
    for pattern in ignore_patterns:
        if pattern.startswith("re:"):
            if re.match(pattern[3:], module_name):
                return True
        elif module_name == pattern:
            return True
    return False


@pytest.mark.unit
def test_backup_ignore_excludes_mapping_covered_names():
    original_ignore = [
        "re:.*proj$",
        "model.layers.0.self_attn.q_proj",
        "re:.*lm_head$",
    ]
    modifier = SimpleNamespace(
        ignore=list(original_ignore),
        mappings=[
            AWQMapping(
                smooth_layer="re:.*input_layernorm$",
                balance_layers=["re:.*q_proj$", "re:.*k_proj$"],
            )
        ],
    )

    with _backup_ignore(cast(SmoothGPTQModifier, modifier)):
        new_ignore = modifier.ignore

        assert "model.layers.0.self_attn.q_proj" not in new_ignore
        assert not _is_ignored(new_ignore, "model.layers.0.self_attn.q_proj")
        assert not _is_ignored(new_ignore, "model.layers.0.self_attn.k_proj")

        assert _is_ignored(new_ignore, "model.layers.0.self_attn.o_proj")
        assert _is_ignored(new_ignore, "model.lm_head")

    assert modifier.ignore == original_ignore


@pytest.mark.unit
def test_on_event_sequential_keeps_quantization_disabled(monkeypatch):
    modifier = SmoothGPTQModifier()
    modifier.started_ = True
    modifier.ended_ = False

    call_order = []
    monkeypatch.setattr(
        SmoothGPTQModifier,
        "_apply_smoothing",
        lambda self, _model: call_order.append("smooth"),
    )
    monkeypatch.setattr(
        SmoothGPTQModifier,
        "compress_modules",
        lambda self: call_order.append("compress"),
    )

    state = SimpleNamespace(model=_FakeModel())
    modifier.on_event(cast(State, state), Event(type_=EventType.SEQUENTIAL_EPOCH_END))

    assert call_order == ["smooth", "compress"]
    assert state.model.applied == [smooth_gptq_base.disable_quantization]


@pytest.mark.unit
def test_on_event_calibration_end_runs_finalize_path(monkeypatch):
    modifier = SmoothGPTQModifier()
    modifier.started_ = True
    modifier.ended_ = False

    call_order = []
    monkeypatch.setattr(
        SmoothGPTQModifier,
        "_apply_smoothing",
        lambda self, _model: call_order.append("smooth"),
    )
    monkeypatch.setattr(
        SmoothGPTQModifier,
        "compress_modules",
        lambda self: call_order.append("compress"),
    )
    monkeypatch.setattr(
        SmoothGPTQModifier,
        "on_end",
        lambda self, _state, _event, **kwargs: call_order.append("end"),
    )

    state = SimpleNamespace(model=_FakeModel())
    modifier.on_event(cast(State, state), Event(type_=EventType.CALIBRATION_EPOCH_END))

    assert call_order == ["smooth", "compress", "end"]
    assert state.model.applied == [smooth_gptq_base.disable_quantization]
