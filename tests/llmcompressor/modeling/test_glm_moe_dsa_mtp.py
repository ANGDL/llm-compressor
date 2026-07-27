from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast  # noqa: F401

import llmcompressor.modeling.glm_moe_dsa_mtp as glm_mtp
from llmcompressor.pipelines.sequential.helpers import trace_subgraphs

# CausalLMOutputWithPast must remain in globals for the recompiled fake forward.


class _RecordingMTPLayer(nn.Module):
    instances = []

    def __init__(self, **kwargs):
        super().__init__()
        self.calls = []
        self.instances.append(self)

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["token_hidden_states"]


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 1)
        self.rotary_calls = []

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        inputs = input_ids if input_ids is not None else inputs_embeds[..., 0]
        batch_size, seq_len = inputs.shape[:2]
        hidden_states = torch.arange(
            batch_size * seq_len,
            dtype=torch.float32,
            device=inputs.device,
        ).view(batch_size, seq_len, 1)
        return SimpleNamespace(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def rotary_emb(self, hidden_states, position_ids):
        self.rotary_calls.append((hidden_states.clone(), position_ids.clone()))
        return position_ids, position_ids


class _CausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            num_hidden_layers=4,
            vocab_size=32,
            _attn_implementation="eager",
        )
        self.model = _Backbone()
        self.lm_head = nn.Identity()

    @property
    def device(self):
        return self.anchor.device


def _attach_recording_mtp(monkeypatch):
    _RecordingMTPLayer.instances = []
    monkeypatch.setattr(glm_mtp, "_load_mtp_tensors", lambda *_: {})
    monkeypatch.setattr(glm_mtp, "GlmMoeDsaMTPLayer", _RecordingMTPLayer)
    model = _CausalLM()
    glm_mtp.attach_mtp_layer(model, "unused")
    return model, _RecordingMTPLayer.instances[0]


def test_attached_mtp_uses_shifted_inputs_and_explicit_causal_mask(monkeypatch):
    model, mtp = _attach_recording_mtp(monkeypatch)
    input_ids = torch.tensor([[5, 6, 7, 8, 9], [10, 11, 12, 13, 14]])
    position_ids = torch.arange(5).unsqueeze(0)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]])

    model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )

    call = mtp.calls[0]
    backbone_hidden = torch.arange(10, dtype=torch.float32).view(2, 5, 1)
    torch.testing.assert_close(call["input_ids"], input_ids[:, 1:])
    torch.testing.assert_close(
        call["token_hidden_states"], backbone_hidden[:, :-1, :]
    )
    torch.testing.assert_close(call["position_ids"], torch.arange(1, 5).unsqueeze(0))
    assert call["cache_position"] is None

    mask_value = -1e4
    expected_mask = torch.tensor(
        [
            [0, mask_value, mask_value, mask_value],
            [0, 0, mask_value, mask_value],
            [0, 0, 0, mask_value],
            [0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(call["attention_mask"][0, 0], expected_mask)
    expected_padded_mask = expected_mask.clone()
    expected_padded_mask[:, 0] = mask_value
    torch.testing.assert_close(
        call["attention_mask"][1, 0], expected_padded_mask
    )


def test_attached_mtp_supports_inputs_embeds(monkeypatch):
    model, mtp = _attach_recording_mtp(monkeypatch)
    inputs_embeds = torch.arange(5, dtype=torch.float32).view(1, 5, 1)

    model(inputs_embeds=inputs_embeds)

    call = mtp.calls[0]
    assert call["input_ids"] is None
    torch.testing.assert_close(call["input_embeds"], inputs_embeds[:, 1:])


def test_attached_forward_exposes_mtp_to_sequential_tracer(monkeypatch):
    model, _ = _attach_recording_mtp(monkeypatch)
    sample_input = {
        "input_ids": torch.tensor([[5, 6, 7, 8, 9]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }

    subgraphs = trace_subgraphs(
        model,
        sample_input,
        sequential_targets=["_RecordingMTPLayer"],
        ignore=[],
    )

    traced_targets = [
        node.target
        for subgraph in subgraphs
        for node in subgraph.graph.find_nodes(op="call_module")
    ]
    assert "mtp" in traced_targets
