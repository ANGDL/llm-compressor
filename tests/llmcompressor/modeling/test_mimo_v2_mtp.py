from types import SimpleNamespace

import torch
import torch.nn as nn

import llmcompressor.modeling.mimo_v2_mtp as mimo_v2_mtp
from llmcompressor.modeling.mimo_v2_mtp import MiMoV2MTPLayer


class _CaptureProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, value):
        self.input = value
        return value[..., :1]


class _ZeroAttention(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return torch.zeros_like(hidden_states), None


class _ZeroMLP(nn.Module):
    def forward(self, hidden_states):
        return torch.zeros_like(hidden_states)


def test_mtp_layer_concatenates_token_embedding_before_target_hidden():
    layer = MiMoV2MTPLayer.__new__(MiMoV2MTPLayer)
    nn.Module.__init__(layer)
    layer.enorm = nn.Identity()
    layer.hnorm = nn.Identity()
    layer.eh_proj = _CaptureProjection()
    layer.input_layernorm = nn.Identity()
    layer.self_attn = _ZeroAttention()
    layer.pre_mlp_layernorm = nn.Identity()
    layer.mlp = _ZeroMLP()
    layer.final_layernorm = nn.Identity()

    embed_tokens = nn.Embedding(8, 1)
    embed_tokens.weight.data.copy_(torch.arange(8, dtype=torch.float32).view(-1, 1))
    previous_hidden_states = torch.tensor([[[10.0], [20.0]]])

    layer(
        previous_hidden_states=previous_hidden_states,
        input_ids=torch.tensor([[2, 3]]),
        embed_tokens=embed_tokens,
        position_ids=torch.tensor([[0, 1]]),
        cache_position=None,
        position_embeddings=(torch.empty(0), torch.empty(0)),
    )

    torch.testing.assert_close(
        layer.eh_proj.input,
        torch.tensor([[[2.0, 10.0], [3.0, 20.0]]]),
    )


class _RecordingMTPLayer(nn.Module):
    instances = []

    def __init__(self, **kwargs):
        super().__init__()
        self.calls = []
        self.instances.append(self)

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["previous_hidden_states"] + 1000


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 1)
        self.rotary_calls = []

    def forward(self, input_ids, **kwargs):
        hidden_states = torch.arange(
            input_ids.shape[1], dtype=torch.float32, device=input_ids.device
        ).view(1, -1, 1)
        return SimpleNamespace(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def swa_rotary_emb(self, hidden_states, position_ids):
        self.rotary_calls.append((hidden_states.clone(), position_ids.clone()))
        return position_ids, position_ids


class _CausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(num_hidden_layers=4, vocab_size=32)
        self.model = _Backbone()
        self.lm_head = nn.Identity()


def test_attach_mtp_layer_uses_shifted_tokens_and_backbone_hidden(monkeypatch):
    _RecordingMTPLayer.instances = []
    monkeypatch.setattr(mimo_v2_mtp, "_count_mtp_layers", lambda _: 3)
    monkeypatch.setattr(mimo_v2_mtp, "_load_mtp_tensors", lambda *_: {})
    monkeypatch.setattr(mimo_v2_mtp, "MiMoV2MTPLayer", _RecordingMTPLayer)

    model = _CausalLM()
    mimo_v2_mtp.attach_mtp_layer(model, "unused")

    input_ids = torch.tensor([[5, 6, 7, 8, 9]])
    position_ids = torch.arange(5).unsqueeze(0)
    cache_position = torch.arange(5)
    model(
        input_ids=input_ids,
        position_ids=position_ids,
        cache_position=cache_position,
    )

    backbone_hidden = torch.arange(5, dtype=torch.float32).view(1, -1, 1)
    for offset, layer in enumerate(_RecordingMTPLayer.instances):
        call = layer.calls[0]
        prediction_distance = offset + 1
        expected_length = input_ids.shape[1] - prediction_distance
        torch.testing.assert_close(
            call["previous_hidden_states"],
            backbone_hidden[:, :expected_length, :],
        )
        torch.testing.assert_close(
            call["input_ids"], input_ids[:, prediction_distance:]
        )
        torch.testing.assert_close(
            call["position_ids"], position_ids[..., :expected_length]
        )
        torch.testing.assert_close(
            call["cache_position"], cache_position[:expected_length]
        )

    assert [call[0].shape[1] for call in model.model.rotary_calls] == [4, 3, 2]
