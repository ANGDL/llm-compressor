from __future__ import annotations

import gc
import json
import weakref

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor.streaming import (
    SafetensorsWeightSource,
    TargetWeightLoader,
    build_meta_model,
)


class TinyBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.register_buffer("step", torch.tensor(3, dtype=torch.int64))

    def forward(self, inputs):
        return self.norm(self.proj(inputs)) + self.step


class TinyModel(nn.Module):
    def __init__(self, hidden_size: int = 4, layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, hidden_size)
        self.layers = nn.ModuleList(
            [TinyBlock(hidden_size) for _ in range(layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, 8, bias=False)


class RemoteStyleBlock(nn.Module):
    """Fixture standing in for a checkpoint-defined modeling class."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate = nn.Parameter(torch.empty(hidden_size))
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, inputs):
        return self.projection(inputs) * self.gate


def _write_checkpoint(model: nn.Module, path, *, overrides=None):
    tensors = {
        name: tensor.detach().clone().contiguous()
        for name, tensor in model.state_dict().items()
    }
    tensors.update(overrides or {})
    save_file(tensors, path)


@pytest.fixture
def tiny_checkpoint(tmp_path):
    torch.manual_seed(7)
    model = TinyModel()
    path = tmp_path / "model.safetensors"
    _write_checkpoint(model, path)
    return model, path


def _assert_all_meta(module):
    assert all(tensor.is_meta for tensor in module.parameters())
    assert all(tensor.is_meta for tensor in module.buffers())


def _real_storage_bytes(module):
    tensors = [*module.parameters(), *module.buffers()]
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in tensors
        if not tensor.is_meta
    )


def test_build_meta_model_does_not_allocate_real_weights():
    model = build_meta_model(TinyModel, hidden_size=16, layers=20)

    _assert_all_meta(model)
    assert len(model.layers) == 20


def test_tiny_llama_decoder_target_runs_and_unloads(tmp_path):
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    torch.manual_seed(11)
    reference = LlamaForCausalLM(config)
    path = tmp_path / "model.safetensors"
    _write_checkpoint(reference, path)
    model = build_meta_model(LlamaForCausalLM, config)
    loader = TargetWeightLoader(model, SafetensorsWeightSource(path))
    inputs = torch.randn(1, 5, config.hidden_size)
    position_ids = torch.arange(5).unsqueeze(0)
    position_embeddings = reference.model.rotary_emb(inputs, position_ids)
    expected = reference.model.layers[0](
        inputs, position_embeddings=position_embeddings
    )

    with loader.loaded(
        "model.layers.0",
        device=torch.device("cpu"),
        dtype=torch.float32,
    ) as layer:
        actual = layer(inputs, position_embeddings=position_embeddings)
        assert torch.allclose(actual, expected)
        _assert_all_meta(model.model.layers[1])

    _assert_all_meta(model)


def test_loads_one_target_for_forward_then_restores_meta(tiny_checkpoint):
    reference, path = tiny_checkpoint
    model = build_meta_model(TinyModel)
    loader = TargetWeightLoader(model, SafetensorsWeightSource(path))
    inputs = torch.randn(2, 4)
    expected = reference.layers[0](inputs)

    with loader.loaded(
        "layers.0", device=torch.device("cpu"), dtype=torch.float32
    ) as layer:
        assert all(not parameter.is_meta for parameter in layer.parameters())
        assert all(not buffer.is_meta for buffer in layer.buffers())
        assert torch.allclose(layer(inputs), expected)
        _assert_all_meta(model.layers[1])
        loaded_parameter = next(layer.parameters())
        reference_to_parameter = weakref.ref(loaded_parameter)

    del loaded_parameter
    gc.collect()
    _assert_all_meta(model)
    assert reference_to_parameter() is None
    assert next(model.layers[0].parameters()).dtype == torch.float32


def test_materialized_storage_does_not_grow_with_model_depth(tmp_path):
    active_bytes = []
    for layer_count in (2, 20):
        reference = TinyModel(hidden_size=32, layers=layer_count)
        path = tmp_path / f"model-{layer_count}.safetensors"
        _write_checkpoint(reference, path)
        del reference
        model = build_meta_model(TinyModel, hidden_size=32, layers=layer_count)
        loader = TargetWeightLoader(model, SafetensorsWeightSource(path))

        with loader.loaded(
            "layers.0", device=torch.device("cpu"), dtype=torch.float32
        ):
            active_bytes.append(_real_storage_bytes(model))
            assert _real_storage_bytes(model.layers[1]) == 0

        assert _real_storage_bytes(model) == 0

    assert active_bytes[0] == active_bytes[1]


def test_exception_exit_restores_target_to_meta(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = build_meta_model(TinyModel)
    loader = TargetWeightLoader(model, SafetensorsWeightSource(path))

    with pytest.raises(RuntimeError, match="forward failed"):
        with loader.loaded(
            "layers.0", device=torch.device("cpu"), dtype=torch.float32
        ):
            raise RuntimeError("forward failed")

    _assert_all_meta(model)
    with loader.loaded(
        "layers.1", device=torch.device("cpu"), dtype=torch.float32
    ):
        assert not next(model.layers[1].parameters()).is_meta


@pytest.mark.parametrize(
    "target_name", ["embed_tokens", "layers.0", "final_norm", "lm_head"]
)
def test_supports_model_boundary_target_types(tiny_checkpoint, target_name):
    _, path = tiny_checkpoint
    model = build_meta_model(TinyModel)
    loader = TargetWeightLoader(model, SafetensorsWeightSource(path))

    with loader.loaded(
        target_name, device=torch.device("cpu"), dtype=torch.float32
    ) as target:
        assert all(not parameter.is_meta for parameter in target.parameters())

    _assert_all_meta(model)


def test_tied_parameters_load_once_and_remain_tied(tmp_path, monkeypatch):
    class TiedTarget(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Linear(4, 4, bias=False)
            self.right = nn.Linear(4, 4, bias=False)
            self.right.weight = self.left.weight

    reference = TiedTarget()
    path = tmp_path / "model.safetensors"
    save_file({"left.weight": reference.left.weight.detach().clone()}, path)
    model = build_meta_model(TiedTarget)
    source = SafetensorsWeightSource(path)
    loaded_names = []
    original_load = source.load_tensors

    def recording_load(names, *, device):
        names = list(names)
        loaded_names.extend(names)
        return original_load(names, device=device)

    monkeypatch.setattr(source, "load_tensors", recording_load)
    loader = TargetWeightLoader(model, source)

    with loader.loaded("", device=torch.device("cpu"), dtype=torch.float32):
        assert model.left.weight is model.right.weight
        assert loaded_names.count("left.weight") == 1

    assert model.left.weight is model.right.weight
    _assert_all_meta(model)


def test_remote_style_model_target_can_run(tmp_path):
    torch.manual_seed(3)
    reference = RemoteStyleBlock(4)
    path = tmp_path / "model.safetensors"
    _write_checkpoint(reference, path)
    model = build_meta_model(RemoteStyleBlock, 4)
    loader = TargetWeightLoader(model, SafetensorsWeightSource(path))
    inputs = torch.randn(2, 4)

    with loader.loaded("", device=torch.device("cpu"), dtype=torch.float32):
        assert torch.allclose(model(inputs), reference(inputs))

    _assert_all_meta(model)


def test_rejects_unknown_target_and_missing_weight(tiny_checkpoint, tmp_path):
    _, path = tiny_checkpoint
    model = build_meta_model(TinyModel)
    loader = TargetWeightLoader(model, SafetensorsWeightSource(path))

    with pytest.raises(ValueError, match="Unknown target"):
        with loader.loaded("layers.99", device=torch.device("cpu")):
            pass

    incomplete = tmp_path / "incomplete.safetensors"
    save_file({"layers.0.proj.weight": torch.ones(4, 4)}, incomplete)
    loader = TargetWeightLoader(model, SafetensorsWeightSource(incomplete))
    with pytest.raises(KeyError, match="layers.0.proj.bias"):
        with loader.loaded("layers.0", device=torch.device("cpu")):
            pass


def test_rejects_shape_mismatch_before_installing(tiny_checkpoint, tmp_path):
    reference, _ = tiny_checkpoint
    path = tmp_path / "wrong-shape.safetensors"
    _write_checkpoint(
        reference, path, overrides={"layers.0.proj.weight": torch.ones(3, 4)}
    )
    model = build_meta_model(TinyModel)
    loader = TargetWeightLoader(model, SafetensorsWeightSource(path))

    with pytest.raises(ValueError, match="model expects"):
        with loader.loaded(
            "layers.0", device=torch.device("cpu"), dtype=torch.float32
        ):
            pass

    _assert_all_meta(model)


def test_rejects_fused_checkpoint_tensor_with_clear_error(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"fused_qkv": torch.ones(12, 4)}, path)
    model = build_meta_model(nn.Linear, 4, 4)
    loader = TargetWeightLoader(model, SafetensorsWeightSource(path))

    with pytest.raises(KeyError, match="Fused or renamed checkpoint tensors"):
        with loader.loaded("", device=torch.device("cpu")):
            pass
