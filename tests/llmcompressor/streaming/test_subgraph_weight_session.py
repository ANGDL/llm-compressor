from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from torch.fx import Graph

from llmcompressor.pipelines.sequential.helpers import Subgraph
from llmcompressor.streaming import (
    SafetensorsWeightSource,
    SubgraphWeightSession,
    build_meta_model,
)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)

    def forward(self, value):
        return self.norm(self.proj(value))


class SessionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.layers = nn.ModuleList([Block(), Block()])
        self.final_norm = nn.LayerNorm(4)


class SharedRuntimeBufferBlock(nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.register_buffer("runtime", shared, persistent=False)
        self.runtime_alias = self.runtime

    def forward(self, value):
        return self.proj(value) + self.runtime


class SharedRuntimeBufferModel(nn.Module):
    def __init__(self):
        super().__init__()
        shared = torch.empty(4)
        self.layers = nn.ModuleList(
            [SharedRuntimeBufferBlock(shared) for _ in range(2)]
        )

    def _reinitialize_non_persistent_buffers(self):
        for index, layer in enumerate(self.layers):
            layer._buffers["runtime"] = torch.full(
                (4,), float(index + 1), device=layer.runtime.device
            )
            layer.runtime_alias = layer.runtime


def checkpoint_for(model, tmp_path):
    first = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name.startswith(("embed.", "layers.0."))
    }
    second = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name.startswith(("layers.1.", "final_norm."))
    }
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_file(first, checkpoint / "model-00001-of-00002.safetensors")
    save_file(second, checkpoint / "model-00002-of-00002.safetensors")
    weight_map = {
        **{name: "model-00001-of-00002.safetensors" for name in first},
        **{name: "model-00002-of-00002.safetensors" for name in second},
    }
    import json

    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return checkpoint


def subgraph_for(*module_names):
    graph = Graph()
    value = graph.placeholder("value")
    current = value
    for name in module_names:
        current = graph.call_module(name, (current,))
    graph.output({"output": current})
    return Subgraph(graph, {"value"}, {"value"})


def assert_all_meta(module):
    assert all(value.is_meta for value in module.parameters())
    assert all(value.is_meta for value in module.buffers())


def test_loads_traced_working_set_and_restores_meta(tmp_path):
    torch.manual_seed(9)
    reference = SessionModel()
    checkpoint = checkpoint_for(reference, tmp_path)
    model = build_meta_model(SessionModel)
    session = SubgraphWeightSession(
        model, SafetensorsWeightSource(checkpoint)
    )
    subgraph = subgraph_for("layers.0")
    inputs = torch.randn(2, 4)

    assert session.working_set(subgraph) == ("layers.0",)
    with session.loaded(
        subgraph, device="cpu", dtype=torch.float32
    ) as loaded:
        output = subgraph.forward(model, value=inputs)["output"]
        assert torch.allclose(output, reference.layers[0](inputs))
        assert loaded.module_names == ("layers.0",)
        assert_all_meta(model.layers[1])

    assert_all_meta(model)


def test_merges_modifier_working_set_across_checkpoint_shards(tmp_path, monkeypatch):
    reference = SessionModel()
    checkpoint = checkpoint_for(reference, tmp_path)
    model = build_meta_model(SessionModel)
    source = SafetensorsWeightSource(checkpoint)
    requests = []
    original_load = source.load_tensors

    def recording_load(names, *, device):
        names = tuple(names)
        requests.append(names)
        return original_load(names, device=device)

    monkeypatch.setattr(source, "load_tensors", recording_load)
    session = SubgraphWeightSession(model, source)
    subgraph = subgraph_for("layers.0")

    with session.loaded(
        subgraph,
        device="cpu",
        dtype=torch.float32,
        include_modules=("final_norm",),
    ) as loaded:
        assert loaded.module_names == ("layers.0", "final_norm")
        assert not next(model.layers[0].parameters()).is_meta
        assert not next(model.final_norm.parameters()).is_meta
        assert_all_meta(model.layers[1])

    loaded_names = {name for request in requests for name in request}
    assert any(name.startswith("layers.0.") for name in loaded_names)
    assert any(name.startswith("final_norm.") for name in loaded_names)
    assert not any(name.startswith("layers.1.") for name in loaded_names)
    assert_all_meta(model)


def test_collapses_descendant_modules_under_target(tmp_path):
    reference = SessionModel()
    checkpoint = checkpoint_for(reference, tmp_path)
    model = build_meta_model(SessionModel)
    session = SubgraphWeightSession(
        model, SafetensorsWeightSource(checkpoint)
    )
    subgraph = subgraph_for("layers.0")

    assert session.working_set(
        subgraph, include_modules=("layers.0.proj", "layers.0.norm")
    ) == ("layers.0",)


def test_state_tensors_include_modifier_output_and_skip_runtime_buffers(tmp_path):
    reference = SessionModel()
    checkpoint = checkpoint_for(reference, tmp_path)
    model = build_meta_model(SessionModel)
    session = SubgraphWeightSession(
        model, SafetensorsWeightSource(checkpoint)
    )
    subgraph = subgraph_for("layers.0")

    with session.loaded(
        subgraph, device="cpu", dtype=torch.float32
    ) as loaded:
        model.layers[0].proj.register_buffer(
            "weight_scale", torch.ones(4), persistent=True
        )
        model.layers[0].proj.register_buffer(
            "scratch", torch.zeros(4), persistent=False
        )
        state = dict(loaded.state_tensors())
        assert "layers.0.proj.weight_scale" in state
        assert "layers.0.proj.scratch" not in state

    assert_all_meta(model)


def test_exception_restores_every_module_in_working_set(tmp_path):
    reference = SessionModel()
    checkpoint = checkpoint_for(reference, tmp_path)
    model = build_meta_model(SessionModel)
    session = SubgraphWeightSession(
        model, SafetensorsWeightSource(checkpoint)
    )

    with pytest.raises(RuntimeError, match="failed"):
        with session.loaded(
            subgraph_for("layers.0"),
            device="cpu",
            include_modules=("final_norm",),
        ):
            raise RuntimeError("failed")

    assert_all_meta(model)


def test_shared_runtime_buffer_aliases_materialize_independently(tmp_path):
    reference = SharedRuntimeBufferModel()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_file(reference.state_dict(), checkpoint / "model.safetensors")
    model = build_meta_model(SharedRuntimeBufferModel)
    session = SubgraphWeightSession(
        model, SafetensorsWeightSource(checkpoint)
    )
    subgraph = subgraph_for("layers.0")

    with session.loaded(
        subgraph,
        device="cpu",
        dtype=torch.float32,
        include_modules=("layers",),
    ):
        assert not model.layers[0].runtime.is_meta
        assert not model.layers[1].runtime.is_meta
        assert model.layers[0].runtime.data_ptr() != (
            model.layers[1].runtime.data_ptr()
        )
        assert torch.equal(model.layers[0].runtime, torch.ones(4))
        assert torch.equal(model.layers[1].runtime, torch.full((4,), 2.0))
        assert model.layers[0].runtime_alias is model.layers[0].runtime

    assert model.layers[0].runtime.is_meta
    assert model.layers[1].runtime.is_meta
    assert model.layers[0].runtime_alias.is_meta
