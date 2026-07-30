from operator import getitem

import torch
from safetensors.torch import save_file
from torch import nn
from torch.fx import Graph

from llmcompressor.pipelines.sequential.helpers import Subgraph
from llmcompressor.pipelines.sequential.plan import SequentialExecutionPlan
from llmcompressor.streaming import SafetensorsWeightSource, build_meta_model
from llmcompressor.streaming.loading import SubgraphWeightSession, TargetWeightLoader
from llmcompressor.streaming.tracing import (
    TracedBoundaryAdapter,
    _project_through_target,
)


class _TupleTarget(nn.Module):
    def forward(self, value):
        return value + 1, value + 2


class _MultimodalPrefixModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.vision_tower = nn.Linear(4, 4)
        self.mm_projector = nn.Linear(4, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4)])


def test_target_projection_preserves_tuple_unpack_nodes():
    model = nn.Module()
    model.layer = _TupleTarget()
    graph = Graph()
    value = graph.placeholder("value")
    output = graph.call_module("layer", (value,))
    first = graph.call_function(getitem, (output, 0))
    second = graph.call_function(getitem, (output, 1))
    graph.output({first.name: first, second.name: second})
    subgraph = Subgraph(graph=graph, input_names={"value"}, consumed_names=set())

    projected = _project_through_target(subgraph, "layer")
    result = projected.forward(model, value=torch.tensor(3))

    assert result == {first.name: torch.tensor(4), second.name: torch.tensor(5)}


def test_prefix_runtime_discovers_autowrapped_checkpoint_dependencies(tmp_path):
    torch.manual_seed(11)
    reference = _MultimodalPrefixModel()
    checkpoint = tmp_path / "model.safetensors"
    save_file(
        {
            name: tensor.detach().clone()
            for name, tensor in reference.state_dict().items()
        },
        checkpoint,
    )
    model = build_meta_model(_MultimodalPrefixModel)

    def hidden_prefix(input_ids, pixel_values):
        text = model.embed_tokens(input_ids)
        image = model.mm_projector(model.vision_tower(pixel_values))
        return text + image.unsqueeze(1)

    graph = Graph()
    input_ids = graph.placeholder("input_ids")
    pixel_values = graph.placeholder("pixel_values")
    hidden = graph.call_function(hidden_prefix, (input_ids, pixel_values))
    graph.output({"hidden": hidden})
    prefix = Subgraph(
        graph=graph,
        input_names={"input_ids", "pixel_values"},
        consumed_names=set(),
    )
    target_graph = Graph()
    target_input = target_graph.placeholder("hidden")
    target_output = target_graph.call_module("layers.0", (target_input,))
    target_graph.output({"hidden": target_output})
    target_subgraph = Subgraph(
        graph=target_graph,
        input_names={"hidden"},
        consumed_names=set(),
    )
    plan = SequentialExecutionPlan(
        subgraphs=(prefix, target_subgraph),
        target_names=("layers.0",),
        target_subgraph_indices=(1,),
    )
    source = SafetensorsWeightSource(checkpoint)
    weight_session = SubgraphWeightSession(model, source)
    adapter = TracedBoundaryAdapter(
        model=model,
        plan=plan,
        loader=TargetWeightLoader(model, source),
        weight_session=weight_session,
        device=torch.device("cpu"),
        dtype=torch.float32,
        _target_subgraphs=(target_subgraph,),
    )
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "pixel_values": torch.randn(1, 4),
    }
    expected = reference.embed_tokens(batch["input_ids"]) + reference.mm_projector(
        reference.vision_tower(batch["pixel_values"])
    ).unsqueeze(1)

    boundaries = list(adapter.calibration_boundaries(iter((batch,))))

    assert torch.allclose(boundaries[0]["hidden"], expected)
    assert adapter._prefix_runtime_modules == (
        "embed_tokens",
        "vision_tower",
        "mm_projector",
    )
    assert all(parameter.is_meta for parameter in model.parameters())
