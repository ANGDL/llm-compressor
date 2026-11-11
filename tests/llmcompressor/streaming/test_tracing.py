from operator import getitem

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from torch.fx import Graph
from transformers import PretrainedConfig

from llmcompressor.pipelines.sequential.helpers import (
    Subgraph,
    partition_graph,
    trace_consumed_names,
)
from llmcompressor.pipelines.sequential.plan import SequentialExecutionPlan
from llmcompressor.streaming import SafetensorsWeightSource, build_meta_model
from llmcompressor.streaming.loading import SubgraphWeightSession, TargetWeightLoader
from llmcompressor.streaming.tracing import (
    TracedBoundaryAdapter,
    _project_through_target,
    trace_streaming_boundaries,
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


class _DeclaredParameterPrefixModel(_MultimodalPrefixModel):
    def __init__(self):
        super().__init__()
        self.prefix_bias = nn.Parameter(torch.randn(4))

    def streaming_runtime_parameter_dependencies(self, stage):
        if stage == "prefix":
            return ("prefix_bias",)
        return ()


def _specialized_offset(value, offset):
    if offset > 0:
        return value + offset
    return value - offset


class _NumericMetadataModel(_MultimodalPrefixModel):
    def __init__(self, specialize=False):
        super().__init__()
        self.config = PretrainedConfig()
        self.device = torch.device("cpu")
        self.specialize = specialize

    def forward(self, input_ids, offset, gain):
        hidden = self.embed_tokens(input_ids) * gain
        if self.specialize:
            hidden = _specialized_offset(hidden, offset)
        else:
            hidden = hidden + offset
        return self.layers[0](hidden)


@pytest.mark.parametrize("specialize", [False, True])
def test_traced_numeric_metadata_preserves_values_and_specialized_branches(
    tmp_path, specialize
):
    reference = _NumericMetadataModel(specialize)
    checkpoint = tmp_path / "model.safetensors"
    save_file(reference.state_dict(), checkpoint)
    batches = [
        {"input_ids": torch.tensor([[1, 2]]), "offset": 2, "gain": 0.5},
        {"input_ids": torch.tensor([[3, 4, 5]]), "offset": -3, "gain": 1.25},
    ]
    model = build_meta_model(_NumericMetadataModel, specialize)
    adapter = trace_streaming_boundaries(
        model=model,
        source=SafetensorsWeightSource(checkpoint),
        sample_batch=batches[0],
        sequential_targets=["layers.0"],
        device="cpu",
        dtype=torch.float32,
    )
    assert ("offset" in adapter.specialized_input_names) is specialize
    if specialize:
        with pytest.raises(ValueError, match="traced input contract"):
            list(adapter.calibration_boundaries(batches))
    else:
        boundaries = list(adapter.calibration_boundaries(batches))
        for batch, boundary in zip(batches, boundaries):
            subgraph = adapter.target_subgraphs[0]
            with adapter.weight_session.loaded(
                subgraph, device="cpu", dtype=torch.float32
            ):
                output = subgraph.forward(
                    model, **{name: boundary[name] for name in subgraph.input_names}
                )
            torch.testing.assert_close(next(iter(output.values())), reference(**batch))
    assert all(parameter.is_meta for parameter in model.parameters())


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


def test_target_projection_preserves_suffix_dependencies():
    model = nn.Module()
    model.layer = _TupleTarget()
    graph = Graph()
    value = graph.placeholder("value")
    output = graph.call_module("layer", (value,))
    first = graph.call_function(getitem, (output, 0))
    scaled = graph.call_function(torch.mul, (first, 3))
    second = graph.call_function(getitem, (output, 1))
    graph.output({scaled.name: scaled, second.name: second})
    subgraph = Subgraph(graph=graph, input_names={"value"}, consumed_names=set())

    projected = _project_through_target(subgraph, "layer")
    result = projected.forward(model, value=torch.tensor(3))

    # A later target needs both the suffix operation and the non-contiguous
    # tuple unpack, not just the first getitem after the target call.
    assert result[scaled.name] == torch.tensor(12)
    assert result[second.name] == torch.tensor(5)


def test_streaming_boundaries_preserve_interleaved_tuple_and_suffix(
    tmp_path, monkeypatch
):
    model = nn.Module()
    model.first = _TupleTarget()
    model.last = nn.Linear(4, 4)
    model.unused_head = nn.Linear(4, 4)
    checkpoint = tmp_path / "model.safetensors"
    save_file(model.state_dict(), checkpoint)
    graph = Graph()
    value = graph.placeholder("value")
    first = graph.call_module("first", (value,))
    first_item = graph.call_function(getitem, (first, 0))
    scaled = graph.call_function(torch.mul, (first_item, 3))
    second_item = graph.call_function(getitem, (first, 1))
    combined = graph.call_function(torch.add, (scaled, second_item))
    last = graph.call_module("last", (combined,))
    head = graph.call_module("unused_head", (last,))
    output = graph.output(head)
    subgraphs = partition_graph(
        model,
        [
            [value],
            [first, first_item, scaled, second_item],
            [combined, last, head, output],
        ],
    )
    trace_consumed_names(subgraphs)
    plan = SequentialExecutionPlan(
        subgraphs=tuple(subgraphs),
        target_names=("first", "last"),
        target_subgraph_indices=(1, 2),
    )
    monkeypatch.setattr(
        "llmcompressor.streaming.tracing.trace_sequential_plan",
        lambda *args, **kwargs: plan,
    )
    data = torch.randn(2, 4)
    expected = model.last((data + 1) * 3 + data + 2)
    model.to("meta")
    adapter = trace_streaming_boundaries(
        model=model,
        source=SafetensorsWeightSource(checkpoint),
        sample_batch={"value": data},
        device="cpu",
        dtype=torch.float32,
    )

    boundary = next(adapter.calibration_boundaries([{"value": data}]))
    for target_name in adapter.targets:
        target = model.get_submodule(target_name)
        target._streaming_target_name = target_name
        subgraph = adapter.target_subgraphs[adapter.targets.index(target_name)]
        with adapter.weight_session.loaded(
            subgraph, device=torch.device("cpu"), dtype=torch.float32
        ):
            boundary = adapter.forward_target(target, boundary)

    torch.testing.assert_close(boundary[last.name], expected)
    assert set(boundary) == {last.name}
    assert model.unused_head.weight.is_meta
    assert all(
        node.target != "unused_head"
        for node in adapter.target_subgraphs[-1].graph.nodes
    )


def test_prefix_runtime_discovers_autowrapped_checkpoint_dependencies(tmp_path):
    torch.manual_seed(11)
    reference = _DeclaredParameterPrefixModel()
    checkpoint = tmp_path / "model.safetensors"
    save_file(
        {
            name: tensor.detach().clone()
            for name, tensor in reference.state_dict().items()
        },
        checkpoint,
    )
    model = build_meta_model(_DeclaredParameterPrefixModel)

    def hidden_prefix(input_ids, pixel_values):
        text = model.embed_tokens(input_ids)
        image = model.mm_projector(model.vision_tower(pixel_values))
        return text + image.unsqueeze(1) + model.prefix_bias

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
    expected = (
        reference.embed_tokens(batch["input_ids"])
        + reference.mm_projector(
            reference.vision_tower(batch["pixel_values"])
        ).unsqueeze(1)
        + reference.prefix_bias
    )

    boundaries = list(adapter.calibration_boundaries(iter((batch,))))

    assert torch.allclose(boundaries[0]["hidden"], expected)
    assert adapter._prefix_runtime_modules == (
        "embed_tokens",
        "vision_tower",
        "mm_projector",
    )
    assert adapter._prefix_runtime_parameters == ("prefix_bias",)
    assert all(parameter.is_meta for parameter in model.parameters())
