from operator import getitem

import torch
from torch import nn
from torch.fx import Graph

from llmcompressor.pipelines.sequential.helpers import Subgraph
from llmcompressor.streaming.tracing import _project_through_target


class _TupleTarget(nn.Module):
    def forward(self, value):
        return value + 1, value + 2


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
