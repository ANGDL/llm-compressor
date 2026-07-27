import math
from unittest.mock import patch

import pytest
import torch
import torch.fx
from transformers import AutoModelForCausalLM, PretrainedConfig

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.pipelines.sequential.helpers import (
    get_sequential_ancestors,
    topological_partition,
    trace_subgraphs,
)
from llmcompressor.utils.dev import skip_weights_download, skip_weights_initialize


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(torch.nn.Linear(10, 20), torch.nn.ReLU())
        self.fc = torch.nn.Linear(20, 5)

    def forward(self, x):
        x = self.seq(x)
        return self.fc(x)


class DummyModelMultipleSequentialLayers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = torch.nn.Linear(10, 10)
        self.layer3 = torch.nn.Linear(10, 10)
        self.layer4 = torch.nn.Linear(10, 10)
        self.layer5 = torch.nn.Linear(10, 10)
        self.layer6 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x


class _SequentialTarget(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1


class _DataDependentTargetCall(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.target = _SequentialTarget()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.numel() > 0:
            value = self.target(value)
        return value


class _ModelWithHiddenSequentialTarget(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig()
        self.device = torch.device("cpu")
        self.visible_target = _SequentialTarget()
        self.data_dependent = _DataDependentTargetCall()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.visible_target(value)
        return self.data_dependent(value)


def test_get_sequential_ancestors():
    with skip_weights_initialize():
        model = DummyModel()

    assert get_sequential_ancestors(model, set()) == set()
    assert get_sequential_ancestors(model, {model}) == set()
    assert get_sequential_ancestors(model, {model.fc}) == {model}
    assert get_sequential_ancestors(model, {model.seq[0]}) == {model, model.seq}
    assert get_sequential_ancestors(model, {model.seq[1]}) == {model, model.seq}


def test_topological_partition_default():
    with skip_weights_initialize():
        model = DummyModelMultipleSequentialLayers()

    targets = {
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        model.layer5,
        model.layer6,
    }
    gm = torch.fx.symbolic_trace(model)

    assert len(topological_partition(gm, targets)) == 7


def test_topological_partition_multiple_targets():
    with skip_weights_initialize():
        model = DummyModelMultipleSequentialLayers()

    gm = torch.fx.symbolic_trace(model)
    targets = {
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        model.layer5,
        model.layer6,
    }

    assert len(topological_partition(gm, targets, 2)) == 4


def test_topological_partition_invalid():
    with skip_weights_initialize():
        model = DummyModelMultipleSequentialLayers()

    gm = torch.fx.symbolic_trace(model)
    targets = {
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        model.layer5,
        model.layer6,
    }

    with pytest.raises(ValueError):
        topological_partition(gm, targets, 0)


def test_trace_subgraphs_warns_when_matched_target_is_hidden_by_autowrap():
    model = _ModelWithHiddenSequentialTarget()

    with patch(
        "llmcompressor.pipelines.sequential.helpers.logger.warning"
    ) as warning:
        subgraphs = trace_subgraphs(
            model,
            {"value": torch.ones(1)},
            sequential_targets=["_SequentialTarget"],
            ignore=[],
        )

    covered_modules = {
        module for subgraph in subgraphs for module in subgraph.submodules(model)
    }
    assert model.visible_target in covered_modules
    assert model.data_dependent.target not in covered_modules
    messages = [call.args[0] for call in warning.call_args_list]
    message = next(
        message for message in messages if message.startswith("Sequential tracing left")
    )
    assert "1/2 matched target modules outside all subgraphs" in message
    assert "data_dependent.target" in message
    assert "pipeline='independent'" in message


@pytest.mark.parametrize("targets_per_subgraph", [1, 2, 3, 4, 5])
def test_trace_subgraphs(targets_per_subgraph):
    target = "Qwen3DecoderLayer"

    with skip_weights_download(AutoModelForCausalLM):
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype="auto")

    subgraphs = trace_subgraphs(
        model,
        model.dummy_inputs,
        sequential_targets=[target],
        ignore=DatasetArguments().tracing_ignore,
        targets_per_subgraph=targets_per_subgraph,
    )

    # +1 refers to preamble before first target
    min_num_subgraphs = len(model.model.layers) // targets_per_subgraph + 1
    max_num_subgraphs = math.ceil(len(model.model.layers) / targets_per_subgraph) + 1
    assert min_num_subgraphs <= len(subgraphs) <= max_num_subgraphs
    for subgraph in subgraphs[1:-1]:  # only check middle, ends can can be non-divisible
        subgraph_modules = subgraph.submodules(model)
        num_targets_present = len(
            [
                module
                for module in subgraph_modules
                if module.__class__.__name__ == target
            ]
        )
        assert num_targets_present == targets_per_subgraph
