import operator
import os
from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.fx import Graph

from llmcompressor.core import LifecycleCallbacks, create_session
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.sequential.helpers import Subgraph
from llmcompressor.pipelines.sequential.plan import SequentialExecutionPlan
from llmcompressor.recipe import Recipe
from llmcompressor.streaming.activations import InMemoryBoundaryActivationStore
from llmcompressor.streaming.admission import DeviceMemoryBudget
from llmcompressor.streaming.artifacts import ArtifactStore
from llmcompressor.streaming.checkpoint import SafetensorsWeightSource
from llmcompressor.streaming.gptq_adapter import GPTQStreamingMemoryAdapter
from llmcompressor.streaming.loading import (
    HostMemoryBudget,
    SubgraphWeightSession,
    TargetWeightLoader,
    build_meta_model,
)
from llmcompressor.streaming.materialization import CastWeightMaterializer
from llmcompressor.streaming.memory_adapters import modifier_calibration_context
from llmcompressor.streaming.parallel.expert import ExpertParallelPlan
from llmcompressor.streaming.parallel.region import (
    ExpertRegionGraph,
    UnsupportedExpertRegion,
    coalesce_expert_regions,
    execute_expert_region,
    profile_expert_region,
)
from llmcompressor.streaming.pipeline import run_subgraph_streaming_pipeline
from llmcompressor.streaming.tracing import (
    TracedBoundaryAdapter,
    _split_target_subgraph,
)
from llmcompressor.utils.helpers import DisableQuantization


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(4, 8, bias=False)
        self.down = nn.Linear(8, 4, bias=False)

    def forward(self, hidden):
        return self.down(self.up(hidden).relu())


class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([Expert() for _index in range(4)])


def subgraph():
    graph = Graph()
    hidden = graph.placeholder("hidden")
    indices = graph.placeholder("indices")
    output = graph.call_function(torch.zeros_like, (hidden,))
    for index in range(4):
        selected = graph.call_function(operator.eq, (indices, index))
        matches = graph.call_function(torch.where, (selected,))
        rows = graph.call_function(operator.getitem, (matches, 0))
        inputs = graph.call_method("index_select", (hidden, 0, rows))
        result = graph.call_module(f"experts.{index}", (inputs,))
        graph.call_method("index_add_", (output, 0, rows, result))
    graph.output({"result": output})
    return Subgraph(graph, {"hidden", "indices"}, {"hidden", "indices"})


def values():
    generator = torch.Generator().manual_seed(14)
    return [
        {
            "hidden": torch.randn(10, 4, generator=generator),
            "indices": torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 2, 2]),
        },
        {
            "hidden": torch.randn(7, 4, generator=generator),
            "indices": torch.tensor([0, 0, 0, 0, 1, 1, 2]),
        },
    ]


def quantizer():
    return GPTQModifier(
        config_groups={
            "weights": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "channel",
                },
            }
        },
        actorder=None,
        block_size=4,
        batched_quantization=None,
    )


def make_plan(model):
    return ExpertParallelPlan.from_model(
        model,
        tuple(f"experts.{index}" for index in range(4)),
        ("cpu:0", "cpu:1"),
    )


def test_region_profile_reads_real_indices_without_executing_experts():
    model = MoE()
    plan = make_plan(model)
    graph = ExpertRegionGraph.build(model, plan, (subgraph(),))
    boundaries = InMemoryBoundaryActivationStore()
    for index, value in enumerate(values()):
        boundaries.put(0, index, value)
    for expert in model.experts:
        expert.register_forward_pre_hook(
            lambda *_args: pytest.fail("profiling must not execute an expert")
        )
    profile = profile_expert_region(graph, boundaries, 0, (0, 1))
    assert profile.token_counts == {
        "experts.0": 10,
        "experts.1": 4,
        "experts.2": 3,
        "experts.3": 0,
    }
    assert profile.batch_counts["experts.0"] == (6, 4)


def test_region_rejects_experts_with_dependent_inputs():
    model = MoE()
    graph = Graph()
    hidden = graph.placeholder("hidden")
    first = graph.call_module("experts.0", (hidden,))
    second = graph.call_module("experts.1", (first,))
    graph.output({"result": second})
    with pytest.raises(UnsupportedExpertRegion, match="another expert"):
        ExpertRegionGraph.build(
            model, make_plan(model), (Subgraph(graph, {"hidden"}, set()),)
        )


def test_region_coalescing_crosses_fixed_trace_groups():
    model = MoE()
    targets = tuple(f"experts.{index}" for index in range(4))
    adapter = SimpleNamespace(
        model=model,
        targets=targets,
        target_groups=((0, 1), (2, 3)),
        target_subgraphs=_split_target_subgraph(model, subgraph(), targets),
    )
    assert coalesce_expert_regions(adapter, make_plan(model)) == ((0, 1, 2, 3),)


def reference_quantization(model, batches):
    with create_session() as session:
        session.state.layerwise_calibration = True
        session.initialize(model=model, recipe=[quantizer()], start=-1, copy_data=False)
        modifier = session.lifecycle.recipe.modifiers[0]
        LifecycleCallbacks.calibration_start()
        modules = tuple(model.modules())
        modifier.start_layerwise_calibration(model, list(modules))
        with torch.no_grad(), DisableQuantization(model):
            for value in batches:
                subgraph().forward(model, **value)
        statistics = {
            modifier._module_names[module]: (
                hessian.clone(),
                modifier._num_samples[module].clone(),
            )
            for module, hessian in modifier._hessians.items()
        }
        LifecycleCallbacks.sequential_epoch_end(list(modules))
        modifier.end_layerwise_calibration(list(modules))
        with (
            torch.no_grad(),
            DisableQuantization(model),
            HooksMixin.disable_hooks_local(),
        ):
            outputs = [
                subgraph().forward(model, **value)["result"] for value in batches
            ]
        tensors = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        LifecycleCallbacks.calibration_end()
        session.finalize()
    return statistics, tensors, outputs


@pytest.mark.parametrize(
    "token_budget,memory_bytes",
    [("auto", 2**20), (2, 2**20), (4, 2**20), ("auto", 5000)],
)
def test_region_matches_serial_gptq_statistics_weights_and_propagation(
    tmp_path, token_budget, memory_bytes
):
    torch.manual_seed(19)
    source = MoE().eval()
    batches = values()
    expected_statistics, expected_tensors, expected_outputs = reference_quantization(
        deepcopy(source), deepcopy(batches)
    )
    save_file(source.state_dict(), tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text("{}")
    model = build_meta_model(MoE).eval()
    plan = make_plan(model)
    graph = ExpertRegionGraph.build(model, plan, (subgraph(),))
    boundaries = InMemoryBoundaryActivationStore()
    for index, value in enumerate(batches):
        boundaries.put(0, index, value)
    profile = profile_expert_region(graph, boundaries, 0, (0, 1))
    adapter = SimpleNamespace(
        model=model,
        weight_session=SubgraphWeightSession(model, SafetensorsWeightSource(tmp_path)),
    )
    statistics, tensors = {}, {}
    dispatches = []
    model.experts[0].register_forward_pre_hook(
        lambda _module, args: dispatches.append(args[0].shape[0])
    )
    with create_session() as session, ExitStack() as stack:
        session.state.layerwise_calibration = True
        session.initialize(model=model, recipe=[quantizer()], start=-1, copy_data=False)
        modifier = session.lifecycle.recipe.modifiers[0]
        stack.enter_context(modifier_calibration_context([modifier]))
        LifecycleCallbacks.calibration_start()
        memory_adapter = GPTQStreamingMemoryAdapter(modifier)
        original_update = memory_adapter.update_owner

        def update(owner, modules):
            for module in modules:
                if module in modifier._hessians:
                    statistics[modifier._module_names[module]] = (
                        modifier._hessians[module].clone(),
                        modifier._num_samples[module].clone(),
                    )
            return original_update(owner, modules)

        memory_adapter.update_owner = update

        def save(window, loaded, _modules):
            for name, tensor in loaded.state_tensors():
                tensors[name] = tensor.clone()

        result = execute_expert_region(
            graph=graph,
            profile=profile,
            adapter=adapter,
            parallel=plan,
            modifier_adapter=memory_adapter,
            modifiers=(modifier,),
            boundaries=boundaries,
            boundary_index=0,
            destination_boundary=4,
            batches=(0, 1),
            device_budget=DeviceMemoryBudget(available_bytes=memory_bytes),
            host_budget=HostMemoryBudget(),
            target_dtype=torch.float32,
            token_budget=token_budget,
            label="test",
            save_window=save,
        )
        assert result["placement"]["policy"] == "activation-aware-v1"
        if memory_bytes == 5000:
            assert len(result["windows"]) >= 4
        assert not modifier._hessians
        assert not modifier._num_samples
        assert all(parameter.is_meta for parameter in model.parameters())
        LifecycleCallbacks.calibration_end()
        session.finalize()
    assert statistics.keys() == expected_statistics.keys()
    for name, (hessian, count) in expected_statistics.items():
        torch.testing.assert_close(statistics[name][0], hessian, msg=name)
        torch.testing.assert_close(statistics[name][1], count, msg=name)
    assert tensors.keys() == expected_tensors.keys()
    for name, expected in expected_tensors.items():
        torch.testing.assert_close(tensors[name], expected, msg=name)
    for index, expected in enumerate(expected_outputs):
        torch.testing.assert_close(boundaries.get(4, index)["result"], expected)
    if token_budget == 4:
        assert dispatches == [4, 4, 2, 4, 4, 2]


class PipelineMoE(MoE):
    def __init__(self):
        super().__init__()
        self.post = nn.Linear(4, 4, bias=False)


@pytest.mark.parametrize(
    "devices",
    [
        ("cpu:0", "cpu:1"),
        pytest.param(
            ("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
            marks=pytest.mark.multi_gpu,
        ),
    ],
)
def test_pipeline_uses_routed_windows_and_preserves_downstream_gptq(
    tmp_path, monkeypatch, devices
):
    import llmcompressor.streaming.pipeline as pipeline

    hardware = devices[0].startswith("cuda")
    if hardware and (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "4,5,6,7"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < 4
    ):
        pytest.skip("This pipeline validation requires physical devices 4-7")
    torch.manual_seed(29)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_file(PipelineMoE().state_dict(), checkpoint / "model.safetensors")
    (checkpoint / "config.json").write_text("{}")
    monkeypatch.setattr(
        pipeline,
        "DeviceMemoryBudget",
        lambda: DeviceMemoryBudget(available_bytes=2**22),
    )
    targets = (*tuple(f"experts.{index}" for index in range(4)), "post")
    prefix = Graph()
    hidden = prefix.placeholder("hidden")
    indices = prefix.placeholder("indices")
    prefix.output({"hidden": hidden, "indices": indices})
    prefix = Subgraph(prefix, {"hidden", "indices"}, set())
    tail = Graph()
    result = tail.placeholder("result")
    final = tail.call_module("post", (result,))
    tail.output({"final": final})
    tail = Subgraph(tail, {"result"}, {"result"})
    outputs = []
    for parallel in (False, True):
        execution_device = torch.device(devices[0] if parallel else "cpu")
        model = build_meta_model(PipelineMoE).eval()
        source = SafetensorsWeightSource(checkpoint)
        slices = (*_split_target_subgraph(model, subgraph(), targets[:-1]), tail)
        adapter = TracedBoundaryAdapter(
            model=model,
            plan=SequentialExecutionPlan(
                (prefix, *slices), targets, tuple(range(1, len(targets) + 1))
            ),
            loader=TargetWeightLoader(model, source),
            weight_session=SubgraphWeightSession(model, source),
            device=execution_device,
            dtype=torch.float32,
            _target_subgraphs=slices,
        )
        artifacts, output, _fingerprint = run_subgraph_streaming_pipeline(
            adapter=adapter,
            checkpoint=checkpoint,
            work_dir=tmp_path / ("parallel" if parallel else "serial"),
            calibration_batches=values(),
            recipe=Recipe.create_instance(quantizer()),
            dataset_fingerprint="synthetic-routes",
            materializer=CastWeightMaterializer(),
            device=execution_device,
            target_dtype=torch.float32,
            num_samples=2,
            max_seq_length=None,
            seed=29,
            pack_to_int8=False,
            expert_parallel=(
                ExpertParallelPlan.from_model(model, targets, devices)
                if parallel
                else None
            ),
            expert_parallel_token_budget="auto",
        )
        outputs.append(
            {
                name: tensor
                for shard in output.glob("*.safetensors")
                for name, tensor in load_file(shard).items()
            }
        )
        assert all(parameter.is_meta for parameter in model.parameters())
        if parallel:
            assert (
                ArtifactStore(artifacts).load_ep_token_plan(0)["policy"]
                == "routed-resident-windows-v1"
            )
    assert outputs[0].keys() == outputs[1].keys()
    for name in outputs[0]:
        torch.testing.assert_close(outputs[1][name], outputs[0][name], msg=name)


@pytest.mark.parametrize("dynamic,expected", [(False, False), (True, True)])
def test_window_capability_preserves_activation_observer_semantics(dynamic, expected):
    modifier = quantizer()
    from compressed_tensors.quantization import QuantizationArgs

    modifier.resolved_config.config_groups[
        "weights"
    ].input_activations = QuantizationArgs(
        num_bits=8,
        type="int",
        strategy="token" if dynamic else "tensor",
        dynamic=dynamic,
    )
    assert (
        GPTQStreamingMemoryAdapter(modifier).supports_independent_expert_windows
        is expected
    )


def test_failed_expert_window_releases_its_own_hessians_and_hooks():
    model = MoE().eval()
    with create_session() as session:
        session.state.layerwise_calibration = True
        session.initialize(model=model, recipe=[quantizer()], start=-1, copy_data=False)
        LifecycleCallbacks.calibration_start()
        modifier = session.lifecycle.recipe.modifiers[0]
        adapter = GPTQStreamingMemoryAdapter(modifier)
        other_modules = tuple(model.experts[1].modules())
        failed_modules = tuple(model.experts[0].modules())
        with adapter.expert_window(model, other_modules), torch.no_grad():
            model.experts[1](torch.randn(3, 4))
            other_hooks = set(modifier._calibration_hooks)
            with pytest.raises(RuntimeError, match="injected failure"):
                with adapter.expert_window(model, failed_modules):
                    model.experts[0](torch.randn(2, 4))
                    raise RuntimeError("injected failure")
            assert modifier._calibration_hooks == other_hooks
            assert not any(module in modifier._hessians for module in failed_modules)
            assert all(
                module in modifier._hessians
                for module in other_modules
                if isinstance(module, nn.Linear)
            )
            adapter.update_owner(torch.device("cpu"), other_modules)
        assert not modifier._hessians
        assert not modifier._num_samples
        LifecycleCallbacks.calibration_end()
        session.finalize()
