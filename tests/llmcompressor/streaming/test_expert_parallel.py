from __future__ import annotations

import gc
import json
import operator
import os
import threading
import weakref
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict

import pytest
import torch
from torch import nn
from torch.fx import Graph

from llmcompressor.core import LifecycleCallbacks, create_session
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.pipelines.sequential.helpers import Subgraph
from llmcompressor.streaming.artifacts import ParallelInfo
from llmcompressor.streaming.checkpoint import SafetensorsWeightSource
from llmcompressor.streaming.loading import (
    HostMemoryBudget,
    SubgraphWeightSession,
    build_meta_model,
)
from llmcompressor.streaming.memory_adapters import modifier_calibration_context
from llmcompressor.streaming.parallel.expert import (
    ExpertParallelExecutor,
    ExpertParallelPlan,
    _ExecutionGraph,
    resolve_expert_parallel_devices,
)
from llmcompressor.streaming.parallel.topology import (
    EPConfig,
    EPMode,
    EPTopology,
    build_expert_placement,
    make_expert_state_filter,
)
from llmcompressor.streaming.tracing import _split_target_subgraph
from llmcompressor.utils.helpers import DisableQuantization


class _Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(4, 8, bias=False)
        self.w2 = nn.Linear(8, 4, bias=False)
        self.barrier = None
        self.fail = False
        self.calls = 0

    def forward(self, hidden):
        self.calls += 1
        if self.barrier is not None:
            self.barrier.wait(timeout=10)
        if self.fail:
            raise RuntimeError("expert failure")
        return self.w2(self.w1(hidden).relu())


class _MoE(nn.Module):
    def __init__(self, count=5):
        super().__init__()
        self.experts = nn.ModuleList([_Expert() for _index in range(count)])
        self.shared_expert = _Expert()


def _targets(model):
    return tuple(f"experts.{index}" for index in range(len(model.experts)))


def _graph(model, *, mutation=True):
    graph = Graph()
    hidden = graph.placeholder("hidden")
    indices = graph.placeholder("indices")
    output = graph.call_function(torch.zeros_like, (hidden,))
    for expert_index in range(len(model.experts)):
        selected = graph.call_function(operator.eq, (indices, expert_index))
        rows = graph.call_function(torch.where, (selected,))
        rows = graph.call_function(operator.getitem, (rows, 0))
        inputs = graph.call_method("index_select", (hidden, 0, rows))
        values = graph.call_module(f"experts.{expert_index}", (inputs,))
        if mutation:
            graph.call_method("index_add_", (output, 0, rows, values))
        else:
            contribution = graph.call_method(
                "index_add", (torch.zeros(6, 4), 0, rows, values)
            )
            output = graph.call_function(operator.add, (output, contribution))
    graph.output({"result": output})
    return Subgraph(graph, input_names={"hidden", "indices"}, consumed_names=set())


@pytest.mark.parametrize("slots", [2, 4, 8])
def test_placement_handles_uneven_experts_and_preserves_shared_owner(slots):
    model = _MoE()
    plan = ExpertParallelPlan.from_model(
        model, _targets(model), [f"cpu:{index}" for index in range(slots)]
    )
    assert [entry.owner_rank for entry in plan.placement.entries] == [
        index % slots for index in range(5)
    ]
    assert plan.device_for("shared_expert.w1") == torch.device("cpu:0")
    assert plan.device_for("experts.3.w2") == torch.device(f"cpu:{3 % slots}")
    assert (
        plan.fingerprint
        == ExpertParallelPlan.from_model(
            model, _targets(model), plan.devices
        ).fingerprint
    )
    changed = ExpertParallelPlan.from_model(
        model, _targets(model), tuple(reversed(plan.devices))
    )
    assert changed.fingerprint != plan.fingerprint


def test_placement_balances_expert_bytes_with_stable_ties():
    model = _MoE(4)
    with torch.no_grad():
        model.experts[0].w1.weight = nn.Parameter(torch.randn(16, 4))
        model.experts[1].w1.weight = nn.Parameter(torch.randn(8, 4))
        model.experts[2].w1.weight = nn.Parameter(torch.randn(4, 4))
        model.experts[3].w1.weight = nn.Parameter(torch.randn(2, 4))
    plan = ExpertParallelPlan.from_model(
        model,
        _targets(model),
        ("cpu:0", "cpu:1"),
    )
    owners = [entry.owner_rank for entry in plan.placement.entries]
    assert owners == [0, 1, 1, 0]
    assert dict(plan.expert_bytes)["experts.0"] > dict(plan.expert_bytes)["experts.3"]


def test_external_adapter_collection_can_drive_plan_and_proxy_ids():
    model = _MoE(4)

    class Adapter:
        def validate(self, value):
            assert value is model

        def expert_module_name(self):
            return "experts"

    adapter = Adapter()
    plan = ExpertParallelPlan.from_model(
        model,
        _targets(model),
        ("cpu:0", "cpu:1"),
        adapters=(adapter,),
    )
    assert plan.expert_collections == ("experts",)
    assert plan.local_expert_ids("experts", 0) == (0, 2)


def test_generic_ep_topology_and_state_filter():
    topology = EPTopology.from_config(EPConfig())
    assert topology.config.world_size == 1
    with pytest.raises(ValueError, match="world_size"):
        EPTopology.from_config(EPConfig(ep_size=2, mode=EPMode.ALL_EXPERT))
    placement = build_expert_placement(
        num_experts=4,
        ep_size=2,
        expert_bytes=(10, 7, 6, 1),
    )
    state_filter = make_expert_state_filter(
        placement,
        1,
        lambda name: (
            int(name.split(".experts.", 1)[1].split(".", 1)[0])
            if ".experts." in name
            else None
        ),
    )
    assert state_filter("layers.0.router.weight")
    assert state_filter("layers.0.experts.1.w1.weight")
    assert not state_filter("layers.0.experts.0.w1.weight")


def test_placement_rejects_hidden_experts_and_cross_owner_aliases():
    model = _MoE()
    with pytest.raises(ValueError, match="independently traced"):
        ExpertParallelPlan.from_model(model, ("experts",), ("cpu:0", "cpu:1"))
    model.experts[1].w1.weight = model.experts[0].w1.weight
    with pytest.raises(ValueError, match="shared across expert owners"):
        ExpertParallelPlan.from_model(model, _targets(model), ("cpu:0", "cpu:1"))


@pytest.mark.parametrize(
    "devices,primary,message",
    [
        (["cuda:0"], "cuda:0", "at least two"),
        (["cuda:0", "cuda:0"], "cuda:0", "unique"),
        (["cpu:0", "cpu:1"], "cpu:0", "explicit cuda"),
        (["cuda:0", "cuda:1"], "cuda:1", "first"),
    ],
)
def test_invalid_devices_fail_before_loading(devices, primary, message):
    with pytest.raises(ValueError, match=message):
        resolve_expert_parallel_devices(devices, torch.device(primary))


@pytest.mark.parametrize("split", [False, True])
def test_concurrent_routing_preserves_mutation_and_cross_slice_liveness(split):
    torch.manual_seed(7)
    model = _MoE(2).eval()
    subgraph = _graph(model)
    hidden = torch.randn(6, 4)
    value = {"hidden": hidden, "indices": torch.tensor([0, 1, 0, 1, 1, 0])}
    expected = subgraph.forward(model, **value)["result"]
    subgraphs = (
        _split_target_subgraph(model, subgraph, _targets(model))
        if split
        else [subgraph]
    )
    barrier = threading.Barrier(2)
    for expert in model.experts:
        expert.barrier = barrier
    plan = ExpertParallelPlan.from_model(model, _targets(model), ("cpu:0", "cpu:1"))
    with ExpertParallelExecutor(model, plan) as executor:
        for _repeat in range(2):
            actual = executor.forward(subgraphs, value)
            assert torch.equal(actual["result"], expected)
            assert torch.equal(value["hidden"], hidden)
            assert executor._graphs


def test_forward_many_batches_repeated_experts_with_variable_lengths():
    torch.manual_seed(11)
    model = _MoE(2).eval()
    subgraph = _graph(model)
    values = [
        {
            "hidden": torch.randn(3, 4),
            "indices": torch.tensor([0, 1, 0]),
        },
        {
            "hidden": torch.randn(5, 4),
            "indices": torch.tensor([1, 0, 1, 0, 1]),
        },
    ]
    expected = [subgraph.forward(model, **value)["result"] for value in values]
    for expert in model.experts:
        expert.calls = 0
    plan = ExpertParallelPlan.from_model(model, _targets(model), ("cpu:0", "cpu:1"))
    with ExpertParallelExecutor(model, plan) as executor:
        actual = executor.forward_many([subgraph], values)

    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result["result"], reference)
    assert [expert.calls for expert in model.experts] == [1, 1]


def test_forward_many_respects_per_owner_token_capacity():
    torch.manual_seed(13)
    model = _MoE(2).eval()
    subgraph = _graph(model)
    values = [
        {
            "hidden": torch.randn(3, 4),
            "indices": torch.tensor([0, 1, 0]),
        },
        {
            "hidden": torch.randn(5, 4),
            "indices": torch.tensor([1, 0, 1, 0, 1]),
        },
    ]
    plan = ExpertParallelPlan.from_model(
        model, _targets(model), ("cpu:0", "cpu:1")
    )
    for expert in model.experts:
        expert.calls = 0
    with ExpertParallelExecutor(
        model,
        plan,
        token_capacity={"cpu:0": 3, "cpu:1": 100},
    ) as executor:
        executor.forward_many([subgraph], values)

    assert model.experts[0].calls == 2
    assert model.experts[1].calls == 1


def test_forward_many_splits_interleaved_experts_in_original_order():
    torch.manual_seed(17)
    model = _MoE(4).eval()
    subgraph = _graph(model)
    values = [
        {
            "hidden": torch.randn(length, 4),
            "indices": torch.arange(length) % 4,
        }
        for length in (7, 9, 5)
    ]
    expected = [subgraph.forward(model, **value)["result"] for value in values]
    plan = ExpertParallelPlan.from_model(model, _targets(model), ("cpu:0", "cpu:1"))
    with ExpertParallelExecutor(
        model, plan, token_capacity={"cpu:0": 100, "cpu:1": 100}
    ) as executor:
        actual = executor.forward_many([subgraph], values)
        statistics = executor.dispatch_statistics
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result["result"], reference)
    assert statistics[torch.device("cpu:0")]["tokens"] == 12
    assert statistics[torch.device("cpu:1")]["tokens"] == 9
    assert statistics[torch.device("cpu:0")]["expert_calls"] == 2
    assert statistics[torch.device("cpu:1")]["expert_calls"] == 2


def test_forward_many_rejects_indivisible_call_larger_than_owner_capacity():
    model = _MoE(2).eval()
    plan = ExpertParallelPlan.from_model(model, _targets(model), ("cpu:0", "cpu:1"))
    with ExpertParallelExecutor(
        model, plan, token_capacity={"cpu:0": 2, "cpu:1": 10}
    ) as executor:
        with pytest.raises(MemoryError, match="requires 3 tokens"):
            executor.forward(
                [_graph(model)],
                {"hidden": torch.randn(3, 4), "indices": torch.zeros(3)},
            )


def test_multidimensional_expert_tokens_count_sequence_and_batch_dimensions():
    entry = (None, "experts.0", (torch.empty(2, 3, 4),), {})
    assert ExpertParallelExecutor._entry_token_count(entry) == 6


def test_owner_queue_prepares_only_one_dispatch_per_active_owner(monkeypatch):
    model = _MoE(2).eval()
    plan = ExpertParallelPlan.from_model(model, _targets(model), ("cpu:0", "cpu:1"))
    original = ExpertParallelExecutor._prepare_batch.__func__
    prepared_counts = {device: 0 for device in plan.devices}
    completed_counts = {device: 0 for device in plan.devices}

    def prepare(cls, entries, primary):
        owner = plan.device_for(entries[0][1])
        assert prepared_counts[owner] == completed_counts[owner]
        prepared_counts[owner] += 1
        return original(cls, entries, primary)

    monkeypatch.setattr(ExpertParallelExecutor, "_prepare_batch", classmethod(prepare))
    with ExpertParallelExecutor(
        model, plan, token_capacity={"cpu:0": 4, "cpu:1": 4}
    ) as executor:
        original_forward = executor._expert_forward_batch

        def forward(prepared, event, stream):
            result = original_forward(prepared, event, stream)
            completed_counts[plan.device_for(prepared.names[0])] += 1
            return result

        monkeypatch.setattr(executor, "_expert_forward_batch", forward)
        values = [
            {"hidden": torch.randn(2, 4), "indices": torch.tensor([0, 1])}
            for _batch in range(12)
        ]
        actual = executor.forward_many([_graph(model)], values)
        assert len(actual) == 12
        for device, statistics in executor.dispatch_statistics.items():
            assert statistics["tokens"] == 12
            assert statistics["max_tokens"] <= 4
            assert statistics["dispatches"] == 3
            assert prepared_counts[device] == completed_counts[device] == 3


def test_worker_failure_waits_for_other_workers_and_can_unwind():
    model = _MoE(2)
    barrier = threading.Barrier(2)
    for expert in model.experts:
        expert.barrier = barrier
    model.experts[0].fail = True
    plan = ExpertParallelPlan.from_model(model, _targets(model), ("cpu:0", "cpu:1"))
    with ExpertParallelExecutor(model, plan) as executor:
        with pytest.raises(RuntimeError, match="expert failure"):
            executor.forward(
                [_graph(model)],
                {
                    "hidden": torch.randn(6, 4),
                    "indices": torch.tensor([0, 1, 0, 1, 0, 1]),
                },
            )
    assert all(
        not thread.is_alive()
        for worker in executor._workers.values()
        for thread in worker._threads
    )


def test_execution_schedule_is_cached_without_retaining_batch_tensors(monkeypatch):
    model = _MoE(2).eval()
    subgraphs = [_graph(model)]
    plan = ExpertParallelPlan.from_model(model, _targets(model), ("cpu:0", "cpu:1"))
    original = _ExecutionGraph.build.__func__
    builds = []

    def record_build(cls, graphs):
        builds.append(tuple(id(graph) for graph in graphs))
        return original(cls, graphs)

    monkeypatch.setattr(_ExecutionGraph, "build", classmethod(record_build))
    with ExpertParallelExecutor(model, plan) as executor, torch.no_grad():
        for _repeat in range(3):
            value = {
                "hidden": torch.randn(6, 4),
                "indices": torch.tensor([0, 1, 1, 0, 0, 1]),
            }
            expected = subgraphs[0].forward(model, **value)["result"]
            actual = executor.forward(subgraphs, value)
            assert torch.equal(actual["result"], expected)
            input_reference = weakref.ref(value["hidden"])
            output_reference = weakref.ref(actual["result"])
            del value, actual, expected
            gc.collect()
            assert input_reference() is None
            assert output_reference() is None
        assert len(builds) == 1


def _run_gptq(model, values, devices=None, *, coalesce=False):
    quantizer = GPTQModifier(
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
        ignore=["re:shared_expert\\..*"],
        actorder=None,
        block_size=4,
        batched_quantization=None,
    )
    subgraph = _graph(model)
    with create_session() as session, ExitStack() as calibration_stack:
        session.state.layerwise_calibration = True
        session.initialize(model=model, recipe=[quantizer], start=-1, copy_data=False)
        quantizer = session.lifecycle.recipe.modifiers[0]
        calibration_stack.enter_context(modifier_calibration_context([quantizer]))
        LifecycleCallbacks.calibration_start()
        modules = tuple(
            module for expert in model.experts for module in expert.modules()
        )
        quantizer.start_layerwise_calibration(model, list(modules))
        try:
            with DisableQuantization(model), torch.no_grad():
                if devices is None:
                    for value in values:
                        subgraph.forward(model, **value)
                else:
                    plan = ExpertParallelPlan.from_model(
                        model, _targets(model), devices
                    )
                    with ExpertParallelExecutor(model, plan) as executor:
                        if coalesce:
                            executor.forward_many([subgraph], values)
                        else:
                            for value in values:
                                executor.forward([subgraph], value)
            statistics = {
                quantizer._module_names[module]: (
                    hessian.clone(),
                    quantizer._num_samples[module].clone(),
                )
                for module, hessian in quantizer._hessians.items()
            }
            assert all(
                hessian.device == module.weight.device
                and quantizer._num_samples[module].device == module.weight.device
                for module, hessian in quantizer._hessians.items()
            )
            LifecycleCallbacks.sequential_epoch_end(list(modules))
            quantizer.end_layerwise_calibration(list(modules))
            assert not quantizer._hessians
            assert not quantizer._num_samples
            LifecycleCallbacks.calibration_end()
            session.finalize()
        finally:
            quantizer.remove_hooks()
    return statistics


@pytest.mark.parametrize("coalesce", [False, True])
def test_parallel_gptq_matches_local_hessians_counts_weights_and_qparams(coalesce):
    torch.manual_seed(3)
    reference = _MoE().eval()
    parallel = deepcopy(reference)
    values = [
        {"hidden": torch.randn(6, 4), "indices": torch.tensor([0, 2, 1, 2, 3, 0])}
        for _batch in range(2)
    ]
    expected = _run_gptq(reference, values)
    actual = _run_gptq(
        parallel, values, ("cpu:0", "cpu:1", "cpu:2", "cpu:3"),
        coalesce=coalesce,
    )
    assert expected.keys() == actual.keys()
    for name, (hessian, count) in expected.items():
        torch.testing.assert_close(actual[name][0], hessian, msg=name)
        assert torch.equal(actual[name][1], count), name
    expected_tensors = reference.state_dict()
    actual_tensors = parallel.state_dict()
    assert actual_tensors.keys() == expected_tensors.keys()
    for name in expected_tensors:
        torch.testing.assert_close(
            actual_tensors[name], expected_tensors[name], msg=name
        )


def test_parallel_manifest_roundtrip_records_topology_and_actual_groups():
    manifest = ParallelInfo.from_context(
        {"world_size": 1, "ep_size": 2},
        backend=ExpertParallelPlan.backend,
        devices=("cuda:1", "cuda:2"),
        expert_owners=(("experts.0", 0), ("experts.1", 1)),
        target_groups=((0, 1), (2,)),
        token_budget_policy="ep-tokens-v2",
        token_budget=8192,
    )
    assert ParallelInfo.from_dict(json.loads(json.dumps(asdict(manifest)))) == manifest


def test_recovery_rejects_device_order_and_group_changes(tmp_path):
    from safetensors.torch import save_file

    from llmcompressor.recipe import Recipe
    from llmcompressor.streaming.artifacts import ArtifactCompatibilityError
    from llmcompressor.streaming.materialization import CastWeightMaterializer
    from llmcompressor.streaming.pipeline import _initialize_run

    model = _MoE()
    checkpoint = tmp_path / "source"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    save_file(model.state_dict(), checkpoint / "model.safetensors")
    plan = ExpertParallelPlan.from_model(model, _targets(model), ("cpu:0", "cpu:1"))
    arguments = dict(
        checkpoint=checkpoint,
        artifact_dir=tmp_path / "artifacts",
        recipe=Recipe.create_instance(GPTQModifier(scheme="W4A16")),
        dataset_fingerprint="ep-recovery",
        targets=_targets(model),
        materializer=CastWeightMaterializer(),
        target_dtype=torch.float32,
        num_samples=2,
        max_seq_length=6,
        seed=42,
        pack_to_int8=True,
        expert_parallel=plan,
        target_groups=((0, 1, 2), (3, 4)),
    )
    fingerprint = _initialize_run(**arguments)
    assert _initialize_run(**arguments) == fingerprint
    changed = dict(arguments, target_groups=((0, 1), (2, 3, 4)))
    with pytest.raises(ArtifactCompatibilityError, match="parallel"):
        _initialize_run(**changed)
    with pytest.raises(ArtifactCompatibilityError, match="parallel"):
        _initialize_run(**dict(arguments, expert_parallel_token_budget=8192))
    changed = dict(
        arguments,
        expert_parallel=ExpertParallelPlan.from_model(
            model, _targets(model), ("cpu:1", "cpu:0")
        ),
    )
    with pytest.raises(ArtifactCompatibilityError, match="parallel"):
        _initialize_run(**changed)


def _cuda_devices(count):
    configured = os.environ.get("LLMCOMPRESSOR_TEST_EP_DEVICES")
    devices = (
        configured.split(",")
        if configured
        else [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    )
    if not torch.cuda.is_available() or len(devices) < count:
        pytest.skip("Requires explicitly available CUDA expert owner devices")
    return tuple(torch.device(device) for device in devices[:count])


@pytest.mark.parametrize("count", [2, 4])
@pytest.mark.parametrize("coalesce", [False, True])
def test_cuda_gptq_uses_local_statistics_and_matches_single_device(count, coalesce):
    devices = _cuda_devices(count)
    torch.manual_seed(3)
    reference = _MoE().eval().to(devices[0])
    parallel = deepcopy(reference)
    for expert_index, expert in enumerate(parallel.experts):
        expert.to(devices[expert_index % count])
    values = [
        {
            "hidden": torch.randn(6, 4, device=devices[0]),
            "indices": torch.tensor([0, 2, 1, 2, 3, 0], device=devices[0]),
        }
        for _batch in range(2)
    ]
    expected = _run_gptq(reference, values)
    actual = _run_gptq(parallel, values, devices, coalesce=coalesce)
    for name, (hessian, samples) in expected.items():
        torch.testing.assert_close(actual[name][0].cpu(), hessian.cpu())
        assert torch.equal(actual[name][1].cpu(), samples.cpu())
    expected_tensors = reference.state_dict()
    actual_tensors = parallel.state_dict()
    assert actual_tensors.keys() == expected_tensors.keys()
    for name in expected_tensors:
        torch.testing.assert_close(
            actual_tensors[name].cpu(),
            expected_tensors[name].cpu(),
            rtol=1e-6,
            atol=1e-7,
            msg=name,
        )


def test_cuda_weight_session_installs_only_the_planned_owner_and_unloads(tmp_path):
    devices = _cuda_devices(2)
    from safetensors.torch import save_file

    source_model = _MoE()
    (tmp_path / "config.json").write_text("{}")
    save_file(source_model.state_dict(), tmp_path / "model.safetensors")
    model = build_meta_model(_MoE)
    ownership = ExpertParallelPlan.from_model(model, _targets(model), devices)
    session = SubgraphWeightSession(model, SafetensorsWeightSource(tmp_path))
    load_plan = session.plan_modules(("experts.0", "experts.1", "shared_expert"))
    budget = HostMemoryBudget()
    required = session.estimate_preparation(load_plan, dtype=torch.float32)
    prepared = session.prepare(
        load_plan,
        dtype=torch.float32,
        reservation=budget.reserve("EP weights", required),
    )
    prepared.move_to(
        devices[0],
        target_devices={
            target.name: ownership.device_for(target.name)
            for target in load_plan.targets
        },
    )
    with session.installed(prepared):
        assert model.experts[0].w1.weight.device == devices[0]
        assert model.experts[1].w1.weight.device == devices[1]
        assert model.shared_expert.w1.weight.device == devices[0]
        assert model.experts[2].w1.weight.is_meta
    assert all(parameter.is_meta for parameter in model.parameters())
    assert not prepared.targets
