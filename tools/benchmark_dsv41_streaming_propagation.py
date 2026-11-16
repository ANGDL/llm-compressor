"""Compare MoE routing and activation transfer without loading a full checkpoint."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import statistics
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.fx import GraphModule, Tracer

from llmcompressor.modeling.deepseekv41.model import DeepseekV41MoE, Expert
from llmcompressor.pipelines.sequential.helpers import Subgraph, trace_consumed_names
from llmcompressor.streaming.activations import InMemoryBoundaryActivationStore
from llmcompressor.streaming.parallel.expert import (
    ExpertParallelExecutor,
    ExpertParallelPlan,
)
from llmcompressor.streaming.pipeline import _target_group_input_names
from llmcompressor.streaming.tracing import _split_target_subgraph


def reference_routing(self, hidden, shape, weights, indices):
    output = torch.zeros_like(hidden, dtype=torch.float32)
    for expert_id, expert in enumerate(self.experts):
        rows, slots = torch.where(indices == expert_id)
        route_weights = weights[rows, slots].unsqueeze(-1)
        if self.calibrate_all_experts:
            all_weights = hidden.new_ones(
                (hidden.shape[0], 1), dtype=weights.dtype
            ).index_copy(0, rows, route_weights)
            values = expert(hidden, all_weights).index_select(0, rows)
        else:
            values = expert(hidden.index_select(0, rows), route_weights)
        output = output.index_add_(0, rows, values.float())
    return (output + self.shared_experts(hidden)).to(hidden.dtype).reshape(shape)


class ExpertTracer(Tracer):
    def is_leaf_module(self, module, name):
        return isinstance(module, Expert) or super().is_leaf_module(module, name)


def synchronize(devices):
    for device in devices:
        torch.cuda.synchronize(device)


def benchmark(model, hidden, carry, devices, args, label, selective):
    traced = GraphModule(
        model, ExpertTracer().trace(model, concrete_args={"image_mask": None})
    )
    output = next(node for node in traced.graph.nodes if node.op == "output")
    output.args = ({"result": output.args[0]},)
    targets = tuple(
        str(node.target) for node in traced.graph.nodes
        if node.op == "call_module" and isinstance(
            traced.get_submodule(str(node.target)), Expert
        )
    )
    all_slices = _split_target_subgraph(
        traced,
        Subgraph(
            traced.graph,
            input_names={"x", "image_mask_1"},
            consumed_names=set(),
        ),
        targets,
    )
    trace_consumed_names(all_slices)
    slices = all_slices[args.group_start:args.group_start + args.group_size]
    accumulator = [
        node.name for subgraph in slices for node in subgraph.graph.nodes
        if node.op == "call_method" and node.target == "index_add_"
    ][-1]
    placement = ExpertParallelPlan.from_model(model, targets, devices)
    input_names = _target_group_input_names(slices)
    consumed_names = set().union(*(subgraph.consumed_names for subgraph in slices))
    timings = []
    profiler = cProfile.Profile() if args.profile else None
    result = None
    with ExpertParallelExecutor(traced, placement) as executor, torch.no_grad():
        initial = {"x": hidden, "image_mask_1": None, "carry": carry}
        if args.group_start:
            initial = executor.forward(all_slices[:args.group_start], initial)
        for iteration in range(args.warmup + args.iterations):
            store = InMemoryBoundaryActivationStore(deduplicate_tensors=True)
            store.put(0, 0, initial)
            synchronize(devices)
            if profiler is not None and iteration == args.warmup:
                profiler.enable()
            started = time.perf_counter()
            value = (
                store.get_for_execution(0, 0, input_names, device=devices[0])
                if selective
                else store.get(0, 0, device=devices[0])
            )
            synchronize(devices)
            loaded = time.perf_counter()
            propagated = executor.forward(slices, value)
            synchronize(devices)
            forwarded = time.perf_counter()
            store.commit_propagated_batch(
                0, 1, 0, propagated,
                consumed_names=consumed_names if selective else None,
            )
            committed = time.perf_counter()
            if iteration >= args.warmup:
                timings.append({
                    "get_seconds": loaded - started,
                    "forward_seconds": forwarded - loaded,
                    "commit_seconds": committed - forwarded,
                    "total_seconds": committed - started,
                })
            result = propagated[accumulator].detach().cpu()
            torch.testing.assert_close(
                store.get(1, 0)["carry"], carry, rtol=0, atol=0
            )
            store.delete(1)
            del value, propagated
    if profiler is not None:
        profiler.disable()
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(25)
    summary = {
        "label": label,
        "mean": {
            name: statistics.mean(sample[name] for sample in timings)
            for name in timings[0]
        },
        "samples": timings,
        "graph_operations": dict(Counter(
            str(node.target) for subgraph in slices
            for node in subgraph.graph.nodes if node.op == "call_function"
        )),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--group-start", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--intermediate-size", type=int, default=2304)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--carry-mib", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.group_start < 0 or args.group_size < 1
        or args.group_start + args.group_size >= args.experts
    ):
        parser.error("group must be nonempty and end before the shared expert")
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup nonnegative")
    devices = tuple(map(torch.device, args.devices))
    config = SimpleNamespace(
        dim=args.hidden_size, moe_inter_dim=args.intermediate_size,
        swiglu_limit=10.0, n_routed_experts=args.experts,
        num_experts_per_tok=args.top_k, gate_temp=1.0,
        score_func="softmax", norm_topk_prob=True, route_scale=1.0,
    )
    with torch.device("meta"):
        model = DeepseekV41MoE(config).to(torch.bfloat16).eval()
    model.gate.to_empty(device=devices[0])
    for index, expert in enumerate(
        model.experts[:args.group_start + args.group_size]
    ):
        expert.to_empty(device=devices[index % len(devices)])
    with torch.no_grad():
        for parameter in model.parameters():
            if not parameter.is_meta:
                parameter.normal_(0, 0.01)
    hidden = torch.randn(
        1, args.tokens, args.hidden_size, dtype=torch.bfloat16, device=devices[0]
    )
    carry = torch.zeros(args.carry_mib * 1024 * 1024 // 2, dtype=torch.bfloat16)
    original = DeepseekV41MoE._forward_traceable
    summaries = []
    expected = None
    try:
        for routing, implementation in (
            ("reference-where", reference_routing), ("current", original)
        ):
            DeepseekV41MoE._forward_traceable = implementation
            for selective in (False, True):
                label = f"{routing}/{'selective' if selective else 'full-boundary'}"
                summary, result = benchmark(
                    model, hidden, carry, devices, args, label, selective
                )
                summaries.append(summary)
                if expected is None:
                    expected = result
                else:
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    finally:
        DeepseekV41MoE._forward_traceable = original
    report = {"config": vars(args) | {"output": str(args.output)}, "results": summaries}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
