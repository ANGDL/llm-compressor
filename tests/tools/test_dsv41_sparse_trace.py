import ast
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from tools import dsv41_sparse_trace as tracing
from tools import validate_dsv41_gpu_kernels as validation


@pytest.fixture
def official_source():
    return (Path(__file__).parent / "fixtures/dsv41_sparse_kernel.py.txt").read_text()


def test_instrumentation_preserves_every_original_statement(official_source):
    class StripSnapshots(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            node.args.args = [
                arg for arg in node.args.args if not arg.arg.startswith("trace_")
            ]
            return node

        def visit_Assign(self, node):
            if any(
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id.startswith("trace_")
                for target in node.targets
            ):
                return None
            return node

        def visit_For(self, node):
            self.generic_visit(node)
            return node if node.body else None

    generated = tracing.instrument_official_source(official_source)
    restored = StripSnapshots().visit(ast.parse(generated))
    assert ast.dump(restored) == ast.dump(ast.parse(official_source))
    assert "T.Pipelined(num_blocks, num_stages=num_stages)" in generated
    assert "threads = 256" in generated
    assert generated.count("trace_scores[by, bx, t,") == 3


@pytest.mark.parametrize(
    "old,new",
    [
        ("block = 64", "block = 32"),
        ("acc_s[i, j] *= scale", "acc_s[i, j] *= scale * 2"),
        ("T.copy(acc_s, acc_s_cast)", "T.copy(acc_s, other_weights)"),
    ],
)
def test_changed_official_contract_is_rejected(official_source, old, new):
    with pytest.raises(ValueError):
        tracing.instrument_official_source(official_source.replace(old, new))


def test_source_generation_never_overwrites_official_file(tmp_path, official_source):
    original = tmp_path / "kernel.py"
    original.write_text(official_source)
    with pytest.raises(ValueError, match="must not overwrite"):
        tracing.write_instrumented_source(original, original)
    assert original.read_text() == official_source
    generated = tmp_path / "debug_kernel.py"
    tracing.write_instrumented_source(original, generated)
    assert "trace_scores" in generated.read_text()
    assert original.read_text() == official_source


def make_inputs(k=129):
    torch.manual_seed(924)
    q = torch.randn(2, 7, 4, 64).bfloat16()
    kv = torch.randn(2, 137, 64).bfloat16()
    sink = torch.randn(4)
    indices = torch.randint(137, (2, 7, k), dtype=torch.int32)
    indices[:, 1, ::2] = -1
    indices[:, 2] = -1
    indices[:, 3] = 136
    indices[:, 4] = -1
    indices[:, 4, -1] = 136
    if k > 64:
        indices[:, 6, :64] = -1
    return q, kv, sink, indices, 64**-0.5


@pytest.mark.parametrize("k", [1, 32, 64, 65, 128, 129])
def test_reference_trace_matches_independent_reference_and_project(k):
    inputs = make_inputs(k)
    trace = tracing.reference_trace(*inputs)
    expected = validation.sparse_reference(*inputs)
    modules, _ = validation.load_project(Path(__file__).resolve().parents[2])
    project = modules["kernels"].sparse_attention(*inputs)
    assert torch.equal(trace["final"]["output"], expected)
    assert torch.equal(trace["final"]["output"], project)
    assert len(trace["blocks"]) == (k + 63) // 64
    assert torch.isneginf(trace["blocks"][0]["qk"][:, 2]).all()
    assert not trace["final"]["output"][:, 2].count_nonzero()
    assert torch.isposinf(trace["final"]["denominator_with_sink"][:, 2]).all()
    if k > 64:
        assert not trace["blocks"][0]["denominator"][:, 6].count_nonzero()
        assert torch.isfinite(trace["blocks"][1]["running_max"][:, 6]).all()


def test_stage_comparison_matches_masks_but_rejects_nan_and_changed_infinity():
    a = torch.tensor([float("-inf"), 0.5, float("inf")])
    check = tracing.compare_stage(a, a)
    assert check["passed"] and check["exact"]
    assert check["matching_nonfinite_elements"] == 2
    assert check["nonfinite_mismatches"] == 0
    for bad in (torch.tensor([float("nan"), 0.5, float("inf")]), -a):
        check = tracing.compare_stage(bad, a)
        assert not check["passed"] and not check["exact"]
        assert check["nonfinite_mismatches"] > 0
        assert check["first_nonfinite_mismatch"]["coordinate"] == [0]


def test_first_exact_strict_and_bf16_divergences_are_distinct():
    expected = tracing.reference_trace(*make_inputs())
    actual = deepcopy(expected)
    point = actual["blocks"][0]["qk"][0, 0, 0, 0]
    point.copy_(torch.nextafter(point, torch.full_like(point, torch.inf)))
    actual["blocks"][1]["bf16_weights"][0, 0, 0, 0] = 0.5
    expected["blocks"][1]["bf16_weights"][0, 0, 0, 0] = 1.0
    check = validation.compare(actual["final"]["output"], expected["final"]["output"])
    summary = tracing.summarize_trace(actual, expected, check, attribution_valid=True)
    assert summary["first_exact_divergence"]["stage"] == "qk"
    assert summary["first_exact_divergence"]["block"] == 0
    assert summary["first_strict_divergence"]["stage"] == "bf16_weights"
    assert summary["first_strict_divergence"]["block"] == 1
    assert summary["first_bf16_weight_divergence"]["block"] == 1
    invalid = tracing.summarize_trace(actual, expected, check, attribution_valid=False)
    assert invalid["status"] == "untrusted"
    assert invalid["first_exact_divergence"] is None
    assert invalid["first_strict_divergence"] is None
    assert invalid["stages"]


def test_largest_bf16_steps_and_largest_abs_points_have_independent_trajectories():
    expected = tracing.reference_trace(*make_inputs(65))
    actual = deepcopy(expected)
    expected["final"]["output"][0, 0, 0, :2] = torch.tensor(
        [2.015625, 4e-6], dtype=torch.bfloat16
    )
    actual["final"]["output"][0, 0, 0, :2] = torch.tensor(
        [2.0, 1e-6], dtype=torch.bfloat16
    )
    check = validation.compare(actual["final"]["output"], expected["final"]["output"])
    assert check["largest_difference"]["coordinate"] == [0, 0, 0, 0]
    assert check["largest_bf16_step_difference"]["coordinate"] == [0, 0, 0, 1]
    assert check["max_bf16_steps"] > 1
    summary = tracing.summarize_trace(actual, expected, check, attribution_valid=True)
    assert len(summary["output_points"]) == 2
    assert len(summary["output_points"][1]["blocks"]) == 2
    assert "normalized" in summary["output_points"][1]["final"]


def test_official_head_padding_and_trace_buffer_layout():
    inputs = make_inputs(65)
    reference = tracing.reference_trace(*inputs)
    captured = []

    class FakeOfficial:
        @staticmethod
        def sparse_attn_kernel(heads, dim, scale):
            assert (heads, dim) == (16, 64)

            def kernel(q, kv, output, sink, indices, *buffers):
                assert q.shape[2] == 16 and sink.shape == (16,)
                assert buffers[0].shape == (2, 7, 2, 3, 16, 64)
                assert torch.isnan(buffers[0]).all()
                padded = tracing.reference_trace(q, kv, sink, indices, scale)
                captured.append(padded)
                scores, rows, weights, acc, denominator, normalized = buffers
                for block, trace in enumerate(padded["blocks"]):
                    for slot, name in enumerate(("qk", "scaled_scores", "exp")):
                        scores[:, :, block, slot] = trace[name]
                    for slot, name in enumerate(
                        ("running_max", "rescale", "block_sum", "denominator")
                    ):
                        rows[:, :, block, slot] = trace[name]
                    weights[:, :, block] = trace["bf16_weights"]
                    for slot, name in enumerate(
                        ("rescaled_accumulator", "accumulator")
                    ):
                        acc[:, :, block, slot] = trace[name]
                denominator.copy_(padded["final"]["denominator_with_sink"])
                normalized.copy_(padded["final"]["normalized"])
                output.copy_(padded["final"]["output"])

            return kernel

    traced, _ = tracing.run_official_trace(FakeOfficial(), *inputs)
    assert traced["final"]["output"].shape == inputs[0].shape
    assert torch.equal(traced["final"]["output"], reference["final"]["output"])
    # Padding changes the GEMM shape and can change FP32 intermediates even on
    # CPU. Verify snapshot indexing against the same padded execution.
    padded_reference = {
        "blocks": [
            {name: value[:, :, :4] for name, value in block.items()}
            for block in captured[0]["blocks"]
        ],
        "final": {
            name: value[:, :, :4] for name, value in captured[0]["final"].items()
        },
    }
    summary = tracing.summarize_trace(
        traced,
        padded_reference,
        validation.compare(traced["final"]["output"], reference["final"]["output"]),
        attribution_valid=True,
    )
    assert summary["first_exact_divergence"] is None
