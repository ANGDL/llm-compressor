import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import compare_dsv41_kernel_level as comparison
from tools import dsv41_sparse_trace as tracing


def test_kernel_level_cli_excludes_native_fp4():
    args = comparison.parse_args(["--self-test"])
    assert args.self_test
    assert "fp4" not in comparison.OPERATORS


def test_kernel_level_cpu_self_test(tmp_path):
    report = tmp_path / "kernel-level.json"
    assert comparison.main(["--self-test", "--report", str(report)]) == 0
    data = json.loads(report.read_text())
    assert data["status"] == "passed"
    assert data["native_fp4"]["status"] == "skipped"
    assert data["results"][0]["official_kernel_called"] is False
    assert data["results"][0]["native_fp4_called"] is False


def test_kernel_level_sparse_cases_include_production_shape():
    cases = comparison.kernel_sparse_cases(quick=False)
    assert (64, 512, 640) in cases
    assert (4, 512, 65) in cases


def test_trace_options_require_cuda_sparse_mode():
    for options in (
        ["--self-test", "--trace-sparse"],
        ["--self-test", "--save-sparse-tensors"],
        ["--official-kernel", "kernel.py", "--operators", "hc", "--trace-sparse"],
    ):
        with pytest.raises(SystemExit):
            comparison.parse_args(options)


def test_trace_compile_error_keeps_original_failed_checks(monkeypatch, tmp_path):
    modules, _ = comparison.load_project(Path(__file__).resolve().parents[2])

    class Official:
        @staticmethod
        def sparse_attn(*args):
            output = comparison.sparse_reference(*args)
            output[0, 0, 0, 0] += 1
            return output

    progress = []
    runner = comparison.KernelComparator(
        Official(),
        modules,
        "cpu",
        20261018,
        True,
        trace_sparse=True,
        trace_dir=tmp_path,
        progress=lambda pending: progress.append(deepcopy(pending)),
    )

    def fail(*_args):
        raise RuntimeError("simulated diagnostic JIT failure")

    monkeypatch.setattr(runner, "sparse_diagnostics", fail)
    runner.case("sparse", "h4_d64_k65", lambda: runner.sparse(4, 64, 65))
    row = runner.results[0]
    assert row["status"] == "failed"
    assert not row["checks"]["official_vs_project"]["passed"]
    assert not row["checks"]["sparse_trace_completed"]["passed"]
    assert "simulated diagnostic JIT failure" in row["sparse_trace"]["traceback"]
    assert progress[0]["status"] == "tracing"
    assert not progress[0]["checks"]["official_vs_project"]["passed"]
    assert progress[-1] is None


@pytest.mark.parametrize("perturbation", ["output", "snapshot_nan"])
def test_untrusted_instrumentation_cannot_attribute_first_divergence(
    monkeypatch, tmp_path, perturbation
):
    torch.manual_seed(1)
    q = torch.randn(1, 2, 4, 64).bfloat16()
    kv = torch.randn(1, 137, 64).bfloat16()
    sink = torch.zeros(4)
    indices = torch.randint(137, (1, 2, 65), dtype=torch.int32)
    args = q, kv, sink, indices, 64**-0.5
    expected = tracing.reference_trace(*args)
    actual = deepcopy(expected)
    if perturbation == "output":
        actual["final"]["output"][0, 0, 0, 0] += 1
    else:
        actual["blocks"][0]["qk"][0, 0, 0, 0] = torch.nan
    source = Path(__file__).parent / "fixtures/dsv41_sparse_kernel.py.txt"
    runner = comparison.KernelComparator(None, {}, "cpu", 1, True, trace_dir=tmp_path)
    runner.trace_module = SimpleNamespace(__file__=str(source))
    monkeypatch.setattr(comparison, "run_official_trace", lambda *_: (actual, object()))
    output = expected["final"]["output"]
    checks = {"official_vs_project": comparison.compare(output, output)}
    result = runner.sparse_diagnostics(args, output, output, output, checks)
    assert result["status"] == "untrusted"
    assert not result["attribution_valid"]
    assert result["first_exact_divergence"] is None
    assert not all(check["passed"] for check in checks.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_official_gpu_trace_preserves_unmodified_output(tmp_path):
    source = os.environ.get("DSV41_OFFICIAL_KERNEL")
    if not source:
        pytest.skip("Set DSV41_OFFICIAL_KERNEL to the official inference/kernel.py")
    report = tmp_path / "gpu-trace.json"
    code = comparison.main(
        [
            "--official-kernel",
            source,
            "--operators",
            "sparse",
            "--quick",
            "--trace-sparse",
            "--save-sparse-tensors",
            "--report",
            str(report),
        ]
    )
    data = json.loads(report.read_text())
    assert code in (0, 1) and "fatal_error" not in data
    row = data["results"][0]
    assert row["checks"]["instrumented_vs_official"]["exact"]
    assert row["sparse_trace"]["attribution_valid"]
    assert Path(row["sparse_trace"]["tensor_artifact"]).is_file()
