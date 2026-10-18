import json
from pathlib import Path

import pytest
import torch

from tools import validate_dsv41_gpu_kernels as validation


def test_fp4_midpoints_choose_even_encodings_without_native_fp4():
    # Anchor 6 fixes the E8M0 scale at 1; expected ties are explicit E2M1 values.
    x = torch.tensor(
        [[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] * 4],
        dtype=torch.bfloat16,
    )
    actual, scale = validation.fp4_reference(x, 32, False)
    expected = torch.tensor([[0, 1, 1, 2, 2, 4, 4, 6] * 4], dtype=torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(scale, torch.ones_like(scale), atol=0, rtol=0)


def test_sparse_online_maximum_rescales_previous_value_accumulator():
    q = torch.tensor([[[[1.0]]]], dtype=torch.bfloat16)
    kv = torch.tensor([[[0.0], [2.0]]], dtype=torch.bfloat16)
    indices = torch.full((1, 1, 65), -1, dtype=torch.int32)
    indices[..., 0] = 0
    indices[..., -1] = 1
    actual = validation.sparse_reference(q, kv, torch.zeros(1), indices, 1.0)
    # First value is zero. Two exp(-2) terms in denominator: key 0 and sink.
    expected = torch.tensor([[[[2 / (1 + 2 * torch.exp(torch.tensor(-2.0)))]]]])
    torch.testing.assert_close(actual, expected.bfloat16(), atol=0, rtol=0)


def test_project_sparse_matches_online_reference_across_block_boundary(tmp_path):
    args = validation.parse_args(
        ["--self-test", "--report", str(tmp_path / "report.json")]
    )
    modules, _ = validation.load_project(Path(__file__).resolve().parents[2])

    class Official:
        sparse_attn = staticmethod(validation.sparse_reference)

    runner = validation.Runner(args, {"results": []}, Official(), modules)
    torch.manual_seed(7243)
    result = runner.sparse(4, 64, 129)
    assert result["checks"]["project_vs_online_reference"]["exact"]
    assert result["checks"]["project_all_masked_zero"]["exact"]
    assert result["checks"]["project_full_vs_last_query"]["exact"]


def test_default_sparse_shapes_cover_production_and_small_shape_is_explicit():
    cases = validation.sparse_cases(quick=False)
    assert (64, 512, 640) in cases
    assert (64, 512, 512) in cases
    assert all(d >= 64 for _, d, _ in cases)
    probe = validation.sparse_cases(quick=False, include_small_head_dim=True)
    assert (16, 32, 1) in probe
    assert all(case in probe for case in cases)


def test_single_row_quantization_input_is_not_trivially_zero():
    assert validation.quant_input(1, 32, 32, "cpu").count_nonzero() > 0


def test_bf16_step_metric_handles_signed_zero_and_one_step():
    actual = torch.tensor([-0.0, 1.0, -1.0], dtype=torch.bfloat16)
    expected = torch.tensor([0.0, 1.0078125, -1.0078125], dtype=torch.bfloat16)
    assert validation.bf16_steps(actual, expected) == 1
    assert not validation.compare(actual, expected, exact=True)["passed"]


def test_comparison_rejects_nonfinite_and_wrong_dtype():
    assert not validation.compare(torch.tensor([float("nan")]), torch.ones(1))["passed"]
    assert not validation.compare(torch.ones(1), torch.ones(1).double())["passed"]


def test_project_reference_cannot_dispatch_back_to_official(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_V4_KERNEL_BACKEND", "reference")
    monkeypatch.setenv("DEEPSEEK_V4_REFERENCE_KERNEL_PATH", "/nonexistent/kernel.py")
    modules, info = validation.load_project(Path(__file__).resolve().parents[2])
    assert info["available"]
    assert modules["kernels"]._load_reference_kernel_module() is None


def test_cpu_self_test_report_and_exit_code(tmp_path):
    path = tmp_path / "report.json"
    assert validation.main(["--self-test", "--report", str(path)]) == 0
    report = json.loads(path.read_text())
    assert report["status"] == "self_test_passed"
    assert report["native_fp4"]["status"] == "skipped"
    assert report["native_fp4"]["official_fp4_kernel_called"] is False
    assert report["results"][0]["native_gpu_operators_executed"] is False
    assert report["project"]["available"]


def test_failed_check_is_a_nonzero_exit_code(monkeypatch, tmp_path):
    monkeypatch.setattr(
        validation,
        "cpu_self_test",
        lambda _: {"checks": {"deliberate_mismatch": {"passed": False}}},
    )
    path = tmp_path / "report.json"
    assert validation.main(["--self-test", "--no-project", "--report", str(path)]) == 1
    report = json.loads(path.read_text())
    assert report["summary"]["failed"] == 1
    assert report["status"] == "failed"


def test_initialization_failure_is_recorded(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.get_device_module("cuda"), "is_available", lambda: False)
    path = tmp_path / "report.json"
    code = validation.main(
        ["--official-kernel", "/nonexistent/kernel.py", "--report", str(path)]
    )
    assert code == 2
    report = json.loads(path.read_text())
    assert report["status"] == "incomplete"
    assert "CUDA is required" in report["fatal_error"]


def test_suite_never_calls_native_fp4_and_case_seed_is_stable(tmp_path):
    args = validation.parse_args(
        [
            "--self-test",
            "--quick",
            "--operators",
            "hc",
            "fp4_reference",
            "--report",
            str(tmp_path / "report.json"),
        ]
    )
    modules, _ = validation.load_project(Path(__file__).resolve().parents[2])

    class Official:
        hc_split_sinkhorn = staticmethod(validation.hc_reference)

        def __getattr__(self, name):
            pytest.fail(f"Unexpected official kernel call: {name}")

    report = {"results": []}
    runner = validation.Runner(args, report, Official(), modules)
    runner.suite()
    assert len(report["results"]) == 6
    assert all(row["status"] == "passed" for row in report["results"])
    first = report["results"][0]
    runner.case("hc", first["case"], lambda: runner.hc(1, 1, 2))
    assert report["results"][-1]["seed"] == first["seed"]
    assert report["results"][-1]["checks"] == first["checks"]


def test_runner_preserves_traceback_for_kernel_compile_failure(tmp_path):
    args = validation.parse_args(
        ["--self-test", "--report", str(tmp_path / "report.json")]
    )
    runner = validation.Runner(args, {"results": []})

    def compile_failure():
        raise RuntimeError("simulated compiler failure")

    runner.case("fp8_gemm", "compiler_error", compile_failure)
    result = json.loads(args.report.read_text())["results"][0]
    assert result["status"] == "error"
    assert "simulated compiler failure" in result["traceback"]


def test_nonfinite_kernel_result_still_produces_valid_failure_report(tmp_path):
    args = validation.parse_args(
        ["--self-test", "--report", str(tmp_path / "report.json")]
    )
    runner = validation.Runner(args, {"results": []})
    runner.case(
        "sparse",
        "nan_output",
        lambda: {
            "per_head_max_abs": [float("nan")],
            "checks": {
                "nonfinite": validation.compare(
                    torch.tensor([float("nan")]), torch.ones(1)
                )
            },
        },
    )
    result = json.loads(args.report.read_text())["results"][0]
    assert result["status"] == "failed"
    assert result["per_head_max_abs"] == ["nan"]
    assert not result["checks"]["nonfinite"]["finite"]
