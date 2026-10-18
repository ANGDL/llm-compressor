import json

from tools import compare_dsv41_kernel_level as comparison


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
