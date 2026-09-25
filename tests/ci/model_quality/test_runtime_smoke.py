import json
import sys
from pathlib import Path

import pytest

from ci.model_quality.config import (
    ConfigError,
    evaluation_fingerprint,
    load_model_config,
)
from ci.model_quality.executor import StageError, run_preflight, run_runtime_smoke


def _runtime_model(tmp_path: Path, *, command: list[str] | None) -> dict:
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    (source / "config.json").write_text("{}", encoding="utf-8")
    quantize_script = tmp_path / "quantize.py"
    quantize_script.write_text("", encoding="utf-8")
    runtime = {"enabled": True, "runtime_revision": "test-runtime"}
    if command is not None:
        runtime["command"] = command
    else:
        runtime["python"] = sys.executable
    return {
        "id": "runtime-model",
        "source": {"path": str(source)},
        "resources": {"gpu_count": 1, "estimated_gpu_hours": 1},
        "workflow": {
            "revision": "test-workflow",
            "quantize": [sys.executable, str(quantize_script)],
        },
        "validation": {"profile": "causal_lm"},
        "runtime_smoke": runtime,
        "evaluation": {},
        "upload": {"enabled": False},
    }


def _smoke_script(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "smoke.py"
    script.write_text(body, encoding="utf-8")
    return script


def test_runtime_smoke_command_resolves_placeholders_and_reads_result(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path / "runs"))
    script = _smoke_script(
        tmp_path,
        """
import json
import sys
from pathlib import Path

argv = sys.argv[1:]
output = Path(argv[argv.index("--output") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"status": "PASS", "outputs": [{"prompt": "p", "text": "t"}]}))
""",
    )
    model = _runtime_model(
        tmp_path,
        command=[
            sys.executable,
            str(script),
            "--model",
            "{output_dir}",
            "--output",
            "{reports_dir}/runtime-smoke-output.json",
        ],
    )

    result = run_runtime_smoke(model, "run-a")

    assert result["status"] == "PASS"
    assert result["result"]["outputs"][0]["text"] == "t"
    assert result["runtime_revision"] == "test-runtime"
    log = tmp_path / "runs/run-a/runtime-model/logs/runtime-smoke.log"
    assert log.is_file()


def test_runtime_smoke_command_requires_a_result_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path / "runs"))
    script = _smoke_script(tmp_path, "import sys\nsys.exit(0)\n")
    model = _runtime_model(tmp_path, command=[sys.executable, str(script)])

    with pytest.raises(StageError) as error:
        run_runtime_smoke(model, "run-a")

    assert error.value.reason_code == "RUNTIME_SMOKE_FAILED"


def test_runtime_smoke_command_rejects_non_passing_result(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path / "runs"))
    script = _smoke_script(
        tmp_path,
        """
import json
import sys
from pathlib import Path

argv = sys.argv[1:]
output = Path(argv[argv.index("--output") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"status": "PASS", "outputs": []}))
""",
    )
    model = _runtime_model(
        tmp_path,
        command=[
            sys.executable,
            str(script),
            "--output",
            "{reports_dir}/runtime-smoke-output.json",
        ],
    )

    with pytest.raises(StageError) as error:
        run_runtime_smoke(model, "run-a")

    assert error.value.reason_code == "RUNTIME_SMOKE_FAILED"


def test_runtime_smoke_command_failure_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path / "runs"))
    script = _smoke_script(tmp_path, "import sys\nsys.exit(3)\n")
    model = _runtime_model(tmp_path, command=[sys.executable, str(script)])

    with pytest.raises(StageError) as error:
        run_runtime_smoke(model, "run-a")

    assert error.value.reason_code == "RUNTIME_SMOKE_FAILED"
    assert "3" in str(error.value)


def test_runtime_smoke_command_skips_vllm_preflight(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_QUALITY_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setattr("torch.accelerator.is_available", lambda: True)
    monkeypatch.setattr("torch.accelerator.device_count", lambda: 1)
    model = _runtime_model(tmp_path, command=[sys.executable, "-c", "pass"])
    result = run_preflight(
        model,
        "run-a",
        run_mode="quantize",
        fingerprint="artifact-a",
        evaluation_fingerprint=evaluation_fingerprint(model, "artifact-a"),
    )

    assert result["status"] in {"PASS", "WARN"}
    assert not any("vllm" in failure for failure in result["failures"])


def test_runtime_smoke_command_must_be_argv(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "id": "bad-runtime",
                        "enabled": False,
                        "source": {"path": "/models/a"},
                        "resources": {"gpu_count": 1, "estimated_gpu_hours": 1},
                        "workflow": {"revision": "r", "quantize": ["python3", "q.py"]},
                        "runtime_smoke": {
                            "enabled": True,
                            "runtime_revision": "runtime",
                            "command": "python3 -m smoke",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_model_config(config)


def _stub_sglang_server():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep pytest output quiet
            return

        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            assert self.path == "/v1/completions"
            assert body["temperature"] == 0.0
            payload = json.dumps(
                {
                    "choices": [{"text": " Paris", "finish_reason": "length"}],
                    "usage": {"completion_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError:  # sandboxed runners cannot bind loopback sockets
        pytest.skip("loopback sockets are not permitted in this environment")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_xsgl_smoke_writes_generation_record(tmp_path):
    import subprocess

    server = _stub_sglang_server()
    try:
        output = tmp_path / "runtime-smoke-output.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "ci.model_quality.xsgl_smoke",
                "--model",
                str(tmp_path / "missing-model"),
                "--base-url",
                f"http://127.0.0.1:{server.server_address[1]}",
                "--served-model-name",
                "deepseek-v4-flash",
                "--prompts-json",
                '["The capital of France is"]',
                "--output",
                str(output),
                "--wait-ready-seconds",
                "30",
            ],
            text=True,
            capture_output=True,
            check=False,
            cwd=str(Path(__file__).resolve().parents[3]),
        )
    finally:
        server.shutdown()

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text())
    assert payload["status"] == "PASS"
    assert payload["runtime"] == "sglang"
    assert payload["outputs"] == [
        {
            "prompt": "The capital of France is",
            "text": " Paris",
            "token_ids": None,
            "token_ids_source": None,
            "completion_tokens": 2,
            "finish_reason": "length",
        }
    ]
    assert payload["tokenizer_error"] is not None
