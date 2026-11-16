from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("flock") is None, reason="Requires the remote Linux flock utility"
)

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/model-quality-web-remote"
SERVER = """
import argparse
import json
import os
import signal
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--host")
parser.add_argument("--port", type=int)
args, _extra = parser.parse_known_args()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if os.environ.get("TEST_HEALTHY", "1") == "1" else 503)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def log_message(self, *_args):
        pass

def terminate(_signal, _frame):
    time.sleep(float(os.environ.get("TEST_SHUTDOWN_DELAY", "0")))
    raise SystemExit(0)

signal.signal(signal.SIGTERM, terminate)
time.sleep(float(os.environ.get("TEST_START_DELAY", "0")))
with HTTPServer((args.host, args.port), Handler) as server:
    with Path(os.environ["TEST_START_LOG"]).open("a") as stream:
        stream.write(json.dumps({"pid": os.getpid()}) + "\\n")
    server.serve_forever()
"""


@pytest.fixture
def manager(tmp_path):
    package = tmp_path / "ci/model_quality/web"
    package.mkdir(parents=True)
    for directory in (tmp_path / "ci", package.parent, package):
        (directory / "__init__.py").touch()
    (package / "__main__.py").write_text(SERVER)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    config = tmp_path / "remote.env"
    config.write_text(
        "\n".join(
            f"{key}={shlex.quote(str(value))}"
            for key, value in {
                "MQ_SSH_HOST": "test",
                "MQ_CONTAINER": "test",
                "MQ_REPO": tmp_path,
                "MQ_PYTHON": sys.executable,
                "MQ_LOCAL_PORT": port,
                "MQ_REMOTE_PORT": port,
                "MQ_RUNS_ROOT": tmp_path / "runs",
                "MQ_RUNTIME_DIR": runtime,
                "MQ_CONFIG": "unused.yaml",
            }.items()
        )
        + "\n"
    )
    environment = {
        **os.environ,
        "MQ_WEB_CONFIG": str(config),
        "PYTHONPATH": str(tmp_path),
        "TEST_START_LOG": str(tmp_path / "starts.jsonl"),
    }
    harness = (
        f"source {shlex.quote(str(SCRIPT))} --help >/dev/null\n"
        "load_config\n"
        "container_ok() { return 0; }\n"
        'docker() { shift 2; "$@"; }\n'
        'remote() { eval "$1"; }\n'
    )

    def launch(action, **overrides):
        return subprocess.Popen(
            ["bash", "-c", harness + action],
            env={**environment, **overrides},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def run(action, **overrides):
        process = launch(action, **overrides)
        output, error = process.communicate(timeout=20)
        assert process.returncode == 0, (output, error)
        return output

    result = SimpleNamespace(
        root=tmp_path, runtime=runtime, port=port, env=environment,
        launch=launch, run=run,
    )
    try:
        yield result
    finally:
        run("stop_service")


def _wait_for_start(manager, process):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        assert process.poll() is None
        if (manager.root / "starts.jsonl").exists():
            return
        time.sleep(0.02)
    pytest.fail("test server did not start")


def test_start_waits_for_unhealthy_listener_to_release_port(manager):
    previous = subprocess.Popen(
        [
            sys.executable, "-m", "ci.model_quality.web",
            "--host", "127.0.0.1", "--port", str(manager.port),
        ],
        cwd=manager.root,
        env={
            **manager.env,
            "TEST_HEALTHY": "0",
            "TEST_SHUTDOWN_DELAY": "0.7",
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_start(manager, previous)
        (manager.runtime / "server.pid").write_text(str(previous.pid))
        manager.run("start_service")
        assert previous.wait(timeout=5) == 0
        starts = (manager.root / "starts.jsonl").read_text().splitlines()
        assert len(starts) == 2
        assert "Address already in use" not in (
            manager.runtime / "server.log"
        ).read_text()
    finally:
        if previous.poll() is None:
            previous.kill()
            previous.wait(timeout=5)


def test_concurrent_start_creates_one_listener_and_releases_lock(manager):
    processes = [
        manager.launch("start_service", TEST_START_DELAY="0.6")
        for _attempt in range(2)
    ]
    for process in processes:
        output, error = process.communicate(timeout=20)
        assert process.returncode == 0, (output, error)
    starts = (manager.root / "starts.jsonl").read_text().splitlines()
    assert len(starts) == 1
    manager.run("stop_service")
    manager.run("start_service")


def test_stale_pid_file_never_terminates_unrelated_process(manager):
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"]
    )
    try:
        pid_file = manager.runtime / "server.pid"
        pid_file.write_text(str(unrelated.pid))
        manager.run("start_service")
        assert unrelated.poll() is None
        started = json.loads(
            (manager.root / "starts.jsonl").read_text().splitlines()[0]
        )
        assert int(pid_file.read_text()) == started["pid"]
        pid_file.write_text(str(unrelated.pid))
        manager.run("stop_service")
        assert unrelated.poll() is None
        assert not pid_file.exists()
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)
