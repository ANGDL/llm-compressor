from __future__ import annotations

import io
import json
from pathlib import Path

from ci.model_quality.state import atomic_write_json
from ci.model_quality.web import RunStore, create_app


def _make_run(root: Path) -> None:
    run = root / "20260926T000000Z_local"
    model = run / "demo"
    atomic_write_json(
        run / "execution-plan.json",
        {
            "schema_version": 1,
            "run_id": run.name,
            "attempt_id": "attempt-1",
            "git_sha": "abc123",
            "run_mode": "quantize",
            "budget": {
                "max_gpu_hours": 8,
                "selected_gpu_hours": 2,
                "remaining_gpu_hours": 6,
            },
        },
    )
    atomic_write_json(
        model / "state" / "preflight.json",
        {
            "status": "PASS",
            "artifact_fingerprint": "artifact-1",
            "attempt_id": "attempt-1",
            "recorded_at": "2026-09-26T00:00:00+00:00",
        },
    )
    atomic_write_json(
        model / "artifact-manifest.json",
        {"content_fingerprint": "content-1", "files": []},
    )
    atomic_write_json(run / "events" / "1.json", {"sequence": 1, "status": "RUNNING"})
    atomic_write_json(run / "events" / "2.json", {"sequence": 2, "status": "PASS"})
    atomic_write_json(
        model / "current-attempt.json",
        {
            "attempt_id": "attempt-1",
            "artifact_fingerprint": "artifact-1",
            "evaluation_fingerprint": "evaluation-1",
        },
    )
    atomic_write_json(
        model / "attempts" / "attempt-1" / "state" / "preflight.json",
        {
            "status": "PASS",
            "attempt_id": "attempt-1",
            "recorded_at": "2026-09-26T00:00:00+00:00",
        },
    )
    log = model / "logs" / "attempt-1" / "quantization.log"
    log.parent.mkdir(parents=True)
    log.write_text("first\nsecond\n", encoding="utf-8")


def test_store_lists_runs_and_model(tmp_path: Path) -> None:
    _make_run(tmp_path)
    (tmp_path / "20260926T000000Z_local" / "attempt-plans").mkdir()
    store = RunStore(tmp_path)
    result = store.list_runs()
    assert result["total"] == 1
    assert result["items"][0]["model_count"] == 1
    assert result["items"][0]["run_id"] == "20260926T000000Z_local"
    model = store.get_model("20260926T000000Z_local", "demo")
    assert model["artifact_fingerprint"] == "artifact-1"
    assert model["evaluation_fingerprint"] == "evaluation-1"
    assert model["attempts"][0]["current"] is True

    logs = store.get_logs(
        "20260926T000000Z_local",
        "demo",
        stage="quantize",
        attempt_id="attempt-1",
    )
    assert [item["text"] for item in logs["items"]] == ["first", "second"]


def test_store_tails_logs(tmp_path: Path) -> None:
    _make_run(tmp_path)
    log = (
        tmp_path
        / "20260926T000000Z_local"
        / "demo"
        / "logs"
        / "attempt-1"
        / "quantization.log"
    )
    log.write_text("\n".join(f"line{n}" for n in range(1, 51)) + "\n", encoding="utf-8")
    store = RunStore(tmp_path)
    tailed = store.get_logs(
        "20260926T000000Z_local",
        "demo",
        stage="quantize",
        attempt_id="attempt-1",
        limit=5,
        tail=True,
    )
    assert tailed["total"] == 50
    assert tailed["offset"] == 45
    assert [item["text"] for item in tailed["items"]] == [
        "line46",
        "line47",
        "line48",
        "line49",
        "line50",
    ]



def test_store_truncates_oversized_log_lines(tmp_path: Path) -> None:
    _make_run(tmp_path)
    log = (
        tmp_path
        / "20260926T000000Z_local"
        / "demo"
        / "logs"
        / "attempt-1"
        / "quantization.log"
    )
    log.write_text("x" * 2_000_000 + "\nsmall line\n", encoding="utf-8")
    store = RunStore(tmp_path)
    result = store.get_logs(
        "20260926T000000Z_local",
        "demo",
        stage="quantize",
        attempt_id="attempt-1",
    )
    huge, small = result["items"][0], result["items"][1]
    # The 2 MB line is capped; a normal line is untouched.
    assert len(huge["text"]) < 5000
    assert "truncated" in huge["text"]
    assert small["text"] == "small line"


def test_store_uses_aggregate_and_filters_since(tmp_path: Path) -> None:
    _make_run(tmp_path)
    run = tmp_path / "20260926T000000Z_local"
    atomic_write_json(run / "aggregate-report.json", {"status": "FAIL"})
    store = RunStore(tmp_path)

    assert store.get_run(run.name)["status"] == "FAIL"
    assert store.list_runs(since="2026-09-27T00:00:00+00:00")["total"] == 0
    assert store.list_runs(since="2026-09-25T00:00:00+00:00")["total"] == 1


def test_wsgi_routes_and_sse(tmp_path: Path) -> None:
    _make_run(tmp_path)
    app = create_app(RunStore(tmp_path))

    def request(
        path: str, method: str = "GET", query: str = "", body: str | None = None
    ):
        captured = {}

        def start(status, headers):
            captured["status"] = status
            captured["headers"] = dict(headers)

        payload = body.encode("utf-8") if body is not None else b""
        body = b"".join(
            app(
                {
                    "REQUEST_METHOD": method,
                    "PATH_INFO": path,
                    "QUERY_STRING": query,
                    "CONTENT_LENGTH": str(len(payload)),
                    "wsgi.input": io.BytesIO(payload),
                },
                start,
            )
        )
        return captured, body

    response, body = request("/api/runs")
    assert response["status"] == "200 OK"
    assert json.loads(body)["total"] == 1

    response, body = request("/api/runs/20260926T000000Z_local/events")
    assert response["headers"]["Content-Type"].startswith("text/event-stream")
    assert b"event: state" in body
    assert b"id: 1" in body

    _, body = request("/api/runs/20260926T000000Z_local/events", query="after=1")
    assert b"id: 1" not in body
    assert b"id: 2" in body

    response, _ = request("/api/runs", method="PUT", body="{}")
    assert response["status"] == "405 Method Not Allowed"
    assert response["headers"]["Allow"] == "POST"

    response, body = request(
        "/api/plans/preview", method="POST", body=json.dumps({"run_mode": "quantize"})
    )
    assert response["status"] == "400 Bad Request"
    assert "manifest" in json.loads(body)["error"]


def test_wsgi_serves_browser_ui_and_static_assets(tmp_path: Path) -> None:
    _make_run(tmp_path)
    app = create_app(RunStore(tmp_path))

    def request(path: str):
        captured = {}

        def start(status, headers):
            captured.update(status=status, headers=dict(headers))

        body = b"".join(
            app(
                {
                    "REQUEST_METHOD": "GET",
                    "PATH_INFO": path,
                    "QUERY_STRING": "",
                    "CONTENT_LENGTH": "0",
                    "wsgi.input": io.BytesIO(),
                },
                start,
            )
        )
        return captured, body

    response, index = request("/")
    assert response["status"] == "200 OK"
    assert response["headers"]["Content-Type"].startswith("text/html")
    assert "default-src 'self'" in response["headers"]["Content-Security-Policy"]
    assert b"Model Quality" in index
    assert b'id="runs-view"' in index
    assert b'id="plan-form"' in index
    assert b'id="operations-view"' in index
    assert b'id="run-failures"' in index
    assert b'data-stage-section="inference"' in index
    assert b'id="script-params"' in index
    assert b'id="load-params"' in index
    assert b'class="required-tag"' in index
    assert b'<script src="/ui/app.js?v=' in index
    assert b'id="plan-feedback"' in index
    assert b'id="preview-plan"' in index

    response, css = request("/ui/app.css")
    assert response["headers"]["Content-Type"].startswith("text/css")
    assert b".timeline" in css
    assert b"prefers-color-scheme: dark" in css

    response, javascript = request("/ui/app.js")
    assert response["headers"]["Content-Type"].startswith("text/javascript")
    assert b"/api/plans/preview" in javascript
    assert b"window.setInterval" in javascript
    assert b"window.clearInterval" in javascript
    assert b"loadScriptParams" in javascript
    assert b"Required fields missing" in javascript
    assert b"Submitting plan preview" in javascript
    assert b"showPlanError" in javascript
    assert b"renderFailures" in javascript
    assert b"Open stage log" in javascript
    assert b"Failure history" in javascript
    assert b"renderScriptParams" in javascript
    assert b"textContent" in javascript

    response, body = request("/ui/unknown.js")
    assert response["status"] == "404 Not Found"
    assert "unknown UI asset" in json.loads(body)["error"]
