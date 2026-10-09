#!/bin/bash
# Built-in runtime-smoke harness for the model-quality pipeline.
#
# The worker invokes this (inside the serving container) as:
#   bash runtime_smoke_harness.sh <serve_script> <port> <reports_dir> <output_dir> [extra args...]
#
# It injects the run-scoped model/report paths as environment variables so an
# existing env-based serve script (e.g. MODEL_DIR=${model_dir:-/home/models/...})
# works unchanged, launches that serve script, waits for the OpenAI-compatible
# server on 127.0.0.1:<port>, sends one probe request, and writes the PASS/FAIL
# result JSON the pipeline reads.
set -u

SERVE_SCRIPT="${1:?serve script required}"
PORT="${2:?port required}"
REPORTS_DIR="${3:?reports_dir required}"
OUTPUT_DIR="${4:?output_dir required}"
shift 4 || true

# Inject the quantized model path and report dir under common env names so the
# serve script needs no changes.
export model_dir="$OUTPUT_DIR" MODEL_DIR="$OUTPUT_DIR" output_dir="$OUTPUT_DIR"
export reports_dir="$REPORTS_DIR" REPORTS_DIR="$REPORTS_DIR"
# Also inject the port the harness will probe, so an env-style serve script
# (e.g. PORT=${port:-30000}) binds the configured port instead of its own
# default. Without this the server can come up on a different port and the
# harness waits forever on the wrong one.
export port="$PORT" PORT="$PORT"

RESULT="$REPORTS_DIR/runtime-smoke-output.json"
SERVE_LOG="$REPORTS_DIR/runtime-smoke-serve.log"
mkdir -p "$REPORTS_DIR"
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
write_result() {
  printf '{"status":"%s","message":"%s","model_dir":"%s","port":%s,"checked_at":"%s"}\n' \
    "$1" "$2" "$OUTPUT_DIR" "$PORT" "$(now)" > "$RESULT"
}

echo "[harness] serve=$SERVE_SCRIPT port=$PORT model_dir=$OUTPUT_DIR reports_dir=$REPORTS_DIR"

# Start the server only if nothing is already serving on the port. nohup so it
# survives after this harness exits (the serve script may also nohup itself).
if ! curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "[harness] launching serve script (output -> $SERVE_LOG)"
  # Capture the serve script's own output so a failed launch (e.g. a missing
  # binary) is visible in the reports dir instead of vanishing to /dev/null.
  nohup bash "$SERVE_SCRIPT" "$@" > "$SERVE_LOG" 2>&1 &
fi

echo "[harness] waiting for 127.0.0.1:$PORT (model load can take many minutes)"
ready=0
for _ in $(seq 1 600); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  sleep 5
done
if [ "$ready" != "1" ]; then
  write_result FAIL "server not ready on 127.0.0.1:$PORT"
  echo "[harness] NOT READY — last lines of serve output:"
  tail -n 40 "$SERVE_LOG" 2>/dev/null || true
  exit 1
fi

MODEL_ID=$(curl -sf "http://127.0.0.1:$PORT/v1/models" \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
[ -z "$MODEL_ID" ] && MODEL_ID="$OUTPUT_DIR"
echo "[harness] model id=$MODEL_ID; sending probe completion"
RESP=$(curl -sf "http://127.0.0.1:$PORT/v1/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"1+1=\",\"max_tokens\":8,\"temperature\":0}" 2>/dev/null)
if [ -n "$RESP" ] && printf "%s" "$RESP" | grep -q '"text"'; then
  write_result PASS "completion ok"
  echo "[harness] PASS: $(printf "%s" "$RESP" | head -c 300)"
  exit 0
fi
write_result FAIL "probe completion failed"
echo "[harness] FAIL: $(printf "%s" "$RESP" | head -c 300)"
exit 1
