#!/bin/bash
# Host-side launcher for the SGLang (xSGL) runtime used by runtime_smoke.
#
# The model quality control plane runs inside the quantization container, which
# has no docker access. Runtime smoke therefore queries an SGLang server that is
# started here, from the host, and reached over the shared host network.
#
# usage: xsgl_launcher.sh <model_dir> [port] [container] [ready_timeout_seconds]
set -euo pipefail

MODEL_DIR="${1:?model directory required}"
PORT="${2:-30000}"
CONTAINER="${3:-zhuang_xsgl_0923}"
READY_TIMEOUT="${4:-1800}"
RUN_SH=/workspace/ds_v4/w4a8/run.sh
CONDA_INIT=/root/miniconda/etc/profile.d/conda.sh
CONDA_ENV=python310_torch29_cuda

echo "[xsgl_launcher] container=${CONTAINER} model_dir=${MODEL_DIR} port=${PORT}"
docker exec "${CONTAINER}" bash -lc "
  source ${CONDA_INIT} && conda activate ${CONDA_ENV} &&
  cd \$(dirname ${RUN_SH}) &&
  model_dir='${MODEL_DIR}' port='${PORT}' nohup bash ${RUN_SH} >/dev/null 2>&1 &
"
echo "[xsgl_launcher] launch requested; waiting up to ${READY_TIMEOUT}s for /health"

deadline=$(( $(date +%s) + READY_TIMEOUT ))
while :; do
  if curl -sf -m 10 "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo "[xsgl_launcher] ready after $(( READY_TIMEOUT - (deadline - $(date +%s)) ))s"
    exit 0
  fi
  if [ "$(date +%s)" -ge "${deadline}" ]; then
    echo "[xsgl_launcher] timed out waiting for http://127.0.0.1:${PORT}/health" >&2
    echo "[xsgl_launcher] last log lines:" >&2
    docker exec "${CONTAINER}" bash -lc "tail -40 \$(ls -t /workspace/ds_v4/w4a8/log_*.log | head -1)" >&2 || true
    exit 1
  fi
  sleep 15
done
