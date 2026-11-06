#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/ssd4/models/Qwen3-VL-32B-Instruct}"
ROOT_PATH="${ROOT_PATH:-/data/quant/e2e/qwen3-vl-32b-streaming}"
ALGORITHM="${ALGORITHM:-rtn}"
SAMPLES="${SAMPLES:-8}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-512}"
DEVICE="${DEVICE:-cuda:0}"
STAGE="${STAGE:-all}"
OVERWRITE="${OVERWRITE:-0}"
CONTAINER="${CONTAINER:-llm-quant-base}"
REPO_PATH="${REPO_PATH:-/data/quant/llm-compressor}"

OVERWRITE_ARG=""
if [[ "${OVERWRITE}" == "1" ]]; then
  OVERWRITE_ARG="--overwrite"
fi

if command -v docker >/dev/null 2>&1; then
  exec docker exec \
    -e PYTHONUNBUFFERED=1 \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "${CONTAINER}" \
    bash -lc "cd '${REPO_PATH}' && \
      python scripts/e2e_qwen3_vl32b_streaming.py \
        --model '${MODEL_PATH}' \
        --root '${ROOT_PATH}' \
        --algorithm '${ALGORITHM}' \
        --samples '${SAMPLES}' \
        --max-sequence-length '${MAX_SEQUENCE_LENGTH}' \
        --device '${DEVICE}' \
        --stage '${STAGE}' ${OVERWRITE_ARG}"
fi

cd "${REPO_PATH}"
exec python scripts/e2e_qwen3_vl32b_streaming.py \
  --model "${MODEL_PATH}" \
  --root "${ROOT_PATH}" \
  --algorithm "${ALGORITHM}" \
  --samples "${SAMPLES}" \
  --max-sequence-length "${MAX_SEQUENCE_LENGTH}" \
  --device "${DEVICE}" \
  --stage "${STAGE}" ${OVERWRITE_ARG}
