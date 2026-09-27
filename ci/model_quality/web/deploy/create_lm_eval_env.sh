#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${1:-model_quality_lm_eval}
CONDA_ROOT=${CONDA_ROOT:-/root/miniconda}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
LM_EVAL_VERSION=${LM_EVAL_VERSION:-0.4.9.1}
OPENAI_VERSION=${OPENAI_VERSION:-2.6.1}
TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION:-4.57.1}
PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

conda run -n "${ENV_NAME}" python -m pip install \
    --index-url "${PIP_INDEX_URL}" \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "lm-eval[api]==${LM_EVAL_VERSION}" \
    "openai==${OPENAI_VERSION}" \
    "transformers==${TRANSFORMERS_VERSION}"

conda run -n "${ENV_NAME}" python - <<'PY'
import importlib.metadata
import lm_eval

print("lm-eval", importlib.metadata.version("lm-eval"))
print("openai", importlib.metadata.version("openai"))
print("transformers", importlib.metadata.version("transformers"))
print("lm_eval", lm_eval.__file__)
PY
