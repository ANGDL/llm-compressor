#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${1:-model_quality_evalscope}
CONDA_ROOT=${CONDA_ROOT:-/root/miniconda}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
EVALSCOPE_VERSION=${EVALSCOPE_VERSION:-1.12.0}
TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION:-4.57.1}
PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

conda run -n "${ENV_NAME}" python -m pip install \
    --index-url "${PIP_INDEX_URL}" \
    "evalscope==${EVALSCOPE_VERSION}" \
    "transformers==${TRANSFORMERS_VERSION}"

conda run -n "${ENV_NAME}" python - <<'PY'
import importlib.metadata
import evalscope

print("evalscope", importlib.metadata.version("evalscope"))
print("transformers", importlib.metadata.version("transformers"))
print("evalscope_module", evalscope.__file__)
PY
