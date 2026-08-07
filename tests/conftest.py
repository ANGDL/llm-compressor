# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch


def _ensure_test_accelerator_api():
    if hasattr(torch, "accelerator"):
        return

    class _TestAcceleratorCompat:
        def _backend(self) -> str:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                return "cuda"
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return "xpu"
            if hasattr(torch, "mps") and torch.mps.is_available():
                return "mps"
            return "cpu"

        def current_accelerator(self):
            return SimpleNamespace(type=self._backend())

        def device_count(self):
            backend = self._backend()
            if backend == "cuda":
                return torch.cuda.device_count()
            if backend == "xpu":
                return torch.xpu.device_count()
            if backend == "mps":
                return 1
            return 0

        def is_available(self):
            return self._backend() != "cpu"

    torch.accelerator = _TestAcceleratorCompat()


_ensure_test_accelerator_api()


def pytest_configure(config):
    # Give each torchrun rank its own basetemp to avoid cleanup race conditions
    rank = os.environ.get("LOCAL_RANK")
    if rank is not None and config.option.basetemp is None:
        run_id = os.environ.get("TORCHELASTIC_RUN_ID", str(os.getpid()))
        config.option.basetemp = tempfile.mkdtemp(
            prefix=f"pytest-torchrun-{run_id}-rank{rank}-",
            dir=Path(tempfile.gettempdir()),
        )
