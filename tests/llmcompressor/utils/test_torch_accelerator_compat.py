import importlib.util
import sys
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "llmcompressor"
    / "_torch_accelerator_compat.py"
)


def _load_compat_module():
    spec = importlib.util.spec_from_file_location(
        "llmcompressor_torch_accelerator_compat", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ensure_torch_accelerator_backfills_missing_namespace(monkeypatch):
    compat = _load_compat_module()
    original_accelerator = getattr(torch, "accelerator", None)

    monkeypatch.delattr(torch, "accelerator", raising=False)
    accelerator = compat.ensure_torch_accelerator()

    assert hasattr(torch, "accelerator")
    assert accelerator is torch.accelerator
    assert hasattr(accelerator, "is_available")
    assert hasattr(accelerator, "current_accelerator")
    assert hasattr(accelerator, "current_device_index")
    assert hasattr(accelerator, "device_count")
    assert hasattr(accelerator, "reset_peak_memory_stats")
    assert hasattr(accelerator, "max_memory_allocated")
    assert hasattr(accelerator, "get_memory_info")

    if original_accelerator is not None:
        monkeypatch.setattr(torch, "accelerator", original_accelerator)


def test_ensure_torch_accelerator_preserves_existing_namespace():
    compat = _load_compat_module()
    accelerator = compat.ensure_torch_accelerator()

    assert accelerator is torch.accelerator


def test_import_llmcompressor_backfills_torch_accelerator(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    try:
        for name in list(sys.modules):
            if name == "llmcompressor" or name.startswith("llmcompressor."):
                sys.modules.pop(name)

        monkeypatch.delattr(torch, "accelerator", raising=False)

        import llmcompressor  # noqa: F401

        assert hasattr(torch, "accelerator")
        assert torch.accelerator.current_accelerator().type in {
            "cpu",
            "cuda",
            "xpu",
            "mps",
        }
    finally:
        sys.path.pop(0)
