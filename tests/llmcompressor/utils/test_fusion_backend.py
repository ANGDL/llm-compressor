import sys

import pytest

from llmcompressor.utils.fusion_backend import (
    FusionKDAProvider,
    fusion_kda_adapter,
    load_fusion_provider,
    requested_kda_backend,
)


def test_backend_selector_uses_public_fusion_name(monkeypatch):
    monkeypatch.setenv("LLMCOMPRESSOR_KDA_BACKEND", "fusion")
    assert requested_kda_backend() == "fusion"


def test_triton_is_compatibility_alias(monkeypatch):
    monkeypatch.setenv("LLMCOMPRESSOR_KDA_BACKEND", "triton")
    assert requested_kda_backend() == "fla"


def test_missing_fusion_plugin_is_optional(monkeypatch):
    monkeypatch.delitem(sys.modules, "llmcompressor_fusion_backend", raising=False)
    monkeypatch.setattr(
        "llmcompressor.utils.fusion_backend.FUSION_BACKEND_MODULE",
        "llmcompressor_fusion_backend_missing",
    )
    assert load_fusion_provider() is None


def test_fusion_adapter_preserves_state_layout():
    class Provider:
        name = "fusion"

        def is_available(self):
            return True

        def forward(self, *args, **kwargs):
            return args[0], kwargs["initial_state"]

    provider = Provider()
    assert isinstance(provider, FusionKDAProvider)
    import torch

    state = torch.zeros(1, 2, 3, 4)
    _, result = fusion_kda_adapter(provider)(
        torch.zeros(1, 1, 2, 3),
        torch.zeros(1, 1, 2, 3),
        torch.zeros(1, 1, 2, 4),
        torch.zeros(1, 1, 2, 3),
        torch.zeros(1, 1, 2),
        initial_state=state,
        transpose_state_layout=True,
    )
    assert result.shape == state.shape


def test_invalid_backend_name_is_rejected(monkeypatch):
    monkeypatch.setenv("LLMCOMPRESSOR_KDA_BACKEND", "vendor_internal")
    with pytest.raises(ValueError, match="invalid LLMCOMPRESSOR_KDA_BACKEND"):
        requested_kda_backend()
