"""Tests for _filter_container_modules_from_ignore in compressed_tensors_utils."""

import torch
import torch.nn as nn

from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.quant_scheme import QuantizationScheme

from llmcompressor.transformers.compression.compressed_tensors_utils import (
    _filter_container_modules_from_ignore,
)


def _make_scheme():
    return QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.CHANNEL,
            symmetric=True,
            dynamic=False,
        ),
    )


class TestFilterContainerModulesFromIgnore:
    """Verify that container modules with quantized children are removed from ignore."""

    def test_removes_containers_with_quantized_children(self):
        """ExpertMLPWithGate-style containers should be filtered out."""

        class ExpertContainer(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(16, 32, bias=False)
                self.up_proj = nn.Linear(16, 32, bias=False)
                self.down_proj = nn.Linear(32, 16, bias=False)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = nn.ModuleList(
                    [ExpertContainer() for _ in range(3)]
                )
                self.gate = nn.Linear(16, 3, bias=False)  # router

        model = Model()
        scheme = _make_scheme()
        for i in range(3):
            model.experts[i].gate_proj.quantization_scheme = scheme
            model.experts[i].up_proj.quantization_scheme = scheme
            model.experts[i].down_proj.quantization_scheme = scheme

        ignore = ["experts.0", "experts.1", "experts.2", "gate"]
        filtered = _filter_container_modules_from_ignore(model, ignore)

        assert "gate" in filtered, "Unquantized leaf should stay"
        assert "experts.0" not in filtered
        assert "experts.1" not in filtered
        assert "experts.2" not in filtered

    def test_keeps_unquantized_leaf_modules(self):
        """Leaf modules (no quantized children) should stay in ignore."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.indexer_wk = nn.Linear(16, 16, bias=False)
                self.q_proj = nn.Linear(16, 16, bias=False)

        model = Model()
        model.q_proj.quantization_scheme = _make_scheme()

        ignore = ["indexer_wk"]
        filtered = _filter_container_modules_from_ignore(model, ignore)
        assert filtered == ["indexer_wk"]

    def test_keeps_containers_without_quantized_children(self):
        """Container with NO quantized children should stay in ignore."""

        class Container(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_a = nn.Linear(4, 4)
                self.linear_b = nn.Linear(4, 4)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.container = Container()

        model = Model()
        ignore = ["container"]
        filtered = _filter_container_modules_from_ignore(model, ignore)
        assert filtered == ["container"]

    def test_empty_ignore(self):
        """Empty list passthrough."""
        model = nn.Linear(4, 4)
        assert _filter_container_modules_from_ignore(model, []) == []

    def test_none_ignore(self):
        """None passthrough."""
        model = nn.Linear(4, 4)
        assert _filter_container_modules_from_ignore(model, None) is None

    def test_nonexistent_module_name_kept(self):
        """If a name doesn't exist in the model, keep it (safe fallback)."""
        model = nn.Linear(4, 4)
        ignore = ["nonexistent.path"]
        filtered = _filter_container_modules_from_ignore(model, ignore)
        assert filtered == ["nonexistent.path"]
