from __future__ import annotations

import json

import pytest
import torch
from compressed_tensors.compressors import compress_module
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from llmcompressor.entrypoints.model_free.lifecycle import (
    initialize_quantized_linear,
)
from llmcompressor.modifiers.gptq.gptq_quantize import quantize_weight
from llmcompressor.modifiers.quantization.calibration import (
    apply_calibration_status,
    freeze_module_quantization,
    initialize_observer,
    observe,
)
from llmcompressor.streaming import (
    ArtifactCompatibilityError,
    collect_calibration_statistics,
    quantize_streaming,
)


class TwoLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])


def scheme(*, observer=None):
    return QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            strategy="channel",
            symmetric=True,
            observer=observer,
        ),
    )


def prepare(tmp_path, *, algorithms=("gptq", "imatrix")):
    torch.manual_seed(8)
    model = TwoLayer()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"model_type": "tiny"}), encoding="utf-8"
    )
    state = model.state_dict()
    save_file(
        {
            "layers.0.weight": state["layers.0.weight"],
            "layers.0.bias": state["layers.0.bias"],
        },
        checkpoint / "model-00001-of-00002.safetensors",
    )
    save_file(
        {
            "layers.1.weight": state["layers.1.weight"],
            "layers.1.bias": state["layers.1.bias"],
        },
        checkpoint / "model-00002-of-00002.safetensors",
    )
    index = {
        "weight_map": {
            name: (
                "model-00001-of-00002.safetensors"
                if name.startswith("layers.0")
                else "model-00002-of-00002.safetensors"
            )
            for name in state
        }
    }
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    inputs = [torch.randn(2, 3, 4), torch.randn(1, 3, 4)]
    artifacts = tmp_path / "artifacts"
    collect_calibration_statistics(
        model_factory=TwoLayer,
        checkpoint=checkpoint,
        artifact_dir=artifacts,
        calibration_batches=inputs,
        targets=("layers.0", "layers.1"),
        recipe={"GPTQModifier": {}},
        dataset_fingerprint="c" * 64,
        algorithms=algorithms,
        target_dtype=torch.float32,
    )
    return checkpoint, artifacts, model, inputs


def read_shard(path):
    with safe_open(path, framework="pt", device="cpu") as file:
        return {name: file.get_tensor(name) for name in file.keys()}


def test_gptq_matches_existing_quantize_and_compress_path(tmp_path):
    checkpoint, artifacts, model, _ = prepare(tmp_path)
    output = quantize_streaming(
        checkpoint=checkpoint,
        artifact_dir=artifacts,
        staging_dir=tmp_path / "staging",
        schemes={"layers.0": scheme()},
        target_dtype=torch.float32,
    )
    actual = read_shard(output / "shards/model-00001-of-00002.safetensors")

    reference = initialize_quantized_linear(
        model.layers[0].weight, scheme(), "cpu"
    )
    initialize_observer(reference, "weight")
    apply_calibration_status(reference)
    observe(reference, "weight")
    statistics = __import__(
        "llmcompressor.streaming", fromlist=["ArtifactStore"]
    ).ArtifactStore(artifacts).load_target_statistics(0)
    _, qparams = quantize_weight(
        reference,
        reference.quantization_scheme.weights,
        statistics["layers.0.gptq_hessian"]
        / statistics["layers.0.gptq_num_samples"],
    )
    for name, value in qparams.items():
        setattr(reference, name, nn.Parameter(value, requires_grad=False))
    freeze_module_quantization(reference)
    compress_module(reference)

    expected = reference.state_dict(prefix="layers.0.")
    assert actual.keys() == {"layers.0.bias", *expected.keys()}
    for name, value in expected.items():
        assert torch.equal(actual[name], value)
    assert torch.equal(actual["layers.0.bias"], model.layers[0].bias)


def test_imatrix_and_resume_skip_completed_shard(tmp_path, monkeypatch):
    checkpoint, artifacts, _, _ = prepare(tmp_path)
    staging = tmp_path / "staging"
    schemes = {
        "layers.0": scheme(observer="imatrix_mse"),
        "layers.1": scheme(observer="imatrix_mse"),
    }
    quantize_streaming(
        checkpoint=checkpoint,
        artifact_dir=artifacts,
        staging_dir=staging,
        schemes=schemes,
        use_gptq=False,
        target_dtype=torch.float32,
    )
    first = staging / "shards/model-00001-of-00002.safetensors"
    before = first.read_bytes()

    quantize_streaming(
        checkpoint=checkpoint,
        artifact_dir=artifacts,
        staging_dir=staging,
        schemes=schemes,
        use_gptq=False,
        target_dtype=torch.float32,
    )
    assert first.read_bytes() == before
    assert not (staging / "model.safetensors.index.json").exists()
    assert not (staging / "config.json").exists()

    with pytest.raises(ArtifactCompatibilityError, match="different quantization"):
        quantize_streaming(
            checkpoint=checkpoint,
            artifact_dir=artifacts,
            staging_dir=staging,
            schemes=schemes,
            use_gptq=True,
            target_dtype=torch.float32,
        )


def test_missing_statistics_fails_before_loading_weights(tmp_path, monkeypatch):
    checkpoint, artifacts, _, _ = prepare(tmp_path, algorithms=("imatrix",))

    def fail_load(*args, **kwargs):
        raise AssertionError("weights must not be loaded")

    monkeypatch.setattr(
        "llmcompressor.streaming.quantize.SafetensorsWeightSource.load_tensors",
        fail_load,
    )
    with pytest.raises(RuntimeError, match="gptq_hessian"):
        quantize_streaming(
            checkpoint=checkpoint,
            artifact_dir=artifacts,
            staging_dir=tmp_path / "staging",
            schemes={"layers.0": scheme()},
            target_dtype=torch.float32,
        )
