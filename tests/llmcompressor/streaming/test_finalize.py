from __future__ import annotations

import json

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs
from safetensors import safe_open
from safetensors.torch import save_file

from llmcompressor.streaming import (
    collect_calibration_statistics,
    finalize_streaming_checkpoint,
    quantize_streaming,
)
from tests.llmcompressor.streaming.test_quantize import prepare, scheme


def test_finalize_builds_index_config_and_preserves_auxiliary_files(tmp_path):
    checkpoint, artifacts, _, _ = prepare(tmp_path)
    (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")
    staging = tmp_path / "staging"
    quantize_streaming(
        checkpoint=checkpoint,
        artifact_dir=artifacts,
        staging_dir=staging,
        schemes={"layers.0": scheme()},
        target_dtype=torch.float32,
    )

    output = finalize_streaming_checkpoint(
        checkpoint=checkpoint,
        artifact_dir=artifacts,
        staging_dir=staging,
        output_dir=tmp_path / "output",
        validate_config=False,
    )
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == {
        "layers.0.weight_scale",
        "layers.0.weight_packed",
        "layers.0.weight_shape",
        "layers.0.bias",
        "layers.1.weight",
        "layers.1.bias",
    }
    assert index["metadata"]["total_size"] > 0
    config = json.loads((output / "config.json").read_text())
    assert config["quantization_config"]["quantization_status"] == "compressed"
    assert (output / "tokenizer.json").is_file()
    assert (output / "FINALIZED").is_file()

    for name, shard_name in index["weight_map"].items():
        with safe_open(output / shard_name, framework="pt", device="cpu") as file:
            assert name in file.keys()


def test_finalize_rejects_incomplete_staging_without_publishing(tmp_path):
    checkpoint, artifacts, _, _ = prepare(tmp_path)
    staging = tmp_path / "staging"
    quantize_streaming(
        checkpoint=checkpoint,
        artifact_dir=artifacts,
        staging_dir=staging,
        schemes={"layers.0": scheme()},
        target_dtype=torch.float32,
    )
    (staging / "state/model-00002-of-00002.safetensors.json").unlink()
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="incomplete"):
        finalize_streaming_checkpoint(
            checkpoint=checkpoint,
            artifact_dir=artifacts,
            staging_dir=staging,
            output_dir=output,
            validate_config=False,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".output.*.tmp"))


def test_finalize_refuses_existing_or_source_output(tmp_path):
    checkpoint, artifacts, _, _ = prepare(tmp_path)
    staging = tmp_path / "staging"
    quantize_streaming(
        checkpoint=checkpoint,
        artifact_dir=artifacts,
        staging_dir=staging,
        schemes={"layers.0": scheme()},
        target_dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="differ"):
        finalize_streaming_checkpoint(
            checkpoint=checkpoint,
            artifact_dir=artifacts,
            staging_dir=staging,
            output_dir=checkpoint,
            validate_config=False,
        )


def test_finalize_accepts_int_quantized_dense_weight(tmp_path):
    checkpoint, artifacts, _, _ = prepare(tmp_path)
    dense_scheme = scheme()
    dense_scheme.weights.num_bits = 8
    dense_scheme.input_activations = QuantizationArgs(
        num_bits=8, strategy="token", dynamic=True
    )
    staging = tmp_path / "staging"
    quantize_streaming(
        checkpoint=checkpoint,
        artifact_dir=artifacts,
        staging_dir=staging,
        schemes={"layers.0": dense_scheme},
        target_dtype=torch.float32,
    )

    output = finalize_streaming_checkpoint(
        checkpoint=checkpoint,
        artifact_dir=artifacts,
        staging_dir=staging,
        output_dir=tmp_path / "output",
        quantization_config={"format": "int-quantized"},
        validate_config=False,
    )
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert "layers.0.weight" in index["weight_map"]
    assert "layers.0.weight_packed" not in index["weight_map"]


def test_finalize_omits_identical_declared_tied_weight(tmp_path, monkeypatch):
    checkpoint, _, model, inputs = prepare(tmp_path)
    shard = checkpoint / "model-00002-of-00002.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as file:
        tensors = {name: file.get_tensor(name) for name in file.keys()}
    tensors["lm_head.weight"] = model.layers[0].weight.detach().clone()
    save_file(tensors, shard)
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["lm_head.weight"] = shard.name
    index_path.write_text(json.dumps(index), encoding="utf-8")

    artifacts = tmp_path / "tied-artifacts"
    collect_calibration_statistics(
        model_factory=type(model),
        checkpoint=str(checkpoint),
        artifact_dir=str(artifacts),
        calibration_batches=inputs,
        targets=("layers.0", "layers.1"),
        recipe={"GPTQModifier": {}},
        dataset_fingerprint="d" * 64,
        algorithms=("gptq",),
        target_dtype=torch.float32,
    )
    monkeypatch.setattr(
        "llmcompressor.streaming.quantize.infer_transformers_tied_weights",
        lambda _: {"lm_head.weight": "layers.0.weight"},
    )
    staging = tmp_path / "tied-staging"
    quantize_streaming(
        checkpoint=checkpoint,
        artifact_dir=artifacts,
        staging_dir=staging,
        schemes={"layers.1": scheme()},
        target_dtype=torch.float32,
    )
    output = finalize_streaming_checkpoint(
        checkpoint=checkpoint,
        artifact_dir=artifacts,
        staging_dir=staging,
        output_dir=tmp_path / "tied-output",
        validate_config=False,
    )
    output_index = json.loads(
        (output / "model.safetensors.index.json").read_text()
    )
    assert "layers.0.weight" in output_index["weight_map"]
    assert "lm_head.weight" not in output_index["weight_map"]
