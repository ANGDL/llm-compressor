from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open

from llmcompressor.streaming import finalize_streaming_checkpoint
from tests.llmcompressor.streaming.test_quantize import prepare, scheme
from llmcompressor.streaming import quantize_streaming


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
