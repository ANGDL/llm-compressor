from dataclasses import replace
import json

import pytest
import torch
from safetensors.torch import save_file

from llmcompressor.streaming.artifacts import (
    ArtifactCompatibilityError,
    ArtifactStore,
    CalibrationInfo,
    MaterializerInfo,
    RecipeInfo,
    SequentialInfo,
    SoftwareInfo,
    StreamingRunManifest,
    TargetStatisticsMetadata,
    fingerprint_checkpoint,
    fingerprint_json,
)


def make_manifest(source):
    return StreamingRunManifest(
        source=source,
        recipe=RecipeInfo(fingerprint_json({"GPTQModifier": {}})),
        calibration=CalibrationInfo("dataset-v1", 128, 2048, 42),
        sequential=SequentialInfo(("DecoderLayer",)),
        materializer=MaterializerInfo(
            "llmcompressor.cast", fingerprint_json({"dtype": "bfloat16"})
        ),
        software=SoftwareInfo.from_versions({"llmcompressor": "0.1"}),
    )


def make_metadata(index=0):
    return TargetStatisticsMetadata(
        target_name=f"model.layers.{index}",
        target_index=index,
        algorithms=("gptq",),
        tensor_names=("proj.gptq_hessian", "proj.gptq_num_samples"),
        source_tensor_fingerprints=(("proj.weight", "weight-digest"),),
        completed=False,
    )


@pytest.fixture
def initialized_store(tmp_path):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"model_type": "tiny"}), encoding="utf-8"
    )
    save_file({"weight": torch.ones(1)}, checkpoint / "model.safetensors")

    store = ArtifactStore(tmp_path / "artifacts")
    recipe = {"GPTQModifier": {"block_size": 128}}
    manifest = make_manifest(fingerprint_checkpoint(checkpoint))
    store.initialize(
        manifest, normalized_recipe=recipe, targets=["model.layers.0"]
    )
    return store, manifest, recipe


def test_commit_and_load_target(initialized_store):
    store, _, _ = initialized_store
    statistics = {
        "proj.gptq_hessian": torch.eye(4),
        "proj.gptq_num_samples": torch.tensor(8),
    }

    store.commit_target(make_metadata(), statistics)

    assert store.is_target_complete(0)
    assert store.load_target_metadata(0).completed is True
    loaded = store.load_target_statistics(0)
    assert torch.equal(loaded["proj.gptq_hessian"], torch.eye(4))
    assert loaded["proj.gptq_num_samples"].item() == 8


@pytest.mark.parametrize(
    "missing_name", ["COMPLETE", "metadata.json", "stats.safetensors"]
)
def test_incomplete_target_is_not_resumable(initialized_store, missing_name):
    store, _, _ = initialized_store
    statistics = {
        "proj.gptq_hessian": torch.eye(2),
        "proj.gptq_num_samples": torch.tensor(2),
    }
    store.commit_target(make_metadata(), statistics)
    (store.target_dir(0) / missing_name).unlink()

    assert not store.is_target_complete(0)
    with pytest.raises(RuntimeError, match="not completely committed"):
        store.load_target_statistics(0)


def test_tmp_files_do_not_make_target_complete(initialized_store):
    store, _, _ = initialized_store
    directory = store.target_dir(0)
    directory.mkdir(parents=True)
    (directory / ".stats.safetensors.deadbeef.tmp").write_bytes(b"partial")
    (directory / ".metadata.json.deadbeef.tmp").write_text("{}")

    assert not store.is_target_complete(0)


def test_commit_validates_declared_tensor_names(initialized_store):
    store, _, _ = initialized_store

    with pytest.raises(ValueError, match="tensor_names"):
        store.commit_target(make_metadata(), {"unexpected": torch.ones(1)})


def test_initialize_is_idempotent_and_rejects_changed_manifest(initialized_store):
    store, manifest, recipe = initialized_store
    targets = ["model.layers.0"]

    store.initialize(manifest, normalized_recipe=recipe, targets=targets)

    changed = replace(
        manifest, recipe=replace(manifest.recipe, normalized_sha256="0" * 64)
    )
    with pytest.raises(ArtifactCompatibilityError, match="recipe"):
        store.initialize(changed, normalized_recipe=recipe, targets=targets)


def test_initialize_rejects_changed_recipe_payload(initialized_store):
    store, manifest, _ = initialized_store

    with pytest.raises(ValueError, match="recipe content"):
        store.initialize(
            manifest,
            normalized_recipe={"GPTQModifier": {"block_size": 64}},
            targets=["model.layers.0"],
        )


def test_corrupt_statistics_are_not_resumable(initialized_store):
    store, _, _ = initialized_store
    statistics = {
        "proj.gptq_hessian": torch.eye(2),
        "proj.gptq_num_samples": torch.tensor(2),
    }
    store.commit_target(make_metadata(), statistics)
    (store.target_dir(0) / "stats.safetensors").write_bytes(b"corrupt")

    assert not store.is_target_complete(0)
