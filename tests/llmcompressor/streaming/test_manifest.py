import json

import pytest
import torch
from safetensors.torch import save_file

from llmcompressor.streaming.artifacts import (
    CURRENT_SCHEMA_VERSION,
    ArtifactCompatibilityError,
    CalibrationInfo,
    MaterializerInfo,
    RecipeInfo,
    SequentialInfo,
    SoftwareInfo,
    StreamingRunManifest,
    fingerprint_checkpoint,
    fingerprint_json,
)
from llmcompressor.streaming.artifacts.validation import (
    validate_manifest_compatibility,
)


def make_manifest(source, **overrides):
    values = {
        "source": source,
        "recipe": RecipeInfo(fingerprint_json({"GPTQModifier": {}})),
        "calibration": CalibrationInfo("dataset-v1", 128, 2048, 42),
        "sequential": SequentialInfo(("DecoderLayer",)),
        "materializer": MaterializerInfo(
            "llmcompressor.cast", fingerprint_json({"dtype": "bfloat16"})
        ),
        "software": SoftwareInfo.from_versions(
            {"llmcompressor": "0.1", "torch": "2.0"}
        ),
    }
    values.update(overrides)
    return StreamingRunManifest(**values)


@pytest.fixture
def checkpoint(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "tiny"}), encoding="utf-8"
    )
    save_file({"weight": torch.ones(1)}, model_dir / "model.safetensors")
    return model_dir


def test_manifest_round_trip_and_stable_fingerprint(checkpoint):
    source = fingerprint_checkpoint(checkpoint)
    manifest = make_manifest(source)

    restored = StreamingRunManifest.from_dict(manifest.to_dict())

    assert restored == manifest
    assert source.content_fingerprint == fingerprint_checkpoint(
        checkpoint
    ).content_fingerprint
    assert fingerprint_json({"b": 2, "a": 1}) == fingerprint_json(
        {"a": 1, "b": 2}
    )


def test_manifest_rejects_unknown_schema(checkpoint):
    data = make_manifest(fingerprint_checkpoint(checkpoint)).to_dict()
    data["schema_version"] = CURRENT_SCHEMA_VERSION + 1

    with pytest.raises(ValueError, match="schema version"):
        StreamingRunManifest.from_dict(data)


def test_manifest_rejects_unknown_fields(checkpoint):
    data = make_manifest(fingerprint_checkpoint(checkpoint)).to_dict()
    data["future_field"] = True

    with pytest.raises(ValueError, match="Unexpected fields"):
        StreamingRunManifest.from_dict(data)


@pytest.mark.parametrize("changed_section", ["recipe", "sequential", "materializer"])
def test_compatibility_rejects_result_affecting_changes(
    checkpoint, changed_section
):
    source = fingerprint_checkpoint(checkpoint)
    stored = make_manifest(source)
    replacements = {
        "recipe": RecipeInfo("0" * 64),
        "sequential": SequentialInfo(("Linear",)),
        "materializer": MaterializerInfo("custom", "1" * 64),
    }
    expected = make_manifest(source, **{changed_section: replacements[changed_section]})

    with pytest.raises(ArtifactCompatibilityError, match=changed_section):
        validate_manifest_compatibility(stored, expected)


def test_checkpoint_location_is_not_part_of_content_identity(checkpoint, tmp_path):
    source = fingerprint_checkpoint(checkpoint)
    moved = type(source)(
        location=str(tmp_path / "other-location"),
        config_sha256=source.config_sha256,
        index_sha256=source.index_sha256,
        shards=source.shards,
        strict=source.strict,
    )

    validate_manifest_compatibility(make_manifest(source), make_manifest(moved))


def test_software_versions_are_diagnostic(checkpoint):
    source = fingerprint_checkpoint(checkpoint)
    stored = make_manifest(source)
    upgraded = make_manifest(
        source, software=SoftwareInfo.from_versions({"llmcompressor": "0.2"})
    )

    validate_manifest_compatibility(stored, upgraded)


def test_strict_checkpoint_fingerprint_hashes_shards(checkpoint):
    source = fingerprint_checkpoint(checkpoint, strict=True)

    assert source.strict is True
    assert all(shard.sha256 for shard in source.shards)
