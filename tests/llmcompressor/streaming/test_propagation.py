from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import SmoothQuantModifier
from llmcompressor.recipe import Recipe
from llmcompressor.streaming import streaming_oneshot
from llmcompressor.streaming.pipeline import _initialize_run
from llmcompressor.streaming.propagation import (
    propagation_mode,
    resolve_propagate_error,
)


@pytest.mark.parametrize(
    "modifier_type,default",
    [(QuantizationModifier, False), (GPTQModifier, True)],
)
@pytest.mark.parametrize("option", [None, False])
def test_propagation_defaults_follow_modifier_and_explicit_false_overrides(
    modifier_type, default, option
):
    actual = resolve_propagate_error([modifier_type(scheme="W8A8")], option)
    assert actual is (default if option is None else False)
    assert propagation_mode(actual) == (
        "post-update" if actual else "calibration-outputs"
    )


def test_rtn_explicit_true_downgrades_with_warning():
    with pytest.warns(UserWarning, match="has no effect for QuantizationModifier"):
        actual = resolve_propagate_error(
            [QuantizationModifier(scheme="W8A8")], True
        )
    assert actual is False
    assert propagation_mode(actual) == "calibration-outputs"


def test_gptq_explicit_true_is_preserved():
    actual = resolve_propagate_error([GPTQModifier(scheme="W8A8")], True)
    assert actual is True
    assert propagation_mode(actual) == "post-update"


@pytest.mark.parametrize("option", [0, 1, "false"])
def test_propagation_rejects_non_boolean_options(option):
    with pytest.raises(ValueError, match="True, False, or None"):
        resolve_propagate_error([QuantizationModifier(scheme="W8A8")], option)


@pytest.mark.parametrize("option", [False, True])
def test_explicit_override_rejects_mixed_modifier_recipe(option):
    modifiers = [SmoothQuantModifier(), QuantizationModifier(scheme="W8A8")]
    with pytest.raises(ValueError, match="requires one"):
        resolve_propagate_error(modifiers, option)
    assert resolve_propagate_error(modifiers, None)


@pytest.mark.parametrize("option", [None, False, True])
def test_public_api_forwards_propagation_to_pretrained_mode(
    tmp_path, monkeypatch, option
):
    captured = {}

    def frontend(**kwargs):
        captured.update(kwargs)
        return tmp_path / "output"

    monkeypatch.setattr(
        "llmcompressor.streaming.pretrained.streaming_oneshot_from_pretrained",
        frontend,
    )
    streaming_oneshot(
        model=tmp_path / "model",
        recipe=QuantizationModifier(scheme="W8A8"),
        dataset=[],
        output_dir=tmp_path / "output",
        device="cpu",
        propagate_error=option,
    )
    assert captured["propagate_error"] is option


def test_boundary_mode_rejects_propagation_override(tmp_path):
    with pytest.raises(ValueError, match="requires pretrained"):
        streaming_oneshot(
            output_dir=tmp_path / "output", device="cpu", propagate_error=False
        )


def test_propagation_mode_changes_run_identity_and_manifest(tmp_path, monkeypatch):
    import llmcompressor.streaming.pipeline as pipeline

    class Store:
        def __init__(self, root):
            pass

        def initialize(self, manifest, **kwargs):
            captured.append(manifest)

    captured = []
    monkeypatch.setattr(pipeline, "ArtifactStore", Store)
    monkeypatch.setattr(
        pipeline,
        "fingerprint_checkpoint",
        lambda _path: SimpleNamespace(content_fingerprint="a" * 64),
    )
    materializer = SimpleNamespace(
        manifest_info=lambda **_kwargs: SimpleNamespace(config_sha256="b" * 64)
    )
    kwargs = dict(
        checkpoint=Path("checkpoint"),
        artifact_dir=tmp_path,
        recipe=Recipe.create_instance(QuantizationModifier(scheme="W8A8")),
        dataset_fingerprint="c" * 64,
        targets=("layer",),
        materializer=materializer,
        target_dtype=torch.float32,
        num_samples=1,
        max_seq_length=4,
        seed=42,
        pack_to_int8=True,
    )
    default = _initialize_run(**kwargs)
    reused = _initialize_run(**kwargs, propagate_error=False)
    with pytest.warns(UserWarning, match="has no effect for QuantizationModifier"):
        downgraded = _initialize_run(**kwargs, propagate_error=True)
    assert default == reused == downgraded
    assert captured[-3].execution.propagation_mode == "calibration-outputs"
    assert captured[-2].execution.propagation_mode == "calibration-outputs"
    assert captured[-1].execution.propagation_mode == "calibration-outputs"

    propagated = _initialize_run(
        **{
            **kwargs,
            "recipe": Recipe.create_instance(GPTQModifier(scheme="W8A8")),
        },
        propagate_error=True,
    )
    assert propagated != reused
    assert captured[-1].execution.propagation_mode == "post-update"
