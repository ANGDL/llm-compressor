from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from llmcompressor.streaming import (
    ArtifactStore,
    collect_calibration_statistics,
)


class TwoLayer(nn.Module):
    def __init__(self, features=4):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(features, features), nn.Linear(features, features)]
        )

    def forward(self, value):
        for layer in self.layers:
            value = layer(value)
        return value


def checkpoint_for(tmp_path):
    torch.manual_seed(3)
    reference = TwoLayer()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "tiny"}))
    save_file(
        {name: value.detach() for name, value in reference.state_dict().items()},
        checkpoint / "model.safetensors",
    )
    return checkpoint, reference


def run_collect(
    checkpoint, artifacts, batches, *, forward_target=None, model_factory=TwoLayer
):
    return collect_calibration_statistics(
        model_factory=model_factory,
        checkpoint=str(checkpoint),
        artifact_dir=str(artifacts),
        calibration_batches=lambda: iter(batches),
        targets=("layers.0", "layers.1"),
        recipe={"GPTQModifier": {"scheme": "W8A16"}},
        dataset_fingerprint="a" * 64,
        algorithms=("gptq", "imatrix"),
        target_dtype=torch.float32,
        forward_target=forward_target,
    )


def test_collects_each_target_and_releases_weights(tmp_path):
    checkpoint, _ = checkpoint_for(tmp_path)
    batches = [torch.randn(2, 3, 4), torch.randn(2, 3, 4)]
    artifacts = tmp_path / "artifacts"
    built = []

    def model_factory():
        model = TwoLayer()
        built.append(model)
        return model

    store = run_collect(
        checkpoint, artifacts, batches, model_factory=model_factory
    )

    assert store.is_target_complete(0)
    assert store.is_target_complete(1)
    stats = store.load_target_statistics(0)
    expected = torch.cat(batches).reshape(-1, 4).float()
    torch.testing.assert_close(
        stats["layers.0.gptq_hessian"], 2 * expected.T @ expected
    )
    assert all(parameter.is_meta for parameter in built[0].parameters())
    assert not (artifacts / "boundaries" / "boundary-00000").exists()


def test_resume_skips_completed_target_and_uses_persisted_boundary(tmp_path):
    checkpoint, _ = checkpoint_for(tmp_path)
    batches = [torch.randn(1, 2, 4)]
    artifacts = tmp_path / "artifacts"
    calls = []

    def fail_on_second(target, value):
        calls.append(target)
        if len(calls) == 2:
            raise RuntimeError("stop after target zero commit")
        return target(value)

    # The first invocation commits target zero and leaves boundary one durable.
    with pytest.raises(RuntimeError, match="stop after"):
        run_collect(checkpoint, artifacts, batches, forward_target=fail_on_second)
    calls.clear()
    run_collect(
        checkpoint,
        artifacts,
        batches,
        forward_target=lambda target, value: (
            calls.append(target) or target(value)
        ),
    )
    assert len(calls) == 1
    assert ArtifactStore(artifacts).is_target_complete(1)


@pytest.mark.parametrize("modifier", ["AWQModifier", "SmoothQuantModifier", "AutoRoundModifier"])
def test_rejects_non_reference_recipe(modifier, tmp_path):
    checkpoint, _ = checkpoint_for(tmp_path)
    with pytest.raises(ValueError, match=modifier):
        collect_calibration_statistics(
            model_factory=TwoLayer,
            checkpoint=str(checkpoint),
            artifact_dir=str(tmp_path / "artifacts"),
            calibration_batches=[],
            targets=("layers.0",),
            recipe={modifier: {}},
            dataset_fingerprint="b" * 64,
        )
