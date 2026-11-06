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

    store = run_collect(checkpoint, artifacts, batches, model_factory=model_factory)

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
        forward_target=lambda target, value: calls.append(target) or target(value),
    )
    assert len(calls) == 1
    assert ArtifactStore(artifacts).is_target_complete(1)


@pytest.mark.parametrize(
    "modifier", ["AWQModifier", "SmoothQuantModifier", "AutoRoundModifier"]
)
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


def test_collect_cache_skips_hooks_but_still_propagates_boundaries(
    tmp_path, monkeypatch
):
    from llmcompressor.streaming import CandidateStatisticsCache
    from llmcompressor.streaming.statistics import (
        GPTQStatisticsCollector,
        IMatrixStatisticsCollector,
    )

    checkpoint, _reference = checkpoint_for(tmp_path)
    batches = [torch.randn(2, 3, 4)]
    cache = CandidateStatisticsCache(tmp_path / "cache")

    def run(name):
        return collect_calibration_statistics(
            model_factory=TwoLayer,
            checkpoint=str(checkpoint),
            artifact_dir=str(tmp_path / name),
            calibration_batches=batches,
            targets=("layers.0", "layers.1"),
            recipe={"GPTQModifier": {}},
            dataset_fingerprint="a" * 64,
            target_dtype=torch.float32,
            statistics_cache=cache,
        )

    first = run("first")
    for collector in (GPTQStatisticsCollector, IMatrixStatisticsCollector):
        monkeypatch.setattr(
            collector,
            "attach",
            lambda *_args: pytest.fail("cache hits must not attach collectors"),
        )
    second = run("second")
    for index in (0, 1):
        expected = first.load_target_statistics(index)
        actual = second.load_target_statistics(index)
        assert actual.keys() == expected.keys()
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_collect_cache_resume_binds_actual_inputs_and_transform(tmp_path):
    from llmcompressor.streaming import (
        ArtifactCompatibilityError,
        CandidateStatisticsCache,
    )

    checkpoint, _reference = checkpoint_for(tmp_path)
    batches = [torch.randn(2, 3, 4)]
    arguments = {
        "model_factory": TwoLayer,
        "checkpoint": str(checkpoint),
        "artifact_dir": str(tmp_path / "artifacts"),
        "calibration_batches": batches,
        "targets": ("layers.0",),
        "recipe": {"GPTQModifier": {}},
        "dataset_fingerprint": "a" * 64,
        "target_dtype": torch.float32,
        "statistics_cache": CandidateStatisticsCache(tmp_path / "cache"),
    }
    collect_calibration_statistics(**arguments)
    with pytest.raises(ArtifactCompatibilityError):
        collect_calibration_statistics(
            **arguments, candidate_config={"rotation": "new"}
        )
    batches[0].add_(1)
    with pytest.raises(ArtifactCompatibilityError):
        collect_calibration_statistics(**arguments)
