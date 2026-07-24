from __future__ import annotations

import torch
import yaml

from llmcompressor.streaming import streaming_oneshot
from llmcompressor.utils.dev import resolve_execution_device
from tests.llmcompressor.streaming.test_quantize import TwoLayer, prepare, scheme


def test_pretrained_mode_selects_main_accelerator(monkeypatch, tmp_path):
    captured = {}

    def frontend(**kwargs):
        captured.update(kwargs)
        return tmp_path / "output"

    monkeypatch.setattr(
        "llmcompressor.utils.dev.get_main_device",
        lambda: torch.device("mps"),
    )
    monkeypatch.setattr(
        "llmcompressor.utils.dev.accelerator_device",
        lambda: torch.device("mps:0"),
    )
    monkeypatch.setattr(
        "llmcompressor.streaming.pretrained.streaming_oneshot_from_pretrained",
        frontend,
    )

    streaming_oneshot(
        model=tmp_path / "model",
        dataset=[],
        recipe=[],
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
    )

    assert captured["device"] == torch.device("mps:0")


def test_pretrained_mode_derives_sibling_work_dir(monkeypatch, tmp_path):
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
        dataset=[],
        recipe=[],
        output_dir=tmp_path / "output",
        device="cpu",
    )

    assert captured["work_dir"] == tmp_path / "output.streaming-work"


def test_explicit_indexless_accelerator_is_canonicalized(monkeypatch):
    monkeypatch.setattr(
        "llmcompressor.utils.dev.accelerator_device",
        lambda: torch.device("mps:0"),
    )

    assert resolve_execution_device("mps") == torch.device("mps:0")


def test_explicit_indexed_device_is_preserved(monkeypatch):
    def fail_if_called():
        raise AssertionError("indexed devices must not query current accelerator")

    monkeypatch.setattr(
        "llmcompressor.utils.dev.accelerator_device", fail_if_called
    )

    assert resolve_execution_device("cuda:3") == torch.device("cuda:3")
    assert resolve_execution_device("cpu") == torch.device("cpu")


def test_streaming_oneshot_runs_three_stages_and_resumes(tmp_path):
    checkpoint, _, _, inputs = prepare(tmp_path)
    work = tmp_path / "work"
    output = tmp_path / "output"
    calls = []

    def batches():
        calls.append("dataset")
        return iter(inputs)

    result = streaming_oneshot(
        model_factory=TwoLayer,
        checkpoint=checkpoint,
        output_dir=output,
        work_dir=work,
        calibration_batches=batches,
        targets=("layers.0", "layers.1"),
        recipe={"GPTQModifier": {}},
        dataset_fingerprint="c" * 64,
        schemes={"layers.0": scheme(), "layers.1": scheme()},
        algorithms=("gptq",),
        target_dtype=torch.float32,
        validate_config=False,
    )
    assert result == output
    assert (output / "FINALIZED").is_file()
    assert yaml.safe_load((output / "recipe.yaml").read_text()) == {
        "GPTQModifier": {}
    }
    assert calls == ["dataset"]

    resumed = streaming_oneshot(
        model_factory=TwoLayer,
        checkpoint=checkpoint,
        output_dir=output,
        work_dir=work,
        calibration_batches=batches,
        targets=("layers.0", "layers.1"),
        recipe={"GPTQModifier": {}},
        dataset_fingerprint="c" * 64,
        schemes={"layers.0": scheme(), "layers.1": scheme()},
        algorithms=("gptq",),
        target_dtype=torch.float32,
        validate_config=False,
    )
    assert resumed == output
    assert calls == ["dataset"]
