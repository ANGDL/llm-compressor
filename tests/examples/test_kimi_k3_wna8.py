from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[2] / "examples" / "streaming_oneshot" / "kimi_k3_wNa8.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location("kimi_k3_wna8", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
KIMI_K3_WNA8 = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(KIMI_K3_WNA8)


class _FakeDataset(list):
    def shuffle(self, seed):
        self.shuffle_seed = seed
        return self


def test_calibration_datasets_use_independent_splits(monkeypatch):
    calls = []

    def fake_load_dataset(dataset_id, *, split):
        calls.append((dataset_id, split))
        return _FakeDataset([{"source": dataset_id}])

    monkeypatch.setattr(KIMI_K3_WNA8, "load_dataset", fake_load_dataset)

    multimodal_dataset, text_examples = KIMI_K3_WNA8.load_calibration_datasets(
        dataset_id="vision-dataset",
        dataset_split="test",
        text_dataset_id="text-dataset",
        text_dataset_split="train_sft",
        num_calibration_samples=32,
        text_calibration_samples=128,
    )

    assert calls == [
        ("vision-dataset", "test[:32]"),
        ("text-dataset", "train_sft[:128]"),
    ]
    assert multimodal_dataset.shuffle_seed == 42
    assert text_examples == [{"source": "text-dataset"}]


def test_k3_multimodal_message_uses_image_field():
    image = object()

    messages = KIMI_K3_WNA8._build_multimodal_messages(
        {"image": image, "caption": ["A caption"]}
    )

    image_part = messages[0]["content"][0]
    assert image_part == {"type": "image", "image": image}
    assert "image_url" not in image_part


@pytest.mark.parametrize("samples", [0, 3])
def test_text_dataset_can_be_disabled(monkeypatch, samples):
    calls = []

    def fake_load_dataset(dataset_id, *, split):
        calls.append((dataset_id, split))
        return _FakeDataset([{"source": dataset_id}])

    monkeypatch.setattr(KIMI_K3_WNA8, "load_dataset", fake_load_dataset)
    text_dataset_id = None if samples else "text-dataset"

    _, text_examples = KIMI_K3_WNA8.load_calibration_datasets(
        dataset_id="vision-dataset",
        dataset_split="test",
        text_dataset_id=text_dataset_id,
        text_dataset_split="train_sft",
        num_calibration_samples=2,
        text_calibration_samples=samples,
    )

    assert calls == [("vision-dataset", "test[:2]")]
    assert text_examples == []
