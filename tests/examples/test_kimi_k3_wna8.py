from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

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


def _tiny_text_config(num_experts=2):
    return SimpleNamespace(
        text_config=SimpleNamespace(
            first_k_dense_replace=1,
            moe_layer_freq=1,
            num_experts=num_experts,
            num_hidden_layers=2,
        )
    )


def _write_expert_index(model_dir, *, include_all_packed=True):
    weight_map = {}
    for expert in range(2):
        for projection in ("w1", "w2", "w3"):
            prefix = (
                "language_model.model.layers.1.block_sparse_moe.experts."
                f"{expert}.{projection}"
            )
            weight_map[f"{prefix}.weight_scale"] = "model.safetensors"
            if include_all_packed or (expert, projection) != (1, "w3"):
                weight_map[f"{prefix}.weight_packed"] = "model.safetensors"
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )


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


def test_k3_model_level_output_residual_stays_outside_decoder_targets():
    assert KIMI_K3_WNA8._OTHER_LINEAR_RE.fullmatch(
        "language_model.model.layers.1.self_attention_res_proj"
    )
    assert not KIMI_K3_WNA8._OTHER_LINEAR_RE.fullmatch(
        "language_model.model.output_attn_res_proj"
    )


def test_kda_num_heads_comes_from_text_config():
    config = SimpleNamespace(
        text_config=SimpleNamespace(linear_attn_config={"num_heads": 96})
    )

    assert KIMI_K3_WNA8._kda_num_heads(config) == 96


def test_expert_targets_validate_exact_index_but_emit_structural_regex(tmp_path):
    _write_expert_index(tmp_path)

    pattern, targets = KIMI_K3_WNA8._expert_targets_from_index(
        tmp_path, _tiny_text_config()
    )

    assert pattern == KIMI_K3_WNA8._EXPERT_TARGET_PATTERN
    assert len(pattern) < 150
    compiled = re.compile(pattern.removeprefix("re:"))
    assert len(targets) == 6
    assert all(compiled.fullmatch(target) for target in targets)


def test_expert_targets_reject_scale_without_packed_weight(tmp_path):
    _write_expert_index(tmp_path, include_all_packed=False)

    with pytest.raises(ValueError, match="weight_scale/weight_packed targets differ"):
        KIMI_K3_WNA8._expert_targets_from_index(
            tmp_path, _tiny_text_config()
        )


def test_expert_targets_reject_config_index_mismatch(tmp_path):
    _write_expert_index(tmp_path)

    with pytest.raises(ValueError, match="expert indices disagree"):
        KIMI_K3_WNA8._expert_targets_from_index(
            tmp_path, _tiny_text_config(num_experts=3)
        )


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
