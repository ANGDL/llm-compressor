from __future__ import annotations

import torch
import pytest
from torch.utils.data import DataLoader
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.streaming import streaming_oneshot


def test_qwen3_pretrained_mode_hides_boundary_construction(tmp_path):
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )
    torch.manual_seed(4)
    checkpoint = tmp_path / "checkpoint"
    Qwen3ForCausalLM(config).save_pretrained(checkpoint, safe_serialization=True)
    dataset = DataLoader(
        [
            {
                "input_ids": torch.tensor([1, 2, 3, 4]),
                "attention_mask": torch.ones(4, dtype=torch.long),
            }
        ],
        batch_size=1,
    )

    output = streaming_oneshot(
        model=checkpoint,
        dataset=dataset,
        dataset_fingerprint="a" * 64,
        recipe=[
            IMatrixGatherer(ignore=["lm_head"]),
            QuantizationModifier(
                scheme="W8A8",
                targets=["Linear"],
                weight_observer="imatrix_mse",
                ignore=["lm_head"],
            ),
        ],
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        num_calibration_samples=1,
        max_seq_length=4,
        target_dtype=torch.float32,
    )

    assert (output / "FINALIZED").is_file()
    assert '"lm_head.weight"' not in (
        output / "model.safetensors.index.json"
    ).read_text()


@pytest.mark.parametrize(
    ("config", "model_class"),
    [
        (
            LlamaConfig(
                vocab_size=32,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=32,
                tie_word_embeddings=True,
            ),
            LlamaForCausalLM,
        ),
    ],
)
def test_pretrained_boundary_tracing_is_not_model_specific(
    tmp_path, config, model_class
):
    checkpoint = tmp_path / "checkpoint"
    model_class(config).save_pretrained(checkpoint, safe_serialization=True)
    dataset = DataLoader(
        [
            {
                "input_ids": torch.tensor([1, 2, 3, 4]),
                "attention_mask": torch.ones(4, dtype=torch.long),
            }
        ],
        batch_size=1,
    )

    output = streaming_oneshot(
        model=checkpoint,
        dataset=dataset,
        dataset_fingerprint="b" * 64,
        recipe=[
            IMatrixGatherer(ignore=["lm_head"]),
            QuantizationModifier(
                scheme="W8A8",
                targets=["Linear"],
                weight_observer="imatrix_mse",
                ignore=["lm_head"],
            ),
        ],
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        num_calibration_samples=1,
        max_seq_length=4,
        target_dtype=torch.float32,
    )

    assert (output / "FINALIZED").is_file()
