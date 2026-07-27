from __future__ import annotations

import json
import struct

import pytest
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.recipe import Recipe
from llmcompressor.streaming import (
    DeepSeekV4WeightMaterializer,
    streaming_oneshot,
)
from llmcompressor.streaming.pretrained import _recipe_quantizer


def _write_raw_deepseek_v4_checkpoint(checkpoint):
    """Convert one tiny native shard to raw keys with one FP8+E8M0 weight."""
    tensors = load_file(checkpoint / "model.safetensors")
    encoded_name = "model.layers.0.attn.wq_a.weight"
    entries = []
    dtype_names = {
        torch.bfloat16: "BF16",
        torch.float32: "F32",
        torch.int64: "I64",
    }
    for name, tensor in tensors.items():
        raw_name = name.removeprefix("model.")
        if raw_name.startswith("lm_head."):
            raw_name = f"head.{raw_name.removeprefix('lm_head.')}"
        if name == encoded_name:
            encoded = tensor.to(torch.float8_e4m3fn).contiguous()
            entries.append(
                (
                    raw_name,
                    "F8_E4M3",
                    tuple(encoded.shape),
                    encoded.view(torch.uint8).numpy().tobytes(),
                )
            )
            scale = torch.full(
                (
                    (tensor.shape[0] + 127) // 128,
                    (tensor.shape[1] + 127) // 128,
                ),
                127,
                dtype=torch.uint8,
            )
            entries.append(
                (
                    f"{raw_name.removesuffix('.weight')}.scale",
                    "F8_E8M0",
                    tuple(scale.shape),
                    scale.numpy().tobytes(),
                )
            )
            continue
        dtype_name = dtype_names[tensor.dtype]
        storage = (
            tensor.view(torch.uint16)
            if tensor.dtype == torch.bfloat16
            else tensor
        )
        entries.append(
            (raw_name, dtype_name, tuple(tensor.shape), storage.numpy().tobytes())
        )

    header = {}
    payload = bytearray()
    for name, dtype, shape, storage in entries:
        start = len(payload)
        payload.extend(storage)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    encoded_header = json.dumps(header, separators=(",", ":")).encode()
    encoded_header += b" " * ((8 - len(encoded_header) % 8) % 8)
    (checkpoint / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded_header)) + encoded_header + payload
    )
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    config["model_type"] = "deepseek_v4"
    config["architectures"] = ["DeepseekV4ForCausalLM"]
    config_path.write_text(json.dumps(config))


def test_pretrained_explains_autoround_output_adapter_requirement():
    from llmcompressor.modifiers.autoround import AutoRoundModifier

    recipe = Recipe.from_modifiers(AutoRoundModifier(scheme="W4A16"))
    with pytest.raises(ValueError, match="dedicated streaming output adapter"):
        _recipe_quantizer(recipe)


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


def test_deepseek_v4_preserves_input_ids_and_collects_all_experts(tmp_path):
    from llmcompressor.modeling.deepseekv4.config import ModelConfig
    from llmcompressor.modeling.deepseekv4.model import (
        DeepseekV4NativeForCausalLM,
    )

    config = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_hash_layers=1,
        num_nextn_predict_layers=1,
        num_attention_heads=2,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        q_lora_rank=8,
        head_dim=8,
        qk_rope_head_dim=2,
        o_groups=2,
        o_lora_rank=4,
        sliding_window=4,
        max_position_embeddings=16,
        max_seq_len=16,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        hc_mult=2,
        compress_ratios=[0, 0],
    )
    checkpoint = tmp_path / "checkpoint"
    DeepseekV4NativeForCausalLM(config).save_pretrained(
        checkpoint, safe_serialization=True
    )
    _write_raw_deepseek_v4_checkpoint(checkpoint)
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
        model_config=ModelConfig.from_pretrained(checkpoint),
        dataset=dataset,
        dataset_fingerprint="c" * 64,
        recipe=[
            IMatrixGatherer(ignore=["model.lm_head"]),
            QuantizationModifier(
                scheme="W8A8",
                targets=["Linear"],
                weight_observer="imatrix_mse",
                ignore=["model.lm_head"],
            ),
        ],
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        num_calibration_samples=1,
        max_seq_length=4,
        target_dtype=torch.float32,
        materializer=DeepSeekV4WeightMaterializer(
            fp8_block_size=(128, 128), fp4_block_size=32
        ),
    )

    assert (output / "FINALIZED").is_file()
    assert not list(
        (tmp_path / "work" / "artifacts" / "statistics").glob(
            "target-*"
        )
    )
    assert not (tmp_path / "work" / "staging").exists()
    assert (output / "model-subgraph-00000.safetensors").is_file()
    assert (output / "model-subgraph-00001.safetensors").is_file()
    loaded = AutoModelForCausalLM.from_pretrained(
        output, local_files_only=True, dtype=torch.float32
    ).eval()
    quantized = loaded.model.layers[0].attn.wq_a
    assert quantized.weight.dtype == torch.int8
    assert hasattr(quantized, "weight_scale")
    with torch.no_grad():
        logits = loaded(input_ids=torch.tensor([[1, 2, 3, 4]])).logits
    assert torch.isfinite(logits).all()
