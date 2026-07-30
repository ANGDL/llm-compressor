from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from torch.utils.data import DataLoader

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.streaming import KimiK3WeightMaterializer, streaming_oneshot
from tools.create_tiny_kimi_k3 import create_tiny_checkpoint, kda_import_stubs

MODULE_PATH = (
    Path(__file__).parents[3] / "examples" / "streaming_oneshot" / "kimi_k3_wNa8.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location("kimi_k3_wna8_e2e", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
KIMI_K3_WNA8 = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(KIMI_K3_WNA8)


def _load_output_tensor(output: Path, weight_map: dict[str, str], name: str):
    with safe_open(output / weight_map[name], framework="pt", device="cpu") as file:
        return file.get_tensor(name)


@pytest.mark.integration
def test_tiny_kimi_k3_streaming_wna8_end_to_end(tmp_path):
    source_value = os.environ.get("KIMI_K3_SOURCE")
    if source_value is None:
        pytest.skip("set KIMI_K3_SOURCE to a checkpoint-native Kimi K3 directory")
    source = Path(source_value)
    checkpoint = tmp_path / "tiny"

    with kda_import_stubs():
        create_tiny_checkpoint(source, checkpoint)
        config = KIMI_K3_WNA8._register_k3_model(checkpoint)
        expert_pattern, expert_targets = KIMI_K3_WNA8._expert_targets_from_index(
            checkpoint
        )
        other_pattern, other_targets = KIMI_K3_WNA8._other_targets_from_index(
            checkpoint, expert_targets
        )
        ignores = [
            r"re:^vision_tower(?:\.|$)",
            r"re:^mm_projector(?:\.|$)",
            r"re:.*\.embed_tokens$",
            r"re:.*\.lm_head$",
        ]
        recipe = [
            IMatrixGatherer(ignore=ignores),
            QuantizationModifier(
                config_groups={
                    "routed_experts_w4a8": KIMI_K3_WNA8._int_scheme(
                        4, [expert_pattern]
                    ),
                    "other_linear_w8a8": KIMI_K3_WNA8._int_scheme(8, [other_pattern]),
                },
                ignore=ignores,
            ),
        ]
        image_patches = torch.linspace(
            -1.0,
            1.0,
            steps=4 * 3 * 14 * 14,
            dtype=torch.float32,
        ).reshape(4, 3, 14, 14)
        dataset = DataLoader(
            [
                {
                    "input_ids": torch.tensor([[1, 3, 4, 2]]),
                    "attention_mask": torch.ones((1, 4), dtype=torch.long),
                    "pixel_values": image_patches,
                    "grid_thws": torch.tensor([[1, 2, 2]], dtype=torch.long),
                }
            ],
            batch_size=1,
            collate_fn=lambda batch: batch[0],
        )
        output = streaming_oneshot(
            model=checkpoint,
            model_config=config,
            dataset=dataset,
            dataset_fingerprint="k" * 64,
            recipe=recipe,
            output_dir=tmp_path / "output",
            work_dir=tmp_path / "work",
            num_calibration_samples=1,
            max_seq_length=4,
            batch_size=1,
            shuffle_calibration_samples=False,
            moe_calibrate_all_experts=False,
            materializer=KimiK3WeightMaterializer(
                kda_num_heads=KIMI_K3_WNA8._kda_num_heads(config)
            ),
            target_dtype=torch.float32,
            checkpoint_progress=True,
            pack_to_int8=True,
            overwrite_output=True,
        )

    assert (output / "FINALIZED").is_file()
    output_config = json.loads((output / "config.json").read_text())
    weight_map = json.loads((output / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    groups = output_config["quantization_config"]["config_groups"]
    for name, bits in (("routed_experts_w4a8", 4), ("other_linear_w8a8", 8)):
        scheme = groups[name]
        assert scheme["weights"]["num_bits"] == bits
        assert scheme["weights"]["strategy"] == "channel"
        assert scheme["weights"]["symmetric"] is True
        assert scheme["input_activations"]["strategy"] == "token"
        assert scheme["input_activations"]["symmetric"] is True
        assert scheme["input_activations"]["dynamic"] is True

    expert_name = sorted(expert_targets)[0] + ".weight"
    expert_weight = _load_output_tensor(output, weight_map, expert_name)
    assert expert_weight.dtype == torch.int8
    assert expert_weight.shape[-1] == 8
    assert expert_name.removesuffix(".weight") + ".weight_scale" in weight_map

    shared_name = (
        "language_model.model.layers.1.block_sparse_moe.shared_experts.gate_proj.weight"
    )
    assert shared_name.removesuffix(".weight") in other_targets
    assert _load_output_tensor(output, weight_map, shared_name).dtype == torch.int8

    tail_name = "language_model.model.output_attn_res_proj.weight"
    assert _load_output_tensor(output, weight_map, tail_name).dtype == torch.float32
    assert tail_name.removesuffix(".weight") + ".weight_scale" not in weight_map
