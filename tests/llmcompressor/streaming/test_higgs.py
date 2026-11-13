from __future__ import annotations

import json

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from safetensors.torch import save_file

from llmcompressor.streaming import SafetensorsWeightSource
from llmcompressor.streaming.higgs import StreamingHiggsConfigProvider


def _scheme(bits: int) -> QuantizationScheme:
    return QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=bits,
            type="int",
            strategy="tensor",
            symmetric=True,
        ),
    )


def test_streaming_higgs_reads_only_current_logical_group(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    first = {
        "model.layers.0.mlp.experts.gate_up_proj.weight": torch.randn(2, 8, 4),
        "model.layers.0.mlp.experts.down_proj.weight": torch.randn(2, 4, 4),
    }
    second = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(4, 4),
        "model.lm_head.weight": torch.randn(4, 4),
    }
    save_file(first, checkpoint / "model-00001-of-00002.safetensors")
    save_file(second, checkpoint / "model-00002-of-00002.safetensors")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    **{
                        name: "model-00001-of-00002.safetensors"
                        for name in first
                    },
                    **{
                        name: "model-00002-of-00002.safetensors"
                        for name in second
                    },
                }
            }
        )
    )

    source = SafetensorsWeightSource(checkpoint)
    reads = []
    original_load = source.load_tensors_cpu

    def record_load(names):
        names = tuple(names)
        reads.append(names)
        return original_load(names)

    source.load_tensors_cpu = record_load
    provider = StreamingHiggsConfigProvider(
        [_scheme(4), _scheme(8)],
        ignore=["lm_head"],
        target_dtype=torch.float32,
        device="cpu",
    )
    collector = provider.collect(source=source)

    assert set(collector.mse_matrix) == {
        "model.layers.0.mlp.experts.0.gate_proj",
        "model.layers.0.mlp.experts.0.up_proj",
        "model.layers.0.mlp.experts.0.down_proj",
        "model.layers.0.mlp.experts.1.gate_proj",
        "model.layers.0.mlp.experts.1.up_proj",
        "model.layers.0.mlp.experts.1.down_proj",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
    }
    assert all("model.lm_head.weight" not in names for names in reads)
    assert set().union(*map(set, reads)) == {
        "model.layers.0.mlp.experts.gate_up_proj.weight",
        "model.layers.0.mlp.experts.down_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
    }
    assert all(len(names) <= 2 for names in reads)
    assert len(provider.config_fingerprint) == 64
