import importlib.util
import json
from pathlib import Path

from llmcompressor.modeling.deepseekv4.config import ModelConfig


MODULE_PATH = (
    Path(__file__).parents[2]
    / "examples"
    / "streaming_oneshot"
    / "deepseek_v4_wNa8.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location("deepseek_v4_wna8", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)


def test_transformers_config_schema_is_normalized_for_native_model():
    config = ModelConfig.from_dict(
        {
            "model_type": "deepseek_v4",
            "num_hidden_layers": 4,
            "num_nextn_predict_layers": 1,
            "layer_types": [
                "sliding_attention",
                "compressed_sparse_attention",
                "heavily_compressed_attention",
                "compressed_sparse_attention",
            ],
            "compress_rates": {
                "compressed_sparse_attention": 4,
                "heavily_compressed_attention": 128,
            },
            "mlp_layer_types": ["hash_moe", "hash_moe", "moe", "moe"],
            "rope_parameters": {
                "main": {"rope_theta": 10_000, "rope_type": "default"},
                "compress": {
                    "factor": 16,
                    "original_max_position_embeddings": 65_536,
                    "rope_theta": 160_000,
                    "rope_type": "yarn",
                },
            },
        }
    )

    assert config.compress_ratios == [0, 4, 128, 4]
    assert config.num_hash_layers == 2
    assert config.rope_theta == 10_000
    assert config.compress_rope_theta == 160_000
    assert config.rope_scaling["factor"] == 16


def test_absent_mtp_keys_disable_dspark_without_changing_main_layers():
    config = ModelConfig(
        num_hidden_layers=4,
        num_nextn_predict_layers=1,
        compress_ratios=[0, 4, 128, 4],
        dspark_block_size=5,
        dspark_target_layer_ids=[1, 2, 3],
    )

    disabled = MODULE._disable_absent_mtp(
        config,
        {"model.embed.weight", "model.layers.0.attn.wq_a.weight"},
    )

    assert disabled
    assert config.num_nextn_predict_layers == 0
    assert config.dspark_block_size == 0
    assert config.dspark_target_layer_ids == []
    assert config.compress_ratios == [0, 4, 128, 4]
    assert config.n_mtp_layers == 0


def test_present_mtp_keys_preserve_dspark_config():
    config = ModelConfig(
        num_hidden_layers=1,
        num_nextn_predict_layers=1,
        compress_ratios=[0, 0],
        dspark_block_size=5,
        dspark_target_layer_ids=[0],
    )

    disabled = MODULE._disable_absent_mtp(
        config,
        {"model.layers.0.attn.wq_a.weight", "model.mtp.0.main_proj.weight"},
    )

    assert not disabled
    assert config.dspark_block_size == 5
    assert config.n_mtp_layers == 1


def test_checkpoint_names_can_be_read_from_index_without_shards(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "embed.weight": "missing-00001.safetensors",
                    "layers.0.attn.wq_a.weight": "missing-00001.safetensors",
                }
            }
        )
    )

    names = MODULE._checkpoint_tensor_names(
        tmp_path, MODULE.DeepSeekV4WeightMaterializer()
    )

    assert names == {
        "embed.weight",
        "layers.0.attn.wq_a.weight",
    }
