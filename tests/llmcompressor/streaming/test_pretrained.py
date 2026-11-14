from __future__ import annotations

import json
import struct
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from llmcompressor.modeling.deepseekv41 import (
    DeepseekV41NativeForCausalLM,
)
from llmcompressor.modeling.deepseekv41 import (
    ModelConfig as DeepseekV41ModelConfig,
)
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.pipelines.sequential.helpers import trace_graph
from llmcompressor.recipe import Recipe
from llmcompressor.streaming import (
    DeepSeekV4WeightMaterializer,
    streaming_oneshot,
)
from llmcompressor.streaming.pretrained import (
    _cleanup_successful_work_dir,
    _has_expert_collection,
    _quantized_sequential_targets,
    _recipe_quantizer,
)
from llmcompressor.streaming.tracing import trace_streaming_boundaries
from llmcompressor.utils import ImatrixFallbackStats


class _VisionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)


class _DecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(2, 2)
        self.k_proj = torch.nn.Linear(2, 2)


class _AttentionDecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = _Attention()
        self.mlp = torch.nn.Linear(2, 2)


class _MultimodalModel(torch.nn.Module):
    _no_split_modules = ["_VisionBlock", "_DecoderLayer", "Linear"]

    def __init__(self):
        super().__init__()
        self.vision_tower = torch.nn.ModuleList([_VisionBlock()])
        self.layers = torch.nn.ModuleList([_DecoderLayer(), _DecoderLayer()])


class _AttentionModel(torch.nn.Module):
    _no_split_modules = ["_AttentionDecoderLayer"]

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_AttentionDecoderLayer()])


class _ExpertModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = torch.nn.ModuleList([_DecoderLayer(), _DecoderLayer()])


class _ExpertBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Linear(2, 2)
        self.w2 = torch.nn.Linear(2, 2)
        self.w3 = torch.nn.Linear(2, 2)


class _ExpertOwnerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "experts": torch.nn.ModuleList(
                            [_ExpertBlock(), _ExpertBlock()]
                        ),
                        "shared_experts": _ExpertBlock(),
                    }
                )
            ]
        )


def _tiny_deepseek_v41_config():
    return DeepseekV41ModelConfig(
        text_config=dict(
            vocab_size=32,
            hidden_size=32,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_nextn_predict_layers=1,
            num_attention_heads=2,
            n_routed_experts=4,
            num_experts_per_tok=2,
            q_lora_rank=32,
            head_dim=32,
            qk_rope_head_dim=8,
            o_groups=2,
            o_lora_rank=16,
            sliding_window=4,
            max_position_embeddings=64,
            max_seq_len=64,
            index_n_heads=2,
            index_head_dim=32,
            index_topk=2,
            hc_mult=2,
            compress_ratios=[0, 2, 0],
            kv_source_layer_ids=[1],
            index_source_layer_ids=[1],
            candidate_source_layer_id=1,
            candidate_topk_blocks=2,
            candidate_block_size=2,
            engram_layer_ids=[],
            engram_num_embeddings=[],
            engram_vocab_size=5,
            engram_n_heads=1,
            engram_head_dim=32,
            engram_max_ngram_size=2,
            engram_compressed_vocab_size=32,
            dspark_block_size=3,
            dspark_noise_token_id=31,
            dspark_target_layer_ids=[1],
            dspark_markov_rank=8,
            dspark_n_routed_experts=2,
            dspark_num_experts_per_tok=1,
        ),
        vision_config=dict(
            num_hidden_layers=0,
            hidden_size=32,
            num_attention_heads=2,
            intermediate_size=32,
            patch_size=2,
            downsample_ratio=2,
        ),
    )


def _write_raw_deepseek_v4_checkpoint(checkpoint, *, omit_names=()):
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
        if name in omit_names:
            continue
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


def _tiny_deepseek_v4_config():
    from llmcompressor.modeling.deepseekv4.config import ModelConfig

    return ModelConfig(
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


def _tiny_deepseek_v4_dspark_config():
    from llmcompressor.modeling.deepseekv4.config import ModelConfig

    return ModelConfig(
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=3,
        num_hash_layers=3,
        num_nextn_predict_layers=1,
        num_attention_heads=2,
        n_routed_experts=8,
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
        compress_ratios=[0, 0, 0, 0, 0, 0],
        dspark_block_size=3,
        dspark_noise_token_id=31,
        dspark_target_layer_ids=[0, 1, 2],
        dspark_markov_rank=4,
    )


def test_pretrained_autoround_requires_explicit_pipeline_ownership():
    autoround = pytest.importorskip("llmcompressor.modifiers.autoround")
    modifier = autoround.AutoRoundModifier(
        scheme="W4A16", enable_torch_compile=False
    )
    recipe = Recipe.from_modifiers(modifier)
    assert _recipe_quantizer(recipe) is modifier
    modifier.device_ids = "0,1"
    with pytest.raises(ValueError, match="placement belongs to the pipeline"):
        _recipe_quantizer(recipe)


def test_sequential_targets_only_cover_quantized_model_branches():
    model = _MultimodalModel()
    schemes = {
        "layers.0.proj": Mock(),
        "layers.1.proj": Mock(),
    }

    assert _quantized_sequential_targets(model, schemes) == (
        "layers.0",
        "layers.1",
    )


def test_sequential_targets_include_registered_glm5_mtp_layer():
    from llmcompressor.modeling.glm5_next import (
        Glm5NextMTPLayer,
        _register_mtp_no_split_module,
    )

    model = _MultimodalModel()
    mtp = Glm5NextMTPLayer.__new__(Glm5NextMTPLayer)
    torch.nn.Module.__init__(mtp)
    mtp.proj = torch.nn.Linear(2, 2)
    model.layers.append(mtp)
    _register_mtp_no_split_module(model)
    schemes = {
        "layers.0.proj": Mock(),
        "layers.1.proj": Mock(),
        "layers.2.proj": Mock(),
    }

    assert _quantized_sequential_targets(model, schemes) == (
        "layers.0",
        "layers.1",
        "layers.2",
    )


def test_gptq_targets_split_quantized_modules_within_decoder():
    model = _MultimodalModel()
    schemes = {
        "layers.0.proj": Mock(),
        "layers.1.proj": Mock(),
    }

    assert _quantized_sequential_targets(
        model, schemes, sequential_targets="Linear"
    ) == (
        "layers.0.proj",
        "layers.1.proj",
    )


def test_gptq_targets_group_moe_modules_by_expert_owner():
    model = _ExpertOwnerModel()
    schemes = {
        "layers.0.experts.0.w1": Mock(),
        "layers.0.experts.0.w2": Mock(),
        "layers.0.experts.0.w3": Mock(),
        "layers.0.experts.1.w1": Mock(),
        "layers.0.experts.1.w2": Mock(),
        "layers.0.experts.1.w3": Mock(),
        "layers.0.shared_experts.w1": Mock(),
        "layers.0.shared_experts.w2": Mock(),
        "layers.0.shared_experts.w3": Mock(),
    }

    assert _quantized_sequential_targets(
        model,
        schemes,
        sequential_targets="Linear",
        group_moe_experts=True,
    ) == (
        "layers.0.experts.0",
        "layers.0.experts.1",
        "layers.0.shared_experts",
    )


def test_gptq_targets_promote_attention_leaf():
    model = _AttentionModel()
    schemes = {
        "layers.0.attention.q_proj": Mock(),
        "layers.0.mlp": Mock(),
    }

    assert _quantized_sequential_targets(
        model, schemes, sequential_targets="Linear"
    ) == (
        "layers.0.attention",
        "layers.0.mlp",
    )


def test_gptq_targets_promote_deepseek_v41_dspark_attention_leaf():
    model = DeepseekV41NativeForCausalLM(_tiny_deepseek_v41_config())
    layer = model.model.layers[0]
    schemes = {
        "model.layers.0.attn.wq_a": Mock(),
        "model.layers.0.ffn.experts.0.w1": Mock(),
        "model.layers.0.ffn.experts.0.w2": Mock(),
        "model.layers.0.ffn.experts.0.w3": Mock(),
        "model.layers.0.ffn.experts.1.w1": Mock(),
        "model.layers.0.ffn.experts.1.w2": Mock(),
        "model.layers.0.ffn.experts.1.w3": Mock(),
        "model.mtp.0.attn.wq_a": Mock(),
        "model.mtp.0.ffn.experts.0.w1": Mock(),
        "model.mtp.0.ffn.experts.0.w2": Mock(),
        "model.mtp.0.ffn.experts.0.w3": Mock(),
        "model.mtp.0.ffn.experts.1.w1": Mock(),
        "model.mtp.0.ffn.experts.1.w2": Mock(),
        "model.mtp.0.ffn.experts.1.w3": Mock(),
    }

    targets = _quantized_sequential_targets(
        model,
        schemes,
        sequential_targets="Linear",
        group_moe_experts=True,
    )

    assert "model.layers.0.attn" in targets
    assert "model.layers.0.ffn.experts.0" in targets
    assert "model.mtp.0.attn" in targets
    assert "model.mtp.0.ffn.experts.0" in targets
    assert "model.layers.0" not in targets
    assert layer.attn.__class__.__name__ == "Attention"
    assert model.model.mtp[0].attn.__class__.__name__ == "DSparkAttention"


def test_deepseek_v41_trace_keeps_all_attention_calls_visible():
    model = DeepseekV41NativeForCausalLM(_tiny_deepseek_v41_config()).eval()
    schemes = {
        name: Mock()
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    targets = _quantized_sequential_targets(
        model,
        schemes,
        sequential_targets="Linear",
        group_moe_experts=True,
    )
    graph, _ = trace_graph(
        model,
        {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        },
        list(targets),
        [],
    )

    attention_calls = {
        str(node.target)
        for node in graph.graph.nodes
        if node.op == "call_module" and str(node.target).endswith(".attn")
    }

    assert {"model.layers.0.attn", "model.layers.1.attn", "model.mtp.0.attn"} <= (
        attention_calls
    )


def test_expert_collection_detection_is_model_agnostic():
    assert not _has_expert_collection(_MultimodalModel())
    assert _has_expert_collection(_ExpertModel())


def test_successful_work_cleanup_preserves_unknown_files(tmp_path):
    work = tmp_path / "work"
    for name in ("artifacts", "boundaries", "publish", "staging"):
        directory = work / name
        directory.mkdir(parents=True)
        (directory / "owned").write_text("owned")
    unknown = work / "user-note.txt"
    unknown.write_text("keep")

    _cleanup_successful_work_dir(work)

    assert unknown.read_text() == "keep"
    assert sorted(path.name for path in work.iterdir()) == ["user-note.txt"]


def test_qwen3_pretrained_mode_hides_boundary_construction(tmp_path, monkeypatch):
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

    trace_boundaries = Mock(wraps=trace_streaming_boundaries)
    monkeypatch.setattr(
        "llmcompressor.streaming.pretrained.trace_streaming_boundaries",
        trace_boundaries,
    )
    int4_packer = Mock(side_effect=AssertionError("W8A8 must not use INT4 packing"))
    monkeypatch.setattr(
        "llmcompressor.streaming.output.pack_int4_to_int8_cpu_snapshot",
        int4_packer,
    )
    output = streaming_oneshot(
        model=checkpoint,
        dataset=dataset,
        dataset_fingerprint="a" * 64,
        recipe=[
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
    assert (
        "_update_linear_attn_mask"
        in trace_boundaries.call_args.kwargs["tracing_ignore"]
    )
    assert trace_boundaries.call_args.kwargs["model"].training is False
    assert trace_boundaries.call_args.kwargs["sequential_targets"] == (
        "model.layers.0",
        "model.layers.1",
    )
    assert not (tmp_path / "work").exists()
    int4_packer.assert_not_called()
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


@pytest.mark.parametrize(
    ("save_raw_checkpoint_format", "checkpoint_progress"),
    [(False, False), (True, False), (True, True)],
)
def test_deepseek_v4_preserves_input_ids_and_collects_all_experts(
    tmp_path, save_raw_checkpoint_format, checkpoint_progress
):
    from llmcompressor.modeling.deepseekv4.config import ModelConfig
    from llmcompressor.modeling.deepseekv4.model import (
        DeepseekV4NativeForCausalLM,
    )

    config = _tiny_deepseek_v4_config()
    checkpoint = tmp_path / "checkpoint"
    source_model = DeepseekV4NativeForCausalLM(config)
    source_model.save_pretrained(
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

    stats = ImatrixFallbackStats()
    stats.install_hooks()
    with stats:
        output = streaming_oneshot(
            model=checkpoint,
            model_config=ModelConfig.from_pretrained(checkpoint),
            dataset=dataset,
            dataset_fingerprint="c" * 64,
            recipe=[
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
                fp8_block_size=(128, 128),
                fp4_block_size=32,
                save_raw_checkpoint_format=save_raw_checkpoint_format,
            ),
            checkpoint_progress=checkpoint_progress,
        )

    assert not {
        name for name in stats.all_zero_counter if name.startswith("model.mtp.0")
    }

    assert (output / "FINALIZED").is_file()
    assert not (tmp_path / "work").exists()
    if checkpoint_progress:
        assert list(output.glob("*.safetensors"))
    else:
        for index in range(2):
            assert (
                output / f"model-subgraph-{index:05d}.safetensors"
            ).is_file()
    output_config = json.loads((output / "config.json").read_text())
    assert output_config["architectures"] == ["DeepseekV4ForCausalLM"]
    assert output_config["model_type"] == "deepseek_v4"
    if save_raw_checkpoint_format:
        weight_map = json.loads(
            (output / "model.safetensors.index.json").read_text()
        )["weight_map"]
        assert "mtp.0.ffn.experts.0.w1.weight" in weight_map
        assert "mtp.0.ffn.experts.0.w1.scale" in weight_map
        assert "mtp.0.ffn.shared_experts.w1.weight" in weight_map
        assert "mtp.0.ffn.shared_experts.w1.scale" in weight_map
        assert "head.weight" in weight_map
        assert not any(name.startswith("model.mtp.") for name in weight_map)
        assert "model.lm_head.weight" not in weight_map
        return

    loaded = DeepseekV4NativeForCausalLM.from_pretrained(
        output,
        config=ModelConfig.from_pretrained(output),
        local_files_only=True,
        dtype=torch.float32,
    ).eval()
    assert torch.equal(loaded.model.embed.weight, source_model.model.embed.weight)
    assert loaded.model.mtp[0].embed.weight is loaded.model.embed.weight
    quantized = loaded.model.layers[0].attn.wq_a
    assert quantized.weight.dtype == torch.int8
    assert hasattr(quantized, "weight_scale")
    with torch.no_grad():
        logits = loaded(input_ids=torch.tensor([[1, 2, 3, 4]])).logits
    assert torch.isfinite(logits).all()


def test_deepseek_v4_missing_mtp_norms_fail_before_calibration(tmp_path):
    from llmcompressor.modeling.deepseekv4.model import (
        DeepseekV4NativeForCausalLM,
    )

    config = _tiny_deepseek_v4_config()
    checkpoint = tmp_path / "checkpoint"
    DeepseekV4NativeForCausalLM(config).save_pretrained(
        checkpoint, safe_serialization=True
    )
    _write_raw_deepseek_v4_checkpoint(
        checkpoint,
        omit_names={
            "model.mtp.0.enorm.weight",
            "model.mtp.0.hnorm.weight",
        },
    )
    dataset = DataLoader(
        [{"input_ids": torch.tensor([1, 2, 3, 4])}], batch_size=1
    )

    with pytest.raises(KeyError, match="model.mtp.0.(enorm|hnorm).weight"):
        streaming_oneshot(
            model=checkpoint,
            model_config=config,
            dataset=dataset,
            dataset_fingerprint="m" * 64,
            recipe=[
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
            materializer=DeepSeekV4WeightMaterializer(),
        )


def test_deepseek_v4_dspark_forward_teacher_forces_valid_seed_and_context():
    from llmcompressor.modeling.deepseekv4.model_dspark import (
        DeepseekV4DSparkForCausalLM,
    )

    config = _tiny_deepseek_v4_dspark_config()
    model = DeepseekV4DSparkForCausalLM(config).eval()
    target_outputs = {layer_id: [] for layer_id in config.dspark_target_layer_ids}
    projected_contexts = []
    context_masks = []
    draft_ids = []
    handles = []

    def capture_target(layer_id):
        def hook(_module, _args, output):
            target_outputs[layer_id].append(output[0].mean(dim=2).detach())

        return hook

    for layer_id in config.dspark_target_layer_ids:
        handles.append(
            model.model.layers[layer_id].register_forward_hook(
                capture_target(layer_id)
            )
        )
    handles.append(
        model.model.mtp[0].main_proj.register_forward_pre_hook(
            lambda _module, args: projected_contexts.append(args[0].detach())
        )
    )
    handles.append(
        model.model.mtp[0].attn.register_forward_pre_hook(
            lambda _module, args: context_masks.append(args[2].detach())
        )
    )
    handles.append(
        model.model.mtp[0].ffn.register_forward_pre_hook(
            lambda _module, args: draft_ids.append(args[1].detach())
        )
    )

    try:
        with torch.no_grad():
            unpadded = model(
                input_ids=torch.tensor([[1, 2, 3, 4]]),
                attention_mask=torch.ones((1, 4), dtype=torch.long),
            ).logits
            padded = model(
                input_ids=torch.tensor([[1, 2, 3, 4, 7, 7]]),
                attention_mask=torch.tensor([[1, 1, 1, 1, 0, 0]]),
            ).logits
    finally:
        for handle in handles:
            handle.remove()

    expected_context = torch.cat(
        [
            target_outputs[layer_id][0][:, [0, 0, 1, 2]]
            for layer_id in config.dspark_target_layer_ids
        ],
        dim=-1,
    )
    assert torch.equal(draft_ids[0], torch.tensor([[4, 31, 31]]))
    assert torch.equal(draft_ids[1], draft_ids[0])
    assert torch.equal(
        context_masks[0], torch.tensor([[False, True, True, True]])
    )
    assert torch.equal(context_masks[1], context_masks[0])
    assert torch.allclose(projected_contexts[0], expected_context)
    assert torch.allclose(projected_contexts[1], projected_contexts[0])
    assert torch.allclose(padded[:, :4], unpadded)


def test_deepseek_v4_0731_dspark_streaming_checkpoint_contract(tmp_path):
    from llmcompressor.modeling.deepseekv4.model_dspark import (
        DeepseekV4DSparkForCausalLM,
    )

    config = _tiny_deepseek_v4_dspark_config()
    checkpoint = tmp_path / "checkpoint"
    source_model = DeepseekV4DSparkForCausalLM(config)
    source_model.save_pretrained(checkpoint, safe_serialization=True)
    _write_raw_deepseek_v4_checkpoint(checkpoint)
    dataset = DataLoader(
        [
            {
                # Exercise the same right padding used by calibration datasets.
                "input_ids": torch.tensor([1, 2, 3, 4, 0, 0]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 0, 0]),
            }
        ],
        batch_size=1,
    )
    ignored = [
        "model.lm_head",
        "model.mtp.2.confidence_head.proj",
    ]
    stats = ImatrixFallbackStats()
    stats.install_hooks()
    with stats:
        output = streaming_oneshot(
            model=checkpoint,
            model_config=config,
            model_factory=DeepseekV4DSparkForCausalLM,
            dataset=dataset,
            dataset_fingerprint="d" * 64,
            recipe=[
                QuantizationModifier(
                    scheme="W8A8",
                    targets=["Linear"],
                    weight_observer="imatrix_mse",
                    ignore=ignored,
                ),
            ],
            output_dir=tmp_path / "output",
            work_dir=tmp_path / "work",
            num_calibration_samples=1,
            max_seq_length=6,
            moe_calibrate_all_experts=False,
            target_dtype=torch.float32,
            materializer=DeepSeekV4WeightMaterializer(
                fp8_block_size=(128, 128),
                fp4_block_size=32,
                save_raw_checkpoint_format=True,
            ),
        )

    assert config.n_mtp_layers == 3
    assert not {
        name
        for name in stats.all_zero_counter
        if name.startswith("model.mtp.")
    }
    assert not {
        name
        for name in stats.no_importance_counter
        if name.startswith("model.mtp.")
    }
    weight_map = json.loads(
        (output / "model.safetensors.index.json").read_text()
    )["weight_map"]
    assert "mtp.0.main_proj.weight" in weight_map
    assert "mtp.1.attn.wq_a.weight" in weight_map
    assert "mtp.2.markov_head.markov_w1.weight" in weight_map
    assert "mtp.2.confidence_head.proj.weight" in weight_map
    assert not any("mtp.0.embed" in name for name in weight_map)
    output_config = json.loads((output / "config.json").read_text())
    assert output_config["architectures"] == ["DeepseekV4ForCausalLM"]
    assert output_config["model_type"] == "deepseek_v4"
