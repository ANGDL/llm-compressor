"""Create a tiny checkpoint using Kimi K3's checkpoint-native model code.

The generated checkpoint keeps one dense layer, one latent-MoE layer, routed
and shared experts, MLA, and attention residuals. Routed experts are saved in
K3's native MXFP4 ``weight_packed`` + ``weight_scale`` layout so streaming PTQ
exercises the real K3 materializer before producing W4A8 output.

Example:

    HF_HOME=/tmp/kimi-k3-hf-cache python tools/create_tiny_kimi_k3.py \
        --source /Users/ang/models/K3 \
        --output /tmp/Kimi-K3-Tiny
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import torch
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from llmcompressor.modeling.kimi_k3 import patch_kimi_k3_transformers_compat

_MODEL_FILES = (
    "configuration_kimi_k3.py",
    "modeling_kimi_k3.py",
    "modeling_kimi_linear.py",
)
_FP4_TABLE = torch.tensor(
    (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ),
    dtype=torch.float32,
)


@contextmanager
def kda_import_stubs():
    """Provide import-only FLA/einops symbols for a config with no KDA layers."""
    module_names = (
        "einops",
        "fla",
        "fla.modules",
        "fla.ops",
        "fla.ops.kda",
        "fla.ops.utils",
        "fla.ops.utils.index",
        "fla.utils",
    )
    previous = {name: sys.modules.get(name) for name in module_names}
    try:
        einops = ModuleType("einops")

        def unused_rearrange(*args, **kwargs):
            raise AssertionError("Tiny full-attention K3 unexpectedly used KDA/einops")

        einops.rearrange = unused_rearrange
        sys.modules["einops"] = einops
        for name in module_names[1:]:
            sys.modules[name] = ModuleType(name)

        class UnusedKdaModule:
            def __init__(self, *args, **kwargs):
                raise AssertionError("Tiny full-attention K3 unexpectedly used KDA")

        sys.modules["fla.modules"].FusedRMSNormGated = UnusedKdaModule
        sys.modules["fla.modules"].ShortConvolution = UnusedKdaModule
        sys.modules["fla.ops.kda"].chunk_kda = unused_rearrange
        sys.modules["fla.ops.kda"].fused_recurrent_kda = unused_rearrange
        sys.modules[
            "fla.ops.utils.index"
        ].prepare_cu_seqlens_from_mask = unused_rearrange
        sys.modules["fla.ops.utils.index"].prepare_lens_from_mask = unused_rearrange
        sys.modules["fla.utils"].tensor_cache = lambda function: function
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _tiny_config(source: Path) -> dict:
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    text = config["text_config"]
    text.update(
        {
            "_attn_implementation": "eager",
            "attn_res_block_size": 2,
            "bos_token_id": 1,
            "dtype": "float32",
            "eos_token_id": 2,
            "first_k_dense_replace": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "kv_lora_rank": 8,
            "max_position_embeddings": 64,
            "moe_intermediate_size": 16,
            "num_attention_heads": 4,
            "num_experts": 2,
            "num_experts_per_token": 1,
            "num_hidden_layers": 2,
            "num_key_value_heads": 4,
            "num_shared_experts": 1,
            "pad_token_id": 0,
            "q_lora_rank": 16,
            "qk_nope_head_dim": 4,
            "qk_rope_head_dim": 4,
            "routed_expert_hidden_size": 16,
            "use_cache": False,
            "v_head_dim": 8,
            "vocab_size": 64,
        }
    )
    text["linear_attn_config"] = {
        "full_attn_layers": [1, 2],
        "gate_lower_bound": -5.0,
        "head_dim": 8,
        "kda_layers": [],
        "num_heads": 4,
        "short_conv_kernel_size": 4,
        "use_full_rank_gate": True,
    }
    text.pop("quantization_config", None)

    config.update(
        {
            "bos_token_id": 1,
            "dtype": "float32",
            "eos_token_id": 2,
            "media_placeholder_token_id": 3,
            "pad_token_id": 0,
        }
    )
    config["vision_config"].update(
        {
            "_attn_implementation": "eager",
            "init_pos_emb_height": 4,
            "init_pos_emb_time": 1,
            "init_pos_emb_width": 4,
            "mm_hidden_size": 16,
            "qkv_hidden_size": 24,
            "text_hidden_size": 32,
            "vt_hidden_size": 16,
            "vt_intermediate_size": 32,
            "vt_num_attention_heads": 2,
            "vt_num_hidden_layers": 1,
        }
    )
    return config


def _pack_mxfp4(weight: torch.Tensor, block_size: int = 32):
    weight = weight.float()
    out_features, in_features = weight.shape
    if in_features % 2:
        raise ValueError(f"MXFP4 requires an even input dimension, got {in_features}")
    scales = torch.empty(
        (out_features, (in_features + block_size - 1) // block_size),
        dtype=torch.uint8,
    )
    codes = torch.empty_like(weight, dtype=torch.uint8)
    table = _FP4_TABLE.to(weight.device)
    for block_index, start in enumerate(range(0, in_features, block_size)):
        block = weight[:, start : start + block_size]
        maximum = block.abs().amax(dim=1)
        exponent = torch.ceil(torch.log2((maximum / 6.0).clamp_min(2.0**-126)))
        exponent = exponent.clamp(-126, 127).to(torch.int32)
        scale = torch.pow(2.0, exponent.float()).unsqueeze(1)
        normalized = block / scale
        codes[:, start : start + block.shape[1]] = (
            (normalized.unsqueeze(-1) - table).abs().argmin(dim=-1).to(torch.uint8)
        )
        scales[:, block_index] = (exponent + 127).to(torch.uint8)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.contiguous(), scales.contiguous()


def create_tiny_checkpoint(source: Path, output: Path, seed: int = 42) -> Path:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    missing = [
        name for name in ("config.json", *_MODEL_FILES) if not (source / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"K3 source is missing required files: {missing}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    for name in _MODEL_FILES:
        shutil.copy2(source / name, output / name)
    config_dict = _tiny_config(source)
    (output / "config.json").write_text(
        json.dumps(config_dict, indent=2) + "\n", encoding="utf-8"
    )

    patch_kimi_k3_transformers_compat()
    with kda_import_stubs():
        config = AutoConfig.from_pretrained(
            output, trust_remote_code=True, local_files_only=True
        )
        model_class = get_class_from_dynamic_module(
            config.auto_map["AutoModelForCausalLM"],
            str(output),
            local_files_only=True,
        )
        model_class = patch_kimi_k3_transformers_compat(model_class)
        torch.manual_seed(seed)
        model = model_class(config).eval()

    tensors = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    expert_names = [
        name
        for name in tensors
        if ".block_sparse_moe.experts." in name and name.endswith(".weight")
    ]
    for name in expert_names:
        packed, scale = _pack_mxfp4(tensors.pop(name))
        prefix = name.removesuffix(".weight")
        tensors[f"{prefix}.weight_packed"] = packed
        tensors[f"{prefix}.weight_scale"] = scale

    shard_name = "model-00001-of-00001.safetensors"
    save_file(tensors, output / shard_name)
    total_size = sum(
        tensor.numel() * tensor.element_size() for tensor in tensors.values()
    )
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {name: shard_name for name in tensors},
    }
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Created tiny Kimi K3 at {output}: 2 layers, 2 routed experts, "
        f"{len(tensors)} tensors, {total_size / 1024**2:.2f} MiB"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_tiny_checkpoint(args.source, args.output, args.seed)


if __name__ == "__main__":
    main()
