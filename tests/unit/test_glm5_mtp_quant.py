"""
Create a tiny GlmMoeDsa model (same architecture as GLM-5.1, parameters < 1B)
with random weights, including a proper MTP layer at index `num_hidden_layers`,
save it to disk, then run end-to-end W8A8 GPTQ quantization to verify that
the MTP layer is quantized alongside the main model.

The test mirrors the real ``glm5_w8a8.py`` pipeline:
  load model -> load tokenizer -> preprocess text -> tokenize -> oneshot -> save

Usage::

    # Use the real GLM-5.1 tokenizer (recommended)
    python tests/unit/test_glm5_mtp_quant.py --tokenizer_path /Users/ang/models/GLM-5.1

    # Self-contained (builds a minimal byte-level tokenizer)
    python tests/unit/test_glm5_mtp_quant.py

    # Keep outputs on disk
    python tests/unit/test_glm5_mtp_quant.py --save_dir /tmp/glm5_mini
"""

import argparse
import copy
import os
import tempfile

import pytest
import torch
import torch.nn as nn
from datasets import Dataset
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaDecoderLayer,
    GlmMoeDsaForCausalLM,
    GlmMoeDsaRMSNorm,
)

from llmcompressor import oneshot
from llmcompressor.modeling.glm_moe_dsa import CalibrationGlmMoeDsaMoE  # noqa: F401
from llmcompressor.modeling.glm_moe_dsa_mtp import attach_mtp_layer
from llmcompressor.modifiers.gptq import GPTQModifier

# ---------------------------------------------------------------------------
# Scaled-down config  (same architecture as GLM-5.1, ~29 M parameters)
# ---------------------------------------------------------------------------
# Original -> mini (scale ~= 8-12x)
#
#   hidden_size        6144  ->  512
#   num_attention_heads  64  ->    8
#   q_lora_rank        2048  ->  256
#   kv_lora_rank        512  ->   64
#   qk_nope_head_dim    192  ->   24   qk_head_dim = 24+8 = 32
#   qk_rope_head_dim     64  ->    8
#   v_head_dim          256  ->   32
#   intermediate_size 12288  -> 1536
#   moe_intermediate   2048  ->  256
#   n_routed_experts    256  ->    8
#   num_experts_per_tok   8  ->    2
#   index_head_dim      128  ->   16
#   index_n_heads        32  ->    4
#   index_topk         2048  ->   64
#   num_hidden_layers    78  ->    6   (3 dense + 3 sparse)  + 1 MTP
#   vocab_size       154880  -> 4096

MINI_CONFIG = dict(
    vocab_size=4096,
    hidden_size=512,
    intermediate_size=1536,
    moe_intermediate_size=256,
    num_hidden_layers=6,
    first_k_dense_replace=3,
    num_attention_heads=8,
    num_key_value_heads=8,
    q_lora_rank=256,
    kv_lora_rank=64,
    qk_rope_head_dim=8,
    qk_nope_head_dim=24,
    v_head_dim=32,
    n_shared_experts=1,
    n_routed_experts=8,
    num_experts_per_tok=2,
    n_group=1,
    topk_group=1,
    norm_topk_prob=True,
    routed_scaling_factor=2.5,
    hidden_act="silu",
    max_position_embeddings=512,
    rms_norm_eps=1e-5,
    attention_bias=False,
    attention_dropout=0.0,
    index_head_dim=16,
    index_n_heads=4,
    index_topk=64,
    indexer_rope_interleave=True,
    tie_word_embeddings=False,
    num_nextn_predict_layers=1,
    rope_parameters={"rope_theta": 10000, "rope_type": "default"},
    rope_interleave=True,
    pad_token_id=0,
    eos_token_id=1,
)

# Sample texts used for calibration (mirrors real-world chat-format inputs)
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the way we work and live.",
    "Large language models can generate coherent text across many domains.",
    "Quantization reduces model size while preserving most of the accuracy.",
    "Python is a versatile programming language used in data science.",
    "The weather today is sunny with a high chance of afternoon clouds.",
    "Neural networks learn representations from raw data automatically.",
    "Transformers use self-attention to model long-range dependencies.",
    "Model compression techniques include pruning, distillation, and quantization.",
    "Calibration data should be representative of the target deployment domain.",
    "The earth revolves around the sun once every 365.25 days.",
    "Deep learning has achieved remarkable results in image recognition tasks.",
    "Efficient inference is crucial for deploying LLMs in production systems.",
    "The history of computing dates back to the mid-twentieth century.",
    "Mixture-of-experts models activate only a subset of parameters per token.",
    "Multi-token prediction heads allow models to speculate future tokens.",
]


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def build_minimal_tokenizer(vocab_size: int) -> PreTrainedTokenizerFast:
    """
    Build a byte-level BPE tokenizer trained on SAMPLE_TEXTS with at most
    *vocab_size* tokens.  No external files needed.
    """
    tok_obj = Tokenizer(BPE(unk_token="[UNK]"))
    tok_obj.pre_tokenizer = ByteLevelPreTokenizer()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[EOS]", "[BOS]", "[UNK]"],
        min_frequency=1,
    )
    tok_obj.train_from_iterator(SAMPLE_TEXTS, trainer=trainer)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        pad_token="[PAD]",
        eos_token="[EOS]",
        bos_token="[BOS]",
        unk_token="[UNK]",
        model_max_length=512,
    )
    print(f"[tokenizer] Built minimal BPE tokenizer, vocab_size={tokenizer.vocab_size}")
    return tokenizer


def load_or_build_tokenizer(
    tokenizer_path: str | None, vocab_size: int
) -> PreTrainedTokenizerFast:
    """
    Load tokenizer from *tokenizer_path* if provided, otherwise build a minimal
    byte-level BPE tokenizer.

    When loading an external tokenizer whose vocab_size > *vocab_size*, token IDs
    are remapped to [2, vocab_size-1] via modulo so embedding lookups stay in range.
    """
    if tokenizer_path and os.path.isdir(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(
            f"[tokenizer] Loaded from {tokenizer_path}, "
            f"vocab_size={tokenizer.vocab_size}"
        )
        return tokenizer

    return build_minimal_tokenizer(vocab_size)


def make_calibration_dataset(
    tokenizer: PreTrainedTokenizerBase,
    model_vocab_size: int,
    seq_len: int,
    n: int,
) -> Dataset:
    """
    Tokenize SAMPLE_TEXTS and return a Dataset with ``input_ids``.

    If the tokenizer's vocab_size exceeds *model_vocab_size*, each token ID is
    remapped with ``(id_ % (model_vocab_size - 2)) + 2`` to avoid the special
    token slots 0 (PAD) and 1 (EOS) and keep indices in-bounds for the mini
    model's embedding table.
    """
    rows = []
    for i in range(n):
        text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
        enc = tokenizer(
            text,
            max_length=seq_len,
            truncation=True,
            padding=False,
            add_special_tokens=False,
        )
        ids = enc["input_ids"]

        if tokenizer.vocab_size > model_vocab_size:
            safe_range = model_vocab_size - 2
            ids = [(id_ % safe_range) + 2 for id_ in ids]

        rows.append({"input_ids": ids})

    ds = Dataset.from_list(rows)
    lengths = [len(r["input_ids"]) for r in rows]
    print(f"[dataset] {n} samples, seq_len range [{min(lengths)}, {max(lengths)}]")
    return ds


# ---------------------------------------------------------------------------
# Model construction helpers
# ---------------------------------------------------------------------------

def build_mini_config() -> GlmMoeDsaConfig:
    cfg = GlmMoeDsaConfig(**MINI_CONFIG)
    print(
        f"[config] num_hidden_layers={cfg.num_hidden_layers}, "
        f"mlp_layer_types={cfg.mlp_layer_types}"
    )
    return cfg


def build_main_model(config: GlmMoeDsaConfig) -> GlmMoeDsaForCausalLM:
    torch.manual_seed(42)
    model = GlmMoeDsaForCausalLM(config).to(dtype=torch.bfloat16)
    n = sum(p.numel() for p in model.parameters())
    print(f"[model] Main model: {n / 1e6:.1f}M parameters")
    return model


def build_mtp_state_dict(config: GlmMoeDsaConfig) -> dict[str, torch.Tensor]:
    """
    Build random MTP layer weights (decoder + enorm/hnorm/eh_proj/shared_head_norm)
    keyed with the checkpoint prefix ``model.layers.{num_hidden_layers}.*``.
    """
    mtp_idx = config.num_hidden_layers

    extended = copy.copy(config)
    extended.mlp_layer_types = list(config.mlp_layer_types) + ["sparse"]
    if hasattr(config, "indexer_types") and len(config.indexer_types) <= mtp_idx:
        fill = config.indexer_types[-1] if config.indexer_types else "full"
        extended.indexer_types = list(config.indexer_types) + [fill] * (
            mtp_idx - len(config.indexer_types) + 1
        )

    torch.manual_seed(99)
    decoder = GlmMoeDsaDecoderLayer(extended, layer_idx=mtp_idx)
    enorm = GlmMoeDsaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    hnorm = GlmMoeDsaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
    shared_hd_norm = GlmMoeDsaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    prefix = f"model.layers.{mtp_idx}"
    state: dict[str, torch.Tensor] = {}
    for k, v in decoder.state_dict().items():
        state[f"{prefix}.{k}"] = v.to(torch.bfloat16)
    for k, v in enorm.state_dict().items():
        state[f"{prefix}.enorm.{k}"] = v.to(torch.bfloat16)
    for k, v in hnorm.state_dict().items():
        state[f"{prefix}.hnorm.{k}"] = v.to(torch.bfloat16)
    for k, v in eh_proj.state_dict().items():
        state[f"{prefix}.eh_proj.{k}"] = v.to(torch.bfloat16)
    for k, v in shared_hd_norm.state_dict().items():
        state[f"{prefix}.shared_head.norm.{k}"] = v.to(torch.bfloat16)

    n = sum(t.numel() for t in state.values())
    print(f"[mtp] MTP layer: {n / 1e6:.1f}M parameters ({len(state)} tensors)")
    return state


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: GlmMoeDsaForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    mtp_state: dict[str, torch.Tensor],
    save_dir: str,
) -> None:
    """
    Merge main model weights + MTP weights into a single ``model.safetensors``
    and save alongside ``config.json`` and tokenizer files.

    ``GlmMoeDsaForCausalLM.from_pretrained`` will load the file successfully:
    ``_keys_to_ignore_on_load_unexpected`` silently drops ``model.layers.{N}.*``,
    and ``attach_mtp_layer`` later reads those tensors back from the same file.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    combined: dict[str, torch.Tensor] = {
        k: v.contiguous() for k, v in model.state_dict().items()
    }
    combined.update({k: v.contiguous() for k, v in mtp_state.items()})

    save_file(combined, os.path.join(save_dir, "model.safetensors"))
    total = sum(t.numel() for t in combined.values())
    print(f"[save] {len(combined)} tensors ({total / 1e6:.1f}M params) -> {save_dir}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_mtp_quantized(model: GlmMoeDsaForCausalLM) -> None:
    assert hasattr(model, "mtp"), "model.mtp not found - attach_mtp_layer failed"
    assert isinstance(model.mtp.eh_proj, nn.Linear), "mtp.eh_proj should remain an nn.Linear"

    mtp_linears = {
        name: mod
        for name, mod in model.mtp.named_modules()
        if isinstance(mod, nn.Linear)
    }
    print(f"\n[verify] MTP Linear layers ({len(mtp_linears)} total):")
    quantized = 0
    for name, mod in mtp_linears.items():
        scheme = getattr(mod, "quantization_scheme", None)
        tag = f"quantized ({scheme})" if scheme is not None else "NOT quantized"
        print(f"  mtp.{name}: {tag}")
        if scheme is not None:
            quantized += 1

    assert getattr(model.mtp.eh_proj, "quantization_scheme", None) is None, (
        "mtp.eh_proj should stay in BF16 and be excluded from quantization"
    )

    assert quantized > 0, (
        "No MTP Linear layers were quantized! "
        "Check that the ignore list does not exclude model.mtp.*"
    )
    print(f"\n[verify] PASSED - {quantized}/{len(mtp_linears)} MTP Linear layers quantized.")


def verify_layer78_export(save_dir: str, mtp_idx: int) -> None:
    prefix = f"model.layers.{mtp_idx}."
    eh_proj_weight_key = prefix + "eh_proj.weight"
    bad_keys = []
    dtype_summary: dict[str, torch.dtype] = {}

    with safe_open(os.path.join(save_dir, "model.safetensors"), framework="pt") as f:
        for key in f.keys():
            if not key.startswith(prefix):
                continue
            if key.startswith(prefix + "eh_proj.") and key != eh_proj_weight_key:
                bad_keys.append(key)
            if key in {
                eh_proj_weight_key,
                prefix + "enorm.weight",
                prefix + "hnorm.weight",
                prefix + "input_layernorm.weight",
                prefix + "post_attention_layernorm.weight",
            }:
                dtype_summary[key] = f.get_tensor(key).dtype

    assert eh_proj_weight_key in dtype_summary, "eh_proj.weight missing from exported checkpoint"
    assert dtype_summary[eh_proj_weight_key] == torch.bfloat16, (
        f"Expected {eh_proj_weight_key} to remain bf16, got {dtype_summary[eh_proj_weight_key]}"
    )
    assert not bad_keys, f"eh_proj should not export quantization side tensors: {bad_keys}"

    for key, dtype in dtype_summary.items():
        assert dtype == torch.bfloat16, f"Expected {key} to be bf16, got {dtype}"

    print("[verify] PASSED - layer 78 export keeps eh_proj/norm weights in bf16 only.")


def run_glm5_mtp_quant(tokenizer_path: str | None, save_dir: str | None) -> None:
    seq_len = 64
    n_samples = 8

    config = build_mini_config()
    tokenizer = load_or_build_tokenizer(tokenizer_path, config.vocab_size)

    model = build_main_model(config)
    mtp_state = build_mtp_state_dict(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = save_dir if save_dir else tmpdir

        save_checkpoint(model, tokenizer, mtp_state, ckpt_dir)
        del model

        print(f"\n[load] Reloading from {ckpt_dir} ...")
        model = GlmMoeDsaForCausalLM.from_pretrained(
            ckpt_dir, torch_dtype=torch.bfloat16, device_map=None
        )
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)

        attach_mtp_layer(model, ckpt_dir)

        ds = make_calibration_dataset(
            tokenizer=tokenizer,
            model_vocab_size=config.vocab_size,
            seq_len=seq_len,
            n=n_samples,
        )

        recipe = GPTQModifier(
            targets="Linear",
            scheme="W8A8",
            ignore=[
                "re:^mtp\\.eh_proj$",
                "re:.*mlp.gate$",
                "re:.*embed_tokens$",
                "re:.*indexer.*",
                "lm_head",
            ],
        )

        print("\n[oneshot] Starting quantization ...")
        oneshot(
            model=model,
            dataset=ds,
            recipe=recipe,
            max_seq_length=seq_len,
            num_calibration_samples=n_samples,
            batch_size=1,
        )
        print("[oneshot] Done.")

        verify_mtp_quantized(model)

        if save_dir:
            quant_dir = save_dir.rstrip("/") + "-W8A8-GPTQ"
            print(f"\n[save] Saving quantized model to {quant_dir} ...")
            model.save_pretrained(quant_dir, save_compressed=True)
            tokenizer.save_pretrained(quant_dir)
            verify_layer78_export(quant_dir, config.num_hidden_layers)
            print(f"[save] Done -> {quant_dir}")


@pytest.mark.regression
def test_glm5_mtp_quantization_pipeline():
    run_glm5_mtp_quant(tokenizer_path=None, save_dir=None)


def main(tokenizer_path: str | None, save_dir: str | None) -> None:
    run_glm5_mtp_quant(tokenizer_path=tokenizer_path, save_dir=save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GLM-5 mini MTP end-to-end quantization test"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help=(
            "Path to a tokenizer directory (e.g. /Users/ang/models/GLM-5.1). "
            "If omitted, a minimal byte-level BPE tokenizer is built on-the-fly."
        ),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help=(
            "Directory to save the mini BF16 checkpoint and quantized output. "
            "Uses a temp dir if not specified."
        ),
    )
    args = parser.parse_args()
    main(args.tokenizer_path, args.save_dir)