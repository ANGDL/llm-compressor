from pathlib import Path
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from llmcompressor import oneshot
from llmcompressor.modeling.deepseekv4.config import ModelConfig
from llmcompressor.modeling.deepseekv4.model import DeepseekV4ForCausalLM
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.autosmooth import AutoSmoothModifier
from llmcompressor.modifiers.quantization import QuantizationModifier


def build_tiny_config() -> ModelConfig:
    return ModelConfig(
        max_batch_size=2,
        max_seq_len=16,
        max_position_embeddings=16,
        vocab_size=128,
        hidden_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=5,
        num_hash_layers=1,
        num_nextn_predict_layers=1,
        num_attention_heads=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        q_lora_rank=16,
        head_dim=8,
        qk_rope_head_dim=4,
        o_groups=2,
        o_lora_rank=8,
        sliding_window=8,
        index_n_heads=4,
        index_head_dim=8,
        index_topk=4,
        hc_mult=2,
        compress_ratios=[0, 4],
    )


def main():
    torch.manual_seed(0)
    with torch.no_grad():
        model = DeepseekV4ForCausalLM(build_tiny_config())
        model = model.to(torch.bfloat16)
    # GPTQ's lm_head disabling utility assumes a single-input output head.
    # DeepSeekV4 head has extra arguments, so skip output embedding wrapping.
    model.get_output_embeddings = lambda: None

    sample = torch.randint(0, model.config.vocab_size, (2, 8))
    print("pre-quant logits:", model(input_ids=sample).logits.shape)

    # Save the original bf16/fp model before quantization.
    sym_save_dir = Path("./DeepSeek-V4-Tiny-SYM-model")
    model.save_pretrained(sym_save_dir)
    print(f"#sym:model saved to: {sym_save_dir}")

    # recipe = GPTQModifier(
    #     targets="Linear",
    #     scheme="W8A8",
    #     ignore=["lm_head"],
    #     # offload_hessians=True,
    # )
    recipe = [
        # AutoSmoothModifier(
        #     targets="Linear",
        #     scheme="W8A8",
        #     ignore=["lm_head"],
        # ),
        QuantizationModifier(
            targets="Linear",
            scheme="W8A8",
            ignore=[
                "lm_head", 
                're:.*attn.compressor.wgate$', 
                're:.*attn.compressor.wkv$',
                're:.*attn.indexer.compressor.wgate$',
                're:.*attn.indexer.compressor.wkv$',
                're:.*attn.indexer.weights_proj$',
                ],
        )
    ]

    # Build calibration data in the same preprocess/tokenize style used by
    # examples/quantization_kv_cache/qwen3_dense_int8_chanel_kv_example.py.
    num_calibration_samples = 16
    max_sequence_length = sample.shape[1]

    prompts = [
        "Tell me a short story about quantization.",
        "What is GPTQ calibration used for?",
        "Explain mixture-of-experts in one paragraph.",
        "Give tips for validating compressed checkpoints.",
    ]
    raw_text = [prompts[index % len(prompts)] for index in range(num_calibration_samples)]
    ds = Dataset.from_dict({"text": raw_text}).shuffle(seed=42)

    def preprocess(example):
        return {"text": example["text"]}

    ds = ds.map(preprocess)

    def tokenize(example):
        text = example["text"]
        byte_ids = [byte % model.config.vocab_size for byte in text.encode("utf-8")]
        if not byte_ids:
            byte_ids = [0]
        input_ids = byte_ids[:max_sequence_length]
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    def collate_fn(batch):
        return {
            "input_ids": torch.tensor([row["input_ids"] for row in batch], dtype=torch.long),
            "attention_mask": torch.tensor([row["attention_mask"] for row in batch], dtype=torch.long),
        }

    calibration_dataloader = DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
    )

    class IdentityProcessor:
        def __call__(self, *args, **kwargs):
            return kwargs if kwargs else args

        def save_pretrained(self, *_args, **_kwargs):
            return None

    processor = IdentityProcessor()

    oneshot(
        model=model,
        recipe=recipe,
        dataset=calibration_dataloader,
        processor=processor,
        num_calibration_samples=num_calibration_samples,
        max_seq_length=max_sequence_length,
    )

    save_dir = Path("./DeepSeek-V4-Tiny-W8A8")

    model.save_raw_format = True
    import llmcompressor.transformers.compression.compressed_tensors_utils as ct_utils
    original_from_accelerate = ct_utils.from_accelerate
    ct_utils.from_accelerate = lambda model: ({}, None)
    try:
        model.save_pretrained(save_dir, save_compressed=True)
    finally:
        ct_utils.from_accelerate = original_from_accelerate

    with torch.no_grad():
           reloaded = DeepseekV4ForCausalLM.from_pretrained(sym_save_dir, dtype=torch.bfloat16)
           reloaded = reloaded.to(torch.bfloat16)
    print("post-reload logits:", reloaded(input_ids=sample).logits.shape)
    print(f"saved to: {save_dir}")


if __name__ == "__main__":
    main()