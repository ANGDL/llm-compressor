from __future__ import annotations

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from llmcompressor.streaming import streaming_oneshot


def test_tiny_llama_streaming_checkpoint_loads_and_forwards(tmp_path):
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
    )
    torch.manual_seed(12)
    reference = LlamaForCausalLM(config).eval()
    checkpoint = tmp_path / "checkpoint"
    reference.save_pretrained(checkpoint, safe_serialization=True)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    hidden_states = reference.model.embed_tokens(input_ids)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    position_embeddings = reference.model.rotary_emb(
        hidden_states, position_ids
    )
    batches = [
        {
            "hidden_states": hidden_states,
            "position_ids": position_ids,
            "position_embeddings": position_embeddings,
            "use_cache": False,
        }
    ]

    def forward_layer(layer, value):
        output = layer(**value)
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        return {**value, "hidden_states": hidden}

    module_names = [
        name
        for name, module in reference.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.startswith("model.layers.")
    ]
    schemes = {
        name: QuantizationScheme(
            targets=[name],
            weights=QuantizationArgs(
                num_bits=8,
                strategy="channel",
                symmetric=True,
            ),
        )
        for name in module_names
    }
    output = streaming_oneshot(
        model_factory=LlamaForCausalLM,
        model_args=(config,),
        checkpoint=checkpoint,
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        calibration_batches=batches,
        targets=("model.layers.0", "model.layers.1"),
        recipe={"GPTQModifier": {"targets": module_names}},
        dataset_fingerprint="d" * 64,
        schemes=schemes,
        algorithms=("gptq",),
        forward_target=forward_layer,
        target_dtype=torch.float32,
    )

    loaded = AutoModelForCausalLM.from_pretrained(
        output, local_files_only=True, device_map="cpu"
    ).eval()
    with torch.no_grad():
        logits = loaded(input_ids).logits
    assert logits.shape == (1, 4, config.vocab_size)
    assert torch.isfinite(logits).all()
