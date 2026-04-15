import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling.glm_moe_dsa import CalibrationGlmMoeDsaMoE  # noqa: F401
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.autosmooth import AutoSmoothModifier
from llmcompressor.modifiers.smooth_gptq import SmoothGPTQModifier
from llmcompressor.modifiers.awq import AWQMapping

from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
import torch

# Load the model
model_id = "/ssd4/models/GLM-5"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map=None)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# MoE calibration is now handled automatically by the pipeline.
# The `CalibrationGlmMoeDsaMoE` modules (from `llmcompressor.modeling.glm_moe_dsa`)
# will be applied during calibration to enable proper expert calibration.
# These permanently unpack the fused 3D expert weights into individual nn.Linear
# layers for quantization target matching and vLLM compatibility.

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 32
MAX_SEQUENCE_LENGTH = 8192

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

ignores = [
    # Ignore the output head
    "re:.*mlp\\.gate$",
    "re:.*embed_tokens$",
    "re:.*indexer.*",
    "lm_head",
]

# ---- Quantization config: mixed experts int4 + other linear int8 ----
# Weights: 4-bit, channelwise, symmetric, static
weights_args_4 = QuantizationArgs(
    num_bits=4,
    type=QuantizationType.INT,
    strategy=QuantizationStrategy.CHANNEL,
    symmetric=True,
    dynamic=False,
)

# Weights: 8-bit, channelwise, symmetric, static
weights_args_8 = QuantizationArgs(
    num_bits=8,
    type=QuantizationType.INT,
    strategy=QuantizationStrategy.CHANNEL,
    symmetric=True,
    dynamic=False,
)

# Activations: 8-bit, per-token, asymmetric, dynamic
activations_args = QuantizationArgs(
    num_bits=8,
    type=QuantizationType.INT,
    strategy=QuantizationStrategy.TOKEN,
    symmetric=False,
    dynamic=True,
    observer=None,
)

# Experts and shared experts use int4 weights.
experts_w4_scheme = QuantizationScheme(
    targets=[
        "re:.*mlp\\.experts\\..*\\.(gate_proj|up_proj|down_proj)$",
        "re:.*mlp\\.shared_experts(?:\\..+)?\\.(gate_proj|up_proj|down_proj)$",
    ],
    weights=weights_args_4,
    input_activations=activations_args,
)

# Other linear projections use int8 weights.
other_linear_w8_scheme = QuantizationScheme(
    targets=[
        "re:.*self_attn\\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)$",
        "re:.*mlp\\.(gate_proj|up_proj|down_proj)$",
    ],
    weights=weights_args_8,
    input_activations=activations_args,
)

def build_int4_mappings(config) -> list[AWQMapping]:
    """Build int4-only AWQ mappings with one smooth layer per mapping."""
    num_layers = int(getattr(config, "num_hidden_layers"))
    num_experts = int(
        getattr(
            config,
            "num_local_experts",
            getattr(config, "n_routed_experts", 0),
        )
    )

    mappings: list[AWQMapping] = []
    for layer_idx in range(num_layers):
        layer_prefix = rf"re:.*layers\.{layer_idx}\."

        # LayerNorm -> expert gate/up projections (all int4 MLP entry projections).
        balance_layers = [
            rf"{layer_prefix}mlp\.shared_experts\.gate_proj$",
            rf"{layer_prefix}mlp\.shared_experts\.up_proj$",
        ]
        for expert_idx in range(num_experts):
            balance_layers.append(
                rf"{layer_prefix}mlp\.experts\.{expert_idx}\.gate_proj$"
            )
            balance_layers.append(
                rf"{layer_prefix}mlp\.experts\.{expert_idx}\.up_proj$"
            )

        mappings.append(
            AWQMapping(
                rf"{layer_prefix}post_attention_layernorm$",
                balance_layers,
            )
        )

        # shared_experts up -> down
        mappings.append(
            AWQMapping(
                rf"{layer_prefix}mlp\.shared_experts\.up_proj$",
                [rf"{layer_prefix}mlp\.shared_experts\.down_proj$"],
            )
        )

        # experts.<idx> up -> down (must be per-expert to keep smooth_layer unique)
        for expert_idx in range(num_experts):
            mappings.append(
                AWQMapping(
                    rf"{layer_prefix}mlp\.experts\.{expert_idx}\.up_proj$",
                    [rf"{layer_prefix}mlp\.experts\.{expert_idx}\.down_proj$"],
                )
            )

    return mappings


# AutoSmooth/SmoothGPTQ mappings restricted to int4 layers only.
int4_mapping = build_int4_mappings(model.config)


# Configure the quantization algorithm to run.
#   * quantize experts/shared_experts to int4 and other selected linear layers to int8
recipe = [
    # SmoothGPTQModifier(
    #     activation_scale_type="max", 
    #     norm_func='awq',
    #     mappings=int4_mapping,
    #     config_groups={
    #         "experts_w4a8": experts_w4_scheme,
    #         "other_linear_w8a8": other_linear_w8_scheme,
    #     },
    #     offload_device=torch.device("cuda:1") if torch.cuda.device_count() > 1 else torch.device("cpu"),
    #     ignore=ignores,
    #     muti_gpu_compression=True,
    #     offload_hessians=True,
    # )

    # AutoSmoothModifier(
    #     activation_scale_type="max", 
    #     norm_func='awq',
    #     mappings=int4_mapping,
    #     config_groups={
    #         "experts_w4a8": experts_w4_scheme,
    #     },
    #     offload_device=torch.device("cuda:1") if torch.cuda.device_count() > 1 else torch.device("cpu"),
    #     ignore=ignores,
    # ),
    GPTQModifier(
        config_groups={
            "experts_w4a8": experts_w4_scheme,
            "other_linear_w8a8": other_linear_w8_scheme,
        },
        ignore=ignores,
#        muti_gpu_compression=True,
        offload_hessians=True,
    )
]

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    batch_size=1,
)

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-WNA8-smoothgptq-unpacked"
SAVE_DIR = os.path.join("/ssd3/models", SAVE_DIR)

model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
