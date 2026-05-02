import argparse
import os
from contextlib import contextmanager

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling.glm_moe_dsa import CalibrationGlmMoeDsaMoE  # noqa: F401
from llmcompressor.modeling.glm_moe_dsa_mtp import attach_mtp_layer
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.autosmooth import AutoSmoothModifier
from llmcompressor.modifiers.smooth_gptq import SmoothGPTQModifier
from llmcompressor.modifiers.awq import AWQMapping
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.pipelines.basic import pipeline as basic_pipeline
from llmcompressor.pipelines.data_free import pipeline as data_free_pipeline

from compressed_tensors.offload.dispatch import dispatch_model as _dispatch_model
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="/ssd4/models/GLM-5")
parser.add_argument("--save_dir", type=str, default="/ssd3/models")
parser.add_argument("--observer", type=str, default=None, choices=[None, "imatrix_mse"])
parser.add_argument("--modifier", type=str, default="GPTQ", choices=["GPTQ", "RTN"])
parser.add_argument("--dataset_id", type=str, default="HuggingFaceH4/ultrachat_200k")
parser.add_argument("--dataset_split", type=str, default="train_sft")
parser.add_argument(
    "--skip_restore_from_accelerate",
    action=argparse.BooleanOptionalAction,
    help=(
        "Skip converting model back from accelerate after save_pretrained. "
        "Useful to avoid end-of-run CUDA OOM for very large models."
    ),
    default=True,
)
parser.add_argument(
    "--dispatch_extra_memory_gb",
    type=float,
    default=20.0,
    help="Reserved memory per GPU for dispatch_model. Lower this if dispatch fails.",
)
args = parser.parse_args()


def patch_pipeline_dispatch(extra_memory_gb: float):
    extra_memory_bytes = int(extra_memory_gb * 1024**3)

    def _dispatch_with_cap(model):
        return _dispatch_model(model, extra_memory=extra_memory_bytes)

    basic_pipeline.dispatch_model = _dispatch_with_cap
    data_free_pipeline.dispatch_model = _dispatch_with_cap


patch_pipeline_dispatch(args.dispatch_extra_memory_gb)


@contextmanager
def maybe_skip_from_accelerate(skip_restore: bool):
    if not skip_restore:
        yield
        return

    import llmcompressor.transformers.compression.compressed_tensors_utils as ct_utils

    original_from_accelerate = ct_utils.from_accelerate
    ct_utils.from_accelerate = lambda model: ({}, None)
    try:
        yield
    finally:
        ct_utils.from_accelerate = original_from_accelerate

# Load the model
model_id = args.model_id
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map=None)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# MoE calibration is now handled automatically by the pipeline.
# The `CalibrationGlmMoeDsaMoE` modules (from `llmcompressor.modeling.glm_moe_dsa`)
# will be applied during calibration to enable proper expert calibration.
# These permanently unpack the fused 3D expert weights into individual nn.Linear
# layers for quantization target matching and vLLM compatibility.

# Attach the MTP layer so it participates in the calibration forward pass and
# gets quantized end-to-end along with the main model.
attach_mtp_layer(model, model_id)

# Select calibration dataset.
DATASET_ID = args.dataset_id
DATASET_SPLIT = args.dataset_split

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
    # Keep the MTP projection in BF16 for downstream NextN loaders.
    "re:^mtp\\.eh_proj$",
    "re:.*eh_proj$",
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
config_groups = {
    "experts_w4a8": experts_w4_scheme,
    "other_linear_w8a8": other_linear_w8_scheme,
}

tail_name = "-WNA8"
recipes = []
if args.observer == "imatrix_mse":
    weights_args_4.observer = "imatrix_mse"
    weights_args_8.observer = "imatrix_mse"
    recipes.append(IMatrixGatherer(ignore=ignores))
    tail_name += "-IMatrix"

# Keep these reference configs nearby for future experiments.
# recipe = [
    # SmoothGPTQModifier(
    #     activation_scale_type="max", 
    #     norm_func='awq',
    #     mappings=int4_mapping,
    #     config_groups=config_groups,
    #     offload_device=torch.device("cuda:1") if torch.cuda.device_count() > 1 else torch.device("cpu"),
    #     ignore=ignores,
    #     muti_gpu_compression=True,
    #     offload_hessians=True,
    # )

    # AutoSmoothModifier(
    #     activation_scale_type="max", 
    #     norm_func='awq',
    #     mappings=int4_mapping,
    #     config_groups={"experts_w4a8": experts_w4_scheme},
    #     offload_device=torch.device("cuda:1") if torch.cuda.device_count() > 1 else torch.device("cpu"),
    #     ignore=ignores,
    # ),

if args.modifier == "GPTQ":
    recipes.append(
        GPTQModifier(
            config_groups=config_groups,
            ignore=ignores,
            # muti_gpu_compression=True,
            offload_hessians=True,
        )
    )
    tail_name += "-GPTQ"
elif args.modifier == "RTN":
    recipes.append(
        QuantizationModifier(
            config_groups=config_groups,
            ignore=ignores,
        )
    )
    tail_name += "-RTN"
else:
    raise ValueError(f"Invalid modifier selected: {args.modifier}")

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipes,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    batch_size=1,
)

# Save to disk compressed.
SAVE_NAME = model_id.rstrip("/").split("/")[-1] + tail_name + "-unpacked"
SAVE_DIR = os.path.join(args.save_dir, SAVE_NAME)

with maybe_skip_from_accelerate(args.skip_restore_from_accelerate):
    model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
