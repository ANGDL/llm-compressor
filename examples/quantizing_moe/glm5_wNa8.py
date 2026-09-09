import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager

import torch
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.logger import LoggerConfig, configure_logger
from llmcompressor.modeling.glm_moe_dsa import CalibrationGlmMoeDsaMoE  # noqa: F401
from llmcompressor.modeling.glm_moe_dsa_mtp import attach_mtp_layer
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.pipelines.basic import pipeline as basic_pipeline
from llmcompressor.pipelines.data_free import pipeline as data_free_pipeline
from llmcompressor.utils import ImatrixFallbackStats

from compressed_tensors.offload import load_offloaded_model
from compressed_tensors.offload.dispatch import dispatch_model as _dispatch_model
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

configure_logger(LoggerConfig(console_log_level="DEBUG"))


"""
Usage example for quantizing GLM-5.1 with MoE layers to mixed W4/W8 + A8 using
LLM Compressor. The script performs all three steps below: update
generation_config.json, quantize, and pack the int4 weights for downstream runtimes.
1. 更新权重文件中的 generation_config.json，添加："do_sample": true
2. 量化并保存 unpacked checkpoint
python glm5_wNa8.py --model_id /ssd3/models/GLM-5.1/ --save_dir /ssd2/models/ \
    --modifier RTN --observer imatrix_mse \
    --num_calibration_samples 512 --max_sequence_length 4096 \
    --dataset_id ./ultrachat_200k ./calibration_data.jsonl --dataset_split train_sft \
    --indexer-ignore-mode indexer_all --dispatch_extra_memory_gb 10 \
    --pipeline sequential --use-float32-scale-dtype
3. 自动打包为 packed checkpoint（可使用 --no-pack 保留旧行为）
4. 可选地备份并用原始模型的 config.json 覆盖输出配置
"""


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="/ssd4/models/GLM-5")
parser.add_argument("--save_dir", type=str, default="/ssd3/models")
parser.add_argument(
    "--observer",
    type=str,
    default="",
    choices=["", "mse", "imatrix_mse"],
    help="Observer for weights quantization: empty(default), mse, or imatrix_mse.",
)
parser.add_argument("--modifier", type=str, default="GPTQ", choices=["GPTQ", "RTN"])
parser.add_argument("--dataset_id", type=str, nargs="+",
                    default=["HuggingFaceH4/ultrachat_200k"],
                    help="HuggingFace dataset ID(s) or path(s) to local JSON/JSONL files. "
                         "Multiple sources can be specified and will be concatenated.")
parser.add_argument("--dataset_split", type=str, default="train_sft")
parser.add_argument(
    "--num_calibration_samples",
    type=positive_int,
    default=32,
    help="Number of calibration samples to load from the dataset.",
)
parser.add_argument(
    "--max_sequence_length",
    type=positive_int,
    default=8192,
    help="Maximum sequence length used for tokenization and calibration.",
)
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
    "--offload_folder",
    type=str,
    default="./offload_folder",
    help="Directory used for disk offloading when loading very large models.",
)
parser.add_argument(
    "--dispatch_extra_memory_gb",
    type=float,
    default=20.0,
    help="Reserved memory per GPU for dispatch_model. Lower this if dispatch fails.",
)
parser.add_argument(
    "--indexer-ignore-mode",
    type=str,
    default="indexer_all",
    choices=["weights_proj", "indexer_all"],
    help=(
        "Control indexer ignore scope: 'weights_proj' only ignores "
        "self_attn.indexer.weights_proj; 'indexer_all' ignores all self_attn.indexer*"
    ),
)
parser.add_argument(
    "--moe-calibrate-all-experts",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to calibrate all MoE experts (not just the top-k).",
)
parser.add_argument(
    "--shared-experts-bits",
    type=int,
    default=8,
    choices=[4, 8],
    help=(
        "Quantization bits for mlp.shared_experts.* projections. "
        "Default 4 keeps shared experts in the W4 group (with routed experts); "
        "use 8 to move them into the W8 group together with attention/MLP linears."
    ),
)
parser.add_argument(
    "--use-float32-scale-dtype",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use float32 for weight scale tensors instead of bfloat16.",
)
parser.add_argument(
    "--pipeline",
    type=str,
    default="independent",
    choices=["independent", "sequential", "basic", "data_free"],
    help="Pipeline to use for oneshot compression.",
)
parser.add_argument(
    "--max-memory-cpu-gb",
    type=float,
    default=1500.0,
    help="Max CPU memory in GB for model loading (passed as max_memory={'cpu': <value>e9}).",
)
parser.add_argument(
    "--pack",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Pack int4 weights after quantization (use --no-pack to skip packing).",
)
parser.add_argument(
    "--packed_save_dir",
    type=str,
    default=None,
    help=(
        "Directory for the packed checkpoint. Defaults to --save_dir with a W4A8 name."
    ),
)
parser.add_argument(
    "--use-original-config",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "After quantization and packing, back up each output config.json as "
        "config.json.bak and replace it with the source model config.json."
    ),
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


def ensure_do_sample(model_id: str) -> None:
    """Ensure the source checkpoint's generation config enables sampling."""
    if not os.path.isdir(model_id):
        # Hub checkpoints are updated on the loaded model below instead.
        return

    generation_config_path = os.path.join(model_id, "generation_config.json")
    if os.path.isfile(generation_config_path):
        with open(generation_config_path, "r", encoding="utf-8") as config_file:
            generation_config = json.load(config_file)
    else:
        generation_config = {}
    if generation_config.get("do_sample") is True:
        return

    generation_config["do_sample"] = True
    with open(generation_config_path, "w", encoding="utf-8") as config_file:
        json.dump(generation_config, config_file, ensure_ascii=True, indent=2)
        config_file.write("\n")


def overwrite_with_original_config(model_id: str, output_dir: str) -> None:
    """Back up an output config and replace it with the source config."""
    source_config_path = os.path.join(model_id, "config.json")
    output_config_path = os.path.join(output_dir, "config.json")
    if not os.path.isfile(source_config_path):
        raise FileNotFoundError(
            f"Original config.json not found in model directory: {model_id}"
        )
    if not os.path.isfile(output_config_path):
        raise FileNotFoundError(
            f"Quantized config.json not found in output directory: {output_dir}"
        )
    if os.path.realpath(source_config_path) == os.path.realpath(output_config_path):
        return

    backup_config_path = f"{output_config_path}.bak"
    shutil.copy2(output_config_path, backup_config_path)
    shutil.copy2(source_config_path, output_config_path)
    print(
        f"Backed up quantized config to {backup_config_path} and restored "
        f"original config to {output_config_path}"
    )


# Select calibration dataset.
DATASET_IDS = args.dataset_id
DATASET_SPLIT = args.dataset_split
NUM_CALIBRATION_SAMPLES = args.num_calibration_samples
MAX_SEQUENCE_LENGTH = args.max_sequence_length


# Load dataset and preprocess.
# Support multiple sources (HuggingFace Hub and/or local JSON/JSONL files).
if NUM_CALIBRATION_SAMPLES < len(DATASET_IDS):
    raise ValueError(
        f"num_calibration_samples ({NUM_CALIBRATION_SAMPLES}) must be >= "
        f"number of dataset sources ({len(DATASET_IDS)}): {DATASET_IDS}"
    )

per_source_samples = NUM_CALIBRATION_SAMPLES // len(DATASET_IDS)
remainder = NUM_CALIBRATION_SAMPLES % len(DATASET_IDS)

all_datasets = []
for i, dataset_id in enumerate(DATASET_IDS):
    n_samples = per_source_samples + (1 if i < remainder else 0)
    if os.path.isfile(dataset_id) or dataset_id.endswith((".json", ".jsonl")):
        part = load_dataset(
            "json",
            data_files=dataset_id,
            split=f"train[:{n_samples}]",
        )
    else:
        part = load_dataset(dataset_id, split=f"{DATASET_SPLIT}[:{n_samples}]")
    all_datasets.append(part.shuffle(seed=42))

# Load the model
model_id = args.model_id
ensure_do_sample(model_id)
with load_offloaded_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto_offload",
        offload_folder=args.offload_folder,
        max_memory={"cpu": int(args.max_memory_cpu_gb * 1e9)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
# Apply the same setting for Hub checkpoints or checkpoints without a local
# generation_config.json; save_pretrained will persist it in the output.
if getattr(model, "generation_config", None) is not None:
    model.generation_config.do_sample = True
# MoE calibration is now handled automatically by the pipeline.
# The `CalibrationGlmMoeDsaMoE` modules (from `llmcompressor.modeling.glm_moe_dsa`)
# will be applied during calibration to enable proper expert calibration.
# These permanently unpack the fused 3D expert weights into individual nn.Linear
# layers for quantization target matching and vLLM compatibility.

# Attach the MTP layer so it participates in the calibration forward pass and
# gets quantized end-to-end along with the main model.
attach_mtp_layer(model, model_id)


def preprocess(example):
    if "messages" in example:
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }
    elif "text" in example:
        return {"text": example["text"]}


# Preprocess each source to unified {"text": ...} schema before concatenation,
# since different sources may have different column schemas.
all_datasets = [part.map(preprocess, remove_columns=part.column_names) for part in all_datasets]

if len(all_datasets) == 1:
    ds = all_datasets[0]
else:
    ds = concatenate_datasets(all_datasets)


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

if args.indexer_ignore_mode == "weights_proj":
    INDEXER_WEIGHTS_PROJ_PATTERN = r"re:.*self_attn\.indexer\.weights_proj$"
elif args.indexer_ignore_mode == "indexer_all":
    INDEXER_WEIGHTS_PROJ_PATTERN = r"re:.*self_attn\.indexer*"
else:
    raise ValueError(f"Invalid indexer ignore mode: {args.indexer_ignore_mode}")

ignores = [
    # Keep the MTP projection in BF16 for downstream NextN loaders.
    "re:^mtp\\.eh_proj$",
    "re:.*eh_proj$",
    "re:.*mlp.gate$",
    "re:.*embed_tokens$",
    INDEXER_WEIGHTS_PROJ_PATTERN,
    "lm_head",
]

# ---- Quantization config: mixed experts int4 + other linear int8 ----
scale_dtype = torch.float32 if args.use_float32_scale_dtype else None

# Weights: 4-bit, channelwise, symmetric, static
weights_args_4 = QuantizationArgs(
    num_bits=4,
    type=QuantizationType.INT,
    strategy=QuantizationStrategy.CHANNEL,
    symmetric=True,
    dynamic=False,
    scale_dtype=scale_dtype,
)

# Weights: 8-bit, channelwise, symmetric, static
weights_args_8 = QuantizationArgs(
    num_bits=8,
    type=QuantizationType.INT,
    strategy=QuantizationStrategy.CHANNEL,
    symmetric=True,
    dynamic=False,
    scale_dtype=scale_dtype,
)

# Activations: 8-bit, per-token, asymmetric, dynamic
activations_args = QuantizationArgs(
    num_bits=8,
    type=QuantizationType.INT,
    strategy=QuantizationStrategy.TOKEN,
    symmetric=True,
    dynamic=True,
    observer=None,
    scale_dtype=torch.float32,
)

# Routed experts always use int4 weights; shared experts follow --shared-experts-bits.
shared_experts_pattern = (
    "re:.*mlp\\.shared_experts(?:\\..+)?\\.(gate_proj|up_proj|down_proj)$"
)
experts_w4_targets = [
    "re:.*mlp\\.experts\\..*\\.(gate_proj|up_proj|down_proj)$",
]
other_linear_w8_targets = [
    "re:.*self_attn\\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)$",
    "re:.*mlp\\.(gate_proj|up_proj|down_proj)$",
]
if args.shared_experts_bits == 4:
    experts_w4_targets.append(shared_experts_pattern)
else:
    other_linear_w8_targets.append(shared_experts_pattern)

experts_w4_scheme = QuantizationScheme(
    targets=experts_w4_targets,
    weights=weights_args_4,
    input_activations=activations_args,
)

# Other linear projections use int8 weights.
other_linear_w8_scheme = QuantizationScheme(
    targets=other_linear_w8_targets,
    weights=weights_args_8,
    input_activations=activations_args,
)


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
    imatrix_fallback_stats = ImatrixFallbackStats()
    imatrix_fallback_stats.install_hooks()
    imatrix_kwargs = dict(ignore=ignores)
    if args.pipeline == "sequential":
        imatrix_kwargs["attach_by_initialize"] = False
    recipes.append(IMatrixGatherer(**imatrix_kwargs))
    tail_name += "-IMatrix"
elif args.observer == "mse":
    weights_args_4.observer = "mse"
    weights_args_8.observer = "mse"
    tail_name += "-MSE"

if args.modifier == "GPTQ":
    recipes.append(
        GPTQModifier(
            config_groups=config_groups,
            ignore=ignores,
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
oneshot_kwargs = dict(
    model=model,
    dataset=ds,
    recipe=recipes,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    batch_size=1,
    moe_calibrate_all_experts=args.moe_calibrate_all_experts,
    pipeline=args.pipeline,
)

if args.observer == "imatrix_mse":
    with imatrix_fallback_stats:
        oneshot(**oneshot_kwargs)
else:
    oneshot(**oneshot_kwargs)

# Save to disk compressed.
SAVE_NAME = model_id.rstrip("/").split("/")[-1] + tail_name + "-unpacked"
SAVE_DIR = os.path.join(args.save_dir, SAVE_NAME)

with maybe_skip_from_accelerate(args.skip_restore_from_accelerate):
    model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

if args.pack:
    packed_name = model_id.rstrip("/").split("/")[-1] + tail_name.replace(
        "-WNA8", "-W4A8"
    )
    packed_dir = args.packed_save_dir or os.path.join(args.save_dir, packed_name)
    if os.path.realpath(packed_dir) == os.path.realpath(SAVE_DIR):
        raise ValueError(
            "packed_save_dir must differ from the unpacked save directory"
        )

    # The standalone packing step previously ran after the quantization process
    # exited. Release its large objects before starting the packing subprocess.
    del model, ds, oneshot_kwargs, recipes
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "llmcompressor.utils.pack_int4_to_int8",
            "--model_path",
            SAVE_DIR,
            "--save_path",
            packed_dir,
        ],
        check=True,
    )
    print(f"Saved packed quantized model to {packed_dir}")

if args.use_original_config:
    config_output_dirs = [SAVE_DIR]
    if args.pack:
        config_output_dirs.append(packed_dir)
    for config_output_dir in config_output_dirs:
        overwrite_with_original_config(model_id, config_output_dir)
