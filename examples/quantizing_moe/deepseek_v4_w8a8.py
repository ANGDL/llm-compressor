import argparse
import json
import os
import struct
from contextlib import contextmanager
from glob import glob

import numpy as np
import torch
from datasets import load_dataset
from loguru import logger
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling.deepseek_v4 import CalibrationDeepseekV4MoE  # noqa: F401
from llmcompressor.modeling.deepseek_v4_mtp import attach_mtp_layer
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.pipelines.basic import pipeline as basic_pipeline
from llmcompressor.pipelines.data_free import pipeline as data_free_pipeline
from llmcompressor.utils import ImatrixFallbackStats
from compressed_tensors.offload import get_device_map, load_offloaded_model
from compressed_tensors.offload.dispatch import dispatch_model as _dispatch_model
from compressed_tensors.quantization import preset_name_to_scheme
from llmcompressor.logger import LoggerConfig, configure_logger

configure_logger(LoggerConfig(console_log_level="DEBUG"))


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="/ssd4/models/DeepSeek-V4-Pro")
parser.add_argument("--bf16_save_dir", type=str, default="/ssd3/models")
parser.add_argument("--save_dir", type=str, default="/ssd3/models")
parser.add_argument(
    "--step",
    type=str,
    default="all",
    choices=["all", "bf16", "int8"],
    help="Which step to run: 'bf16' (FP8/FP4→BF16 only), 'int8' (BF16→INT8 only), 'all' (both).",
)
parser.add_argument(
    "--observer",
    type=str,
    default="",
    choices=["", "mse", "imatrix_mse"],
    help="Observer for weights quantization: empty(default), mse, or imatrix_mse.",
)
parser.add_argument("--modifier", type=str, default="GPTQ", choices=["GPTQ", "RTN"])
parser.add_argument("--dataset_id", type=str, default="HuggingFaceH4/ultrachat_200k")
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
    "--dispatch_extra_memory_gb",
    type=float,
    default=20.0,
    help="Reserved memory per GPU for dispatch_model. Lower this if dispatch fails.",
)
parser.add_argument(
    "--offload_folder",
    type=str,
    default="./offload_folder",
    help="Directory used for disk offloading when loading very large models.",
)
parser.add_argument(
    "--moe-calibrate-all-experts",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to calibrate all MoE experts (not just the top-k).",
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
    "--bf16_convert_workers",
    type=int,
    default=4,
    help="Number of parallel workers for BF16 conversion.",
)
args = parser.parse_args()


def patch_pipeline_dispatch(extra_memory_gb: float):
    extra_memory_bytes = int(extra_memory_gb * 1024**3)

    def _has_disk_offload(model) -> bool:
        try:
            device_map = get_device_map(model)
        except Exception:
            return False

        return any(offload == "disk" for _onload, offload in device_map.values())

    def _dispatch_with_cap(model):
        if _has_disk_offload(model):
            logger.info(
                "Preserving existing disk offload map for basic/datafree pipeline "
                "instead of redispatching with CPU-only fallback"
            )
            return model

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


# ===========================================================================
# Module 1: FP8/FP4 → BF16 Conversion
# ===========================================================================
# DeepSeek V4 checkpoint uses mixed precision:
#   - Attention & shared experts: FP8 (F8_E4M3 weight + F8_E8M0 scale, block [128,128])
#   - Routed experts: FP4 (I8 packed weight + F8_E8M0 scale, fp4_block_size=32)
# The standard `convert_checkpoint` API doesn't support F8_E8M0 scale format,
# so we implement custom dequantization operating directly on safetensors files.
# ===========================================================================

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _e8m0_to_float(raw_bytes: bytes, shape: tuple) -> torch.Tensor:
    """Convert F8_E8M0 raw bytes to float32 tensor. Each byte = 2^(byte - 127)."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(shape).copy()
    return torch.from_numpy(arr).float().sub_(127).exp2_()


def _e4m3_to_float(raw_bytes: bytes, shape: tuple) -> torch.Tensor:
    """Convert F8_E4M3 raw bytes to float32 tensor via torch float8_e4m3fn."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(shape).copy()
    t = torch.from_numpy(arr).view(torch.float8_e4m3fn)
    return t.float()


def _dequant_fp8_block(weight_bytes: bytes, weight_shape: tuple,
                       scale_bytes: bytes, scale_shape: tuple,
                       block_size: tuple = (128, 128)) -> torch.Tensor:
    """Dequantize FP8 block-quantized weight to BF16."""
    weight = _e4m3_to_float(weight_bytes, weight_shape)
    scale = _e8m0_to_float(scale_bytes, scale_shape)

    out_dim, in_dim = weight_shape
    bh, bw = block_size
    n_row_blocks = (out_dim + bh - 1) // bh
    n_col_blocks = (in_dim + bw - 1) // bw

    # Pad if needed
    pad_h = n_row_blocks * bh - out_dim
    pad_w = n_col_blocks * bw - in_dim
    if pad_h > 0 or pad_w > 0:
        weight = torch.nn.functional.pad(weight, (0, pad_w, 0, pad_h))

    weight = weight.reshape(n_row_blocks, bh, n_col_blocks, bw).transpose(1, 2)
    weight = weight * scale.unsqueeze(-1).unsqueeze(-1)
    weight = weight.transpose(1, 2).reshape(n_row_blocks * bh, n_col_blocks * bw)

    if pad_h > 0 or pad_w > 0:
        weight = weight[:out_dim, :in_dim]

    return weight.bfloat16()


def _dequant_fp4(weight_bytes: bytes, weight_shape: tuple,
                 scale_bytes: bytes, scale_shape: tuple,
                 fp4_block_size: int = 32) -> torch.Tensor:
    """Dequantize FP4 (e2m1fn packed as I8, 2 values per byte) weight to BF16."""
    out_dim, packed_in_dim = weight_shape
    in_dim = packed_in_dim * 2

    # Read as uint8 to correctly extract nibbles regardless of sign interpretation
    arr = np.frombuffer(weight_bytes, dtype=np.uint8).reshape(weight_shape).copy()
    x = torch.from_numpy(arr)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    x_fp4 = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(1)

    scale = _e8m0_to_float(scale_bytes, scale_shape)
    # scale shape: [out_dim, in_dim / fp4_block_size]
    # Repeat scale to match in_dim
    scale = scale.unsqueeze(-1).expand(-1, -1, fp4_block_size).reshape(out_dim, in_dim)
    result = x_fp4 * scale
    return result.bfloat16()


class SafetensorsReader:
    """Low-level reader for safetensors files that handles F8_E8M0/F8_E4M3/I8 dtypes."""

    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "rb")
        header_size = struct.unpack("<Q", self._f.read(8))[0]
        self.header = json.loads(self._f.read(header_size))
        self._data_start = 8 + header_size
        self.keys = [k for k in self.header if k != "__metadata__"]

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get_tensor_info(self, key: str) -> dict:
        return self.header[key]

    def read_raw(self, key: str) -> bytes:
        info = self.header[key]
        start, end = info["data_offsets"]
        self._f.seek(self._data_start + start)
        return self._f.read(end - start)

    def get_tensor(self, key: str) -> torch.Tensor:
        """Read tensor, handling F8_E8M0, F8_E4M3, I8, BF16, F32 dtypes."""
        info = self.header[key]
        dtype_str = info["dtype"]
        shape = tuple(info["shape"])
        raw = self.read_raw(key)

        if dtype_str == "BF16":
            arr = np.frombuffer(raw, dtype=np.uint16).reshape(shape).copy()
            return torch.from_numpy(arr).view(torch.bfloat16)
        elif dtype_str == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
            return torch.from_numpy(arr)
        elif dtype_str == "F8_E4M3":
            return _e4m3_to_float(raw, shape).bfloat16()
        elif dtype_str == "F8_E8M0":
            return _e8m0_to_float(raw, shape)
        elif dtype_str in ("I8", "U8"):
            arr = np.frombuffer(raw, dtype=np.int8 if dtype_str == "I8" else np.uint8)
            return torch.from_numpy(arr.reshape(shape).copy())
        elif dtype_str == "I64":
            arr = np.frombuffer(raw, dtype=np.int64).reshape(shape).copy()
            return torch.from_numpy(arr)
        elif dtype_str == "I32":
            arr = np.frombuffer(raw, dtype=np.int32).reshape(shape).copy()
            return torch.from_numpy(arr)
        else:
            raise ValueError(f"Unsupported dtype: {dtype_str} for key {key}")


def convert_to_bf16(model_path: str, save_dir: str, max_workers: int = 4):
    """
    Convert DeepSeek V4 mixed-precision checkpoint (FP8+FP4) to BF16.

    Processes each safetensors shard independently:
    - FP8 weights (F8_E4M3 + F8_E8M0 scale): block dequantize to BF16
    - FP4 experts (I8 packed + F8_E8M0 scale): unpack and dequantize to BF16
    - BF16/F32 weights: pass through unchanged
    - .scale tensors: removed (no longer needed after dequantization)
    """
    import shutil
    from concurrent.futures import ProcessPoolExecutor, as_completed

    os.makedirs(save_dir, exist_ok=True)

    # Copy non-weight files
    for fname in os.listdir(model_path):
        if fname.endswith(".safetensors"):
            continue
        src = os.path.join(model_path, fname)
        dst = os.path.join(save_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(src):
            if not os.path.exists(dst):
                shutil.copytree(src, dst)

    # Update config.json: remove quantization_config
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config.pop("quantization_config", None)
    config["torch_dtype"] = "bfloat16"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Process each shard
    shard_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    logger.info(f"Converting {len(shard_files)} shards to BF16 in {save_dir}")

    if max_workers <= 1:
        for shard_path in shard_files:
            _convert_shard(shard_path, save_dir)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_convert_shard, sp, save_dir): sp
                for sp in shard_files
            }
            for future in as_completed(futures):
                sp = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Failed to convert {sp}: {e}")
                    raise

    # Update the index file to remove .scale entries
    index_path = os.path.join(save_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        index["weight_map"] = {
            k: v for k, v in index["weight_map"].items()
            if not k.endswith(".scale")
        }
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    logger.info(f"BF16 conversion complete: {save_dir}")


def _convert_shard(shard_path: str, save_dir: str):
    """Convert a single safetensors shard from FP8/FP4 to BF16."""
    shard_name = os.path.basename(shard_path)
    out_path = os.path.join(save_dir, shard_name)
    logger.info(f"Converting shard: {shard_name}")

    output_tensors = {}

    with SafetensorsReader(shard_path) as reader:
        # Group keys by module to pair weights with their scales
        scale_keys = {k for k in reader.keys if k.endswith(".scale")}
        weight_keys = {k for k in reader.keys if not k.endswith(".scale")}

        for key in sorted(weight_keys):
            info = reader.get_tensor_info(key)
            dtype_str = info["dtype"]
            shape = tuple(info["shape"])
            scale_key = key.rsplit(".", 1)[0] + ".scale" if key.endswith(".weight") else None

            if scale_key and scale_key in scale_keys:
                # This weight has a corresponding scale → dequantize
                scale_info = reader.get_tensor_info(scale_key)
                scale_shape = tuple(scale_info["shape"])
                weight_raw = reader.read_raw(key)
                scale_raw = reader.read_raw(scale_key)

                if dtype_str == "F8_E4M3":
                    tensor = _dequant_fp8_block(
                        weight_raw, shape, scale_raw, scale_shape
                    )
                elif dtype_str in ("I8", "U8"):
                    tensor = _dequant_fp4(
                        weight_raw, shape, scale_raw, scale_shape
                    )
                else:
                    raise ValueError(
                        f"Unexpected quantized dtype {dtype_str} for {key}"
                    )
                output_tensors[key] = tensor
            elif key.endswith(".scale"):
                # Skip standalone scale keys (already handled above)
                continue
            else:
                # Non-quantized tensor: read as-is
                output_tensors[key] = reader.get_tensor(key)

    save_file(output_tensors, out_path)
    logger.info(f"Saved: {shard_name} ({len(output_tensors)} tensors)")


# ===========================================================================
# Module 2: BF16 → INT8 Quantization
# ===========================================================================

BFLOAT16_SAVE_DIR = os.path.join(
    args.bf16_save_dir, args.model_id.rstrip("/").split("/")[-1] + "-bf16"
)

# --- Step 1: Convert to BF16 ---
if args.step in ("all", "bf16"):
    if os.path.exists(BFLOAT16_SAVE_DIR):
        logger.info(f"BF16 directory already exists: {BFLOAT16_SAVE_DIR}, skipping conversion")
    else:
        convert_to_bf16(
            model_path=args.model_id,
            save_dir=BFLOAT16_SAVE_DIR,
            max_workers=args.bf16_convert_workers,
        )

if args.step == "bf16":
    logger.info("Step 'bf16' complete. Exiting.")
    raise SystemExit(0)

# --- Step 2: Load BF16 model and quantize to INT8 ---

# Select calibration dataset.
DATASET_ID = args.dataset_id
DATASET_SPLIT = args.dataset_split
NUM_CALIBRATION_SAMPLES = args.num_calibration_samples
MAX_SEQUENCE_LENGTH = args.max_sequence_length

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

# Load the BF16 model
model_id = BFLOAT16_SAVE_DIR
with load_offloaded_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto_offload",
        offload_folder=args.offload_folder,
        max_memory={"cpu": int(args.max_memory_cpu_gb * 1e9)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

# Attach the MTP layer so it participates in the calibration forward pass and
# gets quantized end-to-end along with the main model.
attach_mtp_layer(model, model_id)


def preprocess(example):
    """Format calibration data using DeepSeek V4 chat encoding."""
    if "messages" in example:
        messages = example["messages"]
        # DeepSeek V4 uses custom encoding: BOS + system + <|User|>msg<|Assistant|></think>response<|EOS|>
        bos = "<｜begin▁of▁sentence｜>"
        eos = "<｜end▁of▁sentence｜>"
        user_token = "<｜User｜>"
        assistant_token = "<｜Assistant｜>"

        parts = [bos]
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(content)
            elif role == "user":
                parts.append(f"{user_token}{content}")
            elif role == "assistant":
                parts.append(f"{assistant_token}</think>{content}{eos}")
        return {"text": "".join(parts)}
    elif "text" in example:
        return {"text": example["text"]}


ds = ds.map(preprocess)


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Ignore list: match the original model's non-quantized layers.
# Original quantized layers (have .scale in checkpoint):
#   - Attention: wq_a, wq_b, wkv, wo_a, wo_b (→ q_a_proj, q_b_proj, kv_proj, o_a_proj, o_b_proj)
#   - Indexer: wq_b (→ indexer.wq_b or indexer.q_b_proj)
#   - Experts: w1, w2, w3 (→ gate_proj, up_proj, down_proj or gate_up_proj, down_proj)
#   - Shared experts: w1, w2, w3 (→ gate_proj, up_proj, down_proj)
#   - MTP: e_proj, h_proj (quantized in original)
#
# Non-quantized layers (no .scale):
#   - embed_tokens, lm_head/head
#   - mlp.gate (router)
#   - compressor.wgate, compressor.wkv (attention compressor)
#   - indexer.weights_proj
#   - All norm layers, attn_sink, ape (non-Linear)
ignores = [
    "lm_head",
    "re:.*embed_tokens$",
    "re:.*mlp\\.gate$",
    "re:.*self_attn\\.compressor.*",
    "re:.*self_attn\\.indexer\\.weights_proj$",
]

# Configure the quantization algorithm to run.
scheme = preset_name_to_scheme("W8A8", ["Linear"])

tail_name = "-W8A8"
recipes = []

if args.observer == "imatrix_mse":
    if scheme.weights is None:
        raise RuntimeError("W8A8 preset missing weights quantization args")
    scheme.weights.observer = "imatrix_mse"
    imatrix_fallback_stats = ImatrixFallbackStats()
    imatrix_fallback_stats.install_hooks()
    imatrix_kwargs = dict(ignore=ignores)
    if args.pipeline == "sequential":
        imatrix_kwargs["attach_by_initialize"] = False
    recipes.append(IMatrixGatherer(**imatrix_kwargs))
    tail_name += "-IMatrix"
elif args.observer == "mse":
    if scheme.weights is None:
        raise RuntimeError("W8A8 preset missing weights quantization args")
    scheme.weights.observer = "mse"
    tail_name += "-MSE"

if args.modifier == "GPTQ":
    recipes.append(
        GPTQModifier(
            config_groups={"group_0": scheme},
            ignore=ignores,
            offload_hessians=True,
        )
    )
    tail_name += "-GPTQ"
elif args.modifier == "RTN":
    recipes.append(
        QuantizationModifier(
            config_groups={"group_0": scheme},
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
SAVE_NAME = args.model_id.rstrip("/").split("/")[-1] + tail_name
SAVE_DIR = os.path.join(args.save_dir, SAVE_NAME)

with maybe_skip_from_accelerate(args.skip_restore_from_accelerate):
    model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)