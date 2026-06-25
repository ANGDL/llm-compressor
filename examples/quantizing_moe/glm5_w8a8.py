import argparse
import os
from contextlib import contextmanager

from datasets import load_dataset
from loguru import logger
from torch.utils._pytree import tree_leaves
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling.glm_moe_dsa import CalibrationGlmMoeDsaMoE  # noqa: F401
from llmcompressor.modeling.glm_moe_dsa_mtp import attach_mtp_layer
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.autosmooth import AutoSmoothModifier
from llmcompressor.modifiers.transform.awq.mappings import AWQMapping
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.pipelines.basic import pipeline as basic_pipeline
from llmcompressor.pipelines.data_free import pipeline as data_free_pipeline
from llmcompressor.utils.pytorch.module import get_module_to_name_dict
from llmcompressor.utils import ImatrixFallbackStats
from compressed_tensors.offload import get_device_map, load_offloaded_model
from compressed_tensors.offload.dispatch import dispatch_model as _dispatch_model
from compressed_tensors.quantization import preset_name_to_scheme
from compressed_tensors.utils import match_modules_set
from llmcompressor.logger import LoggerConfig, configure_logger

configure_logger(LoggerConfig(console_log_level="DEBUG"))


"""
Usage example for quantizing GLM-5.1 with MoE layers to W8A8 using LLM Compressor.
1. 更改权重文件中的generation_config.json, 添加："do_sample": true
2. 量化
python glm5_w8a8.py --model_id /ssd3/models/GLM-5.1/ --save_dir /ssd2/models/ --transform "" --modifier RTN  --observer imatrix_mse  \
    --num_calibration_samples 512 --max_sequence_length 2048 --dataset_id ./ultrachat_200k --dataset_split train_sft  \
    --indexer-ignore-mode indexer_all --dispatch_extra_memory_gb 10 --pipeline sequential
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
parser.add_argument(
    "--transform",
    type=str,
    default="",
    choices=["", "AutoSmooth", "SmoothQuant"],
    help="Optional transform: empty(default), AutoSmooth, or SmoothQuant.",
)
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
    "--print_autosmooth_matches",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Print resolved AutoSmooth mappings before compression for debugging.",
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
    "--autosmooth-activation-scale-type",
    type=str,
    default="minmax",
    choices=["mean", "max", "minmax"],
    help=(
        "AutoSmooth activation scale aggregation method: mean / max / minmax. "
        "Forwarded to AutoSmoothModifier.activation_scale_type."
    ),
)
parser.add_argument(
    "--autosmooth-norm-func",
    type=str,
    default="",
    choices=["", "awq", "adaptive"],
    help=(
        "Normalization function applied to scales during AutoSmooth grid search. "
        "'awq' use the modifier default (standard AWQ normalization, "
        "norm_func=None). 'adaptive' enables adaptive normalization with "
        "alpha/beta from --autosmooth-norm-func-param."
    ),
)
parser.add_argument(
    "--autosmooth-norm-func-param",
    type=str,
    default="6.0,0.15",
    help=(
        "Only used when --autosmooth-norm-func=adaptive. Format 'alpha,beta', "
        "default 6.0,0.15. Ignored otherwise."
    ),
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


# Select calibration dataset.
DATASET_ID = args.dataset_id
DATASET_SPLIT = args.dataset_split
NUM_CALIBRATION_SAMPLES = args.num_calibration_samples
MAX_SEQUENCE_LENGTH = args.max_sequence_length


# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

# Load the model
model_id = args.model_id
with load_offloaded_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto_offload",
        offload_folder=args.offload_folder,
        max_memory={"cpu": int(args.max_memory_cpu_gb * 1e9)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
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


ds = ds.map(preprocess)
print(ds[0])


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

autosmooth_ignores = [
    pattern for pattern in ignores if pattern != INDEXER_WEIGHTS_PROJ_PATTERN
]

# --------------------------------------------------------------------------
# AutoSmooth / SmoothQuant 映射
#
# GLM-5.2 引入了「跨层 top-k 共享」：config.indexer_types 中标记为 "shared"
# 的层不构造自己的 self_attn.indexer.* 子模块，而是复用前一个 "full" 层的 top-k。
# 因此一条混合写法（``re:.*input_layernorm$`` 配 ``re:.*indexer\.wk$``）在
# 5.2 上会让 ``match_modules_set`` 在 shared 层一直找不齐 indexer 子模块，
# 把多层的 input_layernorm 累积进同一个 group，最终触发
# ``Smooth needs to match a single smoothlayer ...`` 错误。
#
# 按 ``indexer_types`` 把映射拆成 full / shared 两组——full 层
# （以及 MTP 解码器，attach_mtp_layer 已根据 checkpoint 把它配为 "full"）继续
# 把 indexer.{wk,weights_proj,wq_b} 加进 balance_layers，shared 层只 balance 主
# 注意力分支。GLM-5.1 的 indexer_types 全是 "full"，shared 组为空，行为退化为原始
# 4 条映射，与原 5.1 行为完全一致。
# --------------------------------------------------------------------------
_indexer_types: list[str] = (
    list(model.config.indexer_types)
    if getattr(model.config, "indexer_types", None) is not None
    else ["full"] * model.config.num_hidden_layers
)
_full_layer_idx = [i for i, t in enumerate(_indexer_types) if t == "full"]
_shared_layer_idx = [i for i, t in enumerate(_indexer_types) if t == "shared"]


def _alt(idxs: list[int]) -> str:
    """把层号列表拼成 ``(?:0|1|2|6)`` 形式的正则非捕获组。"""
    return "(?:" + "|".join(str(i) for i in idxs) + ")"


# full 前缀：匹配主模型里所有 "full" 层 + 通过 attach_mtp_layer 挂上的 mtp.decoder.*
# （MTP 解码器自带完整的 indexer 权重，被强制配为 "full"）。
_full_prefix = (
    rf"(?:^model\.layers\.{_alt(_full_layer_idx)}\.|^mtp\.decoder\.)"
    if _full_layer_idx
    else r"^mtp\.decoder\."
)
# shared 前缀：仅在 GLM-5.2 上非空。
_shared_prefix = (
    rf"^model\.layers\.{_alt(_shared_layer_idx)}\."
    if _shared_layer_idx
    else None
)

autosmooth_mappings: list[AWQMapping] = [
    # ---- full 层 + MTP：input_layernorm → q_a_proj / kv_a_proj_with_mqa / indexer.wk / indexer.weights_proj
    AWQMapping(
        smooth_layer=rf"re:{_full_prefix}input_layernorm$",
        balance_layers=[
            rf"re:{_full_prefix}self_attn\.q_a_proj$",
            rf"re:{_full_prefix}self_attn\.kv_a_proj_with_mqa$",
            rf"re:{_full_prefix}self_attn\.indexer\.wk$",
            # weights_proj 与 wk/q_a/kv_a 共用同一份 input_layernorm 输出，
            # 在这里一并 balance，保证 smoothing 后 indexer 打分路径一致。
            rf"re:{_full_prefix}self_attn\.indexer\.weights_proj$",
        ],
    ),
    # ---- full 层 + MTP：q_a_layernorm → q_b_proj / indexer.wq_b
    AWQMapping(
        smooth_layer=rf"re:{_full_prefix}self_attn\.q_a_layernorm$",
        balance_layers=[
            rf"re:{_full_prefix}self_attn\.q_b_proj$",
            rf"re:{_full_prefix}self_attn\.indexer\.wq_b$",
        ],
    ),
    # ---- 通用：kv_a_layernorm → kv_b_proj（full / shared 层结构相同）
    AWQMapping(
        smooth_layer=r"re:.*kv_a_layernorm$",
        balance_layers=[r"re:.*kv_b_proj$"],
    ),
    # ---- 通用：up_proj → down_proj（dense MLP / shared_experts / 每个 routed expert 都覆盖）
    AWQMapping(
        smooth_layer=r"re:.*up_proj$",
        balance_layers=[r"re:.*down_proj$"],
    ),
]

# GLM-5.2 才会进这两条；5.1 上 _shared_prefix 为 None，跳过即可。
if _shared_prefix is not None:
    autosmooth_mappings.insert(
        1,
        AWQMapping(
            smooth_layer=rf"re:{_shared_prefix}input_layernorm$",
            balance_layers=[
                rf"re:{_shared_prefix}self_attn\.q_a_proj$",
                rf"re:{_shared_prefix}self_attn\.kv_a_proj_with_mqa$",
            ],
        ),
    )
    autosmooth_mappings.insert(
        3,
        AWQMapping(
            smooth_layer=rf"re:{_shared_prefix}self_attn\.q_a_layernorm$",
            balance_layers=[rf"re:{_shared_prefix}self_attn\.q_b_proj$"],
        ),
    )

smoothquant_mappings: list[tuple | list] = [
    [mapping.balance_layers, mapping.smooth_layer] for mapping in autosmooth_mappings
]


def print_autosmooth_matches(model, mappings: list[AWQMapping]):
    module_to_name = get_module_to_name_dict(model)
    matched_weights_proj = False

    print("[AutoSmooth] Resolved mappings:")
    resolved_count = 0
    for mapping in mappings:
        for smooth_layers, *nested_balance_layers in match_modules_set(
            model, (mapping.smooth_layer, *mapping.balance_layers)
        ):
            if len(smooth_layers) != 1:
                continue

            resolved_count += 1
            smooth_name = module_to_name.get(smooth_layers[0], "<unnamed>")
            balance_layers = tree_leaves(nested_balance_layers)
            balance_names = [
                module_to_name.get(layer, "<unnamed>") for layer in balance_layers
            ]

            if any(name.endswith("self_attn.indexer.weights_proj") for name in balance_names):
                matched_weights_proj = True

            print(f"  [{resolved_count}] smooth: {smooth_name}")
            for name in balance_names:
                print(f"       -> {name}")

    print(f"[AutoSmooth] total resolved mappings: {resolved_count}")
    print(f"[AutoSmooth] indexer.weights_proj matched: {matched_weights_proj}")


if args.print_autosmooth_matches:
    print_autosmooth_matches(model, autosmooth_mappings)

# Configure the quantization algorithm to run.
scheme = preset_name_to_scheme("W8A8", ["Linear"])

tail_name = "-W8A8"
recipes = []

if args.transform == "AutoSmooth":
    autosmooth_kwargs = dict(
        ignore=autosmooth_ignores,
        activation_scale_type=args.autosmooth_activation_scale_type,
        mappings=autosmooth_mappings,
    )
    if args.autosmooth_norm_func == "adaptive":
        try:
            alpha_s, beta_s = args.autosmooth_norm_func_param.split(",")
            alpha, beta = float(alpha_s), float(beta_s)
        except ValueError as e:
            raise ValueError(
                f"--autosmooth-norm-func-param 应为 'alpha,beta'，"
                f"收到: {args.autosmooth_norm_func_param!r}"
            ) from e
        autosmooth_kwargs["norm_func"] = "adaptive"
        autosmooth_kwargs["norm_func_param"] = {
            "adaptive": {"alpha": alpha, "beta": beta}
        }
    # "awq" / "" 走默认（norm_func=None）
    recipes.append(AutoSmoothModifier(**autosmooth_kwargs))
    tail_name += "-AutoSmooth"
elif args.transform == "SmoothQuant":
    recipes.append(
        SmoothQuantModifier(
            mappings=smoothquant_mappings,
            ignore=autosmooth_ignores,
            smoothing_strength=0.75,
        ),
    )
    tail_name += "-SmoothQuant"

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
SAVE_NAME = model_id.rstrip("/").split("/")[-1] + tail_name
SAVE_DIR = os.path.join(args.save_dir, SAVE_NAME)

with maybe_skip_from_accelerate(args.skip_restore_from_accelerate):
    model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
