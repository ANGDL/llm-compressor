import argparse
import os
from typing import Any, cast
from contextlib import contextmanager

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling.mimo_v2_flash_mtp import attach_mtp_layer
from llmcompressor.pipelines.basic import pipeline as basic_pipeline
from llmcompressor.pipelines.data_free import pipeline as data_free_pipeline
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from compressed_tensors.quantization import preset_name_to_scheme
from compressed_tensors.offload.dispatch import dispatch_model as _dispatch_model

from compressed_tensors.entrypoints.convert import (
    FP8BlockDequantizer,
    convert_checkpoint,
)


"""
MiMo-V2-Flash quantization script.

Notes vs. mimo2_5_pro_w8a8.py:
- MiMo-V2-Flash stores attention projections as separate q_proj / k_proj / v_proj
  in the FP8 checkpoint (no fused, TP-interleaved `qkv_proj`). FP8 -> BF16
  conversion therefore needs **no QKV de-interleaving** (no `--tp-size` flag,
  no fused-tensor splitting, no V-segment K-padding tricks).
- BUT: the source FP8 quantizer still computed scales per-TP-rank with each
  rank's slice padded up to a 128-row block boundary, even when the weight
  rows themselves are NOT padded in storage. This affects K / V projections
  (and SWA-K) where the per-rank row count is not a multiple of 128 — e.g.
  L0 k_proj has weight rows 768 (naive 6 blocks) but scale shape (8, 32)
  because TP=8 ranks × ceil(96/128)*128 = 8×128 = 1024 padded rows. The
  stock ``FP8BlockDequantizer`` reshapes weights with ``weight.shape//block``
  and broadcasts against the scale, which fails with a shape mismatch on
  these layers. ``MiMoV2FlashFP8BlockDequantizer`` below trusts the scale
  shape (the authoritative source), zero-pads the weight to match, then
  dequantizes and trims back to the original shape.
- MiMo-V2-Flash is intended to be used as the LLM backbone of a multimodal
  stack; this script lives under examples/multimodal_vision and ignores any
  potential vision_tower / mm_projector modules so it is forward-compatible
  with a multimodal release. For the current text-only checkpoint these
  ignores match nothing and behave as no-ops.
- The checkpoint also ships 3 MTP blocks under ``model.mtp.layers.{0,1,2}``,
  which ``MiMoV2FlashForCausalLM`` does not instantiate by itself. We attach
  them as ``model.mtp`` via ``attach_mtp_layer`` so the MTP decoder Linears
  get quantized end-to-end alongside the main stack and the saved checkpoint
  preserves the original ``model.mtp.layers.{i}.*`` naming. The MTP
  ``eh_proj`` is kept in BF16 (excluded from the quantization recipe).

Usage examples:
1. Convert FP8-block checkpoint to BF16:
python mimo2_flash_w8a8.py --convert-bf16 \
    --model_id /ssd1/models/MiMo-V2-Flash \
    --bf16_dir /ssd4/models/MiMo-V2-Flash-bf16 \
    --max_workers 8
2. Quantize the converted BF16 model to W8A8:
python mimo2_flash_w8a8.py --quantize \
    --bf16_dir /ssd4/models/MiMo-V2-Flash-bf16/ \
    --save_dir /ssd4/models/ \
    --modifier RTN --dataset_id ./ultrachat_200k --dataset_split train_sft \
    --num_calibration_samples 256 --observer imatrix_mse
"""


class MiMoV2FlashFP8BlockDequantizer(FP8BlockDequantizer):
    """FP8 block dequantizer that trusts the scale shape over the weight shape.

    The MiMo-V2-Flash FP8 checkpoint stores split q/k/v Linears (no fused
    qkv_proj), so we do NOT need TP-aware QKV de-interleaving. However the
    per-tensor block scales were computed under TP=8 with each rank's slice
    padded up to a 128-row block boundary, while the weight rows themselves
    are stored unpadded. The mismatch surfaces on K / V (and SWA-K) Linears
    where ``rows / TP`` is not a multiple of 128:

        layer 0 k_proj : weight (768, 4096)  scale (8,  32)   # naive 6 vs actual 8
        layer 0 v_proj : weight (512, 4096)  scale (8,  32)   # naive 4 vs actual 8
        SWA   k_proj   : weight (1536,4096)  scale (16, 32)   # naive 12 vs actual 16

    The stock ``FP8BlockDequantizer._create_dequantized_weight`` derives the
    block grid from ``weight.shape``, which collides with the larger scale.
    We override it to derive the padded shape from ``scale.shape *
    block_size`` (the authoritative source), zero-pad the weight, dequantize,
    and trim back. Round-trip accuracy is exact because:

        bf16 = fp8 * scale_inv               # padded zeros stay zero
        fp8' = round(bf16 / scale_inv) = fp8 # for every actual row
    """

    def _create_dequantized_weight(
        self, weight: torch.Tensor, weight_scale_inv: torch.Tensor
    ) -> torch.Tensor:
        R, C = weight.shape
        Sb, Cb = weight_scale_inv.shape
        bh, bw = cast(tuple[int, int], tuple(self.weight_block_size))

        padded_R = Sb * bh
        padded_C = Cb * bw
        if padded_R < R or padded_C < C:
            raise ValueError(
                f"weight_scale_inv shape {(Sb, Cb)} too small for weight "
                f"{(R, C)} with block ({bh}, {bw}); padded ({padded_R}, "
                f"{padded_C}) < weight."
            )

        if padded_R > R or padded_C > C:
            padded = torch.zeros(
                (padded_R, padded_C), dtype=weight.dtype, device=weight.device
            )
            padded[:R, :C] = weight
        else:
            padded = weight

        weight_blocks = padded.reshape(Sb, bh, Cb, bw).transpose(1, 2)
        scale_inv_expanded = weight_scale_inv.unsqueeze(-1).unsqueeze(-1)
        dequantized_blocks = (
            weight_blocks.to(torch.float32)
            * scale_inv_expanded.to(torch.float32)
        ).to(self.dtype)
        dequantized = dequantized_blocks.transpose(1, 2).reshape(padded_R, padded_C)
        return dequantized[:R, :C]


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


def patch_pipeline_dispatch(extra_memory_gb: float):
    extra_memory_bytes = int(extra_memory_gb * 1024**3)

    def _dispatch_with_cap(model):
        return _dispatch_model(model, extra_memory=extra_memory_bytes)

    basic_pipeline.dispatch_model = _dispatch_with_cap
    data_free_pipeline.dispatch_model = _dispatch_with_cap


patch_pipeline_dispatch(5)

MODEL_ID = "/Users/ang/models/MiMo-V2-Flash"
BFLOAT16_SAVE_DIR = os.path.join("/ssd3/models", MODEL_ID.rstrip("/").split("/")[-1] + "-bf16")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert MiMo-V2-Flash from FP8 blocks to BF16 and/or quantize it to W8A8"
    )
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--bf16_dir", type=str, default=BFLOAT16_SAVE_DIR)
    parser.add_argument("--save_dir", type=str, default="/ssd3/models")
    parser.add_argument("--observer", type=str, default=None, choices=[None, "imatrix_mse"])
    parser.add_argument("--modifier", type=str, default="GPTQ", choices=["GPTQ", "RTN"])
    parser.add_argument(
        "--convert-bf16",
        dest="convert_bf16",
        action="store_true",
        help="Convert the source model from FP8 block format back to BF16.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Run W8A8 quantization from the BF16 model directory.",
    )
    parser.add_argument("--dataset_id", type=str, default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--dataset_split", type=str, default="train_sft")
    parser.add_argument("--num_calibration_samples", type=int, default=32)
    parser.add_argument("--max_sequence_length", type=int, default=8192)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument(
        "--skip_restore_from_accelerate",
        action=argparse.BooleanOptionalAction,
        help=(
            "Skip converting model back from accelerate after save_pretrained. "
            "Useful to avoid end-of-run CUDA OOM for very large models."
        ),
        default=True,
    )
    return parser.parse_args()


def preprocess(example, tokenizer):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


def tokenize(sample, tokenizer, max_sequence_length):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=False,
    )


def convert_to_bf16(args):
    convert_checkpoint(
        model_stub=args.model_id,
        save_directory=args.bf16_dir,
        # MiMo-V2-Flash uses split q/k/v projections (no fused qkv_proj), so we
        # don't need TP-aware de-interleaving. We DO need a scale-shape-aware
        # dequantizer because the source quantizer padded each TP rank's slice
        # to the next 128-row block before computing scales — see the
        # MiMoV2FlashFP8BlockDequantizer docstring above.
        converter=MiMoV2FlashFP8BlockDequantizer(
            # All FP8-block-quantized layers found in model.safetensors.index.json:
            #   - dense + MoE expert MLPs (gate/up/down_proj), incl. MTP blocks
            #   - attention q/k/v projections, incl. MTP blocks
            # o_proj is stored as BF16 in the original checkpoint (listed in
            # the source quantization_config.ignored_layers) and is not matched
            # by the targets below.
            targets=[
                r"re:.*mlp(\.experts\.\d+)?\.(gate|up|down)_proj$",
                r"re:.*self_attn\.(q|k|v)_proj$",
            ],
        ),
        max_workers=args.max_workers,
    )


def quantize_model(args):
    if not os.path.isdir(args.bf16_dir):
        raise FileNotFoundError(
            f"BF16 model directory not found: {args.bf16_dir}. "
            f"Run with --convert-bf16 first or pass an existing --bf16_dir."
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.bf16_dir,
        dtype="auto",
        device_map=None,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.bf16_dir, trust_remote_code=True)

    # Attach the 3 MTP blocks (model.mtp.layers.{0,1,2}.*) so the MTP decoder
    # Linears participate in calibration and get quantized end-to-end. The
    # patched state_dict reverses the load-time renames so the saved checkpoint
    # uses the original key naming (incl. pre_mlp_layernorm).
    attach_mtp_layer(model, args.bf16_dir)

    ds = load_dataset(
        args.dataset_id,
        split=f"{args.dataset_split}[:{args.num_calibration_samples}]",
    )
    ds = ds.shuffle(seed=42)
    ds = ds.map(lambda example: preprocess(example, tokenizer))
    column_names = ds.column_names
    if not isinstance(column_names, list):
        raise TypeError(f"Expected dataset column_names to be a list, got {type(column_names)!r}")
    ds = ds.map(
        lambda sample: tokenize(sample, tokenizer, args.max_sequence_length),
        remove_columns=column_names,
    )

    ignores = [
        # MoE router (uses sigmoid scoring, not a quantizable Linear in the usual sense)
        r"re:.*mlp\.gate$",
        # r"re:.*self_attn.o_proj$",
        # Token embedding & language head
        r"re:.*\.embed_tokens$",
        "lm_head",
        # MTP-specific projection — kept in BF16 so downstream NextN loaders
        # (e.g. sglang ``mimo_v2_nextn``) can use it directly. ``re:.*eh_proj$``
        # also matches the attached ``mtp.<i>.eh_proj``.
        r"re:.*eh_proj$",
        # MTP attention output — stored as BF16 in the source FP8 checkpoint
        # (model.mtp.layers.<i>.self_attn.o_proj has no weight_scale_inv).
        # Keep MTP's INT8 quantization scope aligned with the FP8 scope so the
        # round-trip is faithful. Main-model o_proj is still quantized; only
        # MTP's o_proj is excluded here.
        r"re:^model\.mtp\.layers\.\d+\.self_attn\.o_proj$",
        # Vision branch placeholders — no-ops on the current text-only Flash
        # checkpoint, kept for forward-compatibility with a multimodal release.
        r"re:vision_tower\.*",
        r"re:mm_projector\.*",
    ]

    scheme = preset_name_to_scheme("W8A8", ["Linear"])

    tail_name = "-W8A8"
    recipes = []
    if args.observer == "imatrix_mse":
        if scheme.weights is None:
            raise ValueError("W8A8 scheme is missing weight quantization settings")
        scheme.weights.observer = "imatrix_mse"
        recipes.append(IMatrixGatherer(ignore=ignores))
        tail_name += "-IMatrix"

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

    oneshot(
        model=model,
        dataset=cast(Any, ds),
        recipe=recipes,
        max_seq_length=args.max_sequence_length,
        num_calibration_samples=args.num_calibration_samples,
        batch_size=1,
        sequential_targets=['MiMoV2Attention', 'MiMoV2MLP'],
        trust_remote_code_model=True
    )

    save_name = args.bf16_dir.rstrip("/").split("/")[-1] + tail_name
    output_path = os.path.join(args.save_dir, save_name)

    with maybe_skip_from_accelerate(args.skip_restore_from_accelerate):
        model.save_pretrained(output_path, save_compressed=True)
    tokenizer.save_pretrained(output_path)
    print(f"Saved quantized model to {output_path}")


def main():
    args = parse_args()
    run_convert = args.convert_bf16
    run_quantize = args.quantize
    if not run_convert and not run_quantize:
        run_convert = True
        run_quantize = True

    if run_convert:
        convert_to_bf16(args)

    if run_quantize:
        quantize_model(args)


if __name__ == "__main__":
    main()
