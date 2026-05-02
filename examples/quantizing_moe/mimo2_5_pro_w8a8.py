import argparse
import json
import math
import os
import re
from typing import Any, cast
from contextlib import contextmanager

from datasets import load_dataset
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
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
from compressed_tensors.utils.match import match_quantizable_tensors


"""
Usage example:
1. Convert FP8-block checkpoint to BF16:
python mimo2_5_pro_w8a8.py  --convert-bf16 --model_id /ssd1/models/MiMo-V2.5-Pro --bf16_dir /ssd3/models/MiMo-V2.5-Pro-bf16 --max_workers 1 --tp-size 8
2. Quantize the converted BF16 model to W8A8:
python mimo2_5_pro_w8a8.py --quantize --bf16_dir /ssd3/models/MiMo-V2.5-Pro-bf16/ --save_dir /ssd2/models/  \
    --modifier RTN --dataset_id  ./ultrachat_200k --dataset_split train_sft --num_calibration_samples 256 --observer imatrix_mse  --tp-size 8
"""

class MiMoFP8BlockDequantizer(FP8BlockDequantizer):
    def __init__(self, model_stub: str, tp_size: int, verify_roundtrip: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.tp_size = tp_size
        self.verify_roundtrip = verify_roundtrip
        config = AutoConfig.from_pretrained(model_stub, trust_remote_code=True)

        self.default_q_rows, self.default_k_rows, self.default_v_rows = self._get_qkv_rows(
            num_attention_heads=getattr(config, "num_attention_heads"),
            num_key_value_heads=getattr(config, "num_key_value_heads"),
            head_dim=getattr(config, "head_dim"),
            v_head_dim=getattr(config, "v_head_dim", getattr(config, "head_dim")),
        )
        self.swa_q_rows, self.swa_k_rows, self.swa_v_rows = self._get_qkv_rows(
            num_attention_heads=getattr(
                config,
                "swa_num_attention_heads",
                getattr(config, "num_attention_heads"),
            ),
            num_key_value_heads=getattr(
                config,
                "swa_num_key_value_heads",
                getattr(config, "num_key_value_heads"),
            ),
            head_dim=getattr(config, "swa_head_dim", getattr(config, "head_dim")),
            v_head_dim=getattr(
                config,
                "swa_v_head_dim",
                getattr(config, "v_head_dim", getattr(config, "head_dim")),
            ),
        )
        hybrid_pattern = getattr(config, "hybrid_layer_pattern", None)
        self.hybrid_layer_pattern = (
            list(hybrid_pattern) if isinstance(hybrid_pattern, (list, tuple)) else None
        )

    @staticmethod
    def _get_qkv_rows(
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        v_head_dim: int,
    ) -> tuple[int, int, int]:
        return (
            num_attention_heads * head_dim,
            num_key_value_heads * head_dim,
            num_key_value_heads * v_head_dim,
        )

    def _qkv_rows_for_module(self, module_name: str) -> tuple[int, int, int]:
        match = re.search(r"\.layers\.(\d+)\.self_attn\.qkv_proj$", module_name)
        if not match or self.hybrid_layer_pattern is None:
            return self.default_q_rows, self.default_k_rows, self.default_v_rows

        layer_idx = int(match.group(1))
        if layer_idx >= len(self.hybrid_layer_pattern):
            return self.default_q_rows, self.default_k_rows, self.default_v_rows

        is_swa_layer = self.hybrid_layer_pattern[layer_idx] == 1
        if is_swa_layer:
            return self.swa_q_rows, self.swa_k_rows, self.swa_v_rows
        return self.default_q_rows, self.default_k_rows, self.default_v_rows

    def process(self, tensors: dict[str, torch.Tensor]):
        for module_name, name in match_quantizable_tensors(
            tensors, self.ignore, self.targets, allow_nonquantizable=True
        ):
            param_name = name.rsplit(".", 1)[-1]
            if param_name != "weight":
                continue

            scale_name = f"{module_name}.weight_scale_inv"
            if scale_name not in tensors:
                raise KeyError(
                    f"Missing required scale tensor {scale_name} for {module_name}.weight"
                )

            if module_name.endswith("self_attn.qkv_proj"):
                q_rows, k_rows, v_rows = self._qkv_rows_for_module(module_name)
                num_groups = self.tp_size
                weight = tensors[f"{module_name}.weight"]
                scale = tensors[scale_name]
                block_height, block_width = cast(tuple[int, int], self.weight_block_size)

                # Per-group row counts.  The checkpoint is stored in TP-interleaved order:
                #   [Q_heads[0:nq/g], K_head[0], V_head[0],
                #    Q_heads[nq/g:2*nq/g], K_head[1], V_head[1], ...]
                # so that chunk(num_groups)[rank] gives each TP rank exactly its
                # (q_per + k_per + v_per) rows.  The original quantizer padded V per
                # group from v_per to k_per (= k_rows/num_groups) before computing
                # block scales, then trimmed V back; this explains scale.shape[0] > 212.
                q_per, q_rem = divmod(q_rows, num_groups)
                k_per, k_rem = divmod(k_rows, num_groups)
                v_per, v_rem = divmod(v_rows, num_groups)
                if q_rem or k_rem or v_rem:
                    raise ValueError(
                        f"{module_name}: q/k/v rows ({q_rows}/{k_rows}/{v_rows}) "
                        f"not evenly divisible by num_groups={num_groups}"
                    )
                chunk_rows = q_per + k_per + v_per

                blocks_per_group, rem = divmod(scale.shape[0], num_groups)
                if rem != 0:
                    raise ValueError(
                        f"{module_name}: scale rows {scale.shape[0]} not divisible by "
                        f"num_groups={num_groups}; checkpoint may be corrupt"
                    )
                if weight.shape[0] != num_groups * chunk_rows:
                    raise ValueError(
                        f"{module_name}: weight rows {weight.shape[0]} != "
                        f"num_groups×chunk_rows = {num_groups}×{chunk_rows}={num_groups*chunk_rows}"
                    )
                total_padded_per = blocks_per_group * block_height
                v_padded_per = total_padded_per - q_per - k_per
                if v_padded_per < v_per:
                    raise ValueError(
                        f"{module_name}: inferred v_padded_per={v_padded_per} < v_per={v_per}; "
                        f"scale may have too few blocks"
                    )

                # De-interleave: dequantize each group's slice, then cat across groups.
                # _dequant_segment treats scale as authoritative, so padding (chunk_rows <
                # total_padded_per) is zero-filled and trimmed back automatically.
                q_pieces: list[torch.Tensor] = []
                k_pieces: list[torch.Tensor] = []
                v_pieces: list[torch.Tensor] = []
                fp8_q_pieces: list[torch.Tensor] = []
                fp8_k_pieces: list[torch.Tensor] = []
                fp8_v_pieces: list[torch.Tensor] = []

                for g in range(num_groups):
                    w_grp = weight[g * chunk_rows : (g + 1) * chunk_rows]
                    s_grp = scale[g * blocks_per_group : (g + 1) * blocks_per_group]
                    bf16_grp = self._dequant_segment(w_grp, s_grp)

                    q_pieces.append(bf16_grp[:q_per])
                    k_pieces.append(bf16_grp[q_per : q_per + k_per])
                    v_pieces.append(bf16_grp[q_per + k_per :])

                    if self.verify_roundtrip:
                        fp8_q_pieces.append(w_grp[:q_per])
                        fp8_k_pieces.append(w_grp[q_per : q_per + k_per])
                        fp8_v_pieces.append(w_grp[q_per + k_per :])

                attn_module = module_name.rsplit(".", 1)[0]
                tensors[f"{attn_module}.q_proj.weight"] = torch.cat(q_pieces, dim=0)
                tensors[f"{attn_module}.k_proj.weight"] = torch.cat(k_pieces, dim=0)
                tensors[f"{attn_module}.v_proj.weight"] = torch.cat(v_pieces, dim=0)

                if self.verify_roundtrip:
                    fp8_dtype = weight.dtype
                    self._print_roundtrip_stats(
                        f"{attn_module}.q_proj",
                        torch.cat(fp8_q_pieces, dim=0),
                        self._requant_segment(tensors[f"{attn_module}.q_proj.weight"], fp8_dtype),
                    )
                    self._print_roundtrip_stats(
                        f"{attn_module}.k_proj",
                        torch.cat(fp8_k_pieces, dim=0),
                        self._requant_segment(tensors[f"{attn_module}.k_proj.weight"], fp8_dtype),
                    )
                    self._print_roundtrip_stats(
                        f"{attn_module}.v_proj",
                        torch.cat(fp8_v_pieces, dim=0),
                        self._requant_segment(tensors[f"{attn_module}.v_proj.weight"], fp8_dtype),
                    )

                bias_name = f"{module_name}.bias"
                if bias_name in tensors:
                    q_b_pieces, k_b_pieces, v_b_pieces = [], [], []
                    bias = tensors[bias_name]
                    for g in range(num_groups):
                        b_grp = bias[g * chunk_rows : (g + 1) * chunk_rows]
                        q_b_pieces.append(b_grp[:q_per])
                        k_b_pieces.append(b_grp[q_per : q_per + k_per])
                        v_b_pieces.append(b_grp[q_per + k_per :])
                    tensors[f"{attn_module}.q_proj.bias"] = torch.cat(q_b_pieces, dim=0)
                    tensors[f"{attn_module}.k_proj.bias"] = torch.cat(k_b_pieces, dim=0)
                    tensors[f"{attn_module}.v_proj.bias"] = torch.cat(v_b_pieces, dim=0)
                    del tensors[bias_name]

                del tensors[f"{module_name}.weight"]
                del tensors[scale_name]
            else:
                fp8_weight_orig = tensors[f"{module_name}.weight"]
                scale_inv_orig = tensors[scale_name]
                tensors[f"{module_name}.weight"] = self._create_dequantized_weight(
                    fp8_weight_orig,
                    scale_inv_orig,
                )
                if self.verify_roundtrip:
                    self._print_roundtrip_stats(
                        module_name,
                        fp8_weight_orig,
                        self._requant_segment(
                            tensors[f"{module_name}.weight"], fp8_weight_orig.dtype
                        ),
                    )
                del tensors[scale_name]

    def _dequant_segment(
        self, weight: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize a single fp8 weight segment using its block scale tensor.

        The scale tensor is authoritative: padded dimensions are inferred from
        scale.shape * block_size rather than from the weight shape. This handles
        the case where the original quantizer computed scales for a padded weight
        (e.g. V segment padded to K segment height) but stored only the trimmed
        weight rows. Round-trip accuracy (fp8→bf16→fp8) is bit-for-bit exact for
        all actual weight rows since dequant is: bf16 = fp8 * scale_inv, and
        requant is: fp8' = round(bf16 * scale) = round(fp8 * 1.0) = fp8.
        """
        R, C = weight.shape
        block_height, block_width = cast(tuple[int, int], self.weight_block_size)
        Sb, Cb = scale.shape

        expected_Cb = math.ceil(C / block_width)
        if Cb != expected_Cb:
            raise ValueError(
                f"Scale columns {Cb} do not match expected {expected_Cb} "
                f"for weight columns {C} with block_width {block_width}"
            )
        if Sb < math.ceil(R / block_height):
            raise ValueError(
                f"Scale rows {Sb} are insufficient for weight rows {R} "
                f"with block_height {block_height}"
            )

        padded_rows = Sb * block_height
        padded_cols = Cb * block_width

        if padded_rows > R or padded_cols > C:
            padded_weight = torch.zeros(
                (padded_rows, padded_cols),
                dtype=weight.dtype,
                device=weight.device,
            )
            padded_weight[:R, :C] = weight
        else:
            padded_weight = weight

        # Reshape to (Sb, block_height, Cb, block_width), transpose to (Sb, Cb, bh, bw)
        weight_blocks = padded_weight.reshape(Sb, block_height, Cb, block_width).transpose(1, 2)

        dequantized_blocks = (
            weight_blocks.to(torch.float32)
            * scale.unsqueeze(-1).unsqueeze(-1).to(torch.float32)
        ).to(self.dtype)

        dequantized = dequantized_blocks.transpose(1, 2).reshape(padded_rows, padded_cols)
        return dequantized[:R, :C]

    def _requant_segment(
        self, bf16_weight: torch.Tensor, fp8_dtype: torch.dtype
    ) -> torch.Tensor:
        """Re-quantize BF16 weight to FP8 by computing fresh block-wise scales.

        Scale is derived from the BF16 data (not the original scale_inv), so this
        is a real end-to-end validation: if the dequantization was correct the
        re-quantized values should match the original FP8 weights bit-for-bit.
        """
        R, C = bf16_weight.shape
        block_height, block_width = cast(tuple[int, int], self.weight_block_size)
        Sb = math.ceil(R / block_height)
        Cb = math.ceil(C / block_width)

        padded_rows = Sb * block_height
        padded_cols = Cb * block_width

        if padded_rows > R or padded_cols > C:
            padded = torch.zeros(
                (padded_rows, padded_cols), dtype=bf16_weight.dtype, device=bf16_weight.device
            )
            padded[:R, :C] = bf16_weight
        else:
            padded = bf16_weight

        fp8_max = torch.finfo(fp8_dtype).max  # 448.0 for e4m3fn
        blocks = padded.reshape(Sb, block_height, Cb, block_width).transpose(1, 2).to(torch.float32)
        # (Sb, Cb, block_height, block_width)

        block_abs_max = blocks.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
        scale = block_abs_max / fp8_max
        fp8_blocks = (blocks / scale).to(fp8_dtype)

        fp8 = fp8_blocks.transpose(1, 2).reshape(padded_rows, padded_cols)
        return fp8[:R, :C]

    def _print_roundtrip_stats(
        self, name: str, fp8_orig: torch.Tensor, fp8_requant: torch.Tensor,
        warn_threshold: float = 0.01,
    ) -> None:
        """Print element-wise mismatch stats between original and re-quantized FP8 weights.

        A small number of mismatches is expected when using fresh block scales (vs the
        original scale_inv), because scale_inv is stored in BF16 precision and the new
        scale derived from the dequantized BF16 data may differ slightly at block boundaries.
        warn_threshold (default 0.01%) distinguishes negligible precision noise from real bugs.
        """
        total = fp8_orig.numel()
        mismatch = (fp8_orig.view(torch.uint8) != fp8_requant.view(torch.uint8)).sum().item()
        pct = 100.0 * mismatch / total if total > 0 else 0.0
        if mismatch == 0:
            status = "OK"
        elif pct < warn_threshold:
            status = "WARN"
        else:
            status = "FAIL"
        print(f"  [{status}] {name}: {total:,} elements, {mismatch:,} mismatches ({pct:.4f}%)")

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


patch_pipeline_dispatch(20)

MODEL_ID = "/Users/ang/models/MiMo-V2.5-Pro"
BFLOAT16_SAVE_DIR = os.path.join("/ssd3/models", MODEL_ID.rstrip("/").split("/")[-1] + "-bf16")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert MiMo-V2.5 from FP8 blocks to BF16 and/or quantize it to W8A8"
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
    parser.add_argument(
        "--tp-size",
        dest="tp_size",
        type=int,
        required=True,
        help="Tensor-parallel size used when the FP8 checkpoint was created (e.g. 8).",
    )
    parser.add_argument("--dataset_id", type=str, default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--dataset_split", type=str, default="train_sft")
    parser.add_argument("--num_calibration_samples", type=int, default=32)
    parser.add_argument("--max_sequence_length", type=int, default=8192)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument(
        "--verify-roundtrip",
        dest="verify_roundtrip",
        action="store_true",
        help=(
            "After FP8→BF16 dequantization, re-quantize BF16→FP8 and compare "
            "element-wise with original FP8 weights to validate round-trip accuracy."
        ),
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
        converter=MiMoFP8BlockDequantizer(
            model_stub=args.model_id,
            tp_size=args.tp_size,
            verify_roundtrip=args.verify_roundtrip,
            # MiMo-V2.5 fp8-block-quantized layers, found in model.safetensors.index.json
            targets=[
                r"re:.*mlp(\.experts\.\d+)?\.(gate|up|down)_proj$",
                r"re:.*self_attn\.qkv_proj$",
            ],
        ),
        max_workers=args.max_workers,
    )

    # The converted checkpoint stores Q/K/V as separate projections.
    # Ensure config matches the exported parameter layout.
    config_path = os.path.join(args.bf16_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    config_data["attention_projection_layout"] = "split"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=True, indent=2)
        f.write("\n")


def quantize_model(args):
    if not os.path.isdir(args.bf16_dir):
        raise FileNotFoundError(
            f"BF16 model directory not found: {args.bf16_dir}. Run with --convert-bf16 first or pass an existing --bf16_dir."
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.bf16_dir,
        dtype="auto",
        device_map=None,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.bf16_dir, trust_remote_code=True)

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
        # Ignore the output head
        r"re:.*mlp.gate$",
        # r"re:.*self_attn.o_proj$",
        r"re:.*\.embed_tokens$",
        "lm_head",
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
        sequential_targets=['MiMoV2MLP', 'MiMoV2Attention']
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
