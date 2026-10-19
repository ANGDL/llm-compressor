"""Build and text-finetune a full-topology DeepSeek-V4.1 CPU fixture.

Preserves 40 text layers, 32 ViT layers, 3 DSpark stages, both Engram tables,
all CSA2 sources/reusers/reindexers, the real tokenizer and original raw names.
Channels, expert counts and table buckets are reduced to fit a CPU development
machine. Saves both a reloadable native checkpoint and an FP8/FP4 source fixture.

python tools/create_tiny_dsv41.py --source /path/to/DeepSeek-V4.1-Flash \
    --output .cache/tiny-models/DeepSeek-V4.1-Flash-Tiny
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.nn import functional as F
from transformers import AutoTokenizer

from llmcompressor.modeling.deepseekv41 import (
    DeepseekV41NativeForConditionalGeneration,
    ModelConfig,
)
from llmcompressor.modeling.deepseekv41.engram import _next_prime
from llmcompressor.modeling.deepseekv41.model import Head

# The text-only toy corpus from the create-tiny-model finetuning template.
TOY_TEXTS = [
    "According to all known laws of aviation, there is no way a bee should be able "
    "to fly. Its wings are too small to get its fat little body off the ground. "
    "The bee, of course, flies anyway because bees don't care what humans think "
    "is impossible.",
    "I'd just like to interject for a moment. What you're referring to as Linux, "
    "is in fact, GNU/Linux, or as I've recently taken to calling it, GNU plus Linux. "
    "Linux is not an operating system unto itself, but rather another free component "
    "of a fully functioning GNU system made useful by the GNU corelibs, shell "
    "utilities and vital system components comprising a full OS as defined by POSIX.",
    "The FitnessGram Pacer Test is a multistage aerobic capacity test that "
    "progressively gets more difficult as it continues. The 20 meter pacer test "
    "will begin in 30 seconds. Line up at the start. The running speed starts "
    "slowly, but gets faster each minute after you hear this signal.",
    "Did you ever hear the tragedy of Darth Plagueis The Wise? I thought not. "
    "It's not a story the Jedi would tell you. It's a Sith legend. Darth Plagueis "
    "was a Dark Lord of the Sith, so powerful and so wise he could use the Force "
    "to influence the midichlorians to create life.",
    "Hello my name is Inigo Montoya. You killed my father. Prepare to die",
    "The capital of France is Paris. Paris is the largest city in France and "
    "serves as the country's political, economic, and cultural center.",
    "The creator of the theory of relativity was Albert Einstein. Einstein "
    "developed the special theory of relativity in 1905 and the general theory "
    "of relativity in 1915.",
    "The classic game is called rock, paper, scissors. Rock beats scissors, "
    "scissors beats paper, and paper beats rock.",
]


def build_tiny_config(source: Path) -> ModelConfig:
    value = json.loads((source / "config.json").read_text())
    # Layers alone cannot fit the 196B Engram tables. Keep the full layer
    # topology and shrink widths, experts and prime buckets for CPU execution.
    text = value["text_config"]
    text.update(
        hidden_size=64,
        moe_intermediate_size=64,
        n_routed_experts=8,
        q_lora_rank=64,
        head_dim=64,
        qk_rope_head_dim=16,
        num_attention_heads=4,
        o_groups=2,
        o_lora_rank=32,
        index_n_heads=2,
        index_head_dim=32,
        index_topk=4,
        sliding_window=16,
        max_position_embeddings=256,
        max_seq_len=256,
        max_batch_size=2,
        candidate_topk_blocks=4,
        candidate_block_size=2,
        engram_vocab_size=127,
        engram_n_heads=2,
        engram_head_dim=32,
        dspark_n_routed_experts=4,
        dspark_num_experts_per_tok=3,
        dspark_markov_rank=32,
        use_cache=False,
    )
    text["rope_scaling"] = {
        "type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 16,
        "beta_fast": 32,
        "beta_slow": 1,
    }
    seen, sizes = set(), []
    for _ in text["engram_layer_ids"]:
        total = 0
        for _ in range(text["engram_max_ngram_size"] - 1):
            current = text["engram_vocab_size"] - 1
            for _ in range(text["engram_n_heads"]):
                current = _next_prime(current, seen)
                total += current
        sizes.append(total)
    text["engram_num_embeddings"] = sizes
    value["vision_config"].update(
        hidden_size=32, num_attention_heads=2, intermediate_size=64, patch_size=2
    )
    value.pop("quantization_config", None)
    value["source_quantization_config"] = {}
    value["dtype"] = "float32"
    value["use_cache"] = False
    return ModelConfig.from_dict(value)


def toy_batches(tokenizer, length=64):
    batches = []
    for text in TOY_TEXTS:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids = [tokenizer.bos_token_id, *ids, tokenizer.eos_token_id]
        # One token of overlap keeps next-token targets at chunk boundaries.
        for start in range(0, len(ids) - 1, length - 1):
            tokens = ids[start : start + length]
            valid = len(tokens)
            tokens += [tokenizer.pad_token_id] * (length - valid)
            input_ids = torch.tensor([tokens])
            mask = torch.arange(length)[None, :] < valid
            labels = input_ids.masked_fill(~mask, -100)
            batches.append(
                dict(input_ids=input_ids, attention_mask=mask, labels=labels)
            )
    return batches


@torch.no_grad()
def perplexity(model, batches):
    model.eval()
    loss, count = 0.0, 0
    for batch in batches:
        valid = (batch["labels"][:, 1:] != -100).sum().item()
        loss += model(**batch).loss.item() * valid
        count += valid
    return math.exp(loss / count)


def finetune(model, batches, *, max_steps, target, learning_rate):
    model.config.calibrate_dspark = False
    initial = perplexity(model, batches)
    print(f"Initial toy perplexity: {initial:.4f}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)
    history, hits = [], 0
    started = time.monotonic()
    for step in range(1, max_steps + 1):
        model.train()
        batch = batches[(step - 1) % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at step {step}")
        loss.backward()
        bad = [
            name
            for name, param in model.named_parameters()
            if param.grad is not None and not torch.isfinite(param.grad).all()
        ]
        if bad:
            raise RuntimeError(f"Non-finite gradients at step {step}: {bad[:12]}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 25 == 0 or step == max_steps:
            score = perplexity(model, batches)
            history.append(dict(step=step, perplexity=score))
            print(
                f"Step {step}: loss={loss.item():.4f}, toy_ppl={score:.4f}, "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
            hits = hits + 1 if score <= target else 0
            if hits >= 2:
                break
    score = perplexity(model, batches)
    if score > target:
        raise RuntimeError(
            f"Toy perplexity {score:.4f} exceeds {target}; "
            "increase --max-train-steps or adjust --learning-rate"
        )
    model.config.calibrate_dspark = True
    model.eval()
    return dict(
        initial_perplexity=initial,
        final_perplexity=score,
        steps=step,
        target_perplexity=target,
        history=history,
        seconds=time.monotonic() - started,
        corpus="8 create-tiny-model copypastas; training-set metric only",
        dspark_finetuned=False,
        vision_finetuned=False,
    )


def _power_scale(maximum, divisor):
    return torch.exp2(
        torch.ceil(torch.log2(maximum.clamp_min(divisor * 2**-126) / divisor))
    )


def encode_fp8(weight, *, rowwise=False):
    weight = weight.float()
    rows, columns = weight.shape
    if rowwise:
        groups = weight.unflatten(-1, (-1, 32))
        scale = _power_scale(groups.abs().amax(-1), 448)
        encoded = (groups / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
        encoded = encoded.flatten(-2)
    else:
        padded = F.pad(weight, (0, -columns % 32, 0, -rows % 32))
        groups = padded.reshape(-1, 32, padded.shape[1] // 32, 32).transpose(1, 2)
        scale = _power_scale(groups.abs().amax((-1, -2)), 448)
        encoded = (groups / scale[..., None, None]).to(torch.float8_e4m3fn)
        encoded = encoded.transpose(1, 2).reshape(padded.shape)[:rows, :columns]
    return encoded.contiguous(), scale.to(torch.float8_e8m0fnu).contiguous()


def encode_fp4(weight):
    groups = weight.float().unflatten(-1, (-1, 32))
    scale = _power_scale(groups.abs().amax(-1), 6)
    values = (groups / scale.unsqueeze(-1)).clamp(-6, 6)
    table = values.new_tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
    midpoints = (table[:-1] + table[1:]) / 2
    indices = torch.bucketize(values.abs().contiguous(), midpoints)
    lower = indices.clamp(max=6)
    ties = (values.abs() == midpoints[lower]) & (indices.remainder(2) == 1)
    indices = indices + ties.long()
    codes = indices.byte().flatten(-2)
    codes |= (values < 0).byte().flatten(-2) << 3
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).view(torch.int8)
    return packed.contiguous(), scale.to(torch.float8_e8m0fnu).contiguous()


def raw_name(name):
    name = name.removeprefix("model.")
    return (
        "head." + name.removeprefix("lm_head.") if name.startswith("lm_head.") else name
    )


def save_source_fixture(
    model, source: Path, output: Path, max_shard_bytes=16 * 1024**2
):
    output.mkdir()
    original_names = json.loads((source / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    ordinary, tables = [], []
    source_bytes, kinds = 0, {}
    for logical, tensor in model.state_dict().items():
        name = raw_name(logical)
        if name not in original_names:
            raise ValueError(f"Tiny key is absent from the original checkpoint: {name}")
        is_table = ".engram.embed.weight" in name
        is_expert = ".ffn.experts." in name and name.endswith(".weight")
        if is_table:
            weight, scale = encode_fp8(tensor, rowwise=True)
            kind = "engram_fp8_row32"
        elif is_expert:
            weight, scale = encode_fp4(tensor)
            kind = "expert_fp4_group32"
        elif (
            name.endswith(".weight")
            and name.replace(".weight", ".scale") in original_names
        ):
            weight, scale = encode_fp8(tensor)
            kind = "linear_fp8_block32"
        else:
            owner = model.get_submodule(logical.rpartition(".")[0])
            execution_weight = isinstance(
                owner, (torch.nn.Linear, torch.nn.Embedding, Head)
            )
            dtype = (
                torch.bfloat16
                if execution_weight or name.startswith("image_")
                else tensor.dtype
            )
            weight, scale = tensor.detach().to(dtype).contiguous(), None
            kind = str(dtype).removeprefix("torch.")
        entries = {name: weight}
        if scale is not None:
            entries[name.removesuffix(".weight") + ".scale"] = scale
        source_bytes += sum(t.numel() * t.element_size() for t in entries.values())
        kinds[kind] = kinds.get(kind, 0) + 1
        (tables if is_table else ordinary).append(entries)
    groups, current, size = [], {}, 0
    for entries in ordinary:
        entry_size = sum(t.numel() * t.element_size() for t in entries.values())
        if current and size + entry_size > max_shard_bytes:
            groups.append(current)
            current, size = {}, 0
        current.update(entries)
        size += entry_size
    if current:
        groups.append(current)
    groups.extend(tables)  # Dedicated table shards exercise lossless hard-link export.
    mapping = {}
    for index, group in enumerate(groups, 1):
        filename = f"model-{index:05d}-of-{len(groups):05d}.safetensors"
        save_file(group, output / filename)
        mapping.update({name: filename for name in group})
    value = model.config.to_dict()
    value.update(
        model_type="deepseek_v41",
        architectures=["DeepseekV41ForCausalLM"],
        dtype="bfloat16",
        quantization_config={
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        },
    )
    value.pop("source_quantization_config", None)
    (output / "config.json").write_text(json.dumps(value, indent=2) + "\n")
    (output / "model.safetensors.index.json").write_text(
        json.dumps(
            dict(metadata=dict(total_size=source_bytes), weight_map=mapping), indent=2
        )
        + "\n"
    )
    for relative in ("encoding", "chat_template.jinja", "LICENSE"):
        src, dst = source / relative, output / relative
        if src.is_dir():
            shutil.copytree(src, dst)
        elif src.is_file():
            shutil.copy2(src, dst)
    return dict(
        tensors=len(mapping), shards=len(groups), tensor_bytes=source_bytes, kinds=kinds
    )


def rebuild_source_fixture(source: Path, root: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        root / "native-finetuned", local_files_only=True
    )
    model = DeepseekV41NativeForConditionalGeneration.from_pretrained(
        root / "native-finetuned",
        local_files_only=True,
        tokenizer=tokenizer,
        dtype=torch.float32,
    )
    output = root / "source-fp8-fp4"
    stats = save_source_fixture(model, source, output)
    tokenizer.save_pretrained(output)
    manifest = root / "creation.json"
    creation = json.loads(manifest.read_text())
    creation["source_checkpoint"] = stats
    manifest.write_text(json.dumps(creation, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)


def create_fixture(args):
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    config = build_tiny_config(source)
    model = DeepseekV41NativeForConditionalGeneration(config, tokenizer=tokenizer)
    params = model.num_parameters()
    print(
        f"Full topology: {params:,} parameters, "
        f"{config.text_config.n_layers} text + "
        f"{config.vision_config.num_hidden_layers} vision + "
        f"{config.text_config.n_mtp_layers} DSpark layers",
        flush=True,
    )
    training = finetune(
        model,
        toy_batches(tokenizer, args.train_sequence_length),
        max_steps=args.max_train_steps,
        target=args.target_perplexity,
        learning_rate=args.learning_rate,
    )
    native = output / "native-finetuned"
    model.save_pretrained(native, max_shard_size="16MB")
    tokenizer.save_pretrained(native)
    source_output = output / "source-fp8-fp4"
    source_stats = save_source_fixture(model, source, source_output)
    tokenizer.save_pretrained(source_output)
    calibration = output / "calibration.jsonl"
    calibration.write_text(
        "".join(json.dumps(dict(text=text)) + "\n" for text in TOY_TEXTS)
    )
    result = dict(
        source=str(source),
        output=str(output),
        seed=args.seed,
        parameters=params,
        training=training,
        source_checkpoint=source_stats,
        topology=dict(
            text_layers=config.text_config.n_layers,
            vision_layers=config.vision_config.num_hidden_layers,
            dspark_layers=config.text_config.n_mtp_layers,
            kv_sources=config.text_config.kv_source_layer_ids,
            index_sources=config.text_config.index_source_layer_ids,
            engram_layers=config.text_config.engram_layer_ids,
        ),
    )
    (output / "creation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-steps", type=int, default=600)
    parser.add_argument("--target-perplexity", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--train-sequence-length", type=int, default=64)
    parser.add_argument(
        "--rebuild-source",
        action="store_true",
        help="Re-export native-finetuned weights into an absent source directory",
    )
    args = parser.parse_args()
    if args.max_train_steps < 1 or not 2 <= args.train_sequence_length <= 256:
        parser.error("Need positive train steps and a sequence length within 2..256")
    if args.rebuild_source:
        rebuild_source_fixture(args.source.resolve(), args.output.resolve())
    else:
        create_fixture(args)


if __name__ == "__main__":
    main()
