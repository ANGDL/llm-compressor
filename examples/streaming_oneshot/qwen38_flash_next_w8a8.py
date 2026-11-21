"""Stream Qwen3.8-Flash-Next W8A8 / mixed W4A8 quantization.

The INT8 targets use a fixed set of Qwen3.8-Flash-Next module-name regular
expressions.  The model's special convolution, normalization, embedding, router,
vision, and MTP support layers stay in their native dtype by default. The
``--quantize-ple`` switch quantizes PLE key/value projections through the
calibrated W8A8 path and PLE ngram embeddings with data-free per-row INT8 RTN.
The ``--same-as-klx`` switch also quantizes the shared expert, linear attention,
and full attention projections used by the KLX checkpoint layout.
Use ``--no-quantize-ple`` to disable PLE quantization when either of those
options would otherwise enable it.
The ``--w4a8`` switch keeps every other quantized projection at W8A8 and uses
INT4 per-channel symmetric weights for routed expert projections only.

For a full checkpoint, use the Flickr30k multimodal calibration data and append
the text calibration data with ``--text-dataset-id``.  A local tiny checkpoint
can use only ``calibration_data_deepseek-v4-flash-0731.jsonl``.

Weight quantization uses the project's iMatrix observer with RTN. GPTQ is not
used by this example.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from compressed_tensors.quantization import (
    QuantizationScheme,
    preset_name_to_scheme,
)
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

from datasets import Dataset, load_dataset
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.streaming import (
    CastWeightMaterializer,
    SafetensorsWeightSource,
    TensorMetadata,
    streaming_oneshot,
)
from llmcompressor.streaming.checkpoint import CheckpointWeightSource


def _is_local_json(value: str) -> bool:
    path = Path(value).expanduser()
    return path.is_file() or path.suffix.lower() in {".json", ".jsonl"}


def _load_dataset(source: str, split: str, samples: int) -> Dataset:
    if _is_local_json(source):
        return load_dataset("json", data_files=source, split=f"train[:{samples}]")
    return load_dataset(source, split=f"{split}[:{samples}]")


def _normalize_content(content: Any) -> list[dict[str, str]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, dict):
        content = [content]
    normalized = []
    for item in content or []:
        if isinstance(item, str):
            normalized.append({"type": "text", "text": item})
        elif isinstance(item, dict):
            text = item.get("text", item.get("reasoning"))
            if text is not None:
                normalized.append({"type": "text", "text": str(text)})
    return normalized


def _text_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    if "messages" in example:
        return [
            {
                "role": message.get("role", "user"),
                "content": _normalize_content(message.get("content", "")),
            }
            for message in example["messages"]
            if isinstance(message, dict)
        ]
    if "text" in example:
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": example["text"]}],
            }
        ]
    raise ValueError("text calibration rows must contain `messages` or `text`")


def _system_messages_first(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    system_messages = [
        message for message in messages if message.get("role") == "system"
    ]
    other_messages = [
        message for message in messages if message.get("role") != "system"
    ]
    return system_messages + other_messages


def _image_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    buffered = BytesIO()
    example["image"].save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    caption = example["caption"]
    caption = caption[0] if isinstance(caption, (list, tuple)) else caption
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image;base64,{encoded}"},
                {"type": "text", "text": "What does the image show?"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": str(caption)}],
        },
    ]


def _load_multimodal_dataset(source: str, split: str, samples: int) -> Dataset:
    path = Path(source).expanduser()
    if path.is_dir():
        shards = sorted(path.glob("**/flickr30k-test-*.arrow"))
        if not shards:
            raise ValueError(f"no Flickr30k Arrow shard found under {source}")
        dataset = Dataset.from_file(str(shards[0])).select(range(samples))
    else:
        dataset = _load_dataset(source, split, samples)
    return dataset.shuffle(seed=42)


def _processor_inputs(
    processor: Any,
    messages: list[dict[str, Any]],
    max_sequence_length: int,
    multimodal: bool,
) -> dict[str, Any]:
    if multimodal:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
        except ImportError:
            image_inputs = [messages[0]["content"][0]["image"]]
            video_inputs = None
        return dict(
            processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=False,
                max_length=max_sequence_length,
                truncation=True,
            )
        )

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return dict(
        processor(
            text=text,
            padding=False,
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=False,
        )
    )


def _build_calibration_dataset(
    processor: Any,
    multimodal_sources: list[str],
    multimodal_split: str,
    text_sources: list[str],
    text_split: str,
    samples: int,
    max_sequence_length: int,
) -> Dataset:
    rows = []
    if multimodal_sources:
        per_source, remainder = divmod(samples, len(multimodal_sources))
        text_rows = []
        for source in text_sources:
            text_rows.extend(list(_load_dataset(source, text_split, samples)))
        for index, source in enumerate(multimodal_sources):
            count = per_source + int(index < remainder)
            dataset = _load_multimodal_dataset(source, multimodal_split, count)
            for row_index, example in enumerate(dataset):
                messages = _image_messages(example)
                if text_rows:
                    messages.extend(
                        _text_messages(text_rows[row_index % len(text_rows)])
                    )
                messages = _system_messages_first(messages)
                rows.append(
                    _processor_inputs(
                        processor, messages, max_sequence_length, multimodal=True
                    )
                )
    else:
        if not text_sources:
            raise ValueError("at least one calibration dataset is required")
        per_source, remainder = divmod(samples, len(text_sources))
        for index, source in enumerate(text_sources):
            count = per_source + int(index < remainder)
            dataset = _load_dataset(source, text_split, count).shuffle(seed=42)
            for example in dataset:
                rows.append(
                    _processor_inputs(
                        processor,
                        _text_messages(example),
                        max_sequence_length,
                        multimodal=False,
                    )
                )

    if not rows:
        raise ValueError("calibration dataset is empty")
    return Dataset.from_list(rows)


def _data_collator(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if len(batch) != 1:
        raise ValueError("Qwen3.8-Flash-Next calibration requires batch_size=1")
    collated = {}
    for key, value in batch[0].items():
        if value is None:
            continue
        tensor = torch.as_tensor(value)
        if key == "pixel_values":
            tensor = tensor.to(torch.bfloat16).squeeze(0)
        if key in {"input_ids", "attention_mask"} and tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        collated[key] = tensor
    return collated


class _Qwen38FusedExpertSource:
    def __init__(self, checkpoint: str | Path, skip_ple_embeddings: bool = False):
        self._source = SafetensorsWeightSource(checkpoint)
        self._skip_ple_embeddings = skip_ple_embeddings
        self._logical_to_fused: dict[str, tuple[str, int, int, int]] = {}
        names = []
        for name in self._source.tensor_names():
            if name.startswith("mtp."):
                names.append(name)
                continue
            if (
                self._skip_ple_embeddings
                and ".ple.ple_embedding.ngram_embedding.shard_" in name
            ):
                continue
            if name.endswith("mlp.experts.gate_up_proj"):
                metadata = self._source.metadata(name)
                experts, rows, hidden = metadata.shape
                if rows % 2:
                    raise ValueError(f"fused gate_up_proj has an odd row count: {name}")
                intermediate = rows // 2
                for expert in range(experts):
                    prefix = name.removesuffix(".gate_up_proj")
                    for projection, offset in (("gate_proj", 0), ("up_proj", 1)):
                        logical = f"{prefix}.{expert}.{projection}.weight"
                        self._logical_to_fused[logical] = (
                            name,
                            expert,
                            offset,
                            intermediate,
                        )
                        names.append(logical)
                continue
            if name.endswith("mlp.experts.down_proj"):
                metadata = self._source.metadata(name)
                experts, hidden, intermediate = metadata.shape
                for expert in range(experts):
                    logical = (
                        f"{name.removesuffix('.down_proj')}.{expert}.down_proj.weight"
                    )
                    self._logical_to_fused[logical] = (
                        name,
                        expert,
                        0,
                        intermediate,
                    )
                    names.append(logical)
                continue
            names.append(name)
        self._tensor_names = tuple(names)
        self._fused_cache_name: str | None = None
        self._fused_cache: torch.Tensor | None = None

    def tensor_names(self):
        return self._tensor_names

    def metadata(self, name: str) -> TensorMetadata:
        mapping = self._logical_to_fused.get(name)
        if mapping is None:
            return self._source.metadata(name)
        fused_name, _expert, offset, intermediate = mapping
        metadata = self._source.metadata(fused_name)
        if fused_name.endswith("gate_up_proj"):
            shape = (intermediate, metadata.shape[-1])
        else:
            shape = metadata.shape[1:]
        return TensorMetadata(
            name=name,
            shape=shape,
            dtype=metadata.dtype,
            shard=metadata.shard,
            storage_index=metadata.storage_index,
        )

    def _load_fused(self, name: str) -> torch.Tensor:
        if self._fused_cache_name != name:
            self._fused_cache = self._source.load_tensors_cpu([name])[name]
            self._fused_cache_name = name
        assert self._fused_cache is not None
        return self._fused_cache

    def load_tensors_cpu(self, names):
        result = {}
        for name in dict.fromkeys(names):
            mapping = self._logical_to_fused.get(name)
            if mapping is None:
                result[name] = self._source.load_tensors_cpu([name])[name]
                continue
            fused_name, expert, offset, intermediate = mapping
            fused = self._load_fused(fused_name)
            if fused_name.endswith("gate_up_proj"):
                start = offset * intermediate
                result[name] = fused[expert, start : start + intermediate]
            else:
                result[name] = fused[expert]
        return result

    def iter_tensor_groups_cpu(self, groups):
        for group in groups:
            yield self.load_tensors_cpu(group)


class _Qwen38FusedExpertMaterializer(CastWeightMaterializer):
    def __init__(self, skip_ple_embeddings: bool = False):
        self._skip_ple_embeddings = skip_ple_embeddings

    def create_source(self, checkpoint: str) -> CheckpointWeightSource:
        return _Qwen38FusedExpertSource(
            checkpoint, skip_ple_embeddings=self._skip_ple_embeddings
        )


def _allocate_cpu_tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    try:
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
    except RuntimeError:
        return torch.empty(shape, dtype=dtype, device="cpu")


class _Qwen38HostOffloadEmbedding(nn.Module):
    def __init__(self, checkpoint: str | Path, prefix: str):
        super().__init__()
        source = SafetensorsWeightSource(checkpoint)
        shards = []
        for name in source.tensor_names():
            if not (
                name.startswith(f"{prefix}.ngram_embedding.shard_")
                and name.endswith(".weight")
            ):
                continue
            shard_index = int(name.removesuffix(".weight").rsplit("shard_", 1)[1])
            metadata = source.metadata(name)
            shards.append((shard_index, name, metadata))
        if not shards:
            raise RuntimeError(f"no PLE embedding shards found for {prefix}")
        shards.sort()
        offset = 0
        first_dtype = shards[0][2].dtype
        embedding_dim = shards[0][2].shape[1]
        total_rows = sum(metadata.shape[0] for _, _, metadata in shards)
        host_weight = _allocate_cpu_tensor((total_rows, embedding_dim), first_dtype)
        for _shard_index, name, metadata in shards:
            rows = metadata.shape[0]
            with safe_open(
                str(metadata.shard), framework="pt", device="cpu"
            ) as source_file:
                host_weight[offset : offset + rows].copy_(source_file.get_tensor(name))
            offset += rows
        self._host_weight = host_weight
        self.embedding_dim = embedding_dim
        self.register_buffer(
            "weight",
            torch.empty(0, dtype=first_dtype),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        flat_ids = input_ids.reshape(-1).to(device="cpu")
        rows = self._host_weight.index_select(0, flat_ids)
        if not rows.is_pinned():
            rows = _allocate_cpu_tensor(tuple(rows.shape), rows.dtype)
            rows.copy_(self._host_weight.index_select(0, flat_ids))
        return rows.to(
            device=input_ids.device,
            non_blocking=True,
        ).reshape(
            *input_ids.shape,
            self.embedding_dim,
        )


def _install_cpu_ple_embeddings(model: nn.Module, checkpoint: Path) -> None:
    for module_name, module in model.named_modules():
        if type(module).__name__ != "Qwen4ExpTextNGramEmbedding":
            continue
        module.ngram_embedding = _Qwen38HostOffloadEmbedding(checkpoint, module_name)


QWEN38_W8A8_COMMON_IGNORES = [
    r"lm_head",
    r"re:.*visual.*",
    r"re:.*embed_tokens$",
    r"re:.*hyper_connection_mixer\."
    r"(?:input_mix_weight_down|input_mix_weight_up)$",
    r"re:.*attn_hyper_connection\."
    r"(?:block_inject_weight|input_mix_weight_down|input_mix_weight_up)$",
    r"re:.*mlp_hyper_connection\."
    r"(?:block_inject_weight|input_mix_weight_down|input_mix_weight_up)$",
    r"re:.*linear_attn\."
    r"conv1d$",
    r"re:.*mlp\.gate$",
    r"re:.*shared_expert_gate$",
    r"re:.*norm$",
    r"re:.*mtp.*",
]

QWEN38_W8A8_NON_KLX_IGNORES = [
    r"re:.*linear_attn\."
    r"(?:in_proj_a|in_proj_b|in_proj_qkv|in_proj_z|out_proj|out_proj_sp)$",
    r"re:.*linear_attn\.in_proj_qkvzba$",
    r"re:.*mlp\.shared_expert\."
    r"(?:down_proj|gate_proj|up_proj)$",
    r"re:.*self_attn\.indexer\."
    r"index_qk_proj$",
    r"re:.*self_attn\."
    r"(?:k_proj|o_proj|o_proj_sp|q_proj|v_proj)$",
]

QWEN38_W8A8_PLE_IGNORES = [
    r"re:.*ple\.conv1d$",
    r"re:.*ple\.(?:key_proj|value_proj)$",
]

QWEN38_W8A8_PLE_LINEAR_IGNORES = [
    r"re:.*ple\.conv1d$",
]

QWEN38_W8A8_IGNORES = [
    *QWEN38_W8A8_COMMON_IGNORES,
    *QWEN38_W8A8_NON_KLX_IGNORES,
    *QWEN38_W8A8_PLE_IGNORES,
]

QWEN38_ROUTED_EXPERT_TARGETS = [
    r"re:.*\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)$",
]

QWEN38_W4A8_W8_PATTERNS = [
    r"re:.*\.linear_attn\..*$",
    r"re:.*\.self_attn\..*$",
    r"re:.*\.mlp\.shared_expert\..*$",
    r"re:.*\.ple\.(?:key_proj|value_proj)$",
]


def _w8a8_other_targets(same_as_klx: bool, quantize_ple: bool) -> list[str]:
    targets = []
    if same_as_klx:
        targets.extend(
            [
                r"re:.*\.linear_attn\.(?:in_proj_qkvzba|in_proj_qkvz|"
                r"in_proj_ba|in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|"
                r"out_proj|out_proj_sp)$",
                r"re:.*\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj|"
                r"o_proj_sp|indexer\.index_qk_proj)$",
                r"re:.*\.self_attn\.qkv_proj$",
                r"re:.*\.mlp\.shared_expert\."
                r"(?:down_proj|gate_proj|up_proj)$",
                r"re:.*\.mlp\.shared_expert\.gate_up_proj$",
            ]
        )
    if quantize_ple:
        targets.append(r"re:.*\.ple\.(?:key_proj|value_proj)$")
    return targets


def _mixed_w4a8_schemes(
    same_as_klx: bool, quantize_ple: bool
) -> dict[str, QuantizationScheme]:
    weights_4bit = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
        dynamic=False,
        observer="imatrix_mse",
        scale_dtype=torch.float32,
    )
    weights_8bit = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
        dynamic=False,
        observer="imatrix_mse",
        scale_dtype=torch.float32,
    )
    activations_8bit = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.TOKEN,
        symmetric=True,
        dynamic=True,
        observer=None,
        scale_dtype=torch.float32,
    )
    schemes = {
        "experts_w4a8": QuantizationScheme(
            targets=QWEN38_ROUTED_EXPERT_TARGETS,
            weights=weights_4bit,
            input_activations=activations_8bit,
        )
    }
    other_targets = _w8a8_other_targets(same_as_klx, quantize_ple)
    if other_targets:
        schemes["other_w8a8"] = QuantizationScheme(
            targets=other_targets,
            weights=weights_8bit,
            input_activations=activations_8bit,
        )
    return schemes


_PLE_EMBEDDING_WEIGHT = re.compile(
    r"^(?P<prefix>model\.language_model\.layers\.\d+\.ple\."
    r"ple_embedding\.ngram_embedding\.shard_\d+)\.weight$"
)
_PLE_LINEAR_WEIGHT = re.compile(
    r"^model\.language_model\.layers\.\d+\.ple\.(?:key_proj|value_proj)\.weight$"
)


def _quantize_data_free_int8_embedding(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError(
            f"PLE embedding weights must be 2D, got shape {tuple(weight.shape)}"
        )
    weight_float = weight.float()
    scale = weight_float.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    quantized = torch.round(weight_float / scale).clamp(-128, 127).to(torch.int8)
    return quantized, scale.to(torch.float32)


def _quantize_ple_embeddings(source: Path, output: Path) -> int:
    source_index = json.loads((source / "model.safetensors.index.json").read_text())
    output_index_path = output / "model.safetensors.index.json"
    output_index = json.loads(output_index_path.read_text())
    weight_map = output_index["weight_map"]
    source_weight_map = source_index["weight_map"]
    embedding_names = sorted(
        name for name in source_weight_map if _PLE_EMBEDDING_WEIGHT.fullmatch(name)
    )
    if not embedding_names:
        raise RuntimeError("PLE quantization requested but no PLE embeddings found")

    names_by_source_shard: dict[str, list[str]] = {}
    for name in embedding_names:
        names_by_source_shard.setdefault(source_weight_map[name], []).append(name)

    new_weight_map = dict(weight_map)
    added_bytes = 0
    removed_bytes = 0
    old_shards = set()
    for shard_index, (source_shard, names) in enumerate(
        sorted(names_by_source_shard.items())
    ):
        quantized_tensors = {}
        with safe_open(
            str(source / source_shard), framework="pt", device="cpu"
        ) as source_file:
            for name in names:
                scale_name = f"{name.removesuffix('.weight')}.weight_scale"
                if scale_name in weight_map:
                    raise RuntimeError(
                        f"PLE embedding is already quantized in output: {name}"
                    )
                old_shard = weight_map.get(name)
                if old_shard is not None:
                    old_shards.add(old_shard)
                    with safe_open(
                        str(output / old_shard), framework="pt", device="cpu"
                    ) as output_file:
                        old_weight = output_file.get_tensor(name)
                    removed_bytes += old_weight.numel() * old_weight.element_size()
                weight = source_file.get_tensor(name)
                quantized, scale = _quantize_data_free_int8_embedding(weight)
                quantized_tensors[name] = quantized
                quantized_tensors[scale_name] = scale
                added_bytes += (
                    quantized.numel() * quantized.element_size()
                    + scale.numel() * scale.element_size()
                )

        new_shard = f"model-ple-embedding-{shard_index:05d}.safetensors"
        save_file(quantized_tensors, output / new_shard)
        for name in names:
            new_weight_map[name] = new_shard
            scale_name = f"{name.removesuffix('.weight')}.weight_scale"
            new_weight_map[scale_name] = new_shard

    output_index["weight_map"] = new_weight_map
    output_index["metadata"]["total_size"] += added_bytes - removed_bytes
    referenced_shards = set(new_weight_map.values())
    for old_shard in old_shards - referenced_shards:
        (output / old_shard).unlink()
    output_index_path.write_text(json.dumps(output_index, indent=2) + "\n")
    print(
        f"quantized {len(embedding_names)} PLE embedding shards with "
        "data-free INT8 channel scales"
    )
    return len(embedding_names)


def _verify_ple_quantized(output: Path) -> None:
    index = json.loads((output / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    linear_names = sorted(
        name for name in weight_map if _PLE_LINEAR_WEIGHT.fullmatch(name)
    )
    if not linear_names:
        raise RuntimeError("quantized output has no PLE key/value projections")
    for name in linear_names:
        scale_name = f"{name.removesuffix('.weight')}.weight_scale"
        if scale_name not in weight_map:
            raise RuntimeError(f"missing PLE linear scale: {scale_name}")
        with (
            safe_open(
                str(output / weight_map[name]), framework="pt", device="cpu"
            ) as weight_file,
            safe_open(
                str(output / weight_map[scale_name]),
                framework="pt",
                device="cpu",
            ) as scale_file,
        ):
            weight = weight_file.get_tensor(name)
            scale = scale_file.get_tensor(scale_name)
        if weight.dtype != torch.int8:
            raise RuntimeError(f"PLE linear weight is not INT8: {name}")
        if scale.dtype != torch.float32:
            raise RuntimeError(f"PLE linear scale is not FP32: {scale_name}")

    embedding_names = sorted(
        name for name in weight_map if _PLE_EMBEDDING_WEIGHT.fullmatch(name)
    )
    if not embedding_names:
        raise RuntimeError("quantized output has no PLE embedding weights")
    for name in embedding_names:
        scale_name = f"{name.removesuffix('.weight')}.weight_scale"
        if scale_name not in weight_map:
            raise RuntimeError(f"missing PLE embedding scale: {scale_name}")
        with (
            safe_open(
                str(output / weight_map[name]), framework="pt", device="cpu"
            ) as weight_file,
            safe_open(
                str(output / weight_map[scale_name]),
                framework="pt",
                device="cpu",
            ) as scale_file,
        ):
            weight = weight_file.get_tensor(name)
            scale = scale_file.get_tensor(scale_name)
        if weight.dtype != torch.int8:
            raise RuntimeError(f"PLE embedding is not INT8: {name}")
        if scale.dtype != torch.float32:
            raise RuntimeError(f"PLE embedding scale is not FP32: {scale_name}")
        if tuple(scale.shape) != (weight.shape[0], 1):
            raise RuntimeError(
                f"unexpected PLE embedding scale shape for {name}: {tuple(scale.shape)}"
            )
    print(
        f"verified {len(linear_names)} PLE key/value projections and "
        f"{len(embedding_names)} quantized PLE embedding shards"
    )


def _update_ple_embedding_dtype(output: Path, quantized: bool) -> None:
    config_path = output / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise RuntimeError("output config.json has no object-valued text_config")
    if quantized:
        config["ple_embedding_dtype"] = "int8"
        text_config["ple_embedding_dtype"] = "int8"
    else:
        config.pop("ple_embedding_dtype", None)
        text_config.pop("ple_embedding_dtype", None)
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _update_w4a8_runtime_config(output: Path, enabled: bool) -> None:
    config_path = output / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quantization_config = config["quantization_config"]
    if enabled:
        quantization_config["modules_to_not_convert"] = list(QWEN38_W4A8_W8_PATTERNS)
        quantization_config["self_describing"] = True
    else:
        quantization_config.pop("modules_to_not_convert", None)
        quantization_config.pop("self_describing", None)
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _verify_mtp_preserved(source: Path, output: Path) -> None:
    def index(path: Path) -> dict[str, str]:
        return json.loads((path / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]

    source_index = index(source)
    output_index = index(output)
    source_names = {name for name in source_index if name.startswith("mtp.")}
    output_names = {name for name in output_index if name.startswith("mtp.")}
    if source_names != output_names:
        missing = sorted(source_names - output_names)
        extra = sorted(output_names - source_names)
        raise RuntimeError(
            f"MTP checkpoint mismatch, missing={missing[:5]}, extra={extra[:5]}"
        )
    if not source_names:
        raise RuntimeError("source checkpoint has no MTP tensors")
    quantized_mtp = [
        name
        for name in output_names
        if "weight_scale" in name or "weight_scale_inv" in name
    ]
    if quantized_mtp:
        raise RuntimeError(
            f"MTP tensors unexpectedly have quantization scales: {quantized_mtp[:5]}"
        )

    for name in sorted(source_names):
        with (
            safe_open(
                str(source / source_index[name]), framework="pt", device="cpu"
            ) as source_file,
            safe_open(
                str(output / output_index[name]), framework="pt", device="cpu"
            ) as output_file,
        ):
            source_tensor = source_file.get_tensor(name)
            output_tensor = output_file.get_tensor(name)
        if not torch.equal(source_tensor, output_tensor):
            raise RuntimeError(f"MTP tensor changed during quantization: {name}")
    print(f"verified {len(source_names)} preserved, unquantized MTP tensors in output")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--dataset-id", nargs="*", default=[])
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--text-dataset-id", nargs="*", default=[])
    parser.add_argument("--text-dataset-split", default="train")
    parser.add_argument("--num-calibration-samples", type=int, default=128)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument(
        "--quantize-ple",
        action="store_true",
        help="quantize PLE key/value projections and data-free ngram embeddings",
    )
    parser.add_argument(
        "--same-as-klx",
        action="store_true",
        help=(
            "quantize the KLX-aligned shared expert, attention projections, "
            "and PLE tensors"
        ),
    )
    parser.add_argument(
        "--no-quantize-ple",
        action="store_true",
        help="disable PLE quantization even when another option enables it",
    )
    parser.add_argument(
        "--w4a8",
        action="store_true",
        help=(
            "quantize routed experts with INT4 per-channel symmetric weights; "
            "all other quantized projections remain W8A8"
        ),
    )
    parser.add_argument(
        "--ple-embedding-device",
        choices=("gpu", "cpu"),
        default="gpu",
        help="place PLE ngram embeddings on GPU or use CPU shard lookup",
    )
    args = parser.parse_args()
    if args.num_calibration_samples < 1 or args.max_sequence_length < 1:
        parser.error("calibration samples and sequence length must be positive")
    if not args.dataset_id and not args.text_dataset_id:
        parser.error("provide --dataset-id and/or --text-dataset-id")

    source_config = AutoConfig.from_pretrained(args.model_id, local_files_only=True)
    if source_config.model_type != "qwen4_exp":
        raise ValueError(f"expected qwen4_exp, got {source_config.model_type}")
    if args.dataset_id:
        processor = AutoProcessor.from_pretrained(
            args.model_id, local_files_only=True, trust_remote_code=True
        )
    else:
        processor = AutoTokenizer.from_pretrained(
            args.model_id, local_files_only=True, trust_remote_code=True
        )
    dataset = _build_calibration_dataset(
        processor=processor,
        multimodal_sources=args.dataset_id,
        multimodal_split=args.dataset_split,
        text_sources=args.text_dataset_id,
        text_split=args.text_dataset_split,
        samples=args.num_calibration_samples,
        max_sequence_length=args.max_sequence_length,
    )
    calibration_dataloader = DataLoader(
        dataset, batch_size=1, collate_fn=_data_collator
    )

    quantize_ple = (args.quantize_ple or args.same_as_klx) and not args.no_quantize_ple
    ignores = list(QWEN38_W8A8_COMMON_IGNORES)
    if not args.same_as_klx:
        ignores.extend(QWEN38_W8A8_NON_KLX_IGNORES)
    ignores.extend(
        QWEN38_W8A8_PLE_LINEAR_IGNORES if quantize_ple else QWEN38_W8A8_PLE_IGNORES
    )
    if args.w4a8:
        config_groups = _mixed_w4a8_schemes(args.same_as_klx, quantize_ple)
    else:
        scheme = preset_name_to_scheme("W8A8", ["Linear"])
        scheme.weights.observer = "imatrix_mse"
        scheme.weights.scale_dtype = torch.float32
        config_groups = {"group_0": scheme}
    recipe = [
        QuantizationModifier(
            config_groups=config_groups,
            ignore=ignores,
        ),
    ]

    def model_factory(config):
        model = AutoModelForImageTextToText.from_config(config)
        if quantize_ple and args.ple_embedding_device == "cpu":
            _install_cpu_ple_embeddings(model, args.model_id)
        return model

    output = streaming_oneshot(
        model=args.model_id,
        model_config=source_config,
        model_factory=model_factory,
        tokenizer=processor,
        dataset=calibration_dataloader,
        recipe=recipe,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_sequence_length,
        batch_size=1,
        shuffle_calibration_samples=False,
        moe_calibrate_all_experts=True,
        device=args.device,
        materializer=_Qwen38FusedExpertMaterializer(
            skip_ple_embeddings=(quantize_ple and args.ple_embedding_device == "cpu")
        ),
        overwrite_output=args.overwrite_output,
        pack_to_int8=args.w4a8,
    )
    processor.save_pretrained(output)
    if quantize_ple:
        _quantize_ple_embeddings(args.model_id, Path(output))
        _verify_ple_quantized(Path(output))
    _update_ple_embedding_dtype(Path(output), quantize_ple)
    _update_w4a8_runtime_config(Path(output), args.w4a8)
    _verify_mtp_preserved(args.model_id, Path(output))
    precision = "W4A8" if args.w4a8 else "W8A8"
    print(f"saved {precision} checkpoint to {output}")


if __name__ == "__main__":
    main()
