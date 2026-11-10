"""Shared source loading and chat formatting for calibration examples."""

import base64
import io
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from datasets import Dataset, concatenate_datasets, load_dataset

__all__ = [
    "build_calibration_messages",
    "format_calibration_text",
    "image_to_data_url",
    "load_calibration_dataset",
    "load_calibration_datasets",
    "load_calibration_source",
    "normalize_text_content",
]

CalibrationSources = str | Path | Sequence[str | Path]
ImageFormat = Literal["image", "url", "image_url", "image_url_flat"]


def normalize_text_content(content: Any) -> list[dict[str, str]]:
    """Convert strings and text/reasoning blocks to chat text blocks."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, (list, tuple)):
        return []

    normalized = []
    for item in content:
        if isinstance(item, str):
            normalized.append({"type": "text", "text": item})
        elif isinstance(item, dict):
            if "text" in item:
                text = item["text"]
            elif item.get("type") == "reasoning":
                text = item.get("reasoning")
            else:
                continue
            if text is not None:
                normalized.append({"type": "text", "text": str(text)})
    return normalized


def format_calibration_text(example: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract usable messages from a text or messages calibration record."""
    messages = example.get("messages")
    if messages is None:
        text = example.get("text")
        if text is None:
            raise ValueError("Text calibration records must contain messages or text")
        return [{"role": "user", "content": [{"type": "text", "text": str(text)}]}]

    formatted = []
    for message in messages:
        if not isinstance(message, dict) or "role" not in message:
            continue
        content = normalize_text_content(message.get("content", ""))
        if content:
            formatted.append({"role": message["role"], "content": content})
    if not formatted:
        raise ValueError("Text calibration example contains no usable messages")
    return formatted


def image_to_data_url(image: Any) -> str:
    """Encode a PIL image or a datasets bytes/path image record as RGB PNG."""
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            with Image.open(io.BytesIO(image["bytes"])) as source:
                image = source.convert("RGB")
        elif image.get("path"):
            with Image.open(image["path"]) as source:
                image = source.convert("RGB")
        else:
            raise ValueError("Image records must contain bytes or path")
    if not hasattr(image, "save"):
        raise TypeError(f"Unsupported calibration image type: {type(image)!r}")
    buffered = io.BytesIO()
    image.convert("RGB").save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def build_calibration_messages(
    example: dict[str, Any],
    text_example: dict[str, Any] | None = None,
    *,
    image_format: ImageFormat = "image",
    image_prompt: str = "What does the image show?",
) -> list[dict[str, Any]]:
    """Build image/caption chat, optionally appending a separate text record.

    ``image`` passes the original image to the processor. ``url`` uses an image
    block with a PNG data URL; ``image_url`` uses the OpenAI-style nested URL
    block expected by the native DeepSeek encoder. ``image_url_flat`` uses an
    image block with a flat ``image_url`` string for Kimi K2.5/K2.6.
    Text-only records use the same normalized messages as the dataset loader.
    Tokenized records should bypass message construction.
    """
    if image_format not in {"image", "url", "image_url", "image_url_flat"}:
        raise ValueError(f"Unsupported image format: {image_format!r}")
    image = example.get("image")
    if image is None:
        return format_calibration_text(example)
    if image_format == "image":
        image_block = {"type": "image", "image": image}
    elif image_format == "url":
        image_block = {"type": "image", "url": image_to_data_url(image)}
    elif image_format == "image_url":
        image_block = {
            "type": "image_url",
            "image_url": {"url": image_to_data_url(image)},
        }
    else:
        image_block = {"type": "image", "image_url": image_to_data_url(image)}
    caption = example.get("caption", "")
    if isinstance(caption, (list, tuple)):
        caption = caption[0] if caption else ""
    messages = [
        {
            "role": "user",
            "content": [image_block, {"type": "text", "text": image_prompt}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": str(caption)}],
        },
    ]
    if text_example is not None:
        messages.extend(format_calibration_text(text_example))
    return messages


def load_calibration_source(source: str | Path, split: str, samples: int) -> Dataset:
    """Read at most ``samples`` rows from a Hub/local dataset or JSON file.

    JSON/JSONL files use the JSON builder's ``train`` split. Other sources use
    the requested split. This low-level function preserves the source schema.
    """
    if samples <= 0:
        raise ValueError("Calibration sample count must be positive")
    path = Path(source).expanduser()
    if path.is_file() or path.suffix.lower() in {".json", ".jsonl"}:
        return load_dataset("json", data_files=str(path), split=f"train[:{samples}]")
    return load_dataset(str(source), split=f"{split}[:{samples}]")


def load_calibration_dataset(
    sources: CalibrationSources,
    split: str,
    samples: int,
    seed: int = 42,
    *,
    preprocess: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> Dataset:
    """Share a total sample budget across sources and concatenate their rows.

    Text-only sources are rebuilt with a common messages schema, dropping
    unrelated metadata and reconciling JSON versus Arrow chat features.
    Image and tokenized datasets retain their original features. ``preprocess``
    overrides normalization for model-specific encoders and is applied
    separately to each non-tokenized source before concatenation.
    """
    if isinstance(sources, (str, Path)):
        sources = [sources]
    if not sources:
        raise ValueError("At least one calibration dataset source is required")
    if samples < len(sources):
        raise ValueError(
            "Calibration sample count must be at least the number of dataset "
            f"sources ({samples} < {len(sources)})"
        )
    base, remainder = divmod(samples, len(sources))
    parts = []
    for index, source in enumerate(sources):
        part = load_calibration_source(source, split, base + (index < remainder))
        if "input_ids" not in part.column_names:
            if preprocess is not None:
                part = part.map(
                    lambda row: dict(preprocess(row)),
                    remove_columns=part.column_names,
                )
            elif "image" not in part.column_names:
                part = Dataset.from_list(
                    [{"messages": format_calibration_text(row)} for row in part]
                )
        parts.append(part.shuffle(seed=seed))
    if len(parts) == 1:
        return parts[0]
    columns = [set(part.column_names) for part in parts]
    if any(value != columns[0] for value in columns[1:]):
        raise ValueError(
            "Calibration sources have incompatible columns after preprocessing: "
            f"{[part.column_names for part in parts]}"
        )
    return concatenate_datasets(parts)


def load_calibration_datasets(
    sources: CalibrationSources,
    split: str,
    text_sources: CalibrationSources | None,
    text_split: str,
    samples: int,
    text_samples: int,
    seed: int = 42,
    *,
    text_preprocess: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[Dataset, list[dict[str, Any]]]:
    """Load primary calibration rows and optional text from independent splits.

    Text rows are returned separately for cyclic pairing with image rows; they
    do not increase the number of primary calibration batches.
    ``text_preprocess`` overrides normalization of supplementary text sources.
    """
    if text_samples < 0:
        raise ValueError("Text calibration sample count must be non-negative")
    if text_samples and not text_sources:
        raise ValueError("Text calibration samples require text dataset sources")
    primary = load_calibration_dataset(sources, split, samples, seed)
    if text_samples == 0:
        return primary, []
    text = load_calibration_dataset(
        text_sources, text_split, text_samples, seed, preprocess=text_preprocess
    )
    return primary, list(text)
