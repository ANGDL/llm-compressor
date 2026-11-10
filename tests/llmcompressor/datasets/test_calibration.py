import base64
import io
import json

import pytest
from PIL import Image

from datasets import Dataset, Features, Value
from datasets import Image as DatasetImage
from llmcompressor.datasets import (
    build_calibration_messages,
    calibration,
    format_calibration_text,
    image_to_data_url,
    load_calibration_dataset,
    load_calibration_datasets,
    load_calibration_source,
    normalize_text_content,
)


@pytest.fixture(autouse=True)
def isolated_dataset_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("datasets.config.HF_DATASETS_CACHE", tmp_path / "cache")


def test_json_sources_share_budget_and_normalize_different_chat_schemas(tmp_path):
    sources = []
    for source_index in range(2):
        path = tmp_path / f"source-{source_index}.jsonl"
        rows = []
        for row_index in range(4):
            text = f"source {source_index} row {row_index}"
            if source_index == 0:
                rows.append(
                    {
                        "messages": [{"role": "user", "content": text}],
                        "meta": {"language": "zh"},
                    }
                )
            else:
                rows.append({"text": text, "prompt_id": str(row_index)})
        path.write_text("\n".join(json.dumps(row) for row in rows))
        sources.append(path)

    dataset = load_calibration_dataset(sources, "test", 5, seed=7)
    repeated = load_calibration_dataset(sources, "test", 5, seed=7)

    assert dataset.column_names == ["messages"]
    assert list(dataset) == list(repeated)
    contents = [row["messages"][0]["content"][0]["text"] for row in dataset]
    assert sum(text.startswith("source 0") for text in contents) == 3
    assert sum(text.startswith("source 1") for text in contents) == 2


def test_source_preserves_schema(tmp_path):
    path = tmp_path / "calibration.jsonl"
    row = {"text": "hello", "meta": {"language": "en"}}
    path.write_text(json.dumps(row))
    dataset = load_calibration_source(path, "test", 1)

    assert dataset[0] == row


def test_primary_and_text_sources_use_independent_splits(monkeypatch):
    calls = []

    def fake_load(source, *, split):
        calls.append((source, split))
        return Dataset.from_dict({"text": [source]})

    monkeypatch.setattr(calibration, "load_dataset", fake_load)
    primary, text = load_calibration_datasets(
        ["image-source-a", "image-source-b"],
        "test",
        ["text-source-a", "text-source-b"],
        "train_sft",
        5,
        3,
    )

    assert calls == [
        ("image-source-a", "test[:3]"),
        ("image-source-b", "test[:2]"),
        ("text-source-a", "train_sft[:2]"),
        ("text-source-b", "train_sft[:1]"),
    ]
    assert len(primary) == 2
    assert len(text) == 2


def test_zero_text_samples_skip_loading_text_source(monkeypatch):
    calls = []

    def fake_load(source, *, split):
        calls.append(source)
        return Dataset.from_dict({"input_ids": [[1, 2]], "attention_mask": [[1, 1]]})

    monkeypatch.setattr(calibration, "load_dataset", fake_load)
    primary, text = load_calibration_datasets(
        "primary", "test", "unused-text", "train", 1, 0
    )

    assert calls == ["primary"]
    assert primary[0] == {"input_ids": [1, 2], "attention_mask": [1, 1]}
    assert text == []


def test_image_features_survive_source_loading(monkeypatch):
    buffered = io.BytesIO()
    Image.new("RGB", (2, 3), "red").save(buffered, format="PNG")
    features = Features(
        {"image": DatasetImage(decode=False), "caption": Value("string")}
    )
    dataset = Dataset.from_dict(
        {
            "image": [{"bytes": buffered.getvalue(), "path": None}],
            "caption": ["red"],
        },
        features=features,
    )
    monkeypatch.setattr(calibration, "load_dataset", lambda *args, **kwargs: dataset)

    loaded = load_calibration_dataset("images", "test", 1)

    assert loaded.features == features
    messages = build_calibration_messages(loaded[0], image_format="image_url")
    assert messages[0]["content"][0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_model_preprocessor_overrides_common_chat_normalization(monkeypatch):
    dataset = Dataset.from_dict(
        {"messages": [[{"role": "user", "content": "raw chat"}]], "meta": [1]}
    )
    monkeypatch.setattr(calibration, "load_dataset", lambda *args, **kwargs: dataset)

    loaded = load_calibration_dataset(
        "text",
        "train",
        1,
        preprocess=lambda row: {"text": row["messages"][0]["content"]},
    )

    assert loaded[0] == {"text": "raw chat"}


def test_preprocessor_can_return_original_row(monkeypatch):
    dataset = Dataset.from_dict({"text": ["keep this row"], "meta": ["keep"]})
    monkeypatch.setattr(calibration, "load_dataset", lambda *args, **kwargs: dataset)

    loaded = load_calibration_dataset("text", "train", 1, preprocess=lambda row: row)

    assert list(loaded) == list(dataset)


def test_model_preprocessor_is_skipped_for_tokenized_sources(monkeypatch):
    dataset = Dataset.from_dict({"input_ids": [[1, 2]], "attention_mask": [[1, 1]]})
    monkeypatch.setattr(calibration, "load_dataset", lambda *args, **kwargs: dataset)

    def unexpected_preprocess(row):
        pytest.fail("Pretokenized records must bypass model-specific encoding")

    loaded = load_calibration_dataset(
        "tokens", "test", 1, preprocess=unexpected_preprocess
    )

    assert loaded.features == dataset.features
    assert list(loaded) == list(dataset)


@pytest.mark.parametrize(
    ("sources", "samples"),
    [([], 1), (["first", "second"], 1), (["first"], 0), (["first"], -1)],
)
def test_invalid_primary_budgets_fail_before_loading(sources, samples, monkeypatch):
    def unexpected_load(*args, **kwargs):
        pytest.fail("Invalid budgets must fail before loading a dataset")

    monkeypatch.setattr(calibration, "load_dataset", unexpected_load)
    with pytest.raises(ValueError):
        load_calibration_dataset(sources, "train", samples)


@pytest.mark.parametrize(("sources", "samples"), [(None, 1), ([], 1), ([], -1)])
def test_invalid_text_budgets_fail_before_loading(sources, samples, monkeypatch):
    def unexpected_load(*args, **kwargs):
        pytest.fail("Invalid text budgets must fail before loading a dataset")

    monkeypatch.setattr(calibration, "load_dataset", unexpected_load)
    with pytest.raises(ValueError):
        load_calibration_datasets("primary", "test", sources, "train", 1, samples)


def test_incompatible_tokenized_and_chat_sources_report_columns(monkeypatch):
    datasets = [
        Dataset.from_dict({"input_ids": [[1, 2]]}),
        Dataset.from_dict({"text": ["hello"]}),
    ]
    monkeypatch.setattr(
        calibration, "load_dataset", lambda *args, **kwargs: datasets.pop(0)
    )

    with pytest.raises(ValueError, match="incompatible columns"):
        load_calibration_dataset(["tokens", "chat"], "train", 2)


@pytest.mark.parametrize(
    "content",
    [
        "hello",
        {"text": "hello"},
        ["hello"],
        ({"type": "text", "text": "hello"},),
        [{"type": "reasoning", "reasoning": "hello"}],
    ],
)
def test_text_content_formats_are_normalized(content):
    assert normalize_text_content(content) == [{"type": "text", "text": "hello"}]


def test_text_normalization_skips_image_blocks_and_rejects_empty_messages():
    assert normalize_text_content([{"type": "image", "image": object()}]) == []
    with pytest.raises(ValueError, match="no usable messages"):
        format_calibration_text({"messages": [{"role": "user", "content": None}]})


@pytest.mark.parametrize(
    "image_format", ["image", "url", "image_url", "image_url_flat"]
)
def test_image_messages_keep_processor_contract_and_append_text(image_format):
    image = Image.new("RGB", (2, 3), "blue")
    messages = build_calibration_messages(
        {"image": image, "caption": ["blue image", "unused"]},
        {"messages": [{"role": "assistant", "content": "more text"}]},
        image_format=image_format,
    )

    image_block = messages[0]["content"][0]
    if image_format == "image":
        assert image_block == {"type": "image", "image": image}
    else:
        if image_format == "url":
            url = image_block["url"]
        elif image_format == "image_url_flat":
            url = image_block["image_url"]
        else:
            url = image_block["image_url"]["url"]
        assert image_block["type"] == (
            "image_url" if image_format == "image_url" else "image"
        )
        with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as decoded:
            assert decoded.mode == "RGB"
            assert decoded.size == image.size
            assert decoded.getpixel((0, 0)) == image.getpixel((0, 0))
    assert messages[1]["content"] == [{"type": "text", "text": "blue image"}]
    assert messages[2] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "more text"}],
    }


def test_extra_text_is_only_paired_with_image_rows():
    messages = build_calibration_messages(
        {"text": "primary"}, {"text": "image companion"}
    )
    assert messages == [
        {"role": "user", "content": [{"type": "text", "text": "primary"}]}
    ]


def test_supplementary_text_preprocessor_leaves_primary_rows_untouched(monkeypatch):
    primary = Dataset.from_dict({"text": ["primary text"]})
    supplementary = Dataset.from_dict({"text": [" extra text "], "meta": ["ignored"]})
    monkeypatch.setattr(
        calibration,
        "load_dataset",
        lambda source, **kwargs: primary if source == "primary" else supplementary,
    )

    loaded, text = load_calibration_datasets(
        "primary",
        "test",
        "supplementary",
        "train",
        1,
        1,
        text_preprocess=lambda row: {"text": row["text"].strip()},
    )

    assert loaded[0]["messages"][0]["content"] == [
        {"type": "text", "text": "primary text"}
    ]
    assert text == [{"text": "extra text"}]


@pytest.mark.parametrize("storage", ["pil", "bytes", "path"])
def test_image_data_urls_support_dataset_storage_formats(tmp_path, storage):
    image = Image.new("L", (2, 3), 127)
    path = tmp_path / "image.png"
    image.save(path)
    value = {
        "pil": image,
        "bytes": {"bytes": path.read_bytes()},
        "path": {"path": str(path)},
    }[storage]

    url = image_to_data_url(value)

    with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as decoded:
        assert decoded.mode == "RGB"
        assert decoded.getpixel((0, 0)) == (127, 127, 127)
