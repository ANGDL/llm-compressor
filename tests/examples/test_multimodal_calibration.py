import ast
import base64
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from datasets import Dataset
from llmcompressor.datasets import (
    build_calibration_messages,
    calibration,
    format_calibration_text,
    load_calibration_datasets,
)

EXAMPLES = Path(__file__).parents[2] / "examples/multimodal_vision"
SCRIPTS = [
    "glm5_next_wNa8.py",
    "kimi2_5_wNa8.py",
    "qwen_3_5_moe_w8a8_int8_imatrix_rtn.py",
]


def load_data_helpers(script):
    path = EXAMPLES / script
    tree = ast.parse(path.read_text())
    names = {
        "preprocess",
        "preprocess_and_tokenize",
        "tokenize_messages",
        "_select_calibration_example",
        "_prepare_text_example",
    }
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "build_calibration_messages": build_calibration_messages,
        "format_calibration_text": format_calibration_text,
        "load_calibration_datasets": load_calibration_datasets,
    }
    exec(
        compile(ast.Module(body=helpers, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace, tree


class RecordingProcessor:
    def __init__(self):
        self.messages = None
        self.template_kwargs = None
        self.inputs = None
        self.tokenizer = self

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        if kwargs.get("tokenize"):
            return {"input_ids": [1, 2], "pixel_values": [3.0]}
        return "formatted prompt"

    def __call__(self, **kwargs):
        self.inputs = kwargs
        self.messages = kwargs.get("messages", self.messages)
        return {"input_ids": [1, 2], "pixel_values": [3.0]}


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("text_count", [0, 2])
def test_mixed_preprocessing_preserves_image_contract_and_cycles_text(
    script, text_count
):
    namespace, _ = load_data_helpers(script)
    processor = RecordingProcessor()
    image = Image.new("RGB", (3, 2), "red")
    vision_messages = []

    def process_vision_info(messages):
        vision_messages.append(messages)
        return [messages[0]["content"][0]["image"]], None

    namespace.update(
        processor=processor,
        tokenizer=processor,
        args=SimpleNamespace(max_sequence_length=512),
        MAX_SEQUENCE_LENGTH=512,
        MULTIMODAL_MAX_SEQUENCE_LENGTH=512,
        text_examples=[{"text": f"text {index}"} for index in range(text_count)],
        process_vision_info=process_vision_info,
    )
    preprocess = namespace.get("preprocess", namespace.get("preprocess_and_tokenize"))

    output = preprocess({"image": image, "caption": ["caption", "unused"]}, 3)

    assert output["input_ids"] == [1, 2]
    messages = processor.messages
    assert messages[1]["content"] == [{"type": "text", "text": "caption"}]
    assert len(messages) == (3 if text_count else 2)
    if text_count:
        assert messages[2]["content"] == [{"type": "text", "text": "text 1"}]
    image_block = messages[0]["content"][0]
    assert image_block["type"] == "image"
    if script.startswith("qwen"):
        assert image_block["image"] is image
        assert vision_messages == [messages]
        assert processor.inputs == {
            "text": ["formatted prompt"],
            "images": [image],
            "videos": None,
            "padding": False,
            "max_length": 512,
            "truncation": True,
        }
    else:
        field = "image_url" if script.startswith("kimi") else "url"
        url = image_block[field]
        assert url.startswith("data:image/png;base64,")
        with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as decoded:
            assert decoded.size == image.size
            assert decoded.getpixel((0, 0)) == image.getpixel((0, 0))
    if script.startswith("glm"):
        assert processor.template_kwargs == {
            "tokenize": True,
            "return_dict": True,
            "padding": False,
            "max_length": 512,
            "truncation": True,
        }
    elif script.startswith("kimi"):
        assert processor.inputs["max_length"] == 512
        assert processor.inputs["padding"] is False
        assert processor.inputs["truncation"] is True


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("text_enabled", [False, True])
def test_example_loader_uses_independent_splits_and_optional_text(
    script, text_enabled, monkeypatch
):
    namespace, tree = load_data_helpers(script)
    calls = []

    def fake_load(source, *, split):
        calls.append((source, split))
        return Dataset.from_dict({"text": [source]})

    monkeypatch.setattr(calibration, "load_dataset", fake_load)
    namespace.update(
        args=SimpleNamespace(
            dataset_id="vision",
            dataset_split="test",
            text_dataset_id="text",
            text_dataset_split="train_sft",
            num_calibration_samples=4,
            text_calibration_samples=2 if text_enabled else 0,
            enable_non_multimodal_calibration=text_enabled,
        ),
        DATASET_ID="vision",
        DATASET_SPLIT="test",
        TEXT_DATASET_ID="text",
        TEXT_DATASET_SPLIT="train_sft",
        NUM_CALIBRATION_SAMPLES=4,
        NUM_TEXT_CALIBRATION_SAMPLES=2 if text_enabled else 0,
    )
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "load_calibration_datasets"
    )

    exec(
        compile(ast.Module(body=[assignment], type_ignores=[]), script, "exec"),
        namespace,
    )

    expected = [("vision", "test[:4]")]
    if text_enabled:
        text_samples = 4 if script.startswith("qwen") else 2
        expected.append(("text", f"train_sft[:{text_samples}]"))
    assert calls == expected
    assert len(namespace["text_examples"]) == int(text_enabled)


def test_qwen_text_policy_keeps_role_cleanup_and_whitespace_filtering():
    namespace, _ = load_data_helpers(SCRIPTS[2])
    prepare = namespace["_prepare_text_example"]

    row = prepare(
        {
            "messages": [
                {"role": "system", "content": " system "},
                {"role": "tool", "content": " fallback role "},
                {"content": " missing role "},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "reasoning", "reasoning": " think "},
                        {"type": "text", "text": " answer "},
                        {"type": "text", "text": "  "},
                    ],
                },
                {"role": "user", "content": " \n "},
                None,
            ]
        }
    )

    assert row == {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {"role": "user", "content": [{"type": "text", "text": "fallback role"}]},
            {"role": "user", "content": [{"type": "text", "text": "missing role"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "think"},
                    {"type": "text", "text": "answer"},
                ],
            },
        ]
    }
    with pytest.raises(ValueError, match="no usable messages"):
        prepare({"messages": [{"role": "user", "content": "  "}]})


def test_glm_text_only_rows_use_normalized_content_and_keep_tokenized_rows():
    namespace, _ = load_data_helpers(SCRIPTS[0])
    processor = RecordingProcessor()
    namespace.update(
        processor=processor,
        tokenizer=processor,
        args=SimpleNamespace(max_sequence_length=512),
        text_examples=[{"text": "image companion"}],
    )

    namespace["preprocess"]({"messages": [{"role": "user", "content": "text only"}]}, 0)

    assert processor.messages == [
        {"role": "user", "content": [{"type": "text", "text": "text only"}]}
    ]
    assert processor.inputs["text"] == "formatted prompt"
    row = {"input_ids": [1, 2], "attention_mask": [1, 1]}
    assert namespace["preprocess"](row, 0) is row
