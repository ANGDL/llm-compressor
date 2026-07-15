from unittest.mock import patch

from transformers import Kimi_K25Config, Kimi_K25ForConditionalGeneration

from llmcompressor.modeling.kimi_k25 import (
    KimiK25Config,
    KimiK25ForConditionalGeneration,
    load_kimi_k25_model,
)


def test_legacy_names_alias_transformers_classes():
    assert KimiK25Config is Kimi_K25Config
    assert KimiK25ForConditionalGeneration is Kimi_K25ForConditionalGeneration


def test_loader_forces_transformers_implementation():
    sentinel = object()

    with patch.object(
        Kimi_K25ForConditionalGeneration,
        "from_pretrained",
        return_value=sentinel,
    ) as from_pretrained:
        model = load_kimi_k25_model(
            "kimi-checkpoint",
            dtype="auto",
            trust_remote_code=True,
        )

    assert model is sentinel
    from_pretrained.assert_called_once_with(
        "kimi-checkpoint",
        dtype="auto",
        trust_remote_code=False,
    )
