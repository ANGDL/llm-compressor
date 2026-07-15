"""Compatibility helpers for loading Kimi K2.5 with Transformers code."""

from os import PathLike
from typing import Any

from transformers import (
    Kimi_K25Config,
    Kimi_K25ForConditionalGeneration,
    PreTrainedModel,
)


# Early Kimi K2.5 checkpoints use names without the underscore after ``Kimi``.
# Keep those public names available without copying or subclassing the native model.
KimiK25Config = Kimi_K25Config
KimiK25ForConditionalGeneration = Kimi_K25ForConditionalGeneration


def load_kimi_k25_model(
    pretrained_model_name_or_path: str | PathLike[str],
    **kwargs: Any,
) -> PreTrainedModel:
    """Load Kimi K2.5 using the implementation bundled with Transformers.

    Calling the concrete native class bypasses the checkpoint's legacy ``auto_map``
    entries. This is intentional: passing ``trust_remote_code=True`` to an AutoModel
    would otherwise prefer ``modeling_kimi_k25.py`` stored with the checkpoint.
    """
    kwargs.pop("trust_remote_code", None)
    return Kimi_K25ForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=False,
        **kwargs,
    )


__all__ = [
    "KimiK25Config",
    "KimiK25ForConditionalGeneration",
    "load_kimi_k25_model",
]
