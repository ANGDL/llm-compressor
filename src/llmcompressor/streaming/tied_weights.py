"""Discovery of checkpoint tensor aliases declared by Transformers models."""

from __future__ import annotations

from pathlib import Path

import torch


def infer_transformers_tied_weights(checkpoint: str | Path) -> dict[str, str]:
    """Return ``alias -> canonical`` tied-weight names from local config.

    Constructing on ``meta`` makes this independent of checkpoint size.  Failure
    to construct a standard Transformers model is non-fatal: custom model
    factories can still be quantized, but no aliases are removed implicitly.

    Models may expose ``get_expanded_tied_weights_keys()`` to declare aliases
    that are not enumerable from the class-level ``_tied_weights_keys`` attribute
    (for example one entry per MTP layer). Prefer it so this stays consistent
    with the loader's tied-source resolution in
    :class:`~llmcompressor.streaming.loading.target.TargetWeightLoader`.
    """
    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
    except Exception:
        return {}

    expanded = getattr(model, "get_expanded_tied_weights_keys", None)
    declared = (
        expanded() if callable(expanded) else None
    ) or getattr(model, "_tied_weights_keys", None) or {}
    if isinstance(declared, dict):
        return {str(alias): str(canonical) for alias, canonical in declared.items()}
    return {}
