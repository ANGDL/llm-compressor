"""Compatibility wrapper for upstream Transformers MTP modules."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from loguru import logger
from torch import nn

from llmcompressor.transformers.compression.mtp import has_mtp_in_checkpoint
from llmcompressor.transformers.compression.mtp_compat import (
    build_mtp_model,
    get_mtp_config,
    get_streaming_mtp_weight_conversions,
)

__all__ = ["StreamingMTPWrapper"]


_MTP_IGNORE_PATTERNS = (
    r"re:^mtp(?:\.|$)",
    r"re:^model\.mtp(?:\.|$)",
)


def _detach_shared_modules(mtp: nn.Module, model: nn.Module) -> None:
    """Keep shared aliases out of the MTP registered module tree."""

    backbone_modules = set(model.modules())
    for name in ("embed_tokens", "shared_head", "rotary_emb"):
        shared = getattr(mtp, name, None)
        if shared is not None and shared in backbone_modules:
            mtp._modules.pop(name, None)
            object.__setattr__(mtp, name, shared)


def _prepare_mtp_forward_namespace(mtp: nn.Module) -> None:
    """Fill optional annotation globals missing in some Transformers builds."""

    forward = inspect.unwrap(type(mtp).forward)
    namespace = forward.__globals__
    namespace.setdefault("MtpCache", object)
    namespace.setdefault("LogitsProcessorList", object)


@dataclass(frozen=True)
class StreamingMTPWrapper:
    """Describe an optional MTP attachment for streaming model setup.

    Native special models that already expose ``model.mtp`` are returned
    untouched.  The wrapper only constructs upstream ``MtpModel`` when the
    base model declares MTP but has not attached it yet.
    """

    attached: bool
    ignored_targets: tuple[str, ...] = ()
    weight_conversions: tuple[Any, ...] = ()

    @classmethod
    def attach(cls, model: nn.Module) -> StreamingMTPWrapper:
        if hasattr(model, "mtp") or not _declares_mtp(model):
            return cls(attached=False)
        try:
            config = get_mtp_config(model)
            mtp = build_mtp_model(model, config)
            _prepare_mtp_forward_namespace(mtp)
            model.mtp = mtp
            _detach_shared_modules(mtp, model)
            conversions = tuple(
                get_streaming_mtp_weight_conversions(model, mtp)
            )
        except (AttributeError, NotImplementedError, TypeError, ValueError) as error:
            logger.warning(
                "Streaming MTP wrapper could not construct upstream MtpModel; "
                f"leaving the base model unchanged: {error}"
            )
            return cls(attached=False)

        return cls(
            attached=True,
            ignored_targets=_MTP_IGNORE_PATTERNS,
            weight_conversions=conversions,
        )


def _declares_mtp(model: nn.Module) -> bool:
    config = getattr(model, "config", None)
    if config is None or not callable(getattr(config, "get_text_config", None)):
        return False
    return has_mtp_in_checkpoint(model)
