from __future__ import annotations

import threading

import torch

from llmcompressor.streaming.checkpoint import SafetensorsWeightSource
from llmcompressor.streaming.loading import SubgraphPrefetcher, SubgraphWeightSession
from llmcompressor.streaming.loading.target import build_meta_model
from tests.llmcompressor.streaming.test_subgraph_weight_session import (
    SessionModel,
    assert_all_meta,
    checkpoint_for,
    subgraph_for,
)


def test_prefetch_materializes_without_installing_shared_model(tmp_path, monkeypatch):
    reference = SessionModel()
    checkpoint = checkpoint_for(reference, tmp_path)
    model = build_meta_model(SessionModel)
    session = SubgraphWeightSession(
        model, SafetensorsWeightSource(checkpoint)
    )
    plan = session.plan(subgraph_for("layers.0"))
    original_prepare = session.prepare
    started = threading.Event()
    release = threading.Event()

    def blocking_prepare(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(session, "prepare", blocking_prepare)
    prefetcher = SubgraphPrefetcher(session)
    try:
        prefetcher.submit(
            plan,
            device="cpu",
            dtype=torch.float32,
            label="test/weight_materialization",
        )
        assert started.wait(timeout=5)
        assert_all_meta(model)
        release.set()
        prepared = prefetcher.take()
        assert_all_meta(model)
        with session.installed(prepared):
            assert not next(model.layers[0].parameters()).is_meta
    finally:
        release.set()
        prefetcher.close(suppress_errors=True)

    assert_all_meta(model)
