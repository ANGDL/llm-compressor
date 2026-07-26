import torch
from loguru import logger

from llmcompressor.utils import metric_logging
from llmcompressor.utils.metric_logging import CompressionLogger


def test_compression_logger_only_queries_selected_devices(monkeypatch):
    queried = []
    monkeypatch.setattr(metric_logging, "accelerator_is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "is_available", lambda: False)
    monkeypatch.setattr(
        metric_logging,
        "accelerator_max_memory_allocated",
        lambda device_id: queried.append(device_id) or 1,
    )
    monkeypatch.setattr(
        metric_logging,
        "accelerator_get_memory_info",
        lambda _device_id: (9, 10),
    )
    monkeypatch.setattr(
        metric_logging,
        "_get_visible_devices",
        lambda: (_ for _ in ()).throw(
            AssertionError("selected devices must not enumerate all accelerators")
        ),
    )

    with CompressionLogger(torch.nn.Linear(1, 1), device_ids=(3,)):
        pass

    assert queried == [3]


def test_compression_logger_empty_devices_skips_memory_queries(monkeypatch):
    monkeypatch.setattr(metric_logging, "accelerator_is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "is_available", lambda: False)
    monkeypatch.setattr(
        metric_logging,
        "accelerator_max_memory_allocated",
        lambda _device_id: (_ for _ in ()).throw(
            AssertionError("CPU logging must not query accelerator memory")
        ),
    )

    with CompressionLogger(torch.nn.Linear(1, 1), device_ids=()):
        pass


def test_compression_logger_compact_summary(monkeypatch):
    monkeypatch.setattr(metric_logging, "accelerator_is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "is_available", lambda: False)
    monkeypatch.setattr(
        metric_logging,
        "accelerator_max_memory_allocated",
        lambda _device_id: int(2e9),
    )
    monkeypatch.setattr(
        metric_logging,
        "accelerator_get_memory_info",
        lambda _device_id: (0, int(10e9)),
    )
    records = []
    sink = logger.add(lambda message: records.append(message.record), level="METRIC")
    try:
        metrics = CompressionLogger(torch.nn.Linear(1, 1), device_ids=(1,))
        metrics.set_results(
            name="streaming",
            summary="[03/44] model.layers.2 | calibration | device=cuda:1",
        )
        with metrics:
            pass
    finally:
        logger.remove(sink)

    record = records[-1]
    assert record["function"] == "streaming"
    assert record["message"].startswith(
        "[03/44] model.layers.2 | calibration | device=cuda:1 | time="
    )
    assert record["message"].endswith(
        "cuda:1 peak=2.00/10.0 GB (20.0%)"
    )
