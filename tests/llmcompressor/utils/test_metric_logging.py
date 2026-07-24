import torch

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
