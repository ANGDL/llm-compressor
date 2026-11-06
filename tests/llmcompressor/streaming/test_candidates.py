import json
from dataclasses import replace

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from safetensors.torch import save_file
from torch import nn

from llmcompressor.streaming import (
    CandidateStatisticsCache,
    CandidateStatisticsKey,
    StreamingCandidate,
    StreamingDTypePolicy,
    evaluate_streaming_candidates,
)
from llmcompressor.streaming.statistics import GPTQStatisticsCollector


class CandidateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(8, 8), nn.SiLU(), nn.Linear(8, 8))


def _checkpoint(tmp_path):
    torch.manual_seed(28)
    reference = CandidateModel()
    root = tmp_path / "source"
    root.mkdir()
    (root / "config.json").write_text('{"model_type": "tiny"}')
    save_file(reference.state_dict(), root / "model.safetensors")
    return root


def _candidates():
    return tuple(
        StreamingCandidate(
            name=f"w{bits}",
            schemes={
                name: QuantizationScheme(
                    targets=[name],
                    weights=QuantizationArgs(num_bits=bits, strategy="channel"),
                )
                for name in ("block.0", "block.2")
            },
        )
        for bits in (4, 8)
    )


def _evaluate(tmp_path, source, batches, candidates, **kwargs):
    return evaluate_streaming_candidates(
        model_factory=CandidateModel,
        checkpoint=source,
        target_name="block",
        calibration_batches=batches,
        candidates=candidates,
        work_dir=tmp_path / "work",
        dataset_fingerprint="a" * 64,
        target_dtype=torch.float32,
        **kwargs,
    )


def test_candidate_cache_reuses_statistics_and_order_is_independent(
    tmp_path, monkeypatch
):
    source = _checkpoint(tmp_path)
    batches = [torch.randn(2, 9, 8), torch.randn(1, 7, 8)]
    candidates = _candidates()
    cache = CandidateStatisticsCache(tmp_path / "cache")
    first = _evaluate(tmp_path, source, batches, candidates, statistics_cache=cache)
    assert not first.cache_hit
    assert first.best_candidate == "w8"
    assert all(value.mse > 0 and value.error is None for value in first.results)
    assert first.results[0].elements == 200

    def unexpected(*_args, **_kwargs):
        raise AssertionError("Statistics must not be collected on a cache hit")

    monkeypatch.setattr(GPTQStatisticsCollector, "attach", unexpected)
    second = _evaluate(
        tmp_path, source, batches, candidates[::-1], statistics_cache=cache
    )
    assert second.cache_hit
    assert first.statistics_key == second.statistics_key
    assert {value.name: value.mse for value in first.results} == {
        value.name: value.mse for value in second.results
    }
    report = json.loads(second.report_path.read_text())
    assert report["metric"] == "local-calibration-output-mse"
    assert report["best_candidate"] == "w8"


@pytest.mark.parametrize("change", ["input", "prefix", "transform", "weights", "dtype"])
def test_candidate_identity_invalidates_on_actual_changes(tmp_path, change):
    source = _checkpoint(tmp_path)
    batches = [torch.randn(2, 8, 8)]
    candidates = _candidates()
    cache = CandidateStatisticsCache(tmp_path / "cache")
    first = _evaluate(tmp_path, source, batches, candidates, statistics_cache=cache)
    kwargs = {}
    if change == "input":
        batches[0][0, 0, 0] += 1
    elif change == "prefix":
        kwargs["prefix_fingerprint"] = "changed-previous-layer"
    elif change == "transform":
        kwargs["statistics_identity"] = {"rotation": "new"}
    elif change == "weights":
        model = CandidateModel()
        save_file(model.state_dict(), source / "model.safetensors")
    elif change == "dtype":
        kwargs["dtype_policy"] = StreamingDTypePolicy(
            compute_dtype=torch.float32, preserve_source_dtype=False
        )
    second = _evaluate(
        tmp_path, source, batches, candidates, statistics_cache=cache, **kwargs
    )
    assert not second.cache_hit
    assert first.statistics_key != second.statistics_key


def test_failed_candidate_cannot_modify_next_candidate(tmp_path, monkeypatch):
    import llmcompressor.streaming.candidates as implementation

    source = _checkpoint(tmp_path)
    batches = [torch.randn(2, 8, 8)]
    candidates = _candidates()
    reference = _evaluate(tmp_path, source, batches, candidates)
    original = implementation._quantize_module
    calls = 0

    def fail_once(weight, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            weight.fill_(float("nan"))
            raise RuntimeError("injected candidate failure")
        return original(weight, *args, **kwargs)

    monkeypatch.setattr(implementation, "_quantize_module", fail_once)
    actual = _evaluate(tmp_path, source, batches, candidates)
    assert "injected candidate failure" in actual.results[0].error
    assert actual.results[1].mse == reference.results[1].mse
    assert actual.best_candidate == "w8"


def test_imatrix_rtn_candidates_share_only_compatible_raw_statistics(
    tmp_path, monkeypatch
):
    from llmcompressor.streaming.statistics import IMatrixStatisticsCollector

    source = _checkpoint(tmp_path)
    batches = [torch.randn(2, 8, 8)]
    candidates = tuple(
        replace(candidate, use_gptq=False) for candidate in _candidates()
    )
    for candidate in candidates:
        for scheme in candidate.schemes.values():
            scheme.weights.observer = "imatrix_mse"
    cache = CandidateStatisticsCache(tmp_path / "cache")
    first = _evaluate(tmp_path, source, batches, candidates, statistics_cache=cache)
    assert all(value.error is None and value.mse > 0 for value in first.results)
    monkeypatch.setattr(
        IMatrixStatisticsCollector,
        "attach",
        lambda *_args: pytest.fail("cached iMatrix must not collect twice"),
    )
    second = _evaluate(tmp_path, source, batches, candidates, statistics_cache=cache)
    assert second.cache_hit
    assert second.results == first.results


def test_candidate_statistics_budget_rejects_before_collection(tmp_path, monkeypatch):
    source = _checkpoint(tmp_path)
    monkeypatch.setattr(
        GPTQStatisticsCollector,
        "attach",
        lambda *_args: pytest.fail("budget must be checked before collecting H"),
    )
    with pytest.raises(MemoryError, match="candidate budget"):
        _evaluate(
            tmp_path,
            source,
            [torch.ones(1, 8)],
            _candidates(),
            max_statistics_bytes=1,
        )


def _key():
    return CandidateStatisticsKey(
        source_fingerprint="a" * 64,
        dataset_fingerprint="b" * 64,
        target_name="block",
        target_index=0,
        boundary_fingerprint="c" * 64,
        prefix_fingerprint="prefix",
        algorithms=("gptq",),
        observer_fingerprint="d" * 64,
        transform_fingerprint="e" * 64,
        dtype="torch.float32",
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_fingerprint", "f" * 64),
        ("boundary_fingerprint", "f" * 64),
        ("dataset_fingerprint", "f" * 64),
        ("observer_fingerprint", "f" * 64),
        ("transform_fingerprint", "f" * 64),
        ("prefix_fingerprint", "other"),
        ("target_name", "other"),
        ("target_index", 1),
        ("algorithms", ("imatrix",)),
        ("dtype", "torch.bfloat16"),
        ("implementation", "new-counting-rule"),
    ],
)
def test_cache_key_components_never_alias(tmp_path, field, value):
    cache = CandidateStatisticsCache(tmp_path / "cache")
    cache.put(_key(), {"hessian": torch.eye(4)})
    assert cache.get(replace(_key(), **{field: value})) is None


def test_cache_validates_payload_and_enforces_budgets(tmp_path):
    key = _key()
    cache = CandidateStatisticsCache(tmp_path / "cache")
    cache.put(key, {"hessian": torch.eye(4)})
    first = cache.get(key)
    first["hessian"].zero_()
    assert torch.equal(cache.get(key)["hessian"], torch.eye(4))
    assert cache.get(key, expected_names=("count",)) is None
    with pytest.raises(RuntimeError, match="immutable"):
        cache.put(key, {"hessian": torch.ones(4, 4)})
    payload = cache.root / key.digest / "stats.safetensors"
    content = bytearray(payload.read_bytes())
    content[-1] ^= 1
    payload.write_bytes(content)
    with pytest.raises(RuntimeError, match="digest"):
        cache.get(key)
    small = CandidateStatisticsCache(tmp_path / "small", max_entry_bytes=1)
    with pytest.raises(MemoryError):
        small.put(key, {"hessian": torch.eye(4)})
    full = CandidateStatisticsCache(tmp_path / "full", max_bytes=1)
    with pytest.raises(OSError, match="budget"):
        full.put(key, {"hessian": torch.eye(4)})


def test_cache_publish_failure_is_replayable_and_prefix_invalidation(
    tmp_path, monkeypatch
):
    import llmcompressor.streaming.statistics.candidates as implementation

    cache = CandidateStatisticsCache(tmp_path / "cache")
    original = implementation.os.replace

    def fail_publish(source, destination):
        if str(source).endswith(".pending"):
            raise OSError("injected publish failure")
        return original(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(implementation.os, "replace", fail_publish)
        with pytest.raises(OSError, match="injected"):
            cache.put(_key(), {"hessian": torch.eye(4)})
    assert cache.get(_key()) is None
    assert not list(cache.root.glob("*.pending"))
    cache.put(_key(), {"hessian": torch.eye(4)})
    assert cache.invalidate_prefix("other") == 0
    assert cache.invalidate_prefix("prefix") == 1
    assert cache.get(_key()) is None
