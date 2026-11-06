"""Compare independent weight-only candidates at one frozen target boundary."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from compressed_tensors.quantization import QuantizationScheme
from torch import nn

from llmcompressor.utils.helpers import eval_context

from .activations import DiskBoundaryActivationStore
from .admission import DiskSpaceBudget
from .artifacts import fingerprint_json
from .checkpoint import with_work_directory_lock
from .collect import _invoke_target
from .execution import (
    _input_bytes,
    fingerprint_streaming_value,
    fingerprint_streaming_values,
    freeze_calibration_inputs,
    streaming_configuration,
)
from .loading import SubgraphWeightSession, build_meta_model
from .materialization import (
    CastWeightMaterializer,
    StreamingDTypePolicy,
    WeightMaterializer,
)
from .quantize import _quantize_module, _validate_statistics
from .statistics import (
    CandidateStatisticsCache,
    CandidateStatisticsKey,
    GPTQStatisticsCollector,
    IMatrixStatisticsCollector,
    StatisticsCollectorGroup,
)
from .statistics.candidates import _atomic_json

__all__ = [
    "StreamingCandidate",
    "StreamingCandidateResult",
    "StreamingCandidateEvaluation",
    "evaluate_streaming_candidates",
]


@dataclass(frozen=True)
class StreamingCandidate:
    name: str
    schemes: Mapping[str, QuantizationScheme]
    use_gptq: bool = True
    blocksize: int = 128
    dampening_frac: float = 0.01

    def to_dict(self):
        return {
            "name": self.name,
            "schemes": {
                name: scheme.model_dump(mode="json")
                for name, scheme in sorted(self.schemes.items())
            },
            "use_gptq": self.use_gptq,
            "blocksize": self.blocksize,
            "dampening_frac": self.dampening_frac,
        }


@dataclass(frozen=True)
class StreamingCandidateResult:
    name: str
    fingerprint: str
    mse: float | None
    elements: int
    error: str | None = None


@dataclass(frozen=True)
class StreamingCandidateEvaluation:
    results: tuple[StreamingCandidateResult, ...]
    best_candidate: str | None
    statistics_key: str
    cache_hit: bool
    report_path: Path


def _squared_error(actual, expected):
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError("Candidate changed output shape or dtype")
        if actual.is_floating_point():
            difference = actual.detach().cpu().double() - expected.double()
            return difference.square().sum().item(), difference.numel()
        if not torch.equal(actual.cpu(), expected):
            raise ValueError("Candidate changed a non-floating output")
        return 0.0, 0
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        if actual.keys() != expected.keys():
            raise ValueError("Candidate changed output keys")
        pairs = [(actual[name], expected[name]) for name in actual]
    elif isinstance(actual, (tuple, list)) and type(actual) is type(expected):
        if len(actual) != len(expected):
            raise ValueError("Candidate changed output length")
        pairs = zip(actual, expected)
    elif type(actual) is type(expected) and actual == expected:
        return 0.0, 0
    else:
        raise ValueError("Candidate changed output structure")
    total, count = 0.0, 0
    for value, reference in pairs:
        error, elements = _squared_error(value, reference)
        total += error
        count += elements
    return total, count


@with_work_directory_lock
def evaluate_streaming_candidates(
    *,
    model_factory: Callable[..., nn.Module],
    checkpoint: str | Path,
    target_name: str,
    calibration_batches: Iterable[Any] | Callable[[], Iterable[Any]],
    candidates: Sequence[StreamingCandidate],
    work_dir: str | Path,
    dataset_fingerprint: str,
    statistics_cache: CandidateStatisticsCache | None = None,
    prefix_fingerprint: str = "",
    statistics_identity: Any = None,
    forward_target: Callable[[nn.Module, Any], Any] | None = None,
    forward_identity: str = "",
    model_args: Sequence[Any] = (),
    model_kwargs: Mapping[str, Any] | None = None,
    materializer: WeightMaterializer | None = None,
    dtype_policy: StreamingDTypePolicy | None = None,
    device: torch.device | str = "cpu",
    target_dtype: torch.dtype = torch.bfloat16,
    max_statistics_bytes: int = 512 * 1024**2,
) -> StreamingCandidateEvaluation:
    """Score up to eight candidates by local calibration-output MSE.

    Every candidate starts from original checkpoint weights. Only raw compatible
    H/count and iMatrix sum/count are reused. This diagnostic does not publish a
    model or constitute an independent quality evaluation.
    """
    if not 1 <= len(candidates) <= 8:
        raise ValueError("Provide between one and eight local candidates")
    if max_statistics_bytes < 1:
        raise ValueError("max_statistics_bytes must be positive")
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("Candidate names must be unique")
    if forward_target is not None and not forward_identity:
        raise ValueError("Custom forward_target requires an explicit forward_identity")
    selected = set(candidates[0].schemes)
    if not target_name or not selected:
        raise ValueError("Candidates require an explicit target and schemes")
    for candidate in candidates:
        if not candidate.name or set(candidate.schemes) != selected:
            raise ValueError("Candidates must name the same selected modules")
        if (
            candidate.blocksize < 1
            or not math.isfinite(candidate.dampening_frac)
            or candidate.dampening_frac < 0
        ):
            raise ValueError("Invalid candidate GPTQ parameters")
        for name, scheme in candidate.schemes.items():
            if name != target_name and not name.startswith(target_name + "."):
                raise ValueError("Candidate modules must be under the fixed target")
            if (
                scheme.weights is None
                or scheme.input_activations is not None
                or scheme.output_activations is not None
            ):
                raise ValueError("Candidates support weight-only schemes")
    work = Path(work_dir)
    source_path = Path(checkpoint).resolve()
    if work.resolve().is_relative_to(source_path) or source_path.is_relative_to(
        work.resolve()
    ):
        raise ValueError("Candidate work_dir and checkpoint must be separate")
    if statistics_cache is not None:
        cache_root = statistics_cache.root.resolve()
        if cache_root.is_relative_to(work.resolve()) or work.resolve().is_relative_to(
            cache_root
        ):
            raise ValueError("Candidate cache and work_dir must be separate")
    inputs = freeze_calibration_inputs(
        calibration_batches() if callable(calibration_batches) else calibration_batches,
        work / "inputs",
    )
    report = work / "candidates.json"
    _atomic_json(
        report,
        {
            "schema_version": 1,
            "status": "collecting",
            "metric": "local-calibration-output-mse",
            "candidates": [candidate.to_dict() for candidate in candidates],
            "results": [],
            "best_candidate": None,
        },
    )
    teacher = DiskBoundaryActivationStore(work / "teacher")
    teacher.delete(0)
    materializer = materializer or CastWeightMaterializer()
    model = build_meta_model(model_factory, *model_args, **dict(model_kwargs or {}))
    session = SubgraphWeightSession(
        model, materializer.create_source(str(checkpoint)), materializer, dtype_policy
    )
    device = torch.device(device)
    forward = forward_target or _invoke_target
    algorithms = tuple(
        algorithm
        for algorithm, enabled in (
            ("gptq", any(candidate.use_gptq for candidate in candidates)),
            (
                "imatrix",
                any(
                    scheme.weights.observer == "imatrix_mse"
                    for candidate in candidates
                    for scheme in candidate.schemes.values()
                ),
            ),
        )
        if enabled
    )
    statistics = None
    cache_hit = False
    with session.loaded_modules((target_name,), device=device, dtype=target_dtype):
        target = model.get_submodule(target_name)
        modules = {name: model.get_submodule(name) for name in sorted(selected)}
        if any(not isinstance(module, nn.Linear) for module in modules.values()):
            raise ValueError("Candidates support Linear modules only")
        estimated_statistics = sum(
            (4 * module.in_features**2 + 8 if "gptq" in algorithms else 0)
            + (8 * module.in_features + 8 if "imatrix" in algorithms else 0)
            for module in modules.values()
        )
        limit = min(
            max_statistics_bytes,
            statistics_cache.max_entry_bytes
            if statistics_cache is not None
            else max_statistics_bytes,
        )
        if estimated_statistics > limit:
            raise MemoryError("Current target statistics exceed candidate budget")
        key = CandidateStatisticsKey(
            source_fingerprint=fingerprint_streaming_values(
                target.state_dict().items()
            ),
            dataset_fingerprint=dataset_fingerprint,
            target_name=target_name,
            target_index=0,
            boundary_fingerprint=inputs.fingerprint,
            prefix_fingerprint=prefix_fingerprint,
            algorithms=algorithms or ("reference",),
            observer_fingerprint=fingerprint_json(
                {
                    "counting": "raw-unmasked-input-v1",
                    "torch": torch.__version__,
                    "modules": sorted(selected),
                }
            ),
            transform_fingerprint=fingerprint_streaming_value(
                {
                    "declared": statistics_identity,
                    "model_configuration": streaming_configuration(
                        (model_args, model_kwargs)
                    ),
                    "factory": (
                        getattr(model_factory, "__module__", ""),
                        getattr(
                            model_factory,
                            "__qualname__",
                            type(model_factory).__qualname__,
                        ),
                    ),
                    "forward": forward_identity or "default-target-invocation-v1",
                    "classes": [
                        (name, type(module).__module__, type(module).__qualname__)
                        for name, module in target.named_modules()
                    ],
                    "materializer": asdict(
                        materializer.manifest_info(
                            target_dtype=target_dtype, dtype_policy=dtype_policy
                        )
                    ),
                }
            ),
            dtype=str(target_dtype),
        )
        if statistics_cache is not None and algorithms:
            statistics = statistics_cache.get(key)
            cache_hit = statistics is not None
        collectors = []
        if statistics is None:
            if "gptq" in algorithms:
                collectors.append(GPTQStatisticsCollector(modules))
            if "imatrix" in algorithms:
                collectors.append(IMatrixStatisticsCollector(modules))
        group = StatisticsCollectorGroup(collectors) if collectors else None
        with torch.no_grad(), eval_context(model), group or nullcontext():
            for index in range(len(inputs)):
                output = forward(target, inputs.store.get(0, index, device=device))
                DiskSpaceBudget(work).check(
                    "candidate teacher output", 2 * _input_bytes(output) + 4096
                )
                teacher.put(0, index, output)
                del output
        if statistics is None:
            statistics = group.export() if group is not None else {}
            if statistics_cache is not None and statistics:
                statistics_cache.put(key, statistics)
        group = None
        collectors.clear()
    by_module = {
        name: {
            statistic.removeprefix(name + "."): tensor
            for statistic, tensor in statistics.items()
            if statistic.startswith(name + ".")
        }
        for name in selected
    }
    results = []
    for candidate in candidates:
        identity = fingerprint_json(candidate.to_dict())
        originals = {}
        quantized = None
        try:
            with session.loaded_modules(
                (target_name,), device=device, dtype=target_dtype
            ):
                target = model.get_submodule(target_name)
                _validate_statistics(
                    candidate.schemes, by_module, use_gptq=candidate.use_gptq
                )
                try:
                    with torch.no_grad():
                        for name, scheme in candidate.schemes.items():
                            original = model.get_submodule(name)
                            quantized = _quantize_module(
                                original.weight,
                                deepcopy(scheme),
                                by_module[name],
                                device=device,
                                use_gptq=candidate.use_gptq,
                                blocksize=candidate.blocksize,
                                dampening_frac=candidate.dampening_frac,
                                compress=False,
                            )
                            quantized.bias = original.bias
                            originals[name] = original
                            model.set_submodule(name, quantized)
                        target = model.get_submodule(target_name)
                        total, count = 0.0, 0
                        with eval_context(model):
                            for index in range(len(inputs)):
                                output = forward(
                                    target, inputs.store.get(0, index, device=device)
                                )
                                error, elements = _squared_error(
                                    output, teacher.get(0, index)
                                )
                                del output
                                total += error
                                count += elements
                    if not count or not math.isfinite(total):
                        raise ValueError(
                            "Candidate output score is empty or non-finite"
                        )
                    result = StreamingCandidateResult(
                        candidate.name, identity, total / count, count
                    )
                finally:
                    for name, original in originals.items():
                        model.set_submodule(name, original)
                    originals.clear()
                    quantized = None
        except Exception as error:
            result = StreamingCandidateResult(
                candidate.name,
                identity,
                None,
                0,
                f"{type(error).__name__}: {error}",
            )
        results.append(result)
        valid = [value for value in results if value.mse is not None]
        best = (
            min(valid, key=lambda value: (value.mse, value.name)).name
            if valid
            else None
        )
        _atomic_json(
            report,
            {
                "schema_version": 1,
                "status": "complete"
                if len(results) == len(candidates)
                else "evaluating",
                "metric": "local-calibration-output-mse",
                "statistics_key": key.to_dict(),
                "statistics_digest": key.digest,
                "cache_hit": cache_hit,
                "best_candidate": best,
                "results": [asdict(value) for value in results],
            },
        )
    return StreamingCandidateEvaluation(
        tuple(results), best, key.digest, cache_hit, report
    )
