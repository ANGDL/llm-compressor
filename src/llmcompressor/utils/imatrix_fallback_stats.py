"""
Utilities for tracking IMatrixMSEObserver fallback warnings during quantization.

Provides :class:`ImatrixFallbackStats` which installs hooks to intercept
loguru warnings from :class:`IMatrixMSEObserver` and attribute them to
specific model modules.
"""

from collections import Counter, defaultdict
from contextvars import ContextVar
from typing import Any, Optional

from loguru import logger

__all__ = ["ImatrixFallbackStats"]


class ImatrixFallbackStats:
    """Track IMatrixMSEObserver fallback warnings per module.

    The observer falls back to uniform MSE when importance data is unavailable
    or invalid. This class intercepts those warnings and attributes them to
    specific modules, producing a summary of which modules were affected.

    Usage::

        stats = ImatrixFallbackStats()
        stats.install_hooks()
        # ... run quantization with imatrix_mse observer ...
        stats.print_summary()
        stats.remove_hooks()

        # Or as context manager (prints summary on exit):
        with ImatrixFallbackStats() as stats:
            ... run quantization ...

        # Stop immediately when an iMatrix input or its FP32 square/reduction
        # becomes NaN/Inf, while retaining the offending module in the log.
        with ImatrixFallbackStats(check_nonfinite=True):
            ... run quantization ...

    .. note::

        ``install_hooks()`` monkey-patches ``IMatrixGatherer`` and
        ``IMatrixMSEObserver`` at the class level. Only one active
        tracking session is supported at a time.
    """

    # Class-level state shared across all instances because the
    # monkey-patches operate at the class level on IMatrixMSEObserver
    # and IMatrixGatherer.
    _no_importance: Counter = Counter()
    _all_zero: Counter = Counter()
    _other: Counter = Counter()
    _module_name_by_id: dict[int, str] = {}
    _current_module: ContextVar[Optional[str]] = ContextVar(
        "imatrix_current_module", default=None
    )
    _check_nonfinite: bool = False
    _activation_stats: dict[str, dict[str, float | int]] = defaultdict(dict)
    _diagnostic_handles: dict[int, object] = {}
    _sink_id: Optional[int] = None
    _hooks_installed: bool = False

    def __init__(self, check_nonfinite: bool = False):
        """Create a fallback tracker.

        ``check_nonfinite`` inspects each iMatrix input ``x`` and its FP32
        square/per-channel reduction. The first NaN/Inf raises
        :class:`FloatingPointError`, stopping quantization.
        """
        self.check_nonfinite = check_nonfinite

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def install_hooks(self) -> None:
        """Install loguru sink and monkey-patches on observer/gatherer classes.

        Idempotent: subsequent calls are no-ops. The monkey-patches are
        permanent (class-level), while the loguru sink can be removed
        via :meth:`remove_hooks`.
        """
        ImatrixFallbackStats._check_nonfinite = self.check_nonfinite
        if ImatrixFallbackStats._hooks_installed:
            return

        if ImatrixFallbackStats._sink_id is None:
            ImatrixFallbackStats._sink_id = logger.add(
                _imatrix_warning_sink,
                level="WARNING",
                format="{message}",
                enqueue=False,
            )

        _install_class_patches()
        ImatrixFallbackStats._hooks_installed = True

    def remove_hooks(self) -> None:
        """Remove the loguru warning sink.

        Monkey-patches on ``IMatrixGatherer`` and ``IMatrixMSEObserver``
        are left in place (they are harmless and removing them would
        require storing originals globally).
        """
        if ImatrixFallbackStats._sink_id is not None:
            logger.remove(ImatrixFallbackStats._sink_id)
            ImatrixFallbackStats._sink_id = None
        for handle in ImatrixFallbackStats._diagnostic_handles.values():
            handle.remove()
        ImatrixFallbackStats._diagnostic_handles.clear()
        ImatrixFallbackStats._check_nonfinite = False
        ImatrixFallbackStats._hooks_installed = False

    def print_summary(self) -> None:
        """Print a summary of fallback warnings grouped by category and module."""
        print("\n[imatrix_mse] fallback summary")
        self._print_category(
            "no importance data available",
            ImatrixFallbackStats._no_importance,
        )
        self._print_category(
            "all zeros",
            ImatrixFallbackStats._all_zero,
        )
        self._print_category(
            "other",
            ImatrixFallbackStats._other,
        )
        if (
            ImatrixFallbackStats._activation_stats
            or ImatrixFallbackStats._check_nonfinite
        ):
            print("  [activation diagnostics]")
            print(
                f"       check_nonfinite={ImatrixFallbackStats._check_nonfinite} "
                f"modules={len(ImatrixFallbackStats._activation_stats)}"
            )
            for module_name, values in sorted(
                ImatrixFallbackStats._activation_stats.items()
            ):
                print(
                    f"       {module_name}: calls={values.get('calls', 0)} "
                    f"x_nonfinite={values.get('x_nonfinite', 0)} "
                    f"x_nan={values.get('x_nan', 0)} x_inf={values.get('x_inf', 0)} "
                    f"x2_nonfinite={values.get('x2_nonfinite', 0)} "
                    f"x2_nan={values.get('x2_nan', 0)} "
                    f"x2_inf={values.get('x2_inf', 0)} "
                    f"x2_sum_nonfinite={values.get('x2_sum_nonfinite', 0)} "
                    f"imatrix_sum_nonfinite={values.get('imatrix_sum_nonfinite', 0)} "
                    f"max_abs={values.get('max_abs', 0.0):.6g}"
                )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ImatrixFallbackStats":
        ImatrixFallbackStats._no_importance.clear()
        ImatrixFallbackStats._all_zero.clear()
        ImatrixFallbackStats._other.clear()
        ImatrixFallbackStats._activation_stats.clear()
        self.install_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            self.print_summary()
        finally:
            self.remove_hooks()
        return False

    # ------------------------------------------------------------------
    # Counters (read-only access for tests / external consumers)
    # ------------------------------------------------------------------

    @property
    def no_importance_counter(self) -> Counter:
        """Counter of "no importance data available" fallbacks by module name."""
        return ImatrixFallbackStats._no_importance

    @property
    def all_zero_counter(self) -> Counter:
        """Counter of "all zeros" fallbacks by module name."""
        return ImatrixFallbackStats._all_zero

    @property
    def other_counter(self) -> Counter:
        """Counter of other (uncategorized) fallbacks by module name."""
        return ImatrixFallbackStats._other

    @property
    def activation_stats(self) -> dict[str, dict[str, float | int | str]]:
        """Per-module ``x``/``x**2`` diagnostics collected by the active run."""
        return ImatrixFallbackStats._activation_stats

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _print_category(label: str, counts: Counter) -> None:
        print(f"  [{label}]")
        if not counts:
            print("       no modules triggered")
            return

        total_hits = sum(counts.values())
        print(f"       modules_triggered={len(counts)} total_hits={total_hits}")
        for module_name, hit_count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            print(f"       {hit_count:6d}  {module_name}")


# ---------------------------------------------------------------------------
# Module-level warning sink (required by loguru for picklable callable)
# ---------------------------------------------------------------------------


def _imatrix_warning_sink(message):
    """Intercept loguru warnings and attribute them to the current module."""
    text = message.record["message"]
    if not text.startswith("imatrix_mse:"):
        return

    module_name = ImatrixFallbackStats._current_module.get() or "<unknown>"

    if text.startswith(
        "imatrix_mse: no importance data available. Falling back to uniform MSE."
    ):
        ImatrixFallbackStats._no_importance[module_name] += 1
    elif text.startswith("imatrix_mse: all zeros. Falling back to uniform MSE."):
        ImatrixFallbackStats._all_zero[module_name] += 1
    else:
        ImatrixFallbackStats._other[module_name] += 1


def _record_activation_diagnostics(
    module_name: str, args: Any, module: Any = None
) -> None:
    """Record the exact arithmetic used by ``IMatrixMSEObserver``.

    The observer receives ``x`` in a forward-pre hook, converts it to FP32,
    squares it, and reduces all token dimensions into one value per input
    channel. Keeping these checks here (rather than in a separate model hook)
    makes the diagnostics correspond to the module that can later fall back.
    """
    import torch

    if isinstance(args, tuple):
        if not args:
            return
        x = args[0]
    else:
        x = args
    if isinstance(x, tuple):
        x = x[0] if x else None
    if x is None or not isinstance(x, torch.Tensor):
        return

    values = x.detach().to(torch.float32)
    x_nan = int(torch.isnan(values).sum().item())
    x_inf = int(torch.isinf(values).sum().item())
    x_nonfinite = x_nan + x_inf
    x_abs = values.abs()
    finite_abs = x_abs[torch.isfinite(x_abs)]
    max_abs = float(finite_abs.max().item()) if finite_abs.numel() else 0.0

    squared = values.square()
    x2_nan = int(torch.isnan(squared).sum().item())
    x2_inf = int(torch.isinf(squared).sum().item())
    x2_nonfinite = x2_nan + x2_inf

    if values.ndim <= 1:
        reduced = squared
    else:
        reduced = squared.sum(dim=list(range(values.ndim - 1)))
    x2_sum_nonfinite = int((~torch.isfinite(reduced)).sum().item())
    imatrix_sum = getattr(module, "_imatrix_sum", None)
    imatrix_sum_nonfinite = 0
    if isinstance(imatrix_sum, torch.Tensor):
        imatrix_sum_nonfinite = int((~torch.isfinite(imatrix_sum)).sum().item())

    stats = ImatrixFallbackStats._activation_stats[module_name]
    stats["calls"] = int(stats.get("calls", 0)) + 1
    stats["x_nonfinite"] = int(stats.get("x_nonfinite", 0)) + x_nonfinite
    stats["x_nan"] = int(stats.get("x_nan", 0)) + x_nan
    stats["x_inf"] = int(stats.get("x_inf", 0)) + x_inf
    stats["x2_nonfinite"] = int(stats.get("x2_nonfinite", 0)) + x2_nonfinite
    stats["x2_nan"] = int(stats.get("x2_nan", 0)) + x2_nan
    stats["x2_inf"] = int(stats.get("x2_inf", 0)) + x2_inf
    stats["x2_sum_nonfinite"] = int(
        stats.get("x2_sum_nonfinite", 0)
    ) + x2_sum_nonfinite
    stats["imatrix_sum_nonfinite"] = int(
        stats.get("imatrix_sum_nonfinite", 0)
    ) + imatrix_sum_nonfinite
    stats["max_abs"] = max(float(stats.get("max_abs", 0.0)), max_abs)

    if not (
        x_nonfinite
        or x2_nonfinite
        or x2_sum_nonfinite
        or imatrix_sum_nonfinite
    ):
        return

    message = (
        "IMatrix non-finite activation detected in "
        f"{module_name}: x(nan={x_nan}, inf={x_inf}), "
        f"x2(nan={x2_nan}, inf={x2_inf}), "
        f"x2_sum_nonfinite={x2_sum_nonfinite}, "
        f"imatrix_sum_nonfinite={imatrix_sum_nonfinite}, "
        f"max_abs={max_abs:.6g}"
    )
    logger.error(message)
    if ImatrixFallbackStats._check_nonfinite:
        raise FloatingPointError(message)


def _install_activation_hook(module, module_name: str) -> None:
    """Attach one diagnostic pre-hook to a module during iMatrix collection."""
    module_id = id(module)
    if module_id in ImatrixFallbackStats._diagnostic_handles:
        return

    def _hook(_module, args):
        _record_activation_diagnostics(module_name, args, _module)

    ImatrixFallbackStats._diagnostic_handles[module_id] = (
        module.register_forward_pre_hook(_hook)
    )


# ---------------------------------------------------------------------------
# Monkey-patches (applied once, never reverted)
# ---------------------------------------------------------------------------


def _install_class_patches():
    """Monkey-patch IMatrixGatherer and IMatrixMSEObserver to track module names.

    Patches apply at the class level and are guarded by a sentinel attribute
    so they are applied at most once per process.
    """
    from compressed_tensors.utils import match_named_modules

    from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
    from llmcompressor.observers.imatrix import IMatrixMSEObserver

    if getattr(IMatrixMSEObserver, "_imatrix_fallback_stats_installed", False):
        return

    original_gatherer_init = IMatrixGatherer.on_initialize
    original_attach = IMatrixMSEObserver.attach
    original_detach = IMatrixMSEObserver.detach
    original_validate = IMatrixMSEObserver._get_validated_importance

    def _gatherer_init_with_module_names(self, state, **kwargs):
        resolved_targets = (
            self.targets if isinstance(self.targets, list) else [self.targets]
        )
        for module_name, module in match_named_modules(
            state.model,
            resolved_targets,
            self.ignore,
        ):
            ImatrixFallbackStats._module_name_by_id[id(module)] = module_name
        return original_gatherer_init(self, state, **kwargs)

    def _attach_with_module_name(self, module):
        self._imatrix_script_module_name = (
            ImatrixFallbackStats._module_name_by_id.get(
                id(module),
                module.__class__.__name__,
            )
        )
        result = original_attach(self, module)
        if (
            ImatrixFallbackStats._check_nonfinite
            and hasattr(module, "_imatrix_hook")
        ):
            _install_activation_hook(module, self._imatrix_script_module_name)
        return result

    def _detach_with_diagnostics(self, module):
        try:
            return original_detach(self, module)
        finally:
            handle = ImatrixFallbackStats._diagnostic_handles.pop(id(module), None)
            if handle is not None:
                handle.remove()

    def _validate_with_stats(self, observed):
        module_name = getattr(
            self, "_imatrix_script_module_name", "<unknown>"
        )
        token = ImatrixFallbackStats._current_module.set(module_name)
        try:
            return original_validate(self, observed)
        finally:
            ImatrixFallbackStats._current_module.reset(token)

    IMatrixGatherer.on_initialize = _gatherer_init_with_module_names
    IMatrixMSEObserver.attach = _attach_with_module_name
    IMatrixMSEObserver.detach = _detach_with_diagnostics
    IMatrixMSEObserver._get_validated_importance = _validate_with_stats
    IMatrixMSEObserver._imatrix_fallback_stats_installed = True
