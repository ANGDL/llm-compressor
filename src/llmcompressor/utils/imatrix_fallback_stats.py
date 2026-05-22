"""
Utilities for tracking IMatrixMSEObserver fallback warnings during quantization.

Provides :class:`ImatrixFallbackStats` which installs hooks to intercept
loguru warnings from :class:`IMatrixMSEObserver` and attribute them to
specific model modules.
"""

from collections import Counter
from contextvars import ContextVar
from typing import Optional

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
    _sink_id: Optional[int] = None
    _hooks_installed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def install_hooks(self) -> None:
        """Install loguru sink and monkey-patches on observer/gatherer classes.

        Idempotent: subsequent calls are no-ops. The monkey-patches are
        permanent (class-level), while the loguru sink can be removed
        via :meth:`remove_hooks`.
        """
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

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ImatrixFallbackStats":
        ImatrixFallbackStats._no_importance.clear()
        ImatrixFallbackStats._all_zero.clear()
        ImatrixFallbackStats._other.clear()
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
        return original_attach(self, module)

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
    IMatrixMSEObserver._get_validated_importance = _validate_with_stats
    IMatrixMSEObserver._imatrix_fallback_stats_installed = True
