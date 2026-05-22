import io
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from llmcompressor.utils.imatrix_fallback_stats import (
    ImatrixFallbackStats,
    _imatrix_warning_sink,
)

# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


class _FakeMessage:
    """Minimal loguru message-like object for testing the warning sink."""

    def __init__(self, text):
        self.record = {"message": text}


def _reset_class_state():
    """Reset all class-level state between tests."""
    ImatrixFallbackStats._no_importance.clear()
    ImatrixFallbackStats._all_zero.clear()
    ImatrixFallbackStats._other.clear()
    ImatrixFallbackStats._module_name_by_id.clear()
    # Remove any lingering sink
    if ImatrixFallbackStats._sink_id is not None:
        from loguru import logger

        try:
            logger.remove(ImatrixFallbackStats._sink_id)
        except ValueError:
            pass
        ImatrixFallbackStats._sink_id = None
    ImatrixFallbackStats._hooks_installed = False
    # Reset ContextVar to default (may have been left polluted by a
    # test that called .set() without a matching .reset()).
    ImatrixFallbackStats._current_module.set(None)
    # Reset the monkey-patch sentinel so install can be tested fresh
    from llmcompressor.observers.imatrix import IMatrixMSEObserver

    if hasattr(IMatrixMSEObserver, "_imatrix_fallback_stats_installed"):
        delattr(IMatrixMSEObserver, "_imatrix_fallback_stats_installed")


# ------------------------------------------------------------------ #
#  Tests: warning sink
# ------------------------------------------------------------------ #


class TestWarningSink:
    """Unit tests for the _imatrix_warning_sink function."""

    def test_sink_captures_no_importance(self):
        _reset_class_state()
        ImatrixFallbackStats._current_module.set("model.layers.0")
        _imatrix_warning_sink(
            _FakeMessage(
                "imatrix_mse: no importance data available."
                " Falling back to uniform MSE."
            )
        )
        assert ImatrixFallbackStats._no_importance["model.layers.0"] == 1
        assert len(ImatrixFallbackStats._all_zero) == 0
        assert len(ImatrixFallbackStats._other) == 0

    def test_sink_captures_all_zero(self):
        _reset_class_state()
        ImatrixFallbackStats._current_module.set("model.layers.1")
        _imatrix_warning_sink(
            _FakeMessage(
                "imatrix_mse: all zeros. Falling back to uniform MSE."
            )
        )
        assert ImatrixFallbackStats._all_zero["model.layers.1"] == 1
        assert len(ImatrixFallbackStats._no_importance) == 0
        assert len(ImatrixFallbackStats._other) == 0

    def test_sink_captures_other_imatrix_warning(self):
        _reset_class_state()
        ImatrixFallbackStats._current_module.set("model.layers.2")
        _imatrix_warning_sink(
            _FakeMessage(
                "imatrix_mse: contains non-finite values."
                " Falling back to uniform MSE."
            )
        )
        assert ImatrixFallbackStats._other["model.layers.2"] == 1
        assert len(ImatrixFallbackStats._no_importance) == 0
        assert len(ImatrixFallbackStats._all_zero) == 0

    def test_sink_ignores_non_imatrix_messages(self):
        _reset_class_state()
        ImatrixFallbackStats._current_module.set("model.layers.0")
        _imatrix_warning_sink(_FakeMessage("some other warning message"))
        assert len(ImatrixFallbackStats._no_importance) == 0
        assert len(ImatrixFallbackStats._all_zero) == 0
        assert len(ImatrixFallbackStats._other) == 0

    def test_sink_uses_unknown_when_no_module_set(self):
        _reset_class_state()
        # ContextVar default is None, so no module name is set
        _imatrix_warning_sink(
            _FakeMessage(
                "imatrix_mse: no importance data available."
                " Falling back to uniform MSE."
            )
        )
        assert ImatrixFallbackStats._no_importance["<unknown>"] == 1

    def test_sink_accumulates_multiple_hits_same_module(self):
        _reset_class_state()
        ImatrixFallbackStats._current_module.set("model.layers.0")
        _imatrix_warning_sink(
            _FakeMessage(
                "imatrix_mse: no importance data available."
                " Falling back to uniform MSE."
            )
        )
        _imatrix_warning_sink(
            _FakeMessage(
                "imatrix_mse: no importance data available."
                " Falling back to uniform MSE."
            )
        )
        assert ImatrixFallbackStats._no_importance["model.layers.0"] == 2


# ------------------------------------------------------------------ #
#  Tests: install / remove lifecycle
# ------------------------------------------------------------------ #


class TestLifecycle:
    """Tests for install_hooks / remove_hooks."""

    def test_install_is_idempotent(self):
        _reset_class_state()
        stats = ImatrixFallbackStats()
        try:
            stats.install_hooks()
            sink_id_1 = ImatrixFallbackStats._sink_id
            assert sink_id_1 is not None

            stats.install_hooks()
            sink_id_2 = ImatrixFallbackStats._sink_id
            # Same sink, not double-registered
            assert sink_id_2 == sink_id_1
        finally:
            stats.remove_hooks()

    def test_remove_hooks_clears_sink(self):
        _reset_class_state()
        stats = ImatrixFallbackStats()
        stats.install_hooks()
        assert ImatrixFallbackStats._sink_id is not None

        stats.remove_hooks()
        assert ImatrixFallbackStats._sink_id is None

    def test_remove_hooks_twice_is_safe(self):
        _reset_class_state()
        stats = ImatrixFallbackStats()
        stats.install_hooks()
        stats.remove_hooks()
        stats.remove_hooks()  # should not raise

    def test_multiple_instances_share_state(self):
        _reset_class_state()
        stats1 = ImatrixFallbackStats()
        stats2 = ImatrixFallbackStats()
        try:
            stats1.install_hooks()
            assert ImatrixFallbackStats._hooks_installed
            # Second instance can still call install (it's a no-op)
            stats2.install_hooks()
        finally:
            stats1.remove_hooks()


# ------------------------------------------------------------------ #
#  Tests: context manager
# ------------------------------------------------------------------ #


class TestContextManager:
    """Tests for the context manager protocol."""

    def test_context_manager_installs_and_removes(self):
        _reset_class_state()
        with ImatrixFallbackStats():
            assert ImatrixFallbackStats._sink_id is not None
            assert ImatrixFallbackStats._hooks_installed
        assert ImatrixFallbackStats._sink_id is None
        assert not ImatrixFallbackStats._hooks_installed

    def test_context_manager_prints_summary_on_exit(self):
        _reset_class_state()
        buf = io.StringIO()
        with redirect_stdout(buf):
            with ImatrixFallbackStats() as stats:
                token = ImatrixFallbackStats._current_module.set("mod")
                try:
                    _imatrix_warning_sink(
                        _FakeMessage(
                            "imatrix_mse: no importance data available."
                            " Falling back to uniform MSE."
                        )
                    )
                finally:
                    ImatrixFallbackStats._current_module.reset(token)
        output = buf.getvalue()
        assert "[imatrix_mse] fallback summary" in output
        assert "no importance data available" in output
        assert "mod" in output

    def test_context_manager_prints_summary_even_on_exception(self):
        _reset_class_state()
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                with ImatrixFallbackStats():
                    raise RuntimeError("boom")
            except RuntimeError:
                pass
        output = buf.getvalue()
        assert "[imatrix_mse] fallback summary" in output
        # Sink should be cleaned up even after exception
        assert ImatrixFallbackStats._sink_id is None

    def test_context_manager_clears_counters_on_enter(self):
        _reset_class_state()
        # Pollute counters from a "previous run"
        ImatrixFallbackStats._no_importance["old_module"] = 5
        ImatrixFallbackStats._all_zero["old_module"] = 3
        ImatrixFallbackStats._other["old_module"] = 1

        with ImatrixFallbackStats():
            # Counters should have been cleared on __enter__
            assert len(ImatrixFallbackStats._no_importance) == 0
            assert len(ImatrixFallbackStats._all_zero) == 0
            assert len(ImatrixFallbackStats._other) == 0


# ------------------------------------------------------------------ #
#  Tests: print_summary
# ------------------------------------------------------------------ #


class TestPrintSummary:
    """Tests for print_summary output."""

    def test_empty_counters(self):
        _reset_class_state()
        buf = io.StringIO()
        with redirect_stdout(buf):
            ImatrixFallbackStats().print_summary()
        output = buf.getvalue()
        assert "no modules triggered" in output

    def test_non_empty_counters(self):
        _reset_class_state()
        ImatrixFallbackStats._no_importance["layer_a"] = 3
        ImatrixFallbackStats._no_importance["layer_b"] = 1
        ImatrixFallbackStats._all_zero["layer_c"] = 1

        buf = io.StringIO()
        with redirect_stdout(buf):
            ImatrixFallbackStats().print_summary()
        output = buf.getvalue()

        assert "modules_triggered=2" in output  # no_importance category
        assert "modules_triggered=1" in output  # all_zero category
        # Sorted by hit count descending
        assert output.index("layer_a") < output.index("layer_b")


# ------------------------------------------------------------------ #
#  Tests: counter properties
# ------------------------------------------------------------------ #


class TestCounters:
    """Tests for the counter property accessors."""

    def test_properties_return_same_counters(self):
        _reset_class_state()
        stats = ImatrixFallbackStats()
        assert stats.no_importance_counter is ImatrixFallbackStats._no_importance
        assert stats.all_zero_counter is ImatrixFallbackStats._all_zero
        assert stats.other_counter is ImatrixFallbackStats._other


# ------------------------------------------------------------------ #
#  Tests: integration with real observer
# ------------------------------------------------------------------ #


class TestIntegrationWithObserver:
    """End-to-end tests verifying hooks work with real observer instances."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        _reset_class_state()
        yield
        _reset_class_state()

    def test_no_importance_fallback_is_tracked(self):
        """Observer without gatherer falls back and the warning is counted."""
        from llmcompressor.observers.imatrix import IMatrixMSEObserver
        from compressed_tensors.quantization import QuantizationArgs

        stats = ImatrixFallbackStats()
        stats.install_hooks()

        # Register module name so the observer.attach can look it up
        module = nn.Linear(16, 32, bias=False)
        ImatrixFallbackStats._module_name_by_id[id(module)] = "test.linear"

        # Create observer with no importance data → will fall back
        args = QuantizationArgs(
            observer="imatrix_mse",
            strategy="channel",
        )
        observer = IMatrixMSEObserver(base_name="weight", args=args)
        observer.attach(module)

        # Trigger validation with random weights
        observed = module.weight.data.clone()
        observer.update_statistics_from_observed(observed)

        assert ImatrixFallbackStats._no_importance["test.linear"] >= 1

        stats.remove_hooks()

    def test_attach_without_module_name_uses_class_name(self):
        """When no module name is recorded, observer uses class name as fallback."""
        from llmcompressor.observers.imatrix import IMatrixMSEObserver
        from compressed_tensors.quantization import QuantizationArgs

        stats = ImatrixFallbackStats()
        stats.install_hooks()

        # Don't register any module name
        module = nn.Linear(16, 32, bias=False)

        args = QuantizationArgs(
            observer="imatrix_mse",
            strategy="channel",
        )
        observer = IMatrixMSEObserver(base_name="weight", args=args)
        observer.attach(module)
        # After attach, the instance should have the class name fallback
        assert observer._imatrix_script_module_name == "Linear"

        stats.remove_hooks()

    def test_context_var_isolation(self):
        """ContextVar correctly isolates module names between calls."""
        _reset_class_state()
        stats = ImatrixFallbackStats()
        stats.install_hooks()

        mod_a = nn.Linear(16, 32, bias=False)
        mod_b = nn.Linear(32, 64, bias=False)
        ImatrixFallbackStats._module_name_by_id[id(mod_a)] = "layer.a"
        ImatrixFallbackStats._module_name_by_id[id(mod_b)] = "layer.b"

        # Simulate the ContextVar being set for mod_a, then a warning fires
        token_a = ImatrixFallbackStats._current_module.set("layer.a")
        _imatrix_warning_sink(
            _FakeMessage(
                "imatrix_mse: no importance data available."
                " Falling back to uniform MSE."
            )
        )
        ImatrixFallbackStats._current_module.reset(token_a)

        # Now for mod_b
        token_b = ImatrixFallbackStats._current_module.set("layer.b")
        _imatrix_warning_sink(
            _FakeMessage(
                "imatrix_mse: all zeros. Falling back to uniform MSE."
            )
        )
        ImatrixFallbackStats._current_module.reset(token_b)

        assert ImatrixFallbackStats._no_importance["layer.a"] == 1
        assert ImatrixFallbackStats._all_zero["layer.b"] == 1
        # ContextVar should be back to default
        assert ImatrixFallbackStats._current_module.get() is None

        stats.remove_hooks()


# ------------------------------------------------------------------ #
#  Tests: class patches
# ------------------------------------------------------------------ #


class TestClassPatches:
    """Tests for the monkey-patch installation."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        _reset_class_state()
        yield
        _reset_class_state()

    def test_patches_are_applied_once(self):
        from llmcompressor.observers.imatrix import IMatrixMSEObserver
        from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer

        # Store originals before first call
        original_gatherer_init = IMatrixGatherer.on_initialize
        original_attach = IMatrixMSEObserver.attach
        original_validate = IMatrixMSEObserver._get_validated_importance

        stats = ImatrixFallbackStats()
        stats.install_hooks()

        # After first install, methods should be patched
        assert IMatrixGatherer.on_initialize is not original_gatherer_init
        assert IMatrixMSEObserver.attach is not original_attach
        assert IMatrixMSEObserver._get_validated_importance is not original_validate

        patched_gatherer_init = IMatrixGatherer.on_initialize
        patched_attach = IMatrixMSEObserver.attach
        patched_validate = IMatrixMSEObserver._get_validated_importance

        # Second install should not re-patch
        stats.install_hooks()
        assert IMatrixGatherer.on_initialize is patched_gatherer_init
        assert IMatrixMSEObserver.attach is patched_attach
        assert IMatrixMSEObserver._get_validated_importance is patched_validate

        stats.remove_hooks()

    def test_hooks_installed_flag(self):
        _reset_class_state()
        stats = ImatrixFallbackStats()
        assert not ImatrixFallbackStats._hooks_installed
        stats.install_hooks()
        assert ImatrixFallbackStats._hooks_installed
        stats.remove_hooks()
        assert not ImatrixFallbackStats._hooks_installed
