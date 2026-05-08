"""Tests for the Task 5 observability gap-fill: model_confidence /
order_latency_s / per-symbol shakedown_clean_days / per-symbol
daily_pnl_usd_by_symbol metrics + Sentry breadcrumb helper.

Hermetic: every test stubs the metrics pusher and never touches Sentry.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from observability.monitoring import (
    METRIC_NAME_PREFIX,
    MetricsPusher,
    breadcrumb,
    capture_message,
)


# ---------------------------------------------------------------------------
# Stub metrics pusher to capture emissions
# ---------------------------------------------------------------------------


class _RecordingPusher:
    """Behaves like MetricsPusher but records every call instead of pushing."""

    def __init__(self) -> None:
        self.gauge_calls: List[Dict[str, Any]] = []
        self.counter_calls: List[Dict[str, Any]] = []
        self.histogram_calls: List[Dict[str, Any]] = []
        self._enabled = True

    def is_enabled(self) -> bool:
        return self._enabled

    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        self.gauge_calls.append({"name": name, "value": float(value), "labels": labels or {}})

    def counter(
        self, name: str, increment: float = 1.0, labels: Optional[Dict[str, str]] = None
    ) -> None:
        self.counter_calls.append(
            {"name": name, "increment": float(increment), "labels": labels or {}}
        )

    def histogram(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        self.histogram_calls.append(
            {"name": name, "value": float(value), "labels": labels or {}}
        )

    def push(self) -> bool:
        return True


def _names(calls: List[Dict[str, Any]]) -> set[str]:
    return {c["name"] for c in calls}


# ---------------------------------------------------------------------------
# Tick metrics: model_confidence + tick_duration_s + per-symbol gauges
# ---------------------------------------------------------------------------


class TestSupervisorTickMetrics(unittest.TestCase):
    """Run a tick under stub collaborators; assert the new metrics emit."""

    def _supervisor_with_pusher(self, *, predict_fn=None):
        from test_live_supervisor import (
            _build_supervisor,
            StubExchange,
            StubPositionStore,
        )

        store = StubPositionStore()
        exch = StubExchange()
        sup, refs = _build_supervisor(
            position_store=store,
            exchange=exch,
            predict_fn=predict_fn or (lambda s, t: ("buy", 0.85)),
        )
        pusher = _RecordingPusher()
        sup.metrics_pusher = pusher
        return sup, pusher, refs

    def test_tick_emits_model_confidence_histogram_when_finite(self) -> None:
        sup, pusher, _refs = self._supervisor_with_pusher(
            predict_fn=lambda s, t: ("buy", 0.73),
        )
        sup.run_once()
        names = _names(pusher.histogram_calls)
        self.assertIn("model_confidence", names)
        # Find the matching record and confirm value.
        matching = [
            c for c in pusher.histogram_calls if c["name"] == "model_confidence"
        ]
        self.assertEqual(len(matching), 1)
        self.assertAlmostEqual(matching[0]["value"], 0.73, places=4)
        self.assertEqual(matching[0]["labels"], {"symbol": "ETH/USDT"})

    def test_tick_emits_tick_duration_s_histogram(self) -> None:
        sup, pusher, _refs = self._supervisor_with_pusher()
        sup.run_once()
        names = _names(pusher.histogram_calls)
        self.assertIn("tick_duration_s", names)

    def test_tick_skipped_low_confidence_does_not_emit_model_confidence(
        self,
    ) -> None:
        # When confidence is NaN -> action_taken=skipped_low_confidence,
        # model_confidence is None, so the histogram MUST NOT be observed
        # (otherwise we'd be observing None which would crash the pusher).
        sup, pusher, _refs = self._supervisor_with_pusher(
            predict_fn=lambda s, t: ("buy", float("nan")),
        )
        sup.run_once()
        names = _names(pusher.histogram_calls)
        self.assertNotIn("model_confidence", names)


# ---------------------------------------------------------------------------
# Per-symbol shakedown_clean_days + daily_pnl_usd_by_symbol gauges
# ---------------------------------------------------------------------------


class TestSupervisorPerSymbolGauges(unittest.TestCase):
    def _supervisor_with_two_symbols(self):
        from test_live_supervisor import (
            _build_supervisor,
            StubPositionStore,
        )

        store = StubPositionStore()
        sup, refs = _build_supervisor(
            symbols=["ETH/USDT", "BTC/USDT"],
            position_store=store,
        )
        pusher = _RecordingPusher()
        sup.metrics_pusher = pusher
        return sup, pusher, refs

    def test_loop_metrics_emit_per_symbol_shakedown_clean_days(self) -> None:
        sup, pusher, _refs = self._supervisor_with_two_symbols()
        # Seed clean-day counts.
        sup.shakedown_state.get_or_init("ETH/USDT").paper_days_clean = 5
        sup.shakedown_state.get_or_init("BTC/USDT").paper_days_clean = 12
        sup.run_once()
        # Per-symbol gauge should fire once per symbol.
        per_symbol = [
            c for c in pusher.gauge_calls if c["name"] == "shakedown_clean_days"
        ]
        symbols_seen = {c["labels"].get("symbol") for c in per_symbol}
        self.assertEqual(symbols_seen, {"ETH/USDT", "BTC/USDT"})
        # The actual values should match.
        eth = next(c for c in per_symbol if c["labels"]["symbol"] == "ETH/USDT")
        btc = next(c for c in per_symbol if c["labels"]["symbol"] == "BTC/USDT")
        self.assertEqual(eth["value"], 5.0)
        self.assertEqual(btc["value"], 12.0)

    def test_loop_metrics_emit_per_symbol_daily_pnl(self) -> None:
        sup, pusher, _refs = self._supervisor_with_two_symbols()
        sup.run_once()
        per_symbol_pnl = [
            c for c in pusher.gauge_calls if c["name"] == "daily_pnl_usd_by_symbol"
        ]
        symbols_seen = {c["labels"].get("symbol") for c in per_symbol_pnl}
        self.assertEqual(symbols_seen, {"ETH/USDT", "BTC/USDT"})


# ---------------------------------------------------------------------------
# Order placement latency
# ---------------------------------------------------------------------------


class TestSupervisorOrderLatencyMetric(unittest.TestCase):
    def test_live_order_emits_order_latency_s_histogram(self) -> None:
        from test_live_supervisor import (
            _build_supervisor,
            StubExchange,
            StubPositionStore,
        )

        store = StubPositionStore()
        exch = StubExchange(ticker_mid=2_000.0, order_status="filled")
        sup, refs = _build_supervisor(
            mode="live",
            paper_days_clean=14,
            shakedown_min_days=14,
            position_store=store,
            exchange=exch,
        )
        pusher = _RecordingPusher()
        sup.metrics_pusher = pusher
        sup.run_once()
        names = _names(pusher.histogram_calls)
        self.assertIn("order_latency_s", names)
        matching = [
            c for c in pusher.histogram_calls if c["name"] == "order_latency_s"
        ]
        # At least one observation, with the symbol label.
        self.assertGreaterEqual(len(matching), 1)
        self.assertEqual(matching[0]["labels"], {"symbol": "ETH/USDT"})

    def test_order_latency_recorded_even_when_exchange_raises(self) -> None:
        from exchanges.coinbase import ExchangeError
        from test_live_supervisor import (
            _build_supervisor,
            StubExchange,
            StubPositionStore,
        )

        store = StubPositionStore()
        exch = StubExchange(
            ticker_mid=2_000.0,
            raise_on_order=ExchangeError("boom"),
        )
        sup, refs = _build_supervisor(
            mode="live",
            paper_days_clean=14,
            shakedown_min_days=14,
            position_store=store,
            exchange=exch,
        )
        pusher = _RecordingPusher()
        sup.metrics_pusher = pusher
        sup.run_once()
        names = _names(pusher.histogram_calls)
        self.assertIn("order_latency_s", names)


# ---------------------------------------------------------------------------
# Sentry helpers — must never raise even when SDK isn't configured
# ---------------------------------------------------------------------------


class TestSentryHelpersTolerateMissingSDK(unittest.TestCase):
    def test_breadcrumb_returns_silently_without_sentry(self) -> None:
        # No SENTRY_DSN configured in the test env -> the helper should
        # just no-op and not raise.
        breadcrumb(category="x", message="y", data={"k": "v"})

    def test_capture_message_returns_silently_without_sentry(self) -> None:
        capture_message("regression smoke test", level="info")


# ---------------------------------------------------------------------------
# MetricsPusher graceful no-op when prometheus_client isn't configured
# ---------------------------------------------------------------------------


class TestMetricsPusherGracefulDegradation(unittest.TestCase):
    def test_pusher_without_url_is_disabled(self) -> None:
        # No PROMETHEUS_PUSH_URL env, no constructor arg => is_enabled()
        # may still be True if prometheus_client imports OK + push_url
        # falls through to the env var; we assert calling gauge / counter
        # / histogram never raises regardless.
        pusher = MetricsPusher(push_url=None)
        # Force-clear the URL so the disabled path is exercised.
        pusher.push_url = None
        pusher.gauge("test_gauge", 1.0)
        pusher.counter("test_counter", 1.0)
        pusher.histogram("test_histogram", 0.1)
        self.assertFalse(pusher.push())


class TestClusterTierCounters(unittest.TestCase):
    """A4 / A5 cluster-tier extreme-promotion Prometheus counters."""

    def setUp(self) -> None:
        from observability import monitoring

        self._monitoring = monitoring
        self._saved_pusher = monitoring._get_default_metrics_pusher()
        self.pusher = _RecordingPusher()
        monitoring.set_default_metrics_pusher(self.pusher)

    def tearDown(self) -> None:
        # Restore whatever was registered before the test so the module
        # state stays clean for subsequent tests.
        self._monitoring.set_default_metrics_pusher(self._saved_pusher)

    def test_a4_news_cluster_extreme_increments_with_bucket_label(self) -> None:
        from observability.monitoring import incr_a4_news_cluster_extreme

        incr_a4_news_cluster_extreme(10)
        self.assertEqual(len(self.pusher.counter_calls), 1)
        call = self.pusher.counter_calls[0]
        self.assertEqual(call["name"], "a4_news_cluster_extreme_total")
        self.assertAlmostEqual(call["increment"], 1.0)
        self.assertEqual(call["labels"], {"headlines_bucket": "10-19"})

    def test_a4_news_cluster_buckets_handle_each_band(self) -> None:
        from observability.monitoring import incr_a4_news_cluster_extreme

        incr_a4_news_cluster_extreme(15)  # → "10-19"
        incr_a4_news_cluster_extreme(25)  # → "20-49"
        incr_a4_news_cluster_extreme(80)  # → "50+"
        labels = [c["labels"]["headlines_bucket"] for c in self.pusher.counter_calls]
        self.assertEqual(labels, ["10-19", "20-49", "50+"])

    def test_a5_race_cluster_extreme_increments_with_bucket_label(self) -> None:
        from observability.monitoring import incr_a5_race_cluster_extreme

        incr_a5_race_cluster_extreme(50)
        self.assertEqual(len(self.pusher.counter_calls), 1)
        call = self.pusher.counter_calls[0]
        self.assertEqual(call["name"], "a5_race_cluster_extreme_total")
        self.assertAlmostEqual(call["increment"], 1.0)
        self.assertEqual(call["labels"], {"error_bucket": "30-99"})

    def test_a5_race_cluster_buckets_handle_each_band(self) -> None:
        from observability.monitoring import incr_a5_race_cluster_extreme

        incr_a5_race_cluster_extreme(20)   # → "15-29"
        incr_a5_race_cluster_extreme(50)   # → "30-99"
        incr_a5_race_cluster_extreme(150)  # → "100+"
        labels = [c["labels"]["error_bucket"] for c in self.pusher.counter_calls]
        self.assertEqual(labels, ["15-29", "30-99", "100+"])

    def test_helpers_are_no_op_when_pusher_unset(self) -> None:
        """Without a registered pusher the helpers must silently no-op."""
        from observability.monitoring import (
            incr_a4_news_cluster_extreme,
            incr_a5_race_cluster_extreme,
            set_default_metrics_pusher,
        )

        # Clear the default pusher this test set up.
        set_default_metrics_pusher(None)
        # Both helpers must not raise when the default is None.
        incr_a4_news_cluster_extreme(10)
        incr_a5_race_cluster_extreme(15)

    def test_helpers_swallow_pusher_exceptions(self) -> None:
        """A raising pusher must not propagate — observability is best-effort."""
        from observability.monitoring import (
            incr_a4_news_cluster_extreme,
            incr_a5_race_cluster_extreme,
            set_default_metrics_pusher,
        )

        class _ExplodingPusher:
            def is_enabled(self) -> bool:
                return True

            def counter(self, *_args, **_kwargs):  # noqa: ANN001
                raise RuntimeError("simulated metrics outage")

            def gauge(self, *_args, **_kwargs):  # noqa: ANN001
                pass

            def histogram(self, *_args, **_kwargs):  # noqa: ANN001
                pass

            def push(self) -> bool:
                return True

        set_default_metrics_pusher(_ExplodingPusher())
        try:
            # Both must complete without raising.
            incr_a4_news_cluster_extreme(10)
            incr_a5_race_cluster_extreme(15)
        finally:
            set_default_metrics_pusher(None)


if __name__ == "__main__":
    unittest.main()
