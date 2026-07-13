"""Tests for ``scripts/diagnose_calibration_drift.py``.

Mirrors the ``test_cleanup_zombies.py`` pattern: fakeredis + a real
PositionStore + a real TradeContextStore, with the script loaded via
``importlib`` because ``scripts/`` isn't on sys.path under the
prediction-market-scanner runner.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

import fakeredis
import redis.exceptions

from state.position_store import Position, PositionStore
from state.trade_context_store import TradeContextSnapshot, TradeContextStore


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "diagnose_calibration_drift.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "scripts_diagnose_calibration_drift_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["scripts_diagnose_calibration_drift_under_test"] = module
    spec.loader.exec_module(module)
    return module


drift = _load_script_module()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
NOW = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
NAMESPACE = "test-drift"


def _fresh_stores():
    """Build a fresh PositionStore + TradeContextStore on shared fakeredis."""
    client = fakeredis.FakeRedis(decode_responses=True)
    pstore = PositionStore(redis_client=client, namespace=NAMESPACE)
    cstore = TradeContextStore(redis_client=client, namespace=NAMESPACE)
    return pstore, cstore


def _seed_closed_position(
    pstore: PositionStore,
    cstore: Optional[TradeContextStore],
    *,
    position_id: str,
    confidence: Optional[float],
    realized_pnl_usd: float,
    opened_at: datetime,
    symbol: str = "ETH/USD",
    side: str = "long",
    closed_at: Optional[datetime] = None,
    confidence_via: str = "snapshot",
) -> Position:
    """Seed a closed position + (optionally) a signal-snapshot with conf.

    ``confidence_via`` controls how the script will discover the entry
    confidence:
        * ``"snapshot"`` — write a TradeContextSnapshot with the conf.
        * ``"model_meta"`` — write ``position.model_meta["entry_confidence"]``.
        * ``"none"`` — neither (script should treat this as unconfidenced).
    """
    closed_at = closed_at or (opened_at + timedelta(minutes=5))
    model_meta: dict = {}
    if confidence is not None and confidence_via == "model_meta":
        model_meta["entry_confidence"] = float(confidence)

    position = Position(
        position_id=position_id,
        exchange="coinbase-paper",
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        status="closed",
        entry_price=2000.0,
        entry_quote_usd=50.0,
        base_size=0.025,
        exit_price=2010.0 if realized_pnl_usd > 0 else 1990.0,
        exit_quote_usd=50.0 + realized_pnl_usd,
        realized_pnl_usd=float(realized_pnl_usd),
        fees_usd=0.0,
        entry_order_id=f"paper-{position_id[:8]}",
        opened_at_utc=opened_at.isoformat(),
        closed_at_utc=closed_at.isoformat(),
        model_meta=model_meta,
        notes="seed",
    )
    # Persist as closed: write the blob + put it on the closed-set for the
    # closed date. The open_set is left untouched (closed positions don't
    # live there).
    blob = position.model_dump_json()
    pstore._redis.set(  # noqa: SLF001 - test-only seed
        f"{pstore.namespace}:positions:{position.position_id}", blob
    )
    pstore._redis.sadd(  # noqa: SLF001 - test-only seed
        pstore._closed_set_key(closed_at), position.position_id  # noqa: SLF001
    )

    if confidence is not None and confidence_via == "snapshot" and cstore is not None:
        snap = TradeContextSnapshot(
            trade_id=position.position_id,
            symbol=symbol,
            captured_at_utc=opened_at.isoformat(),
            phase="signal",
            feature_buffer={},
            feature_window=None,
            model_probs={},
            model_confidence=float(confidence),
        )
        cstore.record_snapshot(snap)
    return position


def _make_args(**overrides) -> "drift.argparse.Namespace":  # type: ignore[name-defined]
    parser = drift.build_parser()
    defaults = parser.parse_args([])
    for k, v in overrides.items():
        setattr(defaults, k, v)
    return defaults


# ---------------------------------------------------------------------------
# Bucketing primitives
# ---------------------------------------------------------------------------
class BucketAssignmentTests(unittest.TestCase):
    def test_left_inclusive_right_exclusive_default(self) -> None:
        edges = drift.parse_bins(drift.DEFAULT_BINS_CSV)
        # bucket [0.50, 0.55) — confidence 0.50 lands here, 0.55 in next.
        self.assertEqual(drift.assign_bucket(0.50, edges), 0)
        self.assertEqual(drift.assign_bucket(0.5499, edges), 0)
        self.assertEqual(drift.assign_bucket(0.55, edges), 1)
        self.assertEqual(drift.assign_bucket(0.60, edges), 2)

    def test_final_edge_is_inclusive(self) -> None:
        edges = drift.parse_bins(drift.DEFAULT_BINS_CSV)
        # The final bucket [0.95, 1.00] must include 1.00 itself.
        last = len(edges) - 2
        self.assertEqual(drift.assign_bucket(1.00, edges), last)
        self.assertEqual(drift.assign_bucket(0.9999, edges), last)

    def test_out_of_range_returns_none(self) -> None:
        edges = drift.parse_bins(drift.DEFAULT_BINS_CSV)
        self.assertIsNone(drift.assign_bucket(0.49, edges))
        self.assertIsNone(drift.assign_bucket(1.0001, edges))
        self.assertIsNone(drift.assign_bucket(float("nan"), edges))

    def test_parse_bins_rejects_unordered(self) -> None:
        with self.assertRaises(ValueError):
            drift.parse_bins("0.5,0.6,0.55")

    def test_parse_bins_rejects_out_of_unit_interval(self) -> None:
        with self.assertRaises(ValueError):
            drift.parse_bins("-0.1,0.5,1.0")
        with self.assertRaises(ValueError):
            drift.parse_bins("0.0,0.5,1.5")


class WlsFitTests(unittest.TestCase):
    def test_perfect_calibration_slope_one(self) -> None:
        # rows: midpoint == realized_wr -> slope must be 1.0 exactly.
        rows = [
            {"bucket": "a", "lo": 0.5, "hi": 0.6, "midpoint": 0.55,
             "n": 20, "wins": 11, "realized_wr": 0.55, "residual": 0.0},
            {"bucket": "b", "lo": 0.6, "hi": 0.7, "midpoint": 0.65,
             "n": 20, "wins": 13, "realized_wr": 0.65, "residual": 0.0},
            {"bucket": "c", "lo": 0.7, "hi": 0.8, "midpoint": 0.75,
             "n": 20, "wins": 15, "realized_wr": 0.75, "residual": 0.0},
        ]
        slope, intercept, used = drift.wls_slope_intercept(rows, min_n=5)
        self.assertIsNotNone(slope)
        self.assertIsNotNone(intercept)
        self.assertAlmostEqual(slope, 1.0, places=6)
        self.assertAlmostEqual(intercept, 0.0, places=6)
        self.assertEqual(used, 3)

    def test_buckets_below_min_n_excluded(self) -> None:
        # Tiny first bucket should be excluded; the other two carry the fit.
        rows = [
            {"bucket": "a", "lo": 0.5, "hi": 0.6, "midpoint": 0.55,
             "n": 2, "wins": 0, "realized_wr": 0.0, "residual": -0.55},
            {"bucket": "b", "lo": 0.6, "hi": 0.7, "midpoint": 0.65,
             "n": 20, "wins": 13, "realized_wr": 0.65, "residual": 0.0},
            {"bucket": "c", "lo": 0.7, "hi": 0.8, "midpoint": 0.75,
             "n": 20, "wins": 15, "realized_wr": 0.75, "residual": 0.0},
        ]
        slope, intercept, used = drift.wls_slope_intercept(rows, min_n=5)
        self.assertEqual(used, 2)
        self.assertIsNotNone(slope)
        self.assertAlmostEqual(slope, 1.0, places=6)

    def test_single_admitted_bucket_returns_none(self) -> None:
        rows = [
            {"bucket": "a", "lo": 0.5, "hi": 0.6, "midpoint": 0.55,
             "n": 20, "wins": 11, "realized_wr": 0.55, "residual": 0.0},
            {"bucket": "b", "lo": 0.6, "hi": 0.7, "midpoint": 0.65,
             "n": 2, "wins": 1, "realized_wr": 0.5, "residual": -0.15},
        ]
        slope, intercept, used = drift.wls_slope_intercept(rows, min_n=5)
        self.assertEqual(used, 1)
        self.assertIsNone(slope)
        self.assertIsNone(intercept)


class VerdictTests(unittest.TestCase):
    def test_in_tolerance_is_ok(self) -> None:
        self.assertEqual(drift.classify_verdict(0.95, tolerance=0.10), drift.VERDICT_OK)
        self.assertEqual(drift.classify_verdict(1.05, tolerance=0.10), drift.VERDICT_OK)
        # Boundary case — exactly at the tolerance still counts as in.
        self.assertEqual(drift.classify_verdict(0.90, tolerance=0.10), drift.VERDICT_OK)
        self.assertEqual(drift.classify_verdict(1.10, tolerance=0.10), drift.VERDICT_OK)

    def test_out_of_tolerance_alerts(self) -> None:
        self.assertEqual(drift.classify_verdict(0.84, tolerance=0.10), drift.VERDICT_ALERT)
        self.assertEqual(drift.classify_verdict(1.15, tolerance=0.10), drift.VERDICT_ALERT)

    def test_none_slope_is_no_data(self) -> None:
        self.assertEqual(drift.classify_verdict(None, tolerance=0.10), drift.VERDICT_NO_DATA)


# ---------------------------------------------------------------------------
# End-to-end via run()
# ---------------------------------------------------------------------------
class RunEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pstore, self.cstore = _fresh_stores()

    def _well_calibrated(self) -> None:
        """Trades whose per-bucket realized_wr == midpoint -> slope == 1.0.

        Bucket sizes are chosen so wins/n equals the midpoint exactly:
            midpoint 0.525 -> 21/40 = 0.525
            midpoint 0.625 -> 25/40 = 0.625
            midpoint 0.725 -> 29/40 = 0.725
        """
        bucket_specs = [
            (0.525, 40, 21),
            (0.625, 40, 25),
            (0.725, 40, 29),
        ]
        idx = 0
        opened_base = NOW - timedelta(hours=2)
        for mid, n, wins in bucket_specs:
            for i in range(n):
                _seed_closed_position(
                    self.pstore,
                    self.cstore,
                    position_id=f"pos-{idx:04d}",
                    confidence=mid,
                    realized_pnl_usd=0.10 if i < wins else -0.05,
                    opened_at=opened_base + timedelta(seconds=idx),
                )
                idx += 1

    def test_well_calibrated_verdict_ok(self) -> None:
        self._well_calibrated()
        args = _make_args(date="2026-05-18", min_n_per_bucket=5)
        exit_code, blob, report = drift.run(
            args=args,
            store=self.pstore,
            trade_ctx_store=self.cstore,
            now_utc=NOW,
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(blob["verdict"], drift.VERDICT_OK)
        self.assertIsNotNone(blob["reliability_slope"])
        self.assertAlmostEqual(blob["reliability_slope"], 1.0, delta=0.10)
        self.assertIn("OK", report)
        self.assertIn("closed trades: 120", report)

    def test_over_confident_verdict_alert(self) -> None:
        # High-confidence buckets lose more than midpoint -> slope < 0.9.
        # bucket midpoint 0.525 wr 0.40
        # bucket midpoint 0.625 wr 0.30
        # bucket midpoint 0.725 wr 0.20
        bucket_specs = [(0.525, 10, 4), (0.625, 10, 3), (0.725, 10, 2)]
        idx = 0
        opened_base = NOW - timedelta(hours=2)
        for mid, n, wins in bucket_specs:
            for i in range(n):
                _seed_closed_position(
                    self.pstore,
                    self.cstore,
                    position_id=f"oc-{idx:04d}",
                    confidence=mid,
                    realized_pnl_usd=0.10 if i < wins else -0.05,
                    opened_at=opened_base + timedelta(seconds=idx),
                )
                idx += 1
        args = _make_args(date="2026-05-18", min_n_per_bucket=5, alert=True)
        notifier = MagicMock()
        notifier.alert.return_value = True
        exit_code, blob, report = drift.run(
            args=args,
            store=self.pstore,
            trade_ctx_store=self.cstore,
            notifier=notifier,
            now_utc=NOW,
        )
        self.assertEqual(blob["verdict"], drift.VERDICT_ALERT)
        self.assertLess(blob["reliability_slope"], 0.9)
        self.assertEqual(exit_code, 1)
        # --alert was set + verdict ALERT -> notifier.alert was called once.
        notifier.alert.assert_called_once()
        kwargs = notifier.alert.call_args.kwargs
        self.assertEqual(kwargs["severity"], "alert")

    def test_under_confident_verdict_alert(self) -> None:
        # Slope > 1.1: realized winrate runs ABOVE midpoint, and the gap
        # widens with confidence so the regression slope tilts steep.
        # Pairs: x=0.525 y=0.40, x=0.625 y=0.70, x=0.725 y=0.95.
        # rise=0.55 over run=0.20 -> slope ~ 2.75.
        bucket_specs = [(0.525, 10, 4), (0.625, 10, 7), (0.725, 20, 19)]
        idx = 0
        opened_base = NOW - timedelta(hours=2)
        for mid, n, wins in bucket_specs:
            for i in range(n):
                _seed_closed_position(
                    self.pstore,
                    self.cstore,
                    position_id=f"uc-{idx:04d}",
                    confidence=mid,
                    realized_pnl_usd=0.10 if i < wins else -0.05,
                    opened_at=opened_base + timedelta(seconds=idx),
                )
                idx += 1
        args = _make_args(date="2026-05-18", min_n_per_bucket=5)
        exit_code, blob, _report = drift.run(
            args=args,
            store=self.pstore,
            trade_ctx_store=self.cstore,
            now_utc=NOW,
        )
        self.assertEqual(blob["verdict"], drift.VERDICT_ALERT)
        self.assertGreater(blob["reliability_slope"], 1.1)
        # --alert was NOT set, so even on ALERT we exit 0 (stdout-only mode).
        self.assertEqual(exit_code, 0)

    def test_insufficient_data_buckets_excluded_from_fit(self) -> None:
        # Seed: 8 trades at midpoint 0.625 (admitted), 2 trades at
        # midpoint 0.525 (BELOW min_n). The 0.525 bucket must be flagged
        # ``(insufficient)`` in the table and excluded from the WLS fit.
        opened_base = NOW - timedelta(hours=1)
        for i in range(2):
            _seed_closed_position(
                self.pstore,
                self.cstore,
                position_id=f"small-{i:04d}",
                confidence=0.525,
                realized_pnl_usd=-0.05,
                opened_at=opened_base + timedelta(seconds=i),
            )
        for i in range(8):
            _seed_closed_position(
                self.pstore,
                self.cstore,
                position_id=f"big-{i:04d}",
                confidence=0.625,
                realized_pnl_usd=0.10 if i < 5 else -0.05,
                opened_at=opened_base + timedelta(seconds=10 + i),
            )
        args = _make_args(date="2026-05-18", min_n_per_bucket=5)
        _exit_code, blob, report = drift.run(
            args=args,
            store=self.pstore,
            trade_ctx_store=self.cstore,
            now_utc=NOW,
        )
        # The 0.525 bucket has n=2, marked as not included in fit.
        bucket_525 = next(b for b in blob["buckets"] if b["bucket"] == "0.50-0.55")
        self.assertEqual(bucket_525["n"], 2)
        self.assertFalse(bucket_525["included_in_fit"])
        bucket_625 = next(b for b in blob["buckets"] if b["bucket"] == "0.60-0.65")
        self.assertEqual(bucket_625["n"], 8)
        self.assertTrue(bucket_625["included_in_fit"])
        self.assertIn("(insufficient)", report)
        # Only one bucket admitted to fit -> no slope -> NO_DATA verdict.
        self.assertEqual(blob["verdict"], drift.VERDICT_NO_DATA)

    def test_empty_window_no_data(self) -> None:
        args = _make_args(date="2026-05-18")
        exit_code, blob, report = drift.run(
            args=args,
            store=self.pstore,
            trade_ctx_store=self.cstore,
            now_utc=NOW,
        )
        self.assertEqual(blob["verdict"], drift.VERDICT_NO_DATA)
        self.assertEqual(blob["n_closed_trades"], 0)
        self.assertEqual(exit_code, 0)
        self.assertIn("NO_DATA", report)

    def test_alert_flag_not_fired_when_verdict_ok(self) -> None:
        self._well_calibrated()
        notifier = MagicMock()
        args = _make_args(date="2026-05-18", alert=True, min_n_per_bucket=5)
        exit_code, blob, _report = drift.run(
            args=args,
            store=self.pstore,
            trade_ctx_store=self.cstore,
            notifier=notifier,
            now_utc=NOW,
        )
        self.assertEqual(blob["verdict"], drift.VERDICT_OK)
        self.assertEqual(exit_code, 0)
        # No notifier post on OK verdicts.
        notifier.alert.assert_not_called()

    def test_unconfidenced_positions_are_counted_and_skipped(self) -> None:
        # Half the closed trades have a confidence, half don't.
        opened_base = NOW - timedelta(hours=1)
        for i in range(10):
            _seed_closed_position(
                self.pstore,
                self.cstore,
                position_id=f"good-{i:04d}",
                confidence=0.625,
                realized_pnl_usd=0.10 if i < 6 else -0.05,
                opened_at=opened_base + timedelta(seconds=i),
            )
        for i in range(7):
            _seed_closed_position(
                self.pstore,
                self.cstore,
                position_id=f"miss-{i:04d}",
                confidence=None,
                realized_pnl_usd=-0.05,
                opened_at=opened_base + timedelta(seconds=20 + i),
                confidence_via="none",
            )
        args = _make_args(date="2026-05-18")
        _exit_code, blob, _report = drift.run(
            args=args,
            store=self.pstore,
            trade_ctx_store=self.cstore,
            now_utc=NOW,
        )
        self.assertEqual(blob["n_closed_trades"], 17)
        self.assertEqual(blob["n_confidenced"], 10)
        self.assertEqual(blob["n_unconfidenced"], 7)

    def test_model_meta_confidence_path(self) -> None:
        # confidence_via="model_meta" stores the conf directly on the
        # Position blob; no signal snapshot is written. The script should
        # still resolve it.
        opened_base = NOW - timedelta(hours=1)
        for i in range(8):
            _seed_closed_position(
                self.pstore,
                self.cstore,
                position_id=f"mm-{i:04d}",
                confidence=0.725,
                realized_pnl_usd=0.10 if i < 6 else -0.05,
                opened_at=opened_base + timedelta(seconds=i),
                confidence_via="model_meta",
            )
        args = _make_args(date="2026-05-18", min_n_per_bucket=5)
        _exit_code, blob, _report = drift.run(
            args=args,
            store=self.pstore,
            trade_ctx_store=self.cstore,
            now_utc=NOW,
        )
        self.assertEqual(blob["n_confidenced"], 8)
        bucket = next(b for b in blob["buckets"] if b["bucket"] == "0.70-0.75")
        self.assertEqual(bucket["n"], 8)
        self.assertAlmostEqual(bucket["realized_wr"], 6 / 8, places=6)

    def test_symbol_filter(self) -> None:
        opened_base = NOW - timedelta(hours=1)
        for i in range(8):
            _seed_closed_position(
                self.pstore,
                self.cstore,
                position_id=f"eth-{i:04d}",
                confidence=0.625,
                realized_pnl_usd=0.10 if i < 5 else -0.05,
                opened_at=opened_base + timedelta(seconds=i),
                symbol="ETH/USD",
            )
        for i in range(8):
            _seed_closed_position(
                self.pstore,
                self.cstore,
                position_id=f"btc-{i:04d}",
                confidence=0.625,
                realized_pnl_usd=-0.05,
                opened_at=opened_base + timedelta(seconds=20 + i),
                symbol="BTC/USD",
            )
        args = _make_args(date="2026-05-18", symbol="ETH/USD")
        _exit_code, blob, _report = drift.run(
            args=args,
            store=self.pstore,
            trade_ctx_store=self.cstore,
            now_utc=NOW,
        )
        self.assertEqual(blob["symbol"], "ETH/USD")
        self.assertEqual(blob["n_closed_trades"], 8)

    def test_date_wins_over_window_days(self) -> None:
        # Position opened 3 days before NOW. --date=2026-05-15 should pick
        # it up; default window-days=7 would also include it, but the
        # presence of --date should NARROW (not widen) the scope.
        old = NOW - timedelta(days=3)
        _seed_closed_position(
            self.pstore,
            self.cstore,
            position_id="day3-001",
            confidence=0.625,
            realized_pnl_usd=0.10,
            opened_at=old,
        )
        # And one opened today.
        _seed_closed_position(
            self.pstore,
            self.cstore,
            position_id="today-001",
            confidence=0.625,
            realized_pnl_usd=0.10,
            opened_at=NOW - timedelta(hours=1),
        )
        args = _make_args(date="2026-05-15", window_days=7)
        _exit_code, blob, _report = drift.run(
            args=args,
            store=self.pstore,
            trade_ctx_store=self.cstore,
            now_utc=NOW,
        )
        self.assertEqual(blob["n_closed_trades"], 1)
        self.assertEqual(blob["label"], "2026-05-15")

    def test_window_days_default_includes_recent(self) -> None:
        # No --date -> rolling window. Position opened 2 days before NOW
        # should be inside the default 7-day window.
        opened = NOW - timedelta(days=2)
        _seed_closed_position(
            self.pstore,
            self.cstore,
            position_id="window-001",
            confidence=0.625,
            realized_pnl_usd=0.10,
            opened_at=opened,
        )
        args = _make_args()
        _exit_code, blob, report = drift.run(
            args=args,
            store=self.pstore,
            trade_ctx_store=self.cstore,
            now_utc=NOW,
        )
        self.assertEqual(blob["n_closed_trades"], 1)
        self.assertIn("last 7 days", report)

    def test_json_out_writes_blob(self) -> None:
        self._well_calibrated()
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "drift.json"
            # Run via the top-level main() so the --out write path is
            # exercised end-to-end (not just run()).
            argv = [
                "--date",
                "2026-05-18",
                "--min-n-per-bucket",
                "5",
                "--namespace",
                NAMESPACE,
                "--out",
                str(out_path),
            ]
            # Patch the script's store constructors to return our fake-redis
            # backed stores. The script uses module-level imports from
            # state.position_store + state.trade_context_store; we monkey-
            # patch on the loaded module namespace.
            original_pos = drift.PositionStore
            original_ctx = drift.TradeContextStore
            pstore_ref = self.pstore
            cstore_ref = self.cstore
            try:
                drift.PositionStore = lambda **_kw: pstore_ref  # type: ignore[assignment]
                drift.TradeContextStore = lambda **_kw: cstore_ref  # type: ignore[assignment]
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = drift.main(argv)
            finally:
                drift.PositionStore = original_pos  # type: ignore[assignment]
                drift.TradeContextStore = original_ctx  # type: ignore[assignment]
            self.assertEqual(rc, 0)
            self.assertTrue(out_path.exists())
            blob = json.loads(out_path.read_text())
            self.assertIn("verdict", blob)
            self.assertIn("buckets", blob)
            self.assertEqual(blob["verdict"], drift.VERDICT_OK)


# ---------------------------------------------------------------------------
# Redis-down behaviour
# ---------------------------------------------------------------------------
class _ExplodingRedis:
    """Stub redis client that raises ConnectionError on every read."""

    def __init__(self, exc_cls=redis.exceptions.ConnectionError) -> None:
        self._exc = exc_cls

    def smembers(self, *_a, **_k):
        raise self._exc("simulated outage")

    def scan_iter(self, *_a, **_k):
        raise self._exc("simulated outage")

    def get(self, *_a, **_k):
        raise self._exc("simulated outage")

    def set(self, *_a, **_k):
        raise self._exc("simulated outage")

    def sadd(self, *_a, **_k):
        raise self._exc("simulated outage")

    def pipeline(self):
        raise self._exc("simulated outage")


class RedisDownTests(unittest.TestCase):
    def test_main_handles_init_connection_error(self) -> None:
        # main() catches ConnectionError around the PositionStore
        # constructor itself. We force the failure by swapping in a class
        # whose ctor raises.
        def _boom(**_kw):
            raise redis.exceptions.ConnectionError("simulated outage")

        original_pos = drift.PositionStore
        original_ctx = drift.TradeContextStore
        try:
            drift.PositionStore = _boom  # type: ignore[assignment]
            drift.TradeContextStore = _boom  # type: ignore[assignment]
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = drift.main(["--date", "2026-05-18"])
        finally:
            drift.PositionStore = original_pos  # type: ignore[assignment]
            drift.TradeContextStore = original_ctx  # type: ignore[assignment]
        self.assertEqual(rc, 0)
        self.assertIn("Redis unreachable", buf.getvalue())

    def test_main_treats_midscan_connection_error_as_skip(self) -> None:
        # The constructor succeeds (we hand it a real PositionStore on
        # an exploding fake client) but the scan inside run() trips the
        # error. Should still exit 0 + emit the skip message.
        bad_client = _ExplodingRedis()
        pstore = PositionStore.__new__(PositionStore)
        # Hand-roll a minimally-initialised PositionStore so list_open
        # path goes through our exploding client.
        import threading

        pstore._redis = bad_client  # type: ignore[attr-defined]
        pstore.namespace = NAMESPACE  # type: ignore[attr-defined]
        pstore._lock = threading.Lock()  # type: ignore[attr-defined]
        pstore._postmortem_queue = None  # type: ignore[attr-defined]
        pstore._bankroll_provider = None  # type: ignore[attr-defined]
        cstore = TradeContextStore.__new__(TradeContextStore)
        cstore._redis = bad_client  # type: ignore[attr-defined]
        cstore.namespace = NAMESPACE  # type: ignore[attr-defined]
        cstore.ttl_seconds = 1  # type: ignore[attr-defined]
        cstore._lock = threading.Lock()  # type: ignore[attr-defined]

        original_pos = drift.PositionStore
        original_ctx = drift.TradeContextStore
        try:
            drift.PositionStore = lambda **_kw: pstore  # type: ignore[assignment]
            drift.TradeContextStore = lambda **_kw: cstore  # type: ignore[assignment]
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = drift.main(["--date", "2026-05-18"])
        finally:
            drift.PositionStore = original_pos  # type: ignore[assignment]
            drift.TradeContextStore = original_ctx  # type: ignore[assignment]
        self.assertEqual(rc, 0)
        self.assertIn("Redis unreachable", buf.getvalue())

    def test_other_redis_error_exits_two(self) -> None:
        # Non-connection RedisError (e.g. ResponseError) should NOT be
        # treated as a routine outage. Exit code 2 + traceback in logs.
        def _boom(**_kw):
            raise redis.exceptions.ResponseError("WRONGTYPE")

        original_pos = drift.PositionStore
        original_ctx = drift.TradeContextStore
        try:
            drift.PositionStore = _boom  # type: ignore[assignment]
            drift.TradeContextStore = _boom  # type: ignore[assignment]
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = drift.main(["--date", "2026-05-18"])
        finally:
            drift.PositionStore = original_pos  # type: ignore[assignment]
            drift.TradeContextStore = original_ctx  # type: ignore[assignment]
        self.assertEqual(rc, 2)


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
class ParserDefaultsTests(unittest.TestCase):
    def test_defaults_match_spec(self) -> None:
        parser = drift.build_parser()
        args = parser.parse_args([])
        self.assertIsNone(args.date)
        self.assertEqual(args.window_days, drift.DEFAULT_WINDOW_DAYS)
        self.assertEqual(args.bins, drift.DEFAULT_BINS_CSV)
        self.assertIsNone(args.symbol)
        self.assertEqual(args.slope_tolerance, drift.DEFAULT_SLOPE_TOLERANCE)
        self.assertEqual(args.min_n_per_bucket, drift.DEFAULT_MIN_N_PER_BUCKET)
        self.assertFalse(args.alert)
        self.assertIsNone(args.out)


if __name__ == "__main__":
    unittest.main()
