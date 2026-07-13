"""Tests for ``scripts/select_cost_aware_threshold.py``.

Covers the math, the rich-vs-curve meta-shape detection, the n-floor
filter, the dry-run / --write split, and degenerate edge cases (empty
sweep, all-negative net P&L).
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


def _load_script_module():
    """Load ``scripts/select_cost_aware_threshold.py`` by path.

    The script lives in ``scripts/`` which isn't on ``sys.path`` and
    isn't a package, so we resolve it via ``importlib.util`` instead
    of ``import``. We register the module in ``sys.modules`` BEFORE
    executing it because ``@dataclass`` reads ``sys.modules[cls.__module__]``
    during class creation; if the module isn't registered, dataclass
    raises ``AttributeError: 'NoneType' object has no attribute '__dict__'``.
    """
    mod_name = "select_cost_aware_threshold"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "scripts" / "select_cost_aware_threshold.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None  # for mypy / static checkers
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # Don't leak a half-initialized module on import failure.
        sys.modules.pop(mod_name, None)
        raise
    return module


MOD = _load_script_module()


def _make_curve_meta(
    *,
    optimal_threshold: float = 0.55,
    rows=None,
):
    """Build a minimal meta dict in the test_precision_curve shape."""
    if rows is None:
        rows = [
            {"thr": 0.50, "n_trades": 5000, "win_rate": 0.52},
            {"thr": 0.55, "n_trades": 1000, "win_rate": 0.60},
            {"thr": 0.60, "n_trades":   50, "win_rate": 0.70},
        ]
    return {
        "optimal_threshold": optimal_threshold,
        "threshold_status": "ok",
        "test_precision_curve": rows,
    }


def _make_rich_meta(
    *,
    optimal_threshold: float = 0.55,
    entries=None,
):
    """Build a minimal meta dict in the threshold_metrics (rich) shape.

    Values follow ``_simulate_strategy_pnl`` semantics: avg_win/avg_loss
    are in position-size units (1.0 = full position). The trainer-side
    fee is folded in, so these may be a touch under 1.0.
    """
    if entries is None:
        entries = {
            "0.5000": {
                "n_trades": 5000,
                "win_rate": 0.52,
                "avg_win": 0.98,
                "avg_loss": -1.02,
                "sharpe": 0.05,
            },
            "0.5500": {
                "n_trades": 1000,
                "win_rate": 0.60,
                "avg_win": 0.98,
                "avg_loss": -1.02,
                "sharpe": 0.42,
            },
            "0.6000": {
                "n_trades": 50,
                "win_rate": 0.70,
                "avg_win": 0.98,
                "avg_loss": -1.02,
                "sharpe": 0.80,
            },
        }
    return {
        "optimal_threshold": optimal_threshold,
        "threshold_status": "ok",
        "threshold_metrics": entries,
    }


class CostAwarePickerMathTests(unittest.TestCase):
    """Pure-function tests for compute_candidates + _pick_winner."""

    def test_handpicked_winner_curve_shape(self) -> None:
        # symmetric payoff @ 20 bps, notional $50 -> $0.10/win, -$0.10/loss
        # round-trip cost = (5+5)*2/1e4 * 50 = $0.10/trade
        # net @ thr 0.50, wr=0.52: 0.52*.10 - .48*.10 - .10 = .052 - .048 - .10 = -.096
        # net @ thr 0.55, wr=0.60: .060 - .040 - .10 = -.080
        # net @ thr 0.60, wr=0.70: .070 - .030 - .10 = -.060  (but n=50 < 100, filtered)
        # so winner among eligible {0.50, 0.55} = 0.55
        meta = _make_curve_meta()
        rows, _ = MOD._extract_candidates(meta, notional=50.0, target_move_bps=20.0)
        all_rows, eligible = MOD.compute_candidates(
            rows,
            slippage_bps=5.0,
            commission_bps=5.0,
            notional=50.0,
            min_n=100,
        )
        # net values hand-computed above
        net_by_thr = {round(c.thr, 4): c.net_pnl for c in all_rows}
        self.assertAlmostEqual(net_by_thr[0.5000], -0.096, places=4)
        self.assertAlmostEqual(net_by_thr[0.5500], -0.080, places=4)
        self.assertAlmostEqual(net_by_thr[0.6000], -0.060, places=4)
        # thr=0.60 is excluded because n=50 < min_n=100
        eligible_thrs = sorted(c.thr for c in eligible)
        self.assertEqual(eligible_thrs, [0.50, 0.55])
        winner = MOD._pick_winner(eligible)
        self.assertIsNotNone(winner)
        self.assertAlmostEqual(winner.thr, 0.55, places=4)

    def test_handpicked_winner_rich_shape_uses_avg_win_avg_loss(self) -> None:
        # Rich shape: avg_win=0.98, avg_loss=-1.02, notional=$50
        # dollar avg_win  =  0.98 * 50 =  49.0
        # dollar avg_loss = -1.02 * 50 = -51.0
        # round-trip cost = (5+5)*2/1e4 * 50 = $0.10
        # @ thr 0.55, wr=0.60: 0.60*49 - 0.40*51 - 0.10 = 29.4 - 20.4 - .10 = 8.90
        meta = _make_rich_meta()
        rows, label = MOD._extract_candidates(
            meta, notional=50.0, target_move_bps=20.0
        )
        self.assertTrue(label.startswith("threshold_metrics"))
        all_rows, eligible = MOD.compute_candidates(
            rows,
            slippage_bps=5.0,
            commission_bps=5.0,
            notional=50.0,
            min_n=100,
        )
        net_by_thr = {round(c.thr, 4): c.net_pnl for c in all_rows}
        self.assertAlmostEqual(net_by_thr[0.5500], 8.90, places=2)
        # sharpe surfaced from the meta dict, not approximated
        sharpe_by_thr = {round(c.thr, 4): c.sharpe for c in all_rows}
        self.assertAlmostEqual(sharpe_by_thr[0.5500], 0.42, places=4)
        winner = MOD._pick_winner(eligible)
        self.assertIsNotNone(winner)
        # All eligible thrs net positive; 0.55 has the higher net_pnl
        self.assertAlmostEqual(winner.thr, 0.55, places=4)

    def test_min_n_filters_sparse_thresholds(self) -> None:
        # bump min_n to 6000 -> only thr=0.50 survives
        meta = _make_curve_meta()
        rows, _ = MOD._extract_candidates(meta, notional=50.0, target_move_bps=20.0)
        _, eligible = MOD.compute_candidates(
            rows,
            slippage_bps=5.0,
            commission_bps=5.0,
            notional=50.0,
            min_n=6000,
        )
        self.assertEqual(eligible, [])
        # And lower min_n=10 -> all three survive
        _, eligible_all = MOD.compute_candidates(
            rows,
            slippage_bps=5.0,
            commission_bps=5.0,
            notional=50.0,
            min_n=10,
        )
        self.assertEqual(sorted(c.thr for c in eligible_all), [0.50, 0.55, 0.60])


class CostAwarePickerCLITests(unittest.TestCase):
    """End-to-end CLI tests using a tmp meta.json."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.meta_path = Path(self._tmp.name) / "meta.json"

    def _write_meta(self, meta: dict) -> None:
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def test_dry_run_does_not_mutate_meta(self) -> None:
        self._write_meta(_make_curve_meta())
        before = self.meta_path.read_text(encoding="utf-8")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = MOD.main(["--meta", str(self.meta_path)])
        self.assertEqual(rc, 0)
        after = self.meta_path.read_text(encoding="utf-8")
        self.assertEqual(before, after)
        out = buf.getvalue()
        self.assertIn("<-- cost-aware pick", out)
        self.assertIn("<-- current", out)

    def test_write_updates_meta_without_clobbering_optimal_threshold(self) -> None:
        original = _make_curve_meta(optimal_threshold=0.55)
        self._write_meta(original)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = MOD.main(
                [
                    "--meta", str(self.meta_path),
                    "--write",
                    # use min-n=10 so all 3 rows are eligible and we get
                    # a clear winner
                    "--min-n", "10",
                ]
            )
        self.assertEqual(rc, 0)
        written = json.loads(self.meta_path.read_text(encoding="utf-8"))
        # optimal_threshold preserved verbatim
        self.assertEqual(written["optimal_threshold"], 0.55)
        # new keys present
        self.assertIn("cost_aware_threshold", written)
        self.assertIsInstance(written["cost_aware_threshold"], float)
        self.assertIn("cost_aware_threshold_at", written)
        self.assertIsInstance(written["cost_aware_threshold_at"], str)
        self.assertIn("cost_aware_threshold_params", written)
        params = written["cost_aware_threshold_params"]
        self.assertEqual(params["slippage_bps"], 5.0)
        self.assertEqual(params["commission_bps"], 5.0)
        self.assertEqual(params["notional_usd"], 50.0)
        self.assertEqual(params["min_n"], 10)
        self.assertIn("source", params)
        # the original sweep data is still there
        self.assertEqual(
            written["test_precision_curve"], original["test_precision_curve"]
        )

    def test_empty_threshold_metrics_exits_1(self) -> None:
        # Both fields missing entirely
        self._write_meta({"optimal_threshold": 0.5, "threshold_status": "ok"})
        buf_out = io.StringIO()
        buf_err = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = buf_err
        try:
            with redirect_stdout(buf_out):
                rc = MOD.main(["--meta", str(self.meta_path)])
        finally:
            sys.stderr = old_stderr
        self.assertEqual(rc, 1)
        self.assertIn("no threshold sweep data", buf_err.getvalue())

    def test_empty_curve_exits_1(self) -> None:
        # Both shapes present but empty
        self._write_meta(
            {
                "optimal_threshold": 0.5,
                "threshold_status": "ok",
                "threshold_metrics": {},
                "test_precision_curve": [],
            }
        )
        buf_err = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = buf_err
        try:
            with redirect_stdout(io.StringIO()):
                rc = MOD.main(["--meta", str(self.meta_path)])
        finally:
            sys.stderr = old_stderr
        self.assertEqual(rc, 1)
        self.assertIn("no threshold sweep data", buf_err.getvalue())

    def test_all_negative_net_pnl_still_picks_least_bad_with_warning(self) -> None:
        # Hammer costs so every candidate goes negative. Three eligible
        # rows, the highest-wr one is the least-bad winner.
        rows = [
            {"thr": 0.50, "n_trades": 1000, "win_rate": 0.50},
            {"thr": 0.55, "n_trades":  800, "win_rate": 0.55},
            {"thr": 0.60, "n_trades":  600, "win_rate": 0.58},
        ]
        self._write_meta(_make_curve_meta(rows=rows))
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = MOD.main(
                [
                    "--meta", str(self.meta_path),
                    # crank costs way up so net is < 0 for every row.
                    # 100 bps/side * 2 sides = 4% round-trip = $2.00 on
                    # a $50 notional; symmetric 20-bps payoff is only
                    # $0.10/win, so wr=0.58 nets ~0.058*0.10 - 0.42*0.10
                    # - 2.0 ~= -2.0
                    "--slippage-bps", "100",
                    "--commission-bps", "100",
                    "--min-n", "100",
                ]
            )
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        # Still surfaces a WINNER line
        self.assertIn("WINNER:", out)
        # And carries the negative-pnl warning
        self.assertIn("NEGATIVE expected net P&L", out)
        self.assertIn("least-bad", out)

    def test_write_with_no_eligible_winner_returns_1(self) -> None:
        # All rows below min-n -> no eligible winner -> --write refuses
        rows = [
            {"thr": 0.50, "n_trades": 5, "win_rate": 0.60},
        ]
        self._write_meta(_make_curve_meta(rows=rows))
        buf_err = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = buf_err
        try:
            with redirect_stdout(io.StringIO()):
                rc = MOD.main(
                    [
                        "--meta", str(self.meta_path),
                        "--write",
                        "--min-n", "100",
                    ]
                )
        finally:
            sys.stderr = old_stderr
        self.assertEqual(rc, 1)
        # meta untouched
        on_disk = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.assertNotIn("cost_aware_threshold", on_disk)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
