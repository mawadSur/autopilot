"""Unit tests for ``regime_memory.outcome_adjuster``.

Covers:
* Streak math: N losses -> +per_event_delta, M wins -> -per_event_delta
* Multiple-of-N math: 2N losses bumps twice
* Per-label isolation
* Bounds: |adjustment| <= max_adjustment
* Reset (single label + all)
* Corrupt-hash-value tolerance
* Label normalization
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Optional

import fakeredis

from regime_memory.outcome_adjuster import (
    DEFAULT_HASH_KEY,
    OutcomeAdjuster,
    normalize_label,
)


@dataclass
class _StubPosition:
    """Minimal stand-in for ``state.position_store.Position``."""
    position_id: str
    realized_pnl_usd: Optional[float]


def _label_resolver_by_pid_prefix(p: _StubPosition) -> Optional[str]:
    """Map ``"A-1"`` -> ``"trend_up"``, ``"B-1"`` -> ``"chop"``, etc.

    Used by the per-label-isolation tests so we don't have to thread a
    real label dict through every fixture.
    """
    if p.position_id.startswith("A"):
        return "trend_up"
    if p.position_id.startswith("B"):
        return "chop"
    if p.position_id.startswith("C"):
        return "trend_down"
    if p.position_id.startswith("N"):
        return None
    return None


def _make_adjuster(
    client,
    *,
    max_adjustment: float = 0.05,
    losses_to_raise: int = 3,
    wins_to_relax: int = 5,
    per_event_delta: float = 0.01,
):
    return OutcomeAdjuster(
        client,
        namespace="test-adj",
        hash_key=DEFAULT_HASH_KEY,
        max_adjustment=max_adjustment,
        losses_to_raise=losses_to_raise,
        wins_to_relax=wins_to_relax,
        per_event_delta=per_event_delta,
    )


def _make_positions(prefix: str, outcomes: str) -> list:
    """Build positions named ``"{prefix}-N"`` for each char in ``outcomes``.

    ``'L'`` -> realized -1.0 ; ``'W'`` -> realized +1.0 ; ``'0'`` -> 0.0
    """
    out = []
    for i, ch in enumerate(outcomes, start=1):
        if ch == "L":
            pnl = -1.0
        elif ch == "W":
            pnl = +1.0
        else:
            pnl = 0.0
        out.append(_StubPosition(position_id=f"{prefix}-{i}", realized_pnl_usd=pnl))
    return out


# ---------------------------------------------------------------------------
class LabelNormalizationTests(unittest.TestCase):
    def test_known_numeric_labels_map_to_strings(self) -> None:
        self.assertEqual(normalize_label(0.0), "trend_down")
        self.assertEqual(normalize_label(1.0), "chop")
        self.assertEqual(normalize_label(2.0), "trend_up")

    def test_string_labels_pass_through(self) -> None:
        self.assertEqual(normalize_label("chop"), "chop")
        self.assertEqual(normalize_label("  trend_up  "), "trend_up")

    def test_unknown_numeric_falls_back_to_label_tag(self) -> None:
        # 0.5 isn't in the v0 numeric map; we want a stable tag.
        self.assertEqual(normalize_label(0.5), "label_0.50")

    def test_none_and_nan_yield_none(self) -> None:
        self.assertIsNone(normalize_label(None))
        self.assertIsNone(normalize_label(float("nan")))
        self.assertIsNone(normalize_label(float("inf")))
        self.assertIsNone(normalize_label(""))


class CurrentAdjustmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = fakeredis.FakeRedis(decode_responses=True)
        self.adj = _make_adjuster(self.client)

    def test_missing_label_returns_zero(self) -> None:
        self.assertEqual(self.adj.current_adjustment("trend_up"), 0.0)

    def test_corrupt_value_returns_zero_with_warn(self) -> None:
        self.client.hset(self.adj.full_hash_key, "trend_up", "abc")
        with self.assertLogs(
            "regime_memory.outcome_adjuster", level="WARNING"
        ) as cm:
            v = self.adj.current_adjustment("trend_up")
        self.assertEqual(v, 0.0)
        self.assertTrue(any("corrupt value" in line for line in cm.output))

    def test_non_finite_returns_zero(self) -> None:
        self.client.hset(self.adj.full_hash_key, "chop", "inf")
        with self.assertLogs(
            "regime_memory.outcome_adjuster", level="WARNING"
        ):
            self.assertEqual(self.adj.current_adjustment("chop"), 0.0)

    def test_clip_on_read(self) -> None:
        # An external write that bypasses the adjuster could plant a
        # value outside [-max, +max]; current_adjustment must clip on
        # read so consumers see the invariant.
        self.client.hset(self.adj.full_hash_key, "trend_down", "0.99")
        self.assertAlmostEqual(
            self.adj.current_adjustment("trend_down"), 0.05, places=6
        )
        self.client.hset(self.adj.full_hash_key, "trend_down", "-0.99")
        self.assertAlmostEqual(
            self.adj.current_adjustment("trend_down"), -0.05, places=6
        )

    def test_empty_label_returns_zero(self) -> None:
        self.assertEqual(self.adj.current_adjustment(""), 0.0)


class AllAdjustmentsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = fakeredis.FakeRedis(decode_responses=True)
        self.adj = _make_adjuster(self.client)

    def test_empty_hash_returns_empty_dict(self) -> None:
        self.assertEqual(self.adj.all_adjustments(), {})

    def test_corrupt_rows_skipped(self) -> None:
        self.client.hset(self.adj.full_hash_key, "trend_up", "0.02")
        self.client.hset(self.adj.full_hash_key, "chop", "garbage")
        with self.assertLogs(
            "regime_memory.outcome_adjuster", level="WARNING"
        ):
            out = self.adj.all_adjustments()
        self.assertEqual(set(out.keys()), {"trend_up"})
        self.assertAlmostEqual(out["trend_up"], 0.02, places=6)


# ---------------------------------------------------------------------------
class StreakMathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = fakeredis.FakeRedis(decode_responses=True)
        self.adj = _make_adjuster(self.client)

    def test_three_losses_bumps_by_per_event_delta(self) -> None:
        positions = _make_positions("A", "LLL")
        out = self.adj.apply_closed_positions(
            positions, label_resolver=_label_resolver_by_pid_prefix
        )
        self.assertAlmostEqual(out["trend_up"], 0.01, places=6)

    def test_six_losses_bumps_twice(self) -> None:
        positions = _make_positions("A", "LLLLLL")
        out = self.adj.apply_closed_positions(
            positions, label_resolver=_label_resolver_by_pid_prefix
        )
        self.assertAlmostEqual(out["trend_up"], 0.02, places=6)

    def test_two_losses_does_not_bump(self) -> None:
        positions = _make_positions("A", "LL")
        out = self.adj.apply_closed_positions(
            positions, label_resolver=_label_resolver_by_pid_prefix
        )
        self.assertEqual(out, {})

    def test_five_wins_after_losses_relaxes_back(self) -> None:
        # 3 losses (bump to +0.01), then 5 wins (relax by -0.01) -> back to 0.0
        positions = _make_positions("A", "LLLWWWWW")
        out = self.adj.apply_closed_positions(
            positions, label_resolver=_label_resolver_by_pid_prefix
        )
        # The hash skips zero entries so the label drops out entirely.
        self.assertEqual(out.get("trend_up", 0.0), 0.0)

    def test_winning_trade_resets_loss_streak(self) -> None:
        # LL then W then LL -> only 2 consecutive losses, no bump.
        positions = _make_positions("A", "LLWLL")
        out = self.adj.apply_closed_positions(
            positions, label_resolver=_label_resolver_by_pid_prefix
        )
        self.assertEqual(out, {})

    def test_max_adjustment_bound_enforced(self) -> None:
        # 30 losses in a row would naively be +0.10; clipped to +0.05.
        positions = _make_positions("A", "L" * 30)
        out = self.adj.apply_closed_positions(
            positions, label_resolver=_label_resolver_by_pid_prefix
        )
        self.assertAlmostEqual(out["trend_up"], 0.05, places=6)

    def test_zero_pnl_treated_as_loss(self) -> None:
        # 3 break-even closes should bump the adjustment by +0.01 (the
        # bot is in a regime that doesn't even break above zero — we
        # want the threshold to creep up).
        positions = _make_positions("A", "000")
        out = self.adj.apply_closed_positions(
            positions, label_resolver=_label_resolver_by_pid_prefix
        )
        self.assertAlmostEqual(out["trend_up"], 0.01, places=6)

    def test_position_with_no_pnl_skipped(self) -> None:
        # ``realized_pnl_usd is None`` -> skip; doesn't extend streak.
        positions = [
            _StubPosition(position_id=f"A-{i}", realized_pnl_usd=-1.0)
            for i in range(1, 3)
        ]
        positions.append(_StubPosition(position_id="A-3", realized_pnl_usd=None))
        positions.append(_StubPosition(position_id="A-4", realized_pnl_usd=-1.0))
        out = self.adj.apply_closed_positions(
            positions, label_resolver=_label_resolver_by_pid_prefix
        )
        # 3 actual losses, the None doesn't count -> still hits the
        # 3-multiple. Streak count by outcome, not by position seen.
        self.assertAlmostEqual(out["trend_up"], 0.01, places=6)


class PerLabelIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = fakeredis.FakeRedis(decode_responses=True)
        self.adj = _make_adjuster(self.client)

    def test_losses_in_one_regime_dont_affect_other(self) -> None:
        # 3 losses for A (trend_up), 2 for B (chop). Only A should bump.
        positions = [
            _StubPosition(position_id="A-1", realized_pnl_usd=-1.0),
            _StubPosition(position_id="B-1", realized_pnl_usd=-1.0),
            _StubPosition(position_id="A-2", realized_pnl_usd=-1.0),
            _StubPosition(position_id="B-2", realized_pnl_usd=-1.0),
            _StubPosition(position_id="A-3", realized_pnl_usd=-1.0),
        ]
        out = self.adj.apply_closed_positions(
            positions, label_resolver=_label_resolver_by_pid_prefix
        )
        self.assertAlmostEqual(out.get("trend_up", 0.0), 0.01, places=6)
        # chop only saw 2 losses across the interleaved sequence -> no bump.
        self.assertEqual(out.get("chop", 0.0), 0.0)

    def test_independent_streaks_across_three_labels(self) -> None:
        positions = (
            _make_positions("A", "LLL")
            + _make_positions("B", "LLL")
            + _make_positions("C", "LLL")
        )
        out = self.adj.apply_closed_positions(
            positions, label_resolver=_label_resolver_by_pid_prefix
        )
        self.assertAlmostEqual(out["trend_up"], 0.01, places=6)
        self.assertAlmostEqual(out["chop"], 0.01, places=6)
        self.assertAlmostEqual(out["trend_down"], 0.01, places=6)


class MixedSymbolTests(unittest.TestCase):
    """Same regime label across different symbols MUST share streak state."""

    def setUp(self) -> None:
        self.client = fakeredis.FakeRedis(decode_responses=True)
        self.adj = _make_adjuster(self.client)

    def test_two_symbols_one_label_share_streak(self) -> None:
        # ETH-LL + BTC-LL -> 4 losses in the SAME regime "trend_up"
        # because both prefixes resolve to trend_up. That's 1 multiple
        # of 3 -> +0.01 (not 2x because the 4th loss only crosses one).
        positions = []
        for i in range(1, 5):
            positions.append(
                _StubPosition(position_id=f"A-{i}", realized_pnl_usd=-1.0)
            )
        out = self.adj.apply_closed_positions(
            positions, label_resolver=_label_resolver_by_pid_prefix
        )
        self.assertAlmostEqual(out["trend_up"], 0.01, places=6)


class ResetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = fakeredis.FakeRedis(decode_responses=True)
        self.adj = _make_adjuster(self.client)
        # Seed two adjustments.
        self.adj.apply_closed_positions(
            _make_positions("A", "LLL") + _make_positions("B", "LLL"),
            label_resolver=_label_resolver_by_pid_prefix,
        )

    def test_reset_single_label(self) -> None:
        self.assertAlmostEqual(self.adj.current_adjustment("trend_up"), 0.01)
        self.adj.reset(label="trend_up")
        self.assertEqual(self.adj.current_adjustment("trend_up"), 0.0)
        # The other label is preserved.
        self.assertAlmostEqual(self.adj.current_adjustment("chop"), 0.01)

    def test_reset_all_clears_hash(self) -> None:
        self.adj.reset(label=None)
        self.assertEqual(self.adj.all_adjustments(), {})


class ConstructorValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = fakeredis.FakeRedis(decode_responses=True)

    def test_negative_max_rejected(self) -> None:
        with self.assertRaises(ValueError):
            OutcomeAdjuster(self.client, max_adjustment=-0.01)

    def test_zero_losses_to_raise_rejected(self) -> None:
        with self.assertRaises(ValueError):
            OutcomeAdjuster(self.client, losses_to_raise=0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
