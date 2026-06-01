"""Unit tests for ``src/loss_postmortem/sizing_forensics.py`` (Lane E A3).

Five scenarios — one per check axis:

(a) Healthy snapshot → ``innocent``.
(b) Position size > 10 % of bankroll → ``primary_cause``.
(c) Fresh recompute drift > 5 % from snapshot output → ``primary_cause``
    with ``audit_risk_engine_drift`` action.
(d) Correlation cluster (> 3 same-category open positions) → ``contributing``.
(e) Empty ``risk_metrics_input`` → ``unknown`` with the limited-evidence note.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from typing import Any, Dict

import fakeredis

from loss_postmortem.sizing_forensics import SizingForensicsAgent
from models import Market
from risk_management_agent.risk_engine import RiskCalculator
from state.trade_context_store import (
    TradeContextSnapshot,
    TradeContextStore,
    utc_now_iso,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _store() -> TradeContextStore:
    return TradeContextStore(
        redis_client=fakeredis.FakeRedis(decode_responses=True),
        namespace="test",
    )


def _market_dict(
    *,
    implied_prob: float = 0.50,
    category: str = "Politics",
    volume_24h: float = 50_000.0,
    bid: float | None = None,
    ask: float | None = None,
) -> Dict[str, Any]:
    """Match the fields ``models.Market.__init__`` accepts."""
    return {
        "market_id": "test-market",
        "title": "Will the test pass?",
        "category": category,
        "implied_prob": float(implied_prob),
        "bid_price": float(bid if bid is not None else max(0.01, implied_prob - 0.01)),
        "ask_price": float(ask if ask is not None else min(0.99, implied_prob + 0.01)),
        "volume_24h": float(volume_24h),
        "price_history": {"24h_ago": implied_prob, "1h_ago": implied_prob},
        "open_interest": 10_000.0,
        # ISO string is fine — Market.__post_init__ parses it.
        "resolution_date": datetime(2030, 1, 1, tzinfo=timezone.utc).isoformat(),
        "rules_text": "Resolves YES iff the test passes.",
    }


def _record_signal_snapshot(
    store: TradeContextStore,
    *,
    trade_id: str,
    risk_in: Dict[str, Any],
    risk_out: Dict[str, Any],
    symbol: str = "test-symbol",
) -> None:
    snap = TradeContextSnapshot(
        trade_id=trade_id,
        symbol=symbol,
        captured_at_utc=utc_now_iso(),
        phase="signal",
        feature_buffer={},
        feature_window=None,
        model_probs={},
        model_confidence=0.7,
        risk_metrics_input=risk_in,
        risk_metrics_output=risk_out,
        breaker_context={},
        ticker_buffer=[],
        notes=None,
    )
    store.record_snapshot(snap)


def _healthy_inputs() -> Dict[str, Any]:
    """A balanced, in-distribution sizing input set."""
    return {
        "market": _market_dict(implied_prob=0.50, volume_24h=80_000.0),
        "calibrated_true_prob": 0.55,
        "market_price": 0.50,
        "bankroll": 10_000.0,
        "existing_open_positions": [],
    }


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class SizingForensicsAgentTests(unittest.TestCase):
    # ------------------------------------------------------------------
    # (a) healthy snapshot → innocent
    # ------------------------------------------------------------------
    def test_healthy_snapshot_yields_innocent(self) -> None:
        store = _store()
        risk_in = _healthy_inputs()

        # Compute a matching, fee-aware risk_metrics_output via the live
        # calculator — guarantees no recompute drift in this scenario.
        calc = RiskCalculator()
        metrics = calc.calculate_base_metrics(
            market=Market(**risk_in["market"]),
            calibrated_true_prob=risk_in["calibrated_true_prob"],
            bankroll=risk_in["bankroll"],
            market_price=risk_in["market_price"],
        )
        risk_out = metrics.model_dump()

        _record_signal_snapshot(
            store, trade_id="trade-healthy", risk_in=risk_in, risk_out=risk_out
        )

        agent = SizingForensicsAgent(context_store=store)
        finding = agent.investigate("trade-healthy")

        self.assertEqual(finding.agent, "sizing")
        self.assertEqual(finding.verdict, "innocent")
        # Innocent verdicts should not propose a corrective action.
        self.assertIsNone(finding.suggested_action)
        # Position size in this case is well below 5 % bankroll, so no flags.
        self.assertTrue(
            any("recompute matched" in e for e in finding.evidence),
            f"expected recompute-match evidence, got {finding.evidence}",
        )

    # ------------------------------------------------------------------
    # (b) position size > 10 % bankroll → primary_cause
    # ------------------------------------------------------------------
    def test_oversized_position_yields_primary_cause(self) -> None:
        store = _store()
        risk_in = _healthy_inputs()
        # Match snapshot output to inputs (so drift check is clean) but
        # mark the adjusted size at 12 % — > 10 % threshold.
        risk_out = {
            "market_price": 0.50,
            "calibrated_true_prob": 0.55,
            "bankroll": 10_000.0,
            "raw_kelly_size_pct": 30.0,
            "fractional_kelly_size_pct": 12.0,
            "liquidity_penalty_multiplier": 1.0,
            "correlation_penalty_multiplier": 1.0,
            "same_category_open_positions": 0,
            "liquidity_penalty_applied": False,
            "correlation_penalty_applied": False,
            "adjusted_position_size_pct": 12.0,
            "max_loss_if_wrong": 1_200.0,
            # EV close to a fresh recompute is harder here — set a value
            # that's roughly consistent. We don't care about drift in this
            # scenario; we let it surface as its own (non-fatal) flag.
            "expected_value_estimate": 200.0,
        }
        _record_signal_snapshot(
            store, trade_id="trade-big", risk_in=risk_in, risk_out=risk_out
        )

        agent = SizingForensicsAgent(context_store=store)
        finding = agent.investigate("trade-big")

        self.assertEqual(finding.verdict, "primary_cause")
        self.assertGreater(finding.confidence, 0.5)
        # Either size or drift action is acceptable — both fire here. Size
        # is the named cause for this scenario; check evidence mentions it.
        self.assertTrue(
            any("position size" in e for e in finding.evidence),
            f"expected position-size evidence, got {finding.evidence}",
        )

    # ------------------------------------------------------------------
    # (c) recompute drift > 5 % → primary_cause + audit action
    # ------------------------------------------------------------------
    def test_recompute_drift_yields_primary_cause(self) -> None:
        store = _store()
        risk_in = _healthy_inputs()
        # Snapshot output claims a wildly different EV than a fresh
        # recompute would produce — the actual fresh EV is small and
        # positive; we record a snapshot EV 10x larger.
        risk_out = {
            "market_price": 0.50,
            "calibrated_true_prob": 0.55,
            "bankroll": 10_000.0,
            "raw_kelly_size_pct": 10.0,
            "fractional_kelly_size_pct": 2.5,
            "liquidity_penalty_multiplier": 1.0,
            "correlation_penalty_multiplier": 1.0,
            "same_category_open_positions": 0,
            "liquidity_penalty_applied": False,
            "correlation_penalty_applied": False,
            "adjusted_position_size_pct": 2.5,  # 2.5 % bankroll → no size flag
            "max_loss_if_wrong": 250.0,
            "expected_value_estimate": 9999.0,  # absurd → drift > 5 %
        }
        _record_signal_snapshot(
            store, trade_id="trade-drift", risk_in=risk_in, risk_out=risk_out
        )

        agent = SizingForensicsAgent(context_store=store)
        finding = agent.investigate("trade-drift")

        self.assertEqual(finding.verdict, "primary_cause")
        self.assertEqual(
            finding.suggested_action,
            {"type": "audit_risk_engine_drift", "trade_id": "trade-drift"},
        )
        self.assertTrue(
            any("recompute drift" in e for e in finding.evidence),
            f"expected recompute-drift evidence, got {finding.evidence}",
        )

    # ------------------------------------------------------------------
    # (d) correlation cluster → contributing
    # ------------------------------------------------------------------
    def test_correlation_cluster_yields_contributing(self) -> None:
        store = _store()
        risk_in = _healthy_inputs()
        # Inject 4 same-category open positions (> 3 threshold).
        risk_in["existing_open_positions"] = [
            {"category": "Politics", "market_id": f"mkt-{i}"} for i in range(4)
        ]

        # Use a fresh recompute output so no drift / fee / size flags fire.
        calc = RiskCalculator()
        metrics = calc.calculate_base_metrics(
            market=Market(**risk_in["market"]),
            calibrated_true_prob=risk_in["calibrated_true_prob"],
            bankroll=risk_in["bankroll"],
            market_price=risk_in["market_price"],
            existing_open_positions=risk_in["existing_open_positions"],
        )
        risk_out = metrics.model_dump()

        _record_signal_snapshot(
            store, trade_id="trade-cluster", risk_in=risk_in, risk_out=risk_out
        )

        agent = SizingForensicsAgent(context_store=store)
        finding = agent.investigate("trade-cluster")

        self.assertEqual(finding.verdict, "contributing")
        self.assertTrue(
            any("correlation cluster" in e for e in finding.evidence),
            f"expected correlation-cluster evidence, got {finding.evidence}",
        )
        # Suggested action should be the correlation-tightening one because
        # no primary-class flags fired.
        self.assertEqual(
            finding.suggested_action,
            {"type": "tighten_correlation_penalty", "from": 0.5, "to": 0.3},
        )

    # ------------------------------------------------------------------
    # (e) empty risk_metrics_input → unknown
    # ------------------------------------------------------------------
    def test_empty_inputs_yields_unknown(self) -> None:
        store = _store()
        _record_signal_snapshot(
            store,
            trade_id="trade-empty",
            risk_in={},
            risk_out={},
        )

        agent = SizingForensicsAgent(context_store=store)
        finding = agent.investigate("trade-empty")

        self.assertEqual(finding.verdict, "unknown")
        self.assertEqual(finding.confidence, 0.2)
        self.assertTrue(
            any("risk_metrics_input missing" in e for e in finding.evidence),
            f"expected limited-evidence note, got {finding.evidence}",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
