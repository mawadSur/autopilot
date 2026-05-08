"""Unit tests for ``src/loss_postmortem/signal_forensics.py`` (Lane E A1).

Covers the five test cases from the SignalForensicsAgent brief:

(a) Confidence within 5 % of threshold → ``contributing``.
(b) NaN / non-finite feature in the snapshot → ``primary_cause``.
(c) Synthetic OOD features (Mahalanobis > 3σ vs training mean) →
    ``primary_cause``.
(d) Healthy snapshot, high confidence, in-distribution → ``innocent``.
(e) Crashing investigation routes through the safety wrappers and yields
    ``verdict="unknown"``.

We follow the fixture style established by ``test_forensics_base.py``:
fakeredis-backed :class:`TradeContextStore`, snapshots written via the
store's normal write path, and a per-test isolated namespace. Meta JSON
files are written to a tmpdir so we don't pollute the repo's
``model_crypto/`` checked-in artifacts.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any, Dict, List, Mapping, Optional

import fakeredis

from loss_postmortem.base import ForensicsFinding
from loss_postmortem.signal_forensics import (
    DEFAULT_MAHALANOBIS_FLAG_SIGMA,
    SignalForensicsAgent,
)
from state.trade_context_store import (
    TradeContextSnapshot,
    TradeContextStore,
    utc_now_iso,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _store(namespace: str = "test") -> TradeContextStore:
    return TradeContextStore(
        redis_client=fakeredis.FakeRedis(decode_responses=True),
        namespace=namespace,
    )


def _write_meta(
    base_dir: str,
    *,
    symbol_slug: str,
    payload: Mapping[str, Any],
) -> str:
    """Write ``payload`` as ``model_crypto/<slug>/meta.json`` under ``base_dir``."""
    target = os.path.join(base_dir, symbol_slug)
    os.makedirs(target, exist_ok=True)
    meta_path = os.path.join(target, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return meta_path


def _record_signal_snapshot(
    store: TradeContextStore,
    *,
    trade_id: str,
    symbol: str,
    feature_buffer: Optional[Dict[str, float]] = None,
    feature_window: Optional[List[Dict[str, Any]]] = None,
    model_probs: Optional[Dict[str, float]] = None,
    model_confidence: float = 0.6,
) -> None:
    snap = TradeContextSnapshot(
        trade_id=trade_id,
        symbol=symbol,
        captured_at_utc=utc_now_iso(),
        phase="signal",
        feature_buffer=feature_buffer or {},
        feature_window=feature_window,
        model_probs=model_probs or {},
        model_confidence=model_confidence,
    )
    store.record_snapshot(snap)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class SignalForensicsAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.meta_base = self._tmp.name
        self.store = _store(namespace=f"sigtest-{os.urandom(2).hex()}")

    # ------------------------------------------------------------------
    # case (a): confidence barely above threshold → contributing
    # ------------------------------------------------------------------
    def test_confidence_thin_margin_yields_contributing(self) -> None:
        """conf=0.61, threshold=0.6 → within 5 % margin → 1 red flag."""
        _write_meta(
            self.meta_base,
            symbol_slug="eth_usd",
            payload={
                "optimal_threshold": 0.6,
                # No threshold_metrics, no feature_means/stds, no
                # metrics_*.reliability_slope → checks 3 and 4 skip
                # cleanly leaving only check 1 firing.
            },
        )
        _record_signal_snapshot(
            self.store,
            trade_id="trade-thin",
            symbol="ETH/USD",
            model_confidence=0.61,
            # Empty feature_buffer/probs/window → checks 2/3/5 skip.
        )
        agent = SignalForensicsAgent(
            context_store=self.store, meta_base_dir=self.meta_base
        )
        finding = agent.investigate("trade-thin")
        self.assertIsInstance(finding, ForensicsFinding)
        self.assertEqual(finding.agent, "signal")
        self.assertEqual(finding.verdict, "contributing")
        self.assertGreaterEqual(finding.confidence, 0.4)
        self.assertLessEqual(finding.confidence, 0.7)
        # Evidence should mention the thin margin.
        joined = " | ".join(finding.evidence).lower()
        self.assertIn("threshold", joined)
        self.assertIn("margin", joined)
        # Suggested action should propose raising the floor.
        self.assertIsNotNone(finding.suggested_action)
        assert finding.suggested_action is not None  # for type checkers
        self.assertEqual(finding.suggested_action["type"], "raise_floor")

    # ------------------------------------------------------------------
    # case (b): NaN feature → primary_cause (combined with thin margin
    # + OOD to clear the 3-flag bar — one flag alone is "contributing")
    # ------------------------------------------------------------------
    def test_nan_feature_with_other_red_flags_is_primary_cause(self) -> None:
        """NaN feature + thin margin + OOD → 3 red flags → primary_cause."""
        # The brief lists case (b) as "NaN feature → primary_cause". A
        # single red flag alone evaluates to "contributing" per the verdict
        # ladder, so we set up a snapshot where the NaN signal is reinforced
        # by a thin-margin confidence and an OOD feature payload — this is
        # the realistic "the model fired on garbage data" scenario the brief
        # is testing for. The narrower "single flag → contributing" path is
        # covered by case (a).
        _write_meta(
            self.meta_base,
            symbol_slug="btc_usd",
            payload={
                "optimal_threshold": 0.6,
                "feature_means": {"return_1": 0.0, "atr_14": 0.5},
                "feature_stds": {"return_1": 1.0, "atr_14": 0.1},
            },
        )
        _record_signal_snapshot(
            self.store,
            trade_id="trade-nan",
            symbol="BTC/USD",
            model_confidence=0.61,  # thin margin → flag 1
            feature_buffer={
                # NaN/Inf round-trip as None through the snapshot store.
                "return_1": None,  # flag 2: non-finite
                "atr_14": 8.0,  # 75σ from mean — flag 3 (OOD)
            },
        )
        agent = SignalForensicsAgent(
            context_store=self.store, meta_base_dir=self.meta_base
        )
        finding = agent.investigate("trade-nan")
        self.assertEqual(finding.verdict, "primary_cause")
        self.assertGreaterEqual(finding.confidence, 0.7)
        joined = " | ".join(finding.evidence).lower()
        # NaN/inf evidence should be surfaced.
        self.assertTrue(
            "nan/inf/missing" in joined or "non-finite" in joined,
            msg=f"expected NaN evidence, got: {finding.evidence}",
        )

    def test_single_nan_alone_is_contributing(self) -> None:
        """A lone NaN — no other red flags — stays at ``contributing``."""
        _write_meta(
            self.meta_base,
            symbol_slug="eth_usd",
            payload={"optimal_threshold": 0.5},
        )
        _record_signal_snapshot(
            self.store,
            trade_id="trade-lone-nan",
            symbol="ETH/USD",
            # Healthy margin so check 1 doesn't fire.
            model_confidence=0.85,
            feature_buffer={"return_1": None},  # NaN-only red flag
        )
        agent = SignalForensicsAgent(
            context_store=self.store, meta_base_dir=self.meta_base
        )
        finding = agent.investigate("trade-lone-nan")
        self.assertEqual(finding.verdict, "contributing")

    # ------------------------------------------------------------------
    # case (c): synthetic OOD features (Mahalanobis > 3σ) → primary_cause
    # ------------------------------------------------------------------
    def test_extreme_ood_features_flag_primary_cause(self) -> None:
        """Many features, all > 5σ from training mean → primary_cause."""
        feat_means = {f"f{i}": 0.0 for i in range(8)}
        feat_stds = {f"f{i}": 1.0 for i in range(8)}
        _write_meta(
            self.meta_base,
            symbol_slug="eth_usd",
            payload={
                "optimal_threshold": 0.5,
                "feature_means": feat_means,
                "feature_stds": feat_stds,
                # An anti-calibrated reliability_slope on top of OOD
                # gives us 3 distinct red flag domains:
                #   - thin margin (set conf right above 0.5)
                #   - OOD (feature distances)
                #   - anti-calibrated bin
                "metrics_test": {"reliability_slope": -0.7},
            },
        )
        _record_signal_snapshot(
            self.store,
            trade_id="trade-ood",
            symbol="ETH/USD",
            model_confidence=0.51,  # thin margin → flag
            feature_buffer={f"f{i}": 6.0 for i in range(8)},  # 6σ each → OOD
            # model_probs left empty → reliability falls back to
            # model_confidence and the meta-level slope.
        )
        agent = SignalForensicsAgent(
            context_store=self.store, meta_base_dir=self.meta_base
        )
        finding = agent.investigate("trade-ood")
        self.assertEqual(finding.verdict, "primary_cause")
        self.assertGreaterEqual(finding.confidence, 0.7)
        # Mahalanobis evidence should appear.
        joined = " | ".join(finding.evidence).lower()
        self.assertIn("mahalanobis", joined)

    # ------------------------------------------------------------------
    # case (d): healthy snapshot → innocent
    # ------------------------------------------------------------------
    def test_healthy_snapshot_yields_innocent(self) -> None:
        """High confidence, in-distribution features, calibrated model → innocent."""
        _write_meta(
            self.meta_base,
            symbol_slug="eth_usd",
            payload={
                "optimal_threshold": 0.5,
                "feature_means": {"return_1": 0.0, "atr_14": 1.0},
                "feature_stds": {"return_1": 1.0, "atr_14": 0.5},
                "metrics_test": {"reliability_slope": 0.9},  # well-calibrated
                "threshold_metrics": {
                    "0.5": {"reliability_slope": 0.85},
                    "0.6": {"reliability_slope": 0.9},
                },
            },
        )
        _record_signal_snapshot(
            self.store,
            trade_id="trade-healthy",
            symbol="ETH/USD",
            model_confidence=0.85,  # well above threshold → no flag 1
            feature_buffer={"return_1": 0.05, "atr_14": 1.05},  # ~0σ each
            model_probs={"long": 0.85, "short": 0.15},
            feature_window=[
                {"regime": "trend", "return_1": 0.04},
                {"regime": "trend", "return_1": 0.05},
                {"regime": "trend", "return_1": 0.05},
                {"regime": "trend", "return_1": 0.05},
            ],
        )
        agent = SignalForensicsAgent(
            context_store=self.store, meta_base_dir=self.meta_base
        )
        finding = agent.investigate("trade-healthy")
        self.assertEqual(finding.verdict, "innocent")
        self.assertEqual(finding.confidence, 0.0)
        self.assertIsNone(finding.suggested_action)

    # ------------------------------------------------------------------
    # case (e): crashing investigation → safe_investigate → unknown
    # ------------------------------------------------------------------
    def test_crashing_investigation_routes_through_safe_run(self) -> None:
        """Forced crash inside investigate() yields verdict=unknown."""

        class _ExplodingSignalAgent(SignalForensicsAgent):
            def investigate(self, trade_id: str) -> ForensicsFinding:
                raise RuntimeError("boom in signal forensics")

        agent = _ExplodingSignalAgent(
            context_store=self.store, meta_base_dir=self.meta_base
        )
        finding = agent.safe_investigate("trade-crash")
        self.assertEqual(finding.agent, "signal")
        self.assertEqual(finding.verdict, "unknown")
        self.assertIsNotNone(finding.error)
        self.assertIn("boom in signal forensics", finding.error or "")

    # ------------------------------------------------------------------
    # extra: missing snapshot is honest, not a crash
    # ------------------------------------------------------------------
    def test_missing_snapshot_returns_unknown_with_evidence(self) -> None:
        """No signal snapshot recorded → verdict=unknown, single bullet."""
        agent = SignalForensicsAgent(
            context_store=self.store, meta_base_dir=self.meta_base
        )
        finding = agent.investigate("trade-missing")
        self.assertEqual(finding.verdict, "unknown")
        self.assertEqual(finding.error, "missing_signal_snapshot")
        self.assertEqual(len(finding.evidence), 1)

    # ------------------------------------------------------------------
    # extra: empty feature_buffer is gracefully degraded (caveat in brief)
    # ------------------------------------------------------------------
    def test_empty_feature_buffer_skips_checks_without_crashing(self) -> None:
        """Empty feature_buffer (predictor surface limitation) → no crash, evidence bullets."""
        _write_meta(
            self.meta_base,
            symbol_slug="eth_usd",
            payload={
                "optimal_threshold": 0.5,
                "feature_means": {"return_1": 0.0},
                "feature_stds": {"return_1": 1.0},
            },
        )
        _record_signal_snapshot(
            self.store,
            trade_id="trade-empty",
            symbol="ETH/USD",
            model_confidence=0.85,
            feature_buffer={},  # empty
            model_probs={},  # empty
        )
        agent = SignalForensicsAgent(
            context_store=self.store, meta_base_dir=self.meta_base
        )
        finding = agent.investigate("trade-empty")
        # Must NOT be primary_cause solely because data is missing.
        self.assertNotEqual(finding.verdict, "primary_cause")
        joined = " | ".join(finding.evidence).lower()
        self.assertIn("mahalanobis", joined)
        self.assertIn("skipped", joined)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
