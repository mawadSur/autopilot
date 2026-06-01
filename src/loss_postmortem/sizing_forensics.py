"""Sizing forensics agent (Lane E A3).

This agent investigates whether the position sizing math at decision time
contributed to a losing trade. It operates on the ``signal`` snapshot
captured by :class:`TradeContextStore` and runs five orthogonal checks:

1. **Fresh recompute drift.** Re-runs
   :meth:`RiskCalculator.calculate_base_metrics` against the snapshot's
   ``risk_metrics_input`` and compares the freshly computed
   ``expected_value_estimate`` to the snapshot's ``risk_metrics_output``.
   A material drift (> 5 %) is a bug-class red flag — the live engine
   produced a different number from a clean recompute.
2. **Fee deduction applied.** Verifies the snapshot's output reflects the
   Polymarket-fee adjustment. If the EV implies the raw probability was
   used (no fee haircut) on a Polymarket trade, that's a bug-class miss.
3. **Correlation cluster.** Counts how many open positions shared this
   trade's category at decision time. ``> 3`` in-cluster → contributing.
4. **Liquidity penalty mismatch.** If the market's volume / spread make
   it illiquid but the snapshot output shows
   ``liquidity_penalty_applied=False``, that's a contributing factor.
5. **Position size as %% of bankroll.** ``> 5 %`` contributing,
   ``> 10 %`` primary cause.

Defense-in-depth: when ``risk_metrics_input`` is missing or empty (the
sizing pipeline didn't capture inputs for this trade), the agent emits a
single "limited" evidence bullet and leans towards ``verdict="unknown"``
rather than guess.

The agent intentionally does NOT raise on bad input shapes — every check
is wrapped in a try/except so a partial snapshot still yields the
remaining checks. The :class:`BaseForensicsAgent.safe_investigate`
wrapper handles any escaped exception with ``verdict="unknown"``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from loss_postmortem.base import (
    BaseForensicsAgent,
    ForensicsFinding,
)
from state.trade_context_store import (
    TradeContextSnapshot,
    TradeContextStore,
)

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
# Fresh recompute vs snapshot drift: >5% delta on the EV estimate is a bug.
EV_DRIFT_THRESHOLD = 0.05
# Position size as fraction of bankroll thresholds.
POSITION_SIZE_CONTRIBUTING_PCT = 0.05  # > 5 % bankroll = contributing
POSITION_SIZE_PRIMARY_PCT = 0.10  # > 10 % bankroll = primary
# Correlation cluster threshold: > N same-category open positions.
CORRELATION_CLUSTER_THRESHOLD = 3
# Liquidity proxies — used to detect "low-volume market sized as if liquid".
LOW_VOLUME_USD_THRESHOLD = 10_000.0
WIDE_SPREAD_THRESHOLD = 0.05


def _to_float(value: Any, default: float = 0.0) -> float:
    """Coerce ``value`` to float; return ``default`` on failure."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out:  # NaN
        return default
    return out


def _looks_polymarket(snap: TradeContextSnapshot) -> bool:
    """Best-effort heuristic for whether this snapshot is a Polymarket trade.

    Prediction-market snapshots carry ``calibrated_true_prob`` /
    ``market_price`` keys in ``risk_metrics_input``; crypto trades carry
    ``proposed_notional_usd`` / ``side``. The heuristic looks for the
    prediction-market keys and a Market-shaped ``market`` mapping.
    """
    inputs = snap.risk_metrics_input or {}
    if "calibrated_true_prob" in inputs or "market_price" in inputs:
        return True
    market = inputs.get("market")
    if isinstance(market, dict) and "implied_prob" in market:
        return True
    return False


def _maybe_build_market(market_dict: Any) -> Any:
    """Reconstruct a :class:`Market` from a serialized dict, if possible.

    Returns the rebuilt ``Market`` instance or None if the dict is
    incomplete / malformed. Imports are local because the legacy
    ``models`` module pulls in torch — we don't want to pay that cost
    at module import time.
    """
    if not isinstance(market_dict, dict):
        return None
    try:
        from models import Market  # local import: torch is heavy
    except Exception as exc:  # noqa: BLE001 - tolerate missing legacy deps
        LOGGER.debug("sizing_forensics: cannot import Market: %r", exc)
        return None
    try:
        return Market(**market_dict)
    except (TypeError, ValueError) as exc:
        LOGGER.debug("sizing_forensics: Market reconstruction failed: %r", exc)
        return None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class SizingForensicsAgent(BaseForensicsAgent):
    """Forensics specialist that audits the sizing pipeline for one trade.

    Construction mirrors :class:`BaseForensicsAgent`. An optional
    ``position_store`` may be passed so the correlation-cluster check
    can fall back to live position data when the snapshot didn't carry
    a list of open positions.
    """

    agent_name = "sizing"

    def __init__(
        self,
        *,
        context_store: TradeContextStore,
        position_store: Any = None,
        timeout_s: float = 60.0,
    ) -> None:
        super().__init__(context_store=context_store, timeout_s=timeout_s)
        self.position_store = position_store

    # ------------------------------------------------------------------
    # contract
    # ------------------------------------------------------------------
    def investigate(self, trade_id: str) -> ForensicsFinding:
        evidence: List[str] = []
        flags_contributing = 0
        flags_primary = 0
        suggested_action: Optional[Dict[str, Any]] = None

        snap = self.context_store.get_signal_snapshot(trade_id)
        if snap is None:
            # Fall back to breaker phase — sizing inputs sometimes only land
            # there for forced-flat closes.
            snap = self.context_store.get_snapshot(trade_id, "breaker")

        if snap is None:
            return ForensicsFinding(
                agent="sizing",
                verdict="unknown",
                confidence=0.0,
                evidence=[
                    f"no signal/breaker snapshot for trade_id={trade_id}",
                ],
                severity=1,
            )

        risk_in: Dict[str, Any] = dict(snap.risk_metrics_input or {})
        risk_out: Dict[str, Any] = dict(snap.risk_metrics_output or {})

        # ------------------------------------------------------------------
        # Defense-in-depth: empty inputs → limited investigation
        # ------------------------------------------------------------------
        if not risk_in:
            evidence.append(
                "risk_metrics_input missing — sizing checks limited"
            )
            return ForensicsFinding(
                agent="sizing",
                verdict="unknown",
                confidence=0.2,
                evidence=evidence,
                severity=1,
            )

        # ------------------------------------------------------------------
        # Check 5 (run first, drives action choice): position size as % bankroll
        # ------------------------------------------------------------------
        size_action: Optional[Dict[str, Any]] = None
        size_pct = self._position_size_fraction(risk_in, risk_out)
        if size_pct is not None:
            if size_pct > POSITION_SIZE_PRIMARY_PCT:
                flags_primary += 1
                evidence.append(
                    f"position size {size_pct:.1%} of bankroll exceeds "
                    f"{POSITION_SIZE_PRIMARY_PCT:.0%} primary-cause threshold"
                )
                size_action = {
                    "type": "lower_max_kelly_pct",
                    "from": 0.05,
                    "to": 0.025,
                }
            elif size_pct > POSITION_SIZE_CONTRIBUTING_PCT:
                flags_contributing += 1
                evidence.append(
                    f"position size {size_pct:.1%} of bankroll exceeds "
                    f"{POSITION_SIZE_CONTRIBUTING_PCT:.0%} contributing threshold"
                )
                size_action = {
                    "type": "lower_max_kelly_pct",
                    "from": 0.05,
                    "to": 0.025,
                }

        # ------------------------------------------------------------------
        # Check 1: fresh recompute vs snapshot drift
        # ------------------------------------------------------------------
        drift_action: Optional[Dict[str, Any]] = None
        drift_evidence, drift_flag = self._check_recompute_drift(risk_in, risk_out)
        if drift_evidence:
            evidence.append(drift_evidence)
        if drift_flag == "primary":
            flags_primary += 1
            drift_action = {
                "type": "audit_risk_engine_drift",
                "trade_id": trade_id,
            }

        # ------------------------------------------------------------------
        # Check 2: fee deduction was applied
        # ------------------------------------------------------------------
        fee_action: Optional[Dict[str, Any]] = None
        fee_evidence, fee_flag = self._check_fee_deduction(snap, risk_in, risk_out)
        if fee_evidence:
            evidence.append(fee_evidence)
        if fee_flag == "primary":
            flags_primary += 1
            fee_action = {"type": "audit_fee_deduction_call_path"}

        # ------------------------------------------------------------------
        # Check 3: correlation cluster
        # ------------------------------------------------------------------
        corr_action: Optional[Dict[str, Any]] = None
        corr_evidence, corr_flag = self._check_correlation_cluster(snap, risk_in)
        if corr_evidence:
            evidence.append(corr_evidence)
        if corr_flag == "contributing":
            flags_contributing += 1
            corr_action = {
                "type": "tighten_correlation_penalty",
                "from": 0.5,
                "to": 0.3,
            }

        # ------------------------------------------------------------------
        # Check 4: liquidity penalty mismatch
        # ------------------------------------------------------------------
        liq_action: Optional[Dict[str, Any]] = None
        liq_evidence, liq_flag = self._check_liquidity_penalty(risk_in, risk_out)
        if liq_evidence:
            evidence.append(liq_evidence)
        if liq_flag == "contributing":
            flags_contributing += 1
            liq_action = {
                "type": "tighten_liquidity_penalty",
                "scope": "low_volume_markets",
            }

        # ------------------------------------------------------------------
        # Verdict synthesis
        # ------------------------------------------------------------------
        # Pick a single suggested_action: prefer primary-class actions, else
        # the first contributing-class one, in the same priority as the
        # checks ran.
        for candidate in (drift_action, fee_action, size_action, corr_action, liq_action):
            if candidate is not None:
                suggested_action = candidate
                break

        if flags_primary >= 1:
            verdict = "primary_cause"
            confidence = 0.75 if flags_primary == 1 else 0.9
            severity = 4 if flags_primary == 1 else 5
        elif flags_contributing >= 1:
            verdict = "contributing"
            confidence = 0.4 + 0.15 * min(2, flags_contributing)
            severity = 2 + min(2, flags_contributing)
        else:
            verdict = "innocent"
            confidence = 0.6
            severity = 1
            evidence.append(
                "all sizing checks passed; sizing not a contributing cause"
            )

        return ForensicsFinding(
            agent="sizing",
            verdict=verdict,
            confidence=confidence,
            evidence=evidence,
            suggested_action=suggested_action,
            severity=severity,
        )

    # ------------------------------------------------------------------
    # individual checks
    # ------------------------------------------------------------------
    def _position_size_fraction(
        self, risk_in: Dict[str, Any], risk_out: Dict[str, Any]
    ) -> Optional[float]:
        """Return position-size / bankroll as a fraction in [0, 1+].

        Looks at multiple candidate keys to handle both prediction-market
        snapshots (``adjusted_position_size_pct`` / ``bankroll`` /
        ``max_loss_if_wrong``) and crypto snapshots (``proposed_notional_usd``
        / ``equity_current_usd``). Returns None if no shape matches.
        """
        # Prediction-market: percent of bankroll is right there.
        pct = risk_out.get("adjusted_position_size_pct")
        if pct is not None:
            try:
                return float(pct) / 100.0
            except (TypeError, ValueError):
                pass

        # Crypto path: proposed notional vs equity (use whatever bankroll is).
        notional = _to_float(
            risk_in.get("proposed_notional_usd")
            or risk_out.get("position_notional_usd")
            or risk_out.get("max_loss_if_wrong"),
            default=0.0,
        )
        bankroll = _to_float(
            risk_in.get("bankroll")
            or risk_in.get("equity_current_usd")
            or risk_out.get("bankroll"),
            default=0.0,
        )
        if notional > 0.0 and bankroll > 0.0:
            return notional / bankroll
        return None

    def _check_recompute_drift(
        self, risk_in: Dict[str, Any], risk_out: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Re-run the calculator with snapshot inputs; compare EV.

        Returns (evidence_message_or_None, flag_or_None) where flag is
        one of (None, "primary").
        """
        # We only know how to recompute prediction-market shapes.
        market_dict = risk_in.get("market")
        calibrated = risk_in.get("calibrated_true_prob")
        bankroll = risk_in.get("bankroll")
        if market_dict is None or calibrated is None or bankroll is None:
            return None, None

        market = _maybe_build_market(market_dict)
        if market is None:
            return (
                "fresh recompute skipped — market dict could not be reconstructed",
                None,
            )

        try:
            from risk_management_agent.risk_engine import RiskCalculator
        except Exception as exc:  # noqa: BLE001 - defensive
            return (
                f"fresh recompute skipped — RiskCalculator import failed: {exc!r}",
                None,
            )

        try:
            calc = RiskCalculator()
            fresh = calc.calculate_base_metrics(
                market=market,
                calibrated_true_prob=float(calibrated),
                bankroll=float(bankroll),
                market_price=risk_in.get("market_price"),
                existing_open_positions=risk_in.get("existing_open_positions") or [],
            )
        except Exception as exc:  # noqa: BLE001 - defensive
            return (
                f"fresh recompute crashed: {exc!r}",
                None,
            )

        snap_ev = _to_float(risk_out.get("expected_value_estimate"), default=float("nan"))
        if snap_ev != snap_ev:  # missing
            return (
                "fresh recompute ran but snapshot has no expected_value_estimate to compare",
                None,
            )

        fresh_ev = float(fresh.expected_value_estimate)
        # Relative drift; guard zero by using max(|a|, |b|, eps) as denominator.
        denom = max(abs(fresh_ev), abs(snap_ev), 1e-9)
        rel_drift = abs(fresh_ev - snap_ev) / denom
        if rel_drift > EV_DRIFT_THRESHOLD:
            return (
                f"recompute drift: fresh EV={fresh_ev:.4f} vs snapshot EV={snap_ev:.4f} "
                f"({rel_drift:.1%} > {EV_DRIFT_THRESHOLD:.0%} threshold)",
                "primary",
            )
        return (
            f"recompute matched snapshot EV within {EV_DRIFT_THRESHOLD:.0%}",
            None,
        )

    def _check_fee_deduction(
        self,
        snap: TradeContextSnapshot,
        risk_in: Dict[str, Any],
        risk_out: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Detect missing Polymarket fee adjustment.

        When ``apply_polymarket_fees`` is wired in, the fee-adjusted prob
        differs from the raw calibrated prob whenever there's positive
        gross edge. We approximate the fee-adjusted prob from the
        snapshot's ``expected_value_estimate`` and notional; if it
        matches the raw prob to floating-point precision while there's
        a non-zero edge, the fee call was missing.
        """
        if not _looks_polymarket(snap):
            return None, None

        calibrated = _to_float(risk_in.get("calibrated_true_prob"), default=float("nan"))
        market_price = _to_float(
            risk_in.get("market_price")
            or (risk_in.get("market") or {}).get("implied_prob"),
            default=float("nan"),
        )
        if calibrated != calibrated or market_price != market_price:
            return None, None

        gross_edge = calibrated - market_price
        if gross_edge <= 0.0:
            # No gross edge means fees can't haircut anything.
            return None, None

        snap_ev = _to_float(risk_out.get("expected_value_estimate"), default=float("nan"))
        adjusted_pct = _to_float(risk_out.get("adjusted_position_size_pct"), default=float("nan"))
        bankroll = _to_float(risk_in.get("bankroll"), default=float("nan"))
        if (
            snap_ev != snap_ev
            or adjusted_pct != adjusted_pct
            or bankroll != bankroll
            or adjusted_pct <= 0.0
        ):
            return None, None

        # EV = notional * (fee_adjusted_prob - market_price) / market_price
        # → fee_adjusted_prob = EV * market_price / notional + market_price
        notional = bankroll * adjusted_pct / 100.0
        if notional <= 0.0 or market_price <= 0.0:
            return None, None
        implied_fee_prob = snap_ev * market_price / notional + market_price
        # If the implied prob equals the *raw* calibrated prob, no fee was applied.
        if abs(implied_fee_prob - calibrated) < 1e-6 and gross_edge > 1e-4:
            return (
                f"fee deduction missing: snapshot EV implies raw prob {calibrated:.4f} "
                f"on Polymarket trade with gross edge {gross_edge:.4f}",
                "primary",
            )
        return ("fee deduction looks applied", None)

    def _check_correlation_cluster(
        self,
        snap: TradeContextSnapshot,
        risk_in: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Count same-category open positions at decision time."""
        category = ""
        market = risk_in.get("market")
        if isinstance(market, dict):
            category = str(market.get("category") or "").strip().lower()
        if not category:
            # No category to cluster on — skip silently rather than flag noise.
            return None, None

        # Source 1: snapshot inputs.
        positions = risk_in.get("existing_open_positions")

        # Source 2: position store (fallback).
        if positions is None and self.position_store is not None:
            try:
                positions = self.position_store.list_open()
            except Exception as exc:  # noqa: BLE001 - tolerate flaky stores
                LOGGER.debug(
                    "sizing_forensics: position_store.list_open failed: %r", exc
                )
                positions = None

        if not positions:
            return None, None

        same_category = 0
        for pos in positions:
            pos_cat = ""
            if isinstance(pos, dict):
                pos_cat = str(pos.get("category") or pos.get("symbol") or "").strip().lower()
            else:
                pos_cat = str(getattr(pos, "category", None) or getattr(pos, "symbol", "")).strip().lower()
            if pos_cat == category:
                same_category += 1

        if same_category > CORRELATION_CLUSTER_THRESHOLD:
            return (
                f"correlation cluster: {same_category} open positions in category "
                f"'{category}' at decision time (> {CORRELATION_CLUSTER_THRESHOLD})",
                "contributing",
            )
        return (
            f"correlation OK: {same_category} same-category open positions",
            None,
        )

    def _check_liquidity_penalty(
        self,
        risk_in: Dict[str, Any],
        risk_out: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Flag low-volume markets sized as if liquid."""
        market = risk_in.get("market")
        if not isinstance(market, dict):
            return None, None

        volume = _to_float(market.get("volume_24h"), default=float("nan"))
        bid = _to_float(market.get("bid_price"), default=float("nan"))
        ask = _to_float(market.get("ask_price"), default=float("nan"))
        spread = ask - bid if (ask == ask and bid == bid) else float("nan")

        is_low_volume = volume == volume and volume < LOW_VOLUME_USD_THRESHOLD
        is_wide_spread = spread == spread and spread > WIDE_SPREAD_THRESHOLD
        looks_illiquid = is_low_volume or is_wide_spread
        if not looks_illiquid:
            return None, None

        applied = bool(risk_out.get("liquidity_penalty_applied", False))
        if not applied:
            return (
                f"liquidity penalty missing: volume_24h={volume:.0f} "
                f"spread={spread:.4f} but liquidity_penalty_applied=False",
                "contributing",
            )
        return ("liquidity penalty applied as expected", None)


__all__ = ["SizingForensicsAgent"]
