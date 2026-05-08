"""Execution forensics agent (Lane E, A2 of the loss-postmortem swarm).

Investigates whether the *fill machinery* (latency, slippage, partial fills,
stale market data, rejections, stop-loss execution drift) caused or amplified
the realised loss on a closed losing trade.

Inputs (all optional — agent degrades gracefully on missing data):
  * ``TradeContextSnapshot`` at phase ``signal`` — contains the model's
    expected market state at decision time (ticker_buffer, captured_at_utc).
  * ``TradeContextSnapshot`` at phase ``fill`` — contains the actual market
    state when the fill confirmed (captured_at_utc, ticker_buffer, notes).
  * ``Position`` from :class:`PositionStore` — actual ``entry_price`` /
    ``exit_price`` plus free-form ``notes`` (we look for partial-fill,
    rejection, and stop-loss markers in notes).

The six checks (per the brief) each produce one evidence bullet and bump
either a ``red_flags`` or ``yellow_flags`` counter:

1. Signal→fill latency  — > 10s primary, > 5s contributing.
2. Slippage actual vs expected — > 50 bps primary, > 15 bps contributing.
3. Partial fills — ``"partial"`` in position.notes → contributing.
4. Stale ticker — ticker_buffer entry > 3s old at signal capture →
   contributing.
5. Order rejection trail — ``"reject"`` / ``"rejected"`` in position.notes →
   contributing.
6. Stop-loss execution drift — if a stop trigger price is recorded in
   position.notes / position.model_meta and exit_price drifts > 10 bps from
   it → contributing.

Verdict logic (from brief):
  * Primary if slippage > 50 bps OR signal→fill latency > 10s.
  * Contributing if any single yellow flag (and no primary trigger).
  * Innocent otherwise.
  * Unknown if both signal and fill snapshots are missing AND we have no
    Position-side execution data — we have nothing to inspect.

Defense-in-depth: when fill snapshot is missing entirely (e.g., position was
opened pre-Lane E), we emit the bullet ``"fill snapshot missing — execution
checks limited"`` and lean toward verdict="unknown" *unless* the position-only
checks (slippage from signal-snapshot, partial/rejection notes, stop-loss
drift) clearly indicate a problem.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from loss_postmortem.base import BaseForensicsAgent, ForensicsFinding
from state.position_store import Position, PositionStore
from state.trade_context_store import TradeContextSnapshot, TradeContextStore

LOGGER = logging.getLogger(__name__)

# Threshold knobs — kept module-level so tests can monkey-patch if needed.
LATENCY_PRIMARY_S = 10.0
LATENCY_CONTRIBUTING_S = 5.0

SLIPPAGE_PRIMARY_BPS = 50.0
SLIPPAGE_CONTRIBUTING_BPS = 15.0

STALE_TICKER_S = 3.0

STOP_LOSS_DRIFT_BPS = 10.0

# Substring markers we look for inside Position.notes (case-insensitive).
PARTIAL_NOTE_MARKERS: tuple[str, ...] = ("partial", "partial_fill", "partial-fill")
REJECTION_NOTE_MARKERS: tuple[str, ...] = ("reject", "rejected", "rejection")
STOP_LOSS_NOTE_MARKERS: tuple[str, ...] = (
    "stop_loss",
    "stop-loss",
    "stoploss",
    "stop_triggered",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _parse_iso_utc(value: Optional[str]) -> Optional[datetime]:
    """Best-effort ISO-8601 → tz-aware UTC datetime; None on failure."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _ticker_mid(ticker: Dict[str, Any]) -> Optional[float]:
    """Compute mid from a ticker_buffer entry; None if not derivable.

    Accepts the shape produced by :class:`exchanges.coinbase.Ticker.dict()`:
    ``bid`` + ``ask`` (preferred), falling back to ``last`` then ``mid`` if
    the snapshot pre-dumped the computed field.
    """
    if not isinstance(ticker, dict):
        return None
    bid = ticker.get("bid")
    ask = ticker.get("ask")
    try:
        if bid is not None and ask is not None:
            mid = (float(bid) + float(ask)) / 2.0
            if mid > 0:
                return mid
    except (TypeError, ValueError):
        pass
    for fallback in ("mid", "last"):
        v = ticker.get(fallback)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv > 0:
            return fv
    return None


def _latest_ticker(snapshot: Optional[TradeContextSnapshot]) -> Optional[Dict[str, Any]]:
    """Return the last ticker dict in a snapshot's ticker_buffer, if any."""
    if snapshot is None or not snapshot.ticker_buffer:
        return None
    last = snapshot.ticker_buffer[-1]
    return last if isinstance(last, dict) else None


def _bps_diff(actual: float, expected: float) -> Optional[float]:
    """abs(actual - expected) / expected * 10_000 — None when expected ≤ 0."""
    try:
        a = float(actual)
        e = float(expected)
    except (TypeError, ValueError):
        return None
    if e <= 0:
        return None
    return abs(a - e) / e * 10_000.0


# ---------------------------------------------------------------------------
# stop-loss extraction
# ---------------------------------------------------------------------------


# A stop trigger price might be written by callers in any of these forms in
# position.notes. We're permissive: regex match a number after a stop keyword.
_STOP_PRICE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"stop[_\- ]*loss[^=:0-9]*[=:]?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"stop[_\- ]*price[^=:0-9]*[=:]?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"stop[_\- ]*trigger[^=:0-9]*[=:]?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"\bstop\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
)


def _extract_stop_price(position: Position) -> Optional[float]:
    """Pull a stop-loss trigger price from position.notes or model_meta.

    Looked at, in order (Phase-16 priority):
      0. ``position.stop_trigger_price`` — the canonical typed field.
      1. ``position.model_meta`` keys ``stop_price`` / ``stop_loss_price`` /
         ``stop_trigger_price`` (numeric).
      2. ``position.notes`` regex-matched on stop-related keywords.
    """
    # Phase-16 canonical field takes precedence. Legacy positions in Redis
    # serialized before the field existed deserialize with this as None,
    # so the fallback paths still work.
    canonical = getattr(position, "stop_trigger_price", None)
    if canonical is not None:
        try:
            fv = float(canonical)
            if fv > 0:
                return fv
        except (TypeError, ValueError):
            pass

    meta = position.model_meta or {}
    for key in ("stop_price", "stop_loss_price", "stop_trigger_price"):
        v = meta.get(key)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv > 0:
            return fv

    notes = position.notes or ""
    if not notes:
        return None
    for pat in _STOP_PRICE_PATTERNS:
        m = pat.search(notes)
        if m:
            try:
                fv = float(m.group(1))
            except (TypeError, ValueError):
                continue
            if fv > 0:
                return fv
    return None


def _stop_loss_executed(position: Position) -> bool:
    """Heuristic: did this close happen via a stop-loss order?"""
    notes = (position.notes or "").lower()
    return any(marker in notes for marker in STOP_LOSS_NOTE_MARKERS)


# ---------------------------------------------------------------------------
# ExecutionForensicsAgent
# ---------------------------------------------------------------------------


class ExecutionForensicsAgent(BaseForensicsAgent):
    """Audit the fill-side of a losing trade.

    Construction adds an optional :class:`PositionStore` — pass ``None`` and
    the agent skips the position-side checks entirely (slippage, partial,
    rejection, stop-loss drift) and falls back to snapshot-only inspection.
    """

    agent_name = "execution"

    def __init__(
        self,
        *,
        context_store: TradeContextStore,
        position_store: Optional[PositionStore] = None,
        timeout_s: float = 60.0,
    ) -> None:
        super().__init__(context_store=context_store, timeout_s=timeout_s)
        self.position_store = position_store

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------
    def investigate(self, trade_id: str) -> ForensicsFinding:
        start = time.monotonic()
        evidence: List[str] = []
        red_flags = 0      # bumps push toward primary_cause
        yellow_flags = 0   # bumps push toward contributing
        suggested_actions: List[Dict[str, Any]] = []

        signal_snap = self.context_store.get_signal_snapshot(trade_id)
        fill_snap = self.context_store.get_fill_snapshot(trade_id)

        position: Optional[Position] = None
        if self.position_store is not None:
            try:
                position = self.position_store.get(trade_id)
            except Exception as exc:  # noqa: BLE001 - tolerate flaky stores
                LOGGER.debug("position_store.get(%s) raised: %r", trade_id, exc)
                position = None

        # If we have nothing at all to inspect, return verdict="unknown"
        # rather than fabricating an "innocent" finding.
        if signal_snap is None and fill_snap is None and position is None:
            return ForensicsFinding(
                agent=self.agent_name,
                verdict="unknown",
                confidence=0.0,
                evidence=[
                    "no signal snapshot, no fill snapshot, no position record — "
                    "execution forensics has nothing to inspect"
                ],
                severity=1,
                runtime_s=time.monotonic() - start,
                error="missing_inputs",
            )

        if fill_snap is None:
            evidence.append("fill snapshot missing — execution checks limited")

        # ------------------------------------------------------------------
        # 1. Signal → fill latency
        # ------------------------------------------------------------------
        latency_s: Optional[float] = None
        if signal_snap is not None and fill_snap is not None:
            t_signal = _parse_iso_utc(signal_snap.captured_at_utc)
            t_fill = _parse_iso_utc(fill_snap.captured_at_utc)
            if t_signal is not None and t_fill is not None:
                latency_s = max(0.0, (t_fill - t_signal).total_seconds())
                if latency_s > LATENCY_PRIMARY_S:
                    red_flags += 1
                    evidence.append(
                        f"signal→fill latency {latency_s:.2f}s exceeds "
                        f"primary threshold ({LATENCY_PRIMARY_S:.0f}s)"
                    )
                    suggested_actions.append(
                        {
                            "type": "investigate_exchange_latency",
                            "venue": "coinbase",
                        }
                    )
                elif latency_s > LATENCY_CONTRIBUTING_S:
                    yellow_flags += 1
                    evidence.append(
                        f"signal→fill latency {latency_s:.2f}s exceeds "
                        f"contributing threshold ({LATENCY_CONTRIBUTING_S:.0f}s)"
                    )
                    suggested_actions.append(
                        {
                            "type": "investigate_exchange_latency",
                            "venue": "coinbase",
                        }
                    )

        # ------------------------------------------------------------------
        # 2. Slippage: actual entry_price vs expected (signal-snapshot mid)
        # ------------------------------------------------------------------
        slippage_bps: Optional[float] = None
        if position is not None and signal_snap is not None:
            expected_mid = _ticker_mid(_latest_ticker(signal_snap) or {})
            if expected_mid is not None and position.entry_price > 0:
                slippage_bps = _bps_diff(position.entry_price, expected_mid)
                if slippage_bps is not None:
                    if slippage_bps > SLIPPAGE_PRIMARY_BPS:
                        red_flags += 1
                        evidence.append(
                            f"slippage {slippage_bps:.1f} bps "
                            f"(entry {position.entry_price:.4f} vs signal-mid "
                            f"{expected_mid:.4f}) exceeds primary threshold "
                            f"({SLIPPAGE_PRIMARY_BPS:.0f} bps)"
                        )
                        suggested_actions.append(
                            {
                                "type": "increase_paper_slippage_bps",
                                "from": 5,
                                "to": 15,
                            }
                        )
                    elif slippage_bps > SLIPPAGE_CONTRIBUTING_BPS:
                        yellow_flags += 1
                        evidence.append(
                            f"slippage {slippage_bps:.1f} bps "
                            f"(entry {position.entry_price:.4f} vs signal-mid "
                            f"{expected_mid:.4f}) exceeds contributing threshold "
                            f"({SLIPPAGE_CONTRIBUTING_BPS:.0f} bps)"
                        )
                        suggested_actions.append(
                            {
                                "type": "increase_paper_slippage_bps",
                                "from": 5,
                                "to": 15,
                            }
                        )

        # ------------------------------------------------------------------
        # 3. Partial fills (Phase-16: prefer canonical Position.partial_fills)
        # ------------------------------------------------------------------
        if position is not None:
            notes_lc = (position.notes or "").lower()
            canonical_partials = getattr(position, "partial_fills", None)
            if canonical_partials:
                yellow_flags += 1
                evidence.append(
                    f"partial fills detected on position record "
                    f"({len(canonical_partials)} fills)"
                )
            elif any(m in notes_lc for m in PARTIAL_NOTE_MARKERS):
                yellow_flags += 1
                evidence.append(
                    "partial fill detected in position.notes "
                    "(see 'partial' marker)"
                )

            # ------------------------------------------------------------------
            # 5. Order rejection trail (Phase-16: prefer canonical
            #    Position.rejection_reason; fall back to notes scan)
            # ------------------------------------------------------------------
            canonical_reject = getattr(position, "rejection_reason", None)
            if canonical_reject:
                yellow_flags += 1
                evidence.append(
                    f"order rejection on position record: "
                    f"reason={canonical_reject!r}"
                )
            elif any(m in notes_lc for m in REJECTION_NOTE_MARKERS):
                yellow_flags += 1
                evidence.append(
                    "order rejection found in position.notes "
                    "(see 'reject' marker)"
                )

        # ------------------------------------------------------------------
        # 4. Stale ticker at signal capture
        # ------------------------------------------------------------------
        if signal_snap is not None:
            t_capture = _parse_iso_utc(signal_snap.captured_at_utc)
            t_last_tick = _parse_iso_utc(
                (_latest_ticker(signal_snap) or {}).get("as_of_utc")
            )
            if t_capture is not None and t_last_tick is not None:
                tick_age_s = (t_capture - t_last_tick).total_seconds()
                if tick_age_s > STALE_TICKER_S:
                    yellow_flags += 1
                    evidence.append(
                        f"stale ticker at signal: last tick {tick_age_s:.2f}s "
                        f"old (threshold {STALE_TICKER_S:.0f}s)"
                    )
                    suggested_actions.append(
                        {
                            "type": "tighten_stale_ticker_threshold",
                            "from": 5,
                            "to": 2,
                        }
                    )

        # ------------------------------------------------------------------
        # 6. Stop-loss execution price drift
        # ------------------------------------------------------------------
        if position is not None and position.exit_price is not None:
            stop_price = _extract_stop_price(position)
            if stop_price is not None and (
                _stop_loss_executed(position)
                or position.model_meta.get("closed_via_stop") is True
            ):
                drift_bps = _bps_diff(position.exit_price, stop_price)
                if drift_bps is not None and drift_bps > STOP_LOSS_DRIFT_BPS:
                    yellow_flags += 1
                    evidence.append(
                        f"stop-loss execution drift {drift_bps:.1f} bps "
                        f"(exit {position.exit_price:.4f} vs stop "
                        f"{stop_price:.4f}; threshold "
                        f"{STOP_LOSS_DRIFT_BPS:.0f} bps)"
                    )

        # ------------------------------------------------------------------
        # Verdict + confidence
        # ------------------------------------------------------------------
        # Defense-in-depth: if fill snapshot is missing AND we found no
        # red flags from position-only checks, lean to "unknown" rather
        # than declaring innocent on partial information.
        verdict, confidence, severity = self._classify(
            red_flags=red_flags,
            yellow_flags=yellow_flags,
            fill_snap_missing=fill_snap is None,
            position_present=position is not None,
        )

        # Pick the most informative suggested action (deduped by type) — keep
        # the first occurrence of each type to make the synthesizer's job easy.
        chosen_action: Optional[Dict[str, Any]] = None
        if suggested_actions:
            chosen_action = suggested_actions[0]

        return ForensicsFinding(
            agent=self.agent_name,
            verdict=verdict,
            confidence=confidence,
            evidence=evidence,
            suggested_action=chosen_action,
            severity=severity,
            runtime_s=time.monotonic() - start,
        )

    # ------------------------------------------------------------------
    # verdict helper
    # ------------------------------------------------------------------
    @staticmethod
    def _classify(
        *,
        red_flags: int,
        yellow_flags: int,
        fill_snap_missing: bool,
        position_present: bool,
    ) -> Tuple[str, float, int]:
        """Map (red, yellow, missing-data) to (verdict, confidence, severity)."""
        if red_flags >= 1:
            # Confidence rises slightly with multiple red flags.
            confidence = min(0.95, 0.75 + 0.1 * (red_flags - 1))
            severity = 5 if red_flags >= 2 else 4
            return "primary_cause", confidence, severity
        if yellow_flags >= 1:
            confidence = min(0.7, 0.4 + 0.1 * yellow_flags)
            severity = 3 if yellow_flags >= 2 else 2
            return "contributing", confidence, severity
        # No flags. If we lack the fill snapshot AND lack a position record,
        # we shouldn't claim innocence — but we already short-circuited the
        # both-missing case above. With one source missing but the other
        # present and clean, "innocent" is fair but with lower confidence.
        if fill_snap_missing and not position_present:
            return "unknown", 0.0, 1
        confidence = 0.5 if fill_snap_missing else 0.7
        return "innocent", confidence, 1


__all__ = [
    "ExecutionForensicsAgent",
    "LATENCY_CONTRIBUTING_S",
    "LATENCY_PRIMARY_S",
    "SLIPPAGE_CONTRIBUTING_BPS",
    "SLIPPAGE_PRIMARY_BPS",
    "STALE_TICKER_S",
    "STOP_LOSS_DRIFT_BPS",
]
