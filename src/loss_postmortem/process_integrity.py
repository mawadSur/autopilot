"""ProcessIntegrityAgent — Lane E forensics swarm A5.

This agent audits the *system*, not the trade. It looks at the snapshots
captured at signal / fill / breaker phases by ``live_supervisor`` and
asks: did our own machinery behave correctly during this trade?

Six checks are performed (each can produce a single evidence bullet):

1. **Breaker log coherence** — does the breaker snapshot's recorded
   ``recommended_action`` line up with what we observe on the
   :class:`Position` (e.g. the position's notes claim ``force_flat`` was
   applied but the snapshot recorded a permissive ``allow``)?
2. **Kill switch consistency** — was the kill-switch state coherent
   across the snapshot record and the live :class:`CircuitBreakerSet`
   when the agent runs?  A common failure is a kill-switch file getting
   created mid-trade but not surfacing into ``breaker_context`` because
   the breaker check ran *before* the file was placed.
3. **Shakedown coherence** — if a position carried a shakedown gate
   marker (``"live_mode_locked"`` etc.) the loss should have advanced
   the per-symbol shakedown counter; agents downstream of D1 are free
   to short-circuit. The check here only looks for *internal*
   contradictions, not policy.
4. **Race-condition trail** — Lane A's P0 #3 commit moved the per-symbol
   error counter into Redis under
   ``{ns}:errors:by_symbol:{date}``. We probe that hash for the trade's
   date and surface concentration (e.g. >5 entries in a single 5-minute
   window suggests contention). Redis access is best-effort: if the
   client is missing or raises, the check degrades to "unknown" with no
   evidence.
5. **Stop-loss execution price match** — when the breaker snapshot
   records a stop-loss ``trigger_price`` (or the position notes encode
   one), the actual ``exit_price`` should be within a few percent of
   it. If we executed materially off the trigger, that's a bug — likely
   a primary cause.
6. **Paper-vs-live divergence** — every snapshot has a ``notes`` field
   and (for fill) an ``exchange`` marker. They should agree. If signal
   says paper but fill claims live, or vice versa, the system stepped
   on its own foot.

Verdict bias is **conservative**: only declare ``primary_cause`` for
unambiguous bugs. A single small inconsistency is ``contributing``.

Constructor wiring is minimal — pass the :class:`TradeContextStore`
required by the base, and (optionally) the live :class:`PositionStore`
+ :class:`CircuitBreakerSet` so we can compare against the present
state. All optionals degrade gracefully when omitted.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loss_postmortem.base import (
    BaseForensicsAgent,
    DEFAULT_AGENT_TIMEOUT_S,
    ForensicsFinding,
)
from state.trade_context_store import (
    TradeContextSnapshot,
    TradeContextStore,
)

LOGGER = logging.getLogger(__name__)

# Stop-loss execution drift threshold. If the actual exit price diverges
# from the trigger by more than this, we declare a primary-cause bug.
# 1.5% leaves room for normal slippage on a market exit while flagging
# obvious "the stop-loss didn't fire at the right level" cases.
_STOPLOSS_PRICE_DRIFT_PRIMARY = 0.015

# A drift between 0.4% and 1.5% is a contributing factor (worth noting,
# not a definite bug — could be slippage in a fast tape).
_STOPLOSS_PRICE_DRIFT_CONTRIB = 0.004

# Race-condition surfacing: how many error increments in the same UTC
# date qualify as a "concurrency-suspect" cluster. Below this threshold
# we say nothing.
_RACE_CONCURRENCY_THRESHOLD = 5

# Substrings on snapshots indicating paper-mode origin.
_PAPER_MARKERS: tuple[str, ...] = (
    "paper",
    "paper-deferred-fill",
    "coinbase-paper",
)

# Substrings that together with breaker_context mean a forced flat.
_FORCED_FLAT_REASON_SIGNALS: tuple[str, ...] = (
    "force_flat",
    "forced_flat",
    "kill_switch",
    "daily_loss",
    "halt_new_entries",
)


class ProcessIntegrityAgent(BaseForensicsAgent):
    """Forensics specialist focused on system-side integrity (A5).

    Unlike A1-A4 (which look at the trade), this agent looks at the
    machinery that drove the trade. It is the most likely source of
    "we have a bug" findings.
    """

    agent_name = "process"

    def __init__(
        self,
        *,
        context_store: TradeContextStore,
        position_store: Optional[Any] = None,
        circuit_breakers: Optional[Any] = None,
        redis_client: Optional[Any] = None,
        namespace: str = "autopilot",
        timeout_s: float = DEFAULT_AGENT_TIMEOUT_S,
    ) -> None:
        super().__init__(context_store=context_store, timeout_s=timeout_s)
        self.position_store = position_store
        self.circuit_breakers = circuit_breakers
        self.redis_client = redis_client
        self.namespace = namespace

    # ------------------------------------------------------------------
    # public contract
    # ------------------------------------------------------------------
    def investigate(self, trade_id: str) -> ForensicsFinding:
        snapshots = self.context_store.get_snapshots(trade_id)
        evidence: List[str] = []

        # Defense-in-depth: snapshot gaps are themselves a process flag.
        gap_evidence = self._check_snapshot_gaps(snapshots)
        if gap_evidence:
            evidence.extend(gap_evidence)

        signal_snap = snapshots.get("signal")
        fill_snap = snapshots.get("fill")
        breaker_snap = snapshots.get("breaker")

        # 1. Breaker logs vs snapshot coherence.
        flag_breaker, ev = self._check_breaker_coherence(
            breaker_snap, position=self._safe_get_position(trade_id)
        )
        if ev:
            evidence.append(ev)

        # 2. Kill switch consistency.
        flag_killswitch, ev = self._check_kill_switch_consistency(
            breaker_snap, signal_snap
        )
        if ev:
            evidence.append(ev)

        # 3. Shakedown coherence.
        flag_shakedown, ev = self._check_shakedown_coherence(
            signal_snap, self._safe_get_position(trade_id)
        )
        if ev:
            evidence.append(ev)

        # 4. Race-condition trail.
        flag_race, ev = self._check_race_condition_trail(
            signal_snap or fill_snap or breaker_snap
        )
        if ev:
            evidence.append(ev)

        # 5. Stop-loss execution price match.
        flag_stop, ev = self._check_stoploss_price_match(
            breaker_snap, self._safe_get_position(trade_id)
        )
        if ev:
            evidence.append(ev)

        # 6. Paper-vs-live divergence.
        flag_paper, ev = self._check_paper_live_divergence(
            signal_snap, fill_snap, breaker_snap, self._safe_get_position(trade_id)
        )
        if ev:
            evidence.append(ev)

        return self._render_verdict(
            evidence=evidence,
            primary_flags={
                "stop_drift": flag_stop == "primary",
                "kill_switch_supposed_to_trip": flag_killswitch == "primary",
                "breaker_decision_lost": flag_breaker == "primary",
                "paper_live_divergence": flag_paper == "primary",
            },
            contributing_flags={
                "stop_drift": flag_stop == "contributing",
                "kill_switch_minor": flag_killswitch == "contributing",
                "breaker_minor": flag_breaker == "contributing",
                "race": flag_race == "contributing",
                "shakedown": flag_shakedown == "contributing",
                "paper_live_minor": flag_paper == "contributing",
            },
            trade_id=trade_id,
        )

    # ------------------------------------------------------------------
    # individual checks
    # ------------------------------------------------------------------
    def _check_snapshot_gaps(
        self, snapshots: Dict[str, TradeContextSnapshot]
    ) -> List[str]:
        """Emit evidence bullets for missing phases. No verdict on its own."""
        bullets: List[str] = []
        # We expect at least signal+fill for a normal close, or breaker
        # for a forced flat. A trade with ZERO snapshots is an upstream
        # bug — the agent records that but doesn't escalate.
        if not snapshots:
            bullets.append(
                "no snapshots recorded for trade — signal/fill/breaker capture failed upstream"
            )
            return bullets
        if "signal" not in snapshots:
            bullets.append("signal snapshot missing — supervisor capture path skipped it")
        if "fill" not in snapshots and "breaker" not in snapshots:
            bullets.append(
                "neither fill nor breaker snapshot recorded — close path didn't produce one"
            )
        return bullets

    def _check_breaker_coherence(
        self,
        breaker_snap: Optional[TradeContextSnapshot],
        *,
        position: Optional[Any],
    ) -> tuple[Optional[str], Optional[str]]:
        """Was the breaker decision honoured?

        If position notes claim a forced flat but ``breaker_context`` says
        ``recommended_action="allow"``, we have a lost-decision bug.
        """
        if breaker_snap is None:
            # No breaker snapshot doesn't itself indicate breaker
            # incoherence — only forced-flat closes produce one.
            return None, None
        br = breaker_snap.breaker_context or {}
        action = str(br.get("recommended_action") or "").strip()
        reason = str(br.get("reason") or "").strip().lower()
        notes = ""
        if position is not None and getattr(position, "notes", None):
            notes = str(position.notes).lower()

        # Inconsistency: breaker says allow but position closed under a
        # forced-flat reason. That means the breaker recorded a state
        # that didn't match the actual close decision.
        if action == "allow" and any(sig in notes for sig in _FORCED_FLAT_REASON_SIGNALS):
            return (
                "primary",
                f"breaker snapshot recorded action=allow but position closed "
                f"as forced-flat (notes={notes!r})",
            )

        # Inconsistency: breaker recorded force_flat but neither tripped
        # list nor reason populated — that's missing structured data.
        if action == "force_flat":
            tripped = list(br.get("tripped") or [])
            if not tripped and not reason:
                return (
                    "contributing",
                    "breaker recorded force_flat with empty tripped+reason — "
                    "structured logging gap",
                )

        return None, None

    def _check_kill_switch_consistency(
        self,
        breaker_snap: Optional[TradeContextSnapshot],
        signal_snap: Optional[TradeContextSnapshot],
    ) -> tuple[Optional[str], Optional[str]]:
        """Was the kill switch state coherent at the snapshot's record time?

        Two patterns to detect:
        - Breaker_context says ``kill_switch`` tripped but the kill_switch
          file does NOT currently exist (and we have a live
          :class:`CircuitBreakerSet`) — could be transient (file removed
          between trip and now) or a logging mismatch.
        - Breaker snapshot indicates kill_switch was *expected* to trip
          (e.g. notes say ``"kill_switch"`` but breaker_context.tripped
          omits ``"kill_switch"``) — primary bug, the breaker check
          missed a real condition.
        """
        # Trip claimed by breaker_context.
        ks_in_breaker = False
        if breaker_snap is not None:
            br = breaker_snap.breaker_context or {}
            tripped = [str(t) for t in (br.get("tripped") or [])]
            ks_in_breaker = "kill_switch" in tripped

        notes_imply_ks = False
        for snap in (signal_snap, breaker_snap):
            if snap is None:
                continue
            if snap.notes and "kill_switch" in str(snap.notes).lower():
                notes_imply_ks = True

        if notes_imply_ks and not ks_in_breaker:
            return (
                "primary",
                "kill_switch referenced in snapshot notes but breaker_context "
                "did not record it as tripped — supposed to trip but didn't",
            )

        # If the live circuit breaker set is wired, sanity-check current
        # file existence against the breaker_context claim.
        if ks_in_breaker and self.circuit_breakers is not None:
            try:
                live_tripped = bool(self.circuit_breakers.is_kill_switch_tripped())
            except Exception:  # noqa: BLE001 - degrade gracefully
                live_tripped = False
            if not live_tripped:
                # Could be the operator cleared the file post-trade;
                # contributing only.
                return (
                    "contributing",
                    "breaker_context recorded kill_switch trip but file is "
                    "currently absent (cleared post-trade or logging drift)",
                )

        return None, None

    def _check_shakedown_coherence(
        self,
        signal_snap: Optional[TradeContextSnapshot],
        position: Optional[Any],
    ) -> tuple[Optional[str], Optional[str]]:
        """A loss on a shakedown-locked trade should leave a marker.

        We can't audit the shakedown counter from here (no access to the
        live shakedown JSON), so the check is narrow: if the position
        notes say ``live_mode_locked`` but the signal snapshot's
        ``risk_metrics_input`` shows zero proposed_notional (i.e. we
        didn't attempt to size), something inverted the gate.
        """
        if position is None or signal_snap is None:
            return None, None
        notes = str(getattr(position, "notes", "") or "").lower()
        if "live_mode_locked" not in notes:
            return None, None
        rmi = signal_snap.risk_metrics_input or {}
        proposed = float(rmi.get("proposed_notional_usd") or 0.0)
        if proposed <= 0:
            return (
                "contributing",
                "live_mode_locked position has zero proposed_notional in signal "
                "snapshot — shakedown gate may have inverted",
            )
        return None, None

    def _check_race_condition_trail(
        self, any_snap: Optional[TradeContextSnapshot]
    ) -> tuple[Optional[str], Optional[str]]:
        """Probe Redis ``errors:by_symbol:{date}`` for contention.

        Heuristic: a single trade losing on a date with >threshold
        increments to its symbol counter implies repeated errors that
        day — concurrent writes are plausible. We can't see HINCRBY
        timestamps from a HASH, so we only check the count and treat it
        as soft evidence. Degrades to no-op if Redis is unavailable.
        """
        if self.redis_client is None or any_snap is None:
            return None, None
        symbol = any_snap.symbol or ""
        date_part = self._extract_date(any_snap.captured_at_utc)
        if not symbol or not date_part:
            return None, None
        key = f"{self.namespace}:errors:by_symbol:{date_part}"
        try:
            raw = self.redis_client.hget(key, symbol)
        except Exception:  # noqa: BLE001 - degrade gracefully
            return None, None
        if raw is None:
            return None, None
        try:
            count = int(raw)
        except (TypeError, ValueError):
            return None, None
        if count >= _RACE_CONCURRENCY_THRESHOLD:
            return (
                "contributing",
                f"error counter for {symbol} on {date_part} = {count} "
                f"(>= {_RACE_CONCURRENCY_THRESHOLD}); "
                "concurrent error writes plausible",
            )
        return None, None

    def _check_stoploss_price_match(
        self,
        breaker_snap: Optional[TradeContextSnapshot],
        position: Optional[Any],
    ) -> tuple[Optional[str], Optional[str]]:
        """Did the stop-loss exit at (close to) its trigger price?

        Only fires when (a) we have a breaker snapshot whose
        ``risk_metrics_input`` carries a ``stop_loss_trigger`` (or the
        breaker_context records one), and (b) the closed position's
        ``exit_price`` is materially off it.
        """
        if breaker_snap is None or position is None:
            return None, None
        exit_price = getattr(position, "exit_price", None)
        if exit_price is None:
            return None, None
        try:
            exit_price = float(exit_price)
        except (TypeError, ValueError):
            return None, None

        # Try several conventions for surfacing the trigger price.
        candidates: List[float] = []
        rmi = breaker_snap.risk_metrics_input or {}
        for key in ("stop_loss_trigger", "stop_trigger_price", "stop_loss_price"):
            v = rmi.get(key)
            if v is not None:
                try:
                    candidates.append(float(v))
                except (TypeError, ValueError):
                    pass
        br = breaker_snap.breaker_context or {}
        for key in ("stop_loss_trigger", "trigger_price"):
            v = br.get(key)
            if v is not None:
                try:
                    candidates.append(float(v))
                except (TypeError, ValueError):
                    pass

        if not candidates:
            return None, None
        trigger = candidates[0]
        if trigger <= 0:
            return None, None
        drift = abs(exit_price - trigger) / trigger
        if drift >= _STOPLOSS_PRICE_DRIFT_PRIMARY:
            return (
                "primary",
                f"stop-loss exit_price {exit_price:.6f} drifted "
                f"{drift * 100:.2f}% from trigger {trigger:.6f} — "
                "stop did not honour its level",
            )
        if drift >= _STOPLOSS_PRICE_DRIFT_CONTRIB:
            return (
                "contributing",
                f"stop-loss exit_price {exit_price:.6f} drifted "
                f"{drift * 100:.2f}% from trigger {trigger:.6f}",
            )
        return None, None

    def _check_paper_live_divergence(
        self,
        signal_snap: Optional[TradeContextSnapshot],
        fill_snap: Optional[TradeContextSnapshot],
        breaker_snap: Optional[TradeContextSnapshot],
        position: Optional[Any],
    ) -> tuple[Optional[str], Optional[str]]:
        """Do all available signals agree on paper-vs-live mode?

        Paper indicators (any of these):
        - ``snapshot.notes`` contains "paper"
        - ``fill_snap.risk_metrics_output["exchange"]`` contains "paper"
        - ``position.exchange`` contains "paper"
        - ``position.notes`` contains "paper"

        We collect a per-source marker (paper / live / unknown). If two
        non-unknown sources disagree, that's divergence.
        """

        def classify(snap: Optional[TradeContextSnapshot]) -> str:
            if snap is None:
                return "unknown"
            if snap.notes and any(m in str(snap.notes).lower() for m in _PAPER_MARKERS):
                return "paper"
            risk_out = snap.risk_metrics_output or {}
            exchange = str(risk_out.get("exchange") or "").lower()
            if exchange:
                return "paper" if any(m in exchange for m in _PAPER_MARKERS) else "live"
            return "unknown"

        sources: Dict[str, str] = {
            "signal_snapshot": classify(signal_snap),
            "fill_snapshot": classify(fill_snap),
            "breaker_snapshot": classify(breaker_snap),
        }
        if position is not None:
            ex = str(getattr(position, "exchange", "") or "").lower()
            notes = str(getattr(position, "notes", "") or "").lower()
            if ex and any(m in ex for m in _PAPER_MARKERS):
                sources["position"] = "paper"
            elif ex:
                sources["position"] = "live"
            elif any(m in notes for m in _PAPER_MARKERS):
                sources["position"] = "paper"
            else:
                sources["position"] = "unknown"
        else:
            sources["position"] = "unknown"

        decided = {k: v for k, v in sources.items() if v != "unknown"}
        if len(decided) < 2:
            # Not enough signal to call divergence either way.
            return None, None
        unique = set(decided.values())
        if len(unique) == 1:
            return None, None  # all sources agree

        # Disagreement. Treat as primary only when *all* declared sources
        # split (e.g. signal=paper, fill=live), otherwise contributing.
        if len(unique) >= 2 and len(decided) >= 3:
            return (
                "primary",
                f"paper-vs-live divergence across snapshots: {decided!r}",
            )
        return (
            "contributing",
            f"paper-vs-live disagreement between snapshots: {decided!r}",
        )

    # ------------------------------------------------------------------
    # verdict rendering
    # ------------------------------------------------------------------
    def _render_verdict(
        self,
        *,
        evidence: List[str],
        primary_flags: Dict[str, bool],
        contributing_flags: Dict[str, bool],
        trade_id: str,
    ) -> ForensicsFinding:
        primary_count = sum(1 for v in primary_flags.values() if v)
        contrib_count = sum(1 for v in contributing_flags.values() if v)

        if primary_count >= 1:
            confidence = min(0.85, 0.6 + 0.1 * primary_count)
            severity = 4 if primary_count == 1 else 5
            suggested_action = self._suggested_action_for_primary(primary_flags, trade_id)
            return ForensicsFinding(
                agent="process",
                verdict="primary_cause",
                confidence=confidence,
                evidence=evidence,
                suggested_action=suggested_action,
                severity=severity,
            )
        if contrib_count >= 1:
            confidence = min(0.55, 0.3 + 0.1 * contrib_count)
            severity = 2 + min(2, contrib_count - 1)
            return ForensicsFinding(
                agent="process",
                verdict="contributing",
                confidence=confidence,
                evidence=evidence,
                suggested_action={
                    "type": "investigate_breaker_log",
                    "trade_id": trade_id,
                },
                severity=max(2, severity),
            )
        # No flags, but we may still have advisory evidence (snapshot
        # gaps). Innocent verdict, low confidence so synthesizer treats
        # it as "agent had nothing to say".
        return ForensicsFinding(
            agent="process",
            verdict="innocent",
            confidence=0.2,
            evidence=evidence,
            suggested_action=None,
            severity=1,
        )

    @staticmethod
    def _suggested_action_for_primary(
        primary_flags: Dict[str, bool], trade_id: str
    ) -> Dict[str, Any]:
        if primary_flags.get("stop_drift"):
            return {
                "type": "investigate_stoploss_execution",
                "trade_id": trade_id,
                "details": "exit_price drifted materially from trigger",
            }
        if primary_flags.get("kill_switch_supposed_to_trip"):
            return {
                "type": "audit_kill_switch_path",
                "trade_id": trade_id,
            }
        if primary_flags.get("breaker_decision_lost"):
            return {
                "type": "investigate_breaker_log",
                "trade_id": trade_id,
            }
        if primary_flags.get("paper_live_divergence"):
            return {
                "type": "fix_paper_live_divergence",
                "trade_id": trade_id,
            }
        return {"type": "investigate_breaker_log", "trade_id": trade_id}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _safe_get_position(self, trade_id: str) -> Optional[Any]:
        """Best-effort fetch of the closed position. None on any failure."""
        if self.position_store is None:
            return None
        try:
            return self.position_store.get(trade_id)
        except Exception:  # noqa: BLE001 - degrade gracefully
            return None

    @staticmethod
    def _extract_date(captured_at_utc: Optional[str]) -> Optional[str]:
        if not captured_at_utc:
            return None
        try:
            dt = datetime.fromisoformat(str(captured_at_utc))
        except (TypeError, ValueError):
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


__all__ = ["ProcessIntegrityAgent"]
