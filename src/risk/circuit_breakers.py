"""Hard safety circuit breakers for the live trading path.

These are the last line of defense between the model's allow-trade verdict
and the exchange's ``place_order`` call. Each breaker is a binary go/no-go
check; multiple may trip on the same call, in which case the most severe
recommended action wins (``force_flat`` > ``halt_new_entries`` > ``allow``).

The soft-penalty risk layer in
``src.risk_management_agent.risk_engine.RiskCalculator`` shrinks position
size; circuit breakers can REJECT the (already-shrunk) trade outright.

Configured via constructor args, with environment-variable fallbacks
matching the names documented in ``.env.example`` Section 4:

    DAILY_LOSS_LIMIT_USD
    MAX_DRAWDOWN_PCT
    MAX_TOTAL_NOTIONAL_USD
    MAX_PER_SYMBOL_NOTIONAL_USD
    KILL_SWITCH_FILE

When both the constructor arg and the env var are unset, the corresponding
breaker is disabled and a one-shot INFO log is emitted on the first
``check`` call. Disabled breakers always pass.

This module is pure logic: the only I/O is an existence check on the
kill-switch file path.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


__all__ = ["CircuitBreakerSet", "CircuitBreakerVerdict", "DecisionContext"]


_LOGGER = logging.getLogger(__name__)


_RecommendedAction = Literal["allow", "halt_new_entries", "force_flat"]

# Severity ordering, used when multiple breakers trip on the same check.
# Higher number = more severe. ``force_flat`` always wins.
_SEVERITY: Dict[_RecommendedAction, int] = {
    "allow": 0,
    "halt_new_entries": 1,
    "force_flat": 2,
}


class DecisionContext(BaseModel):
    """Read-only snapshot evaluated by each circuit breaker.

    The caller (typically the live-trader loop) is responsible for
    populating these fields from the position store + the proposed trade.
    All monetary fields are in USD.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    symbol: str = Field(..., min_length=1)
    side: Literal["buy", "sell"]
    proposed_notional_usd: float = Field(..., ge=0.0)
    current_open_notional_usd: float = Field(..., ge=0.0)
    current_per_symbol_notional_usd: float = Field(..., ge=0.0)
    daily_realized_pnl_usd: float
    equity_peak_usd: float = Field(..., ge=0.0)
    equity_current_usd: float = Field(..., ge=0.0)
    as_of_utc: str = Field(..., min_length=1)


class CircuitBreakerVerdict(BaseModel):
    """Outcome of evaluating all configured breakers against a context."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    allow: bool
    tripped: List[str] = Field(default_factory=list)
    reason: str = ""
    recommended_action: _RecommendedAction
    details: Dict[str, Any] = Field(default_factory=dict)


def _coerce_float(value: Optional[Union[str, float, int]]) -> Optional[float]:
    """Convert env-string / numeric input to float, returning None on empty."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            _LOGGER.warning(
                "Could not parse circuit-breaker numeric value %r; treating as disabled.",
                value,
            )
            return None
    return float(value)


def _coerce_path(value: Optional[Union[str, Path]]) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    stripped = str(value).strip()
    if not stripped:
        return None
    return Path(stripped)


class CircuitBreakerSet:
    """Container of hard breakers evaluated on every trade decision.

    Each constructor arg falls back to its corresponding environment
    variable when omitted (``daily_loss_limit_usd`` -> ``DAILY_LOSS_LIMIT_USD``,
    etc.). Breakers whose limit is unset in BOTH places are disabled and
    log a one-time INFO notice on the first ``check`` call.
    """

    def __init__(
        self,
        *,
        daily_loss_limit_usd: Optional[float] = None,
        max_drawdown_pct: Optional[float] = None,
        max_total_notional_usd: Optional[float] = None,
        max_per_symbol_notional_usd: Optional[float] = None,
        kill_switch_file: Optional[Union[str, Path]] = None,
    ) -> None:
        self.daily_loss_limit_usd: Optional[float] = (
            float(daily_loss_limit_usd)
            if daily_loss_limit_usd is not None
            else _coerce_float(os.environ.get("DAILY_LOSS_LIMIT_USD"))
        )
        self.max_drawdown_pct: Optional[float] = (
            float(max_drawdown_pct)
            if max_drawdown_pct is not None
            else _coerce_float(os.environ.get("MAX_DRAWDOWN_PCT"))
        )
        self.max_total_notional_usd: Optional[float] = (
            float(max_total_notional_usd)
            if max_total_notional_usd is not None
            else _coerce_float(os.environ.get("MAX_TOTAL_NOTIONAL_USD"))
        )
        self.max_per_symbol_notional_usd: Optional[float] = (
            float(max_per_symbol_notional_usd)
            if max_per_symbol_notional_usd is not None
            else _coerce_float(os.environ.get("MAX_PER_SYMBOL_NOTIONAL_USD"))
        )
        self.kill_switch_file: Optional[Path] = (
            _coerce_path(kill_switch_file)
            if kill_switch_file is not None
            else _coerce_path(os.environ.get("KILL_SWITCH_FILE"))
        )

        # Track which disabled-breaker warnings we've already emitted, so
        # repeated ``check`` calls don't spam the log.
        self._warned_disabled: set[str] = set()

    # ---------------------------------------------------------------- helpers

    def _warn_disabled_once(self, breaker_name: str) -> None:
        if breaker_name in self._warned_disabled:
            return
        self._warned_disabled.add(breaker_name)
        _LOGGER.info(
            "Circuit breaker %r is disabled (no constructor arg and no env override). "
            "All checks for this breaker will pass.",
            breaker_name,
        )

    def is_kill_switch_tripped(self) -> bool:
        """Return True iff the kill-switch file exists on disk."""
        if self.kill_switch_file is None:
            return False
        return self.kill_switch_file.exists()

    def reset_kill_switch(self) -> bool:
        """Remove the kill-switch file.

        Returns True if the file existed (and was removed); False if it
        was not present (or no path was configured).
        """
        if self.kill_switch_file is None:
            return False
        try:
            self.kill_switch_file.unlink()
            return True
        except FileNotFoundError:
            return False

    # ----------------------------------------------------------------- check

    def check(self, ctx: DecisionContext) -> CircuitBreakerVerdict:
        """Evaluate every configured breaker and aggregate the verdict.

        All breakers are evaluated even after the first one trips, so the
        ``details`` dict gives the operator a complete picture.
        """
        if not isinstance(ctx, DecisionContext):  # defensive — pydantic should enforce
            raise TypeError("ctx must be a DecisionContext instance")

        tripped: List[str] = []
        reasons: List[str] = []
        actions: List[_RecommendedAction] = []
        details: Dict[str, Any] = {}

        # 1. Kill switch (highest priority).
        if self.kill_switch_file is None:
            self._warn_disabled_once("kill_switch")
            details["kill_switch"] = {
                "configured": False,
                "tripped": False,
            }
        else:
            ks_tripped = self.is_kill_switch_tripped()
            details["kill_switch"] = {
                "configured": True,
                "path": str(self.kill_switch_file),
                "tripped": ks_tripped,
            }
            if ks_tripped:
                tripped.append("kill_switch")
                actions.append("force_flat")
                reasons.append(
                    f"kill switch file present at {self.kill_switch_file}"
                )

        # 2. Daily realized-loss limit.
        if self.daily_loss_limit_usd is None:
            self._warn_disabled_once("daily_loss")
            details["daily_loss"] = {
                "configured": False,
                "tripped": False,
            }
        else:
            limit = float(self.daily_loss_limit_usd)
            actual = float(ctx.daily_realized_pnl_usd)
            # The limit is expressed as a positive USD number; the bot
            # halts when realized PnL falls to -limit or below.
            dl_tripped = actual <= -limit
            details["daily_loss"] = {
                "configured": True,
                "limit_usd": limit,
                "actual_usd": actual,
                "tripped": dl_tripped,
            }
            if dl_tripped:
                tripped.append("daily_loss")
                actions.append("halt_new_entries")
                reasons.append(
                    f"daily realized PnL {actual:.2f} USD breached limit -{limit:.2f}"
                )

        # 3. Drawdown.
        if self.max_drawdown_pct is None:
            self._warn_disabled_once("drawdown")
            details["drawdown"] = {
                "configured": False,
                "tripped": False,
            }
        else:
            limit_pct = float(self.max_drawdown_pct)
            peak = float(ctx.equity_peak_usd)
            current = float(ctx.equity_current_usd)
            if peak > 0.0:
                dd_pct = (peak - current) / peak * 100.0
            else:
                dd_pct = 0.0
            dd_tripped = peak > 0.0 and dd_pct >= limit_pct
            details["drawdown"] = {
                "configured": True,
                "limit_pct": limit_pct,
                "actual_pct": dd_pct,
                "equity_peak_usd": peak,
                "equity_current_usd": current,
                "tripped": dd_tripped,
            }
            if dd_tripped:
                tripped.append("drawdown")
                actions.append("halt_new_entries")
                reasons.append(
                    f"drawdown {dd_pct:.2f}% breached cap {limit_pct:.2f}%"
                )

        # 4. Total notional cap (only relevant when *adding* exposure).
        if self.max_total_notional_usd is None:
            self._warn_disabled_once("total_notional")
            details["total_notional"] = {
                "configured": False,
                "tripped": False,
            }
        else:
            cap = float(self.max_total_notional_usd)
            current_open = float(ctx.current_open_notional_usd)
            proposed = float(ctx.proposed_notional_usd)
            projected = current_open + proposed
            tn_tripped = proposed > 0.0 and projected > cap
            details["total_notional"] = {
                "configured": True,
                "cap_usd": cap,
                "current_open_usd": current_open,
                "proposed_usd": proposed,
                "projected_usd": projected,
                "tripped": tn_tripped,
            }
            if tn_tripped:
                tripped.append("total_notional")
                actions.append("halt_new_entries")
                reasons.append(
                    f"total notional {projected:.2f} would breach cap {cap:.2f}"
                )

        # 5. Per-symbol notional cap.
        if self.max_per_symbol_notional_usd is None:
            self._warn_disabled_once("per_symbol_notional")
            details["per_symbol_notional"] = {
                "configured": False,
                "tripped": False,
            }
        else:
            cap = float(self.max_per_symbol_notional_usd)
            current_sym = float(ctx.current_per_symbol_notional_usd)
            proposed = float(ctx.proposed_notional_usd)
            projected = current_sym + proposed
            ps_tripped = proposed > 0.0 and projected > cap
            details["per_symbol_notional"] = {
                "configured": True,
                "symbol": ctx.symbol,
                "cap_usd": cap,
                "current_symbol_usd": current_sym,
                "proposed_usd": proposed,
                "projected_usd": projected,
                "tripped": ps_tripped,
            }
            if ps_tripped:
                tripped.append("per_symbol_notional")
                actions.append("halt_new_entries")
                reasons.append(
                    f"{ctx.symbol} notional {projected:.2f} would breach cap {cap:.2f}"
                )

        # Aggregate verdict.
        if not tripped:
            return CircuitBreakerVerdict(
                allow=True,
                tripped=[],
                reason="",
                recommended_action="allow",
                details=details,
            )

        # Pick the most severe action across all tripped breakers.
        worst_action: _RecommendedAction = max(
            actions, key=lambda a: _SEVERITY[a]
        )
        summary = "; ".join(reasons)

        return CircuitBreakerVerdict(
            allow=False,
            tripped=tripped,
            reason=summary,
            recommended_action=worst_action,
            details=details,
        )
