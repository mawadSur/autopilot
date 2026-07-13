"""Exit policy module — Sprint 1 Wave 1B.

This module is the answer to the "no exit policy is THE blocker" finding from
``autopilot-no-exit-policy-blocker-2026-05-16``: the supervisor was opening
positions and then either holding them indefinitely or unwinding via the
breaker force-flat. ``ExitPolicy`` gives the supervisor a single, explicit
seam to decide *every tick* whether an open position should be closed and
*why* — so we get a labelled exit reason on every close (time / SL / TP /
trail / reversal) instead of an undifferentiated breaker dump.

This PR (Wave 1B) is intentionally scoped to the policy + unit tests. The
supervisor wiring lives in Wave 2 and is gated behind
``cfg.EXIT_POLICY_ENABLED`` (default False) so this commit is non-breaking.

Design notes
------------

* The policy is **duck-typed** on its ``position`` and ``tick`` arguments.
  We deliberately do NOT import ``state.position_store.Position`` so the
  policy can run against (a) the live ``Position`` pydantic model, (b) a
  paper-trade position record, and (c) test fixtures built from
  ``SimpleNamespace``. The contract is documented in ``evaluate`` below.

* The constructor takes a small fixed set of percentage / bar thresholds.
  Each is independently toggleable by passing ``None`` (for the float
  thresholds), ``0`` (for the bar threshold), or ``False`` (for the
  reversal flag). When a knob is disabled the corresponding branch is
  skipped — there is no implicit fallback to a global default.

* Priority order, applied top-down inside ``evaluate``:

    1. ``sl``       — stop-loss (lose-more breach)
    2. ``tp``       — take-profit
    3. ``trail``    — trailing stop (uses ``high_water_mark``)
    4. ``time``     — bars-held ceiling
    5. ``reversal`` — model flips against us (long held, signal_prob < 0.5)

  The first branch to fire wins; subsequent checks are skipped. This order
  is deliberately *capital-preservation first* (SL beats TP beats trail)
  followed by time and then reversal. Reversal is last because (a) it is
  off by default until short-side execution exists, and (b) a brief
  signal-prob dip should not unwind a position whose price has not yet
  moved against us.

* No ``except Exception:`` anywhere in this module. If a position is
  missing a required attribute the policy raises ``AttributeError`` — the
  caller is responsible for catching and handling it (typically by
  logging and force-flatting). Silently ignoring a malformed position
  would be the worst possible failure mode for an exit policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional


_ExitReason = Literal["", "time", "sl", "tp", "trail", "reversal"]


@dataclass
class ExitDecision:
    """Result of evaluating an open position against the exit policy.

    ``close`` is the only field a caller strictly needs to read; ``reason``
    is for logging / the decision journal; ``exit_price`` is populated when
    the policy can synthesise a price (today only the trailing-stop branch
    sets it — SL/TP/time use the current tick price and the supervisor
    decides whether to submit a market or marketable-limit order).
    """

    close: bool
    reason: _ExitReason = ""
    exit_price: Optional[float] = None


def _require(obj: Any, attr: str) -> Any:
    """Pull ``attr`` off ``obj``, raising ``AttributeError`` if it is missing.

    We intentionally do not provide a default and we do not swallow the
    error. The exit policy must not silently no-op on a malformed position.
    """

    if not hasattr(obj, attr):
        raise AttributeError(
            f"ExitPolicy: required attribute '{attr}' missing on {type(obj).__name__}"
        )
    value = getattr(obj, attr)
    if value is None:
        raise AttributeError(
            f"ExitPolicy: required attribute '{attr}' is None on {type(obj).__name__}"
        )
    return value


class ExitPolicy:
    """Tick-driven exit policy combining SL, TP, trailing stop, time, and reversal.

    All constructor arguments are independently optional. Passing ``None``
    (for the float thresholds), ``0`` (for the bar threshold), or ``False``
    (for the reversal flag) disables that branch. When every branch is
    disabled :meth:`evaluate` is a no-op that always returns
    ``ExitDecision(close=False, reason="")``.

    Parameters
    ----------
    time_stop_bars
        Close the position once ``position.bars_held`` reaches or exceeds
        this many bars. ``None`` or ``0`` disables.
    stop_loss_pct
        Fractional loss threshold, expressed as a *negative* number (e.g.
        ``-0.004`` = -0.4%). Long positions close when
        ``(price / entry - 1) <= stop_loss_pct``; shorts use the inverse.
        ``None`` disables.
    take_profit_pct
        Fractional gain threshold, expressed as a *positive* number (e.g.
        ``0.008`` = +0.8%). Long positions close when
        ``(price / entry - 1) >= take_profit_pct``; shorts use the inverse.
        ``None`` disables.
    trailing_stop_pct
        Fractional retracement from ``position.high_water_mark`` (for
        longs) or low-water mark (for shorts). Disabled by default — the
        supervisor must opt in once the high-water-mark plumbing is in
        place (Wave 2). ``None`` disables.
    signal_reversal
        Close longs when ``tick.signal_prob < 0.5`` (model flips bearish).
        Disabled by default until short-side execution exists. ``False``
        disables.
    """

    def __init__(
        self,
        time_stop_bars: Optional[int] = 20,
        stop_loss_pct: Optional[float] = -0.004,
        take_profit_pct: Optional[float] = 0.008,
        trailing_stop_pct: Optional[float] = None,
        signal_reversal: bool = False,
    ) -> None:
        # Defensive sign-check on the SL: a positive ``stop_loss_pct`` would
        # silently fire on every tick (since (price/entry - 1) is usually >=
        # a small positive number after a wick). Catch it at construction
        # time rather than at evaluation time.
        if stop_loss_pct is not None and stop_loss_pct > 0:
            raise ValueError(
                f"stop_loss_pct must be <= 0 (got {stop_loss_pct}); "
                "express the threshold as a negative fraction, e.g. -0.004"
            )
        if take_profit_pct is not None and take_profit_pct < 0:
            raise ValueError(
                f"take_profit_pct must be >= 0 (got {take_profit_pct}); "
                "express the threshold as a positive fraction, e.g. 0.008"
            )
        if trailing_stop_pct is not None and trailing_stop_pct <= 0:
            raise ValueError(
                f"trailing_stop_pct must be > 0 (got {trailing_stop_pct}); "
                "express the trail as a positive fraction, e.g. 0.005"
            )
        if time_stop_bars is not None and time_stop_bars < 0:
            raise ValueError(
                f"time_stop_bars must be >= 0 (got {time_stop_bars})"
            )

        self.time_stop_bars = time_stop_bars
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.signal_reversal = bool(signal_reversal)

    # ------------------------------------------------------------------
    # Trailing-stop bookkeeping
    # ------------------------------------------------------------------
    def update_high_water_mark(self, position: Any, tick: Any) -> None:
        """Advance ``position.high_water_mark`` from the current tick price.

        This must be called once per tick *before* :meth:`evaluate` so the
        trailing-stop branch has fresh state. For long positions the high
        water mark is the running maximum of ``tick.price``; for short
        positions it is the running minimum (we still use the same
        attribute name to keep the position schema uniform — see Wave 2).

        If trailing stops are disabled this is a no-op (we still record
        the watermark for free since it is cheap, and so that a future
        operator who flips ``trailing_stop_pct`` on at runtime has a
        usable state to read from).

        The position must expose ``side`` and a mutable ``high_water_mark``
        attribute. ``high_water_mark`` may be ``None`` on the first tick;
        we will initialise it to ``entry_price`` in that case.
        """

        side = _require(position, "side")
        price = float(_require(tick, "price"))

        # Seed the watermark on the first call.
        current = getattr(position, "high_water_mark", None)
        if current is None:
            seed = float(_require(position, "entry_price"))
            position.high_water_mark = seed
            current = seed

        if side == "long":
            if price > float(current):
                position.high_water_mark = price
        elif side == "short":
            if price < float(current):
                position.high_water_mark = price
        else:
            raise ValueError(
                f"ExitPolicy.update_high_water_mark: unknown side {side!r}"
            )

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------
    def evaluate(self, position: Any, tick: Any) -> ExitDecision:
        """Evaluate an open position against all enabled exit branches.

        Returns the *first* triggered :class:`ExitDecision` in priority
        order (SL > TP > trail > time > reversal). If no branch fires,
        returns ``ExitDecision(close=False, reason="")``.

        Required attributes:

        * ``position.entry_price`` — float
        * ``position.side`` — ``"long"`` or ``"short"``
        * ``tick.price`` — float
        * ``position.bars_held`` — int (only if ``time_stop_bars`` is set)
        * ``position.high_water_mark`` — float (only if
          ``trailing_stop_pct`` is set)
        * ``tick.signal_prob`` — float in [0, 1] (only if
          ``signal_reversal`` is True)

        Missing required attributes raise ``AttributeError``. We do not
        catch and we do not default — the caller must surface the bug.
        """

        side = _require(position, "side")
        entry_price = float(_require(position, "entry_price"))
        price = float(_require(tick, "price"))

        # PnL fraction from the position's perspective (positive == winning).
        if side == "long":
            pnl_frac = (price / entry_price) - 1.0
        elif side == "short":
            pnl_frac = 1.0 - (price / entry_price)
        else:
            raise ValueError(f"ExitPolicy.evaluate: unknown side {side!r}")

        # 1. Stop loss --------------------------------------------------
        if self.stop_loss_pct is not None:
            # stop_loss_pct is negative; pnl_frac is negative when losing.
            if pnl_frac <= self.stop_loss_pct:
                return ExitDecision(close=True, reason="sl", exit_price=price)

        # 2. Take profit ------------------------------------------------
        if self.take_profit_pct is not None:
            if pnl_frac >= self.take_profit_pct:
                return ExitDecision(close=True, reason="tp", exit_price=price)

        # 3. Trailing stop ---------------------------------------------
        if self.trailing_stop_pct is not None:
            hwm = float(_require(position, "high_water_mark"))
            if side == "long":
                # Retracement from the running max.
                retracement = (hwm - price) / hwm if hwm > 0 else 0.0
                if retracement >= self.trailing_stop_pct:
                    return ExitDecision(
                        close=True, reason="trail", exit_price=price
                    )
            else:  # short
                # Rebound from the running min.
                rebound = (price - hwm) / hwm if hwm > 0 else 0.0
                if rebound >= self.trailing_stop_pct:
                    return ExitDecision(
                        close=True, reason="trail", exit_price=price
                    )

        # 4. Time stop --------------------------------------------------
        if self.time_stop_bars is not None and self.time_stop_bars > 0:
            bars_held = int(_require(position, "bars_held"))
            if bars_held >= self.time_stop_bars:
                return ExitDecision(close=True, reason="time", exit_price=price)

        # 5. Signal reversal -------------------------------------------
        if self.signal_reversal:
            signal_prob = float(_require(tick, "signal_prob"))
            if side == "long" and signal_prob < 0.5:
                return ExitDecision(
                    close=True, reason="reversal", exit_price=price
                )
            if side == "short" and signal_prob > 0.5:
                return ExitDecision(
                    close=True, reason="reversal", exit_price=price
                )

        return ExitDecision(close=False, reason="", exit_price=None)
