"""Exit-overlay backtester — find the best stop-loss / take-profit for whale-follow.

The whale-follow shadow loop books a symmetric hit-rate but an asymmetric realized
P/L: the LOSERS ride all the way to $0 (a full stake forfeit) while the WINNERS are
capped at resolution. A stop-loss / take-profit overlay fixes that asymmetry — but
only at thresholds that actually help on the REAL forward price paths the whales sat
through. This module is the pure, offline half that searches that grid; the live
read-only data collector lives in ``scripts/optimize_exits.py``.

SCOPE — PURE / NO MONEY MOVES (Constitution: safety first):
    Every function here is a pure computation over numbers the CALLER supplies. There
    is no network, no client, no ledger, no order/sign/redeem/web3 path anywhere in
    this module. It reads nothing and writes nothing — it just simulates what a paper
    position WOULD have booked under a given exit overlay.

No look-ahead / honest reporting:
    :func:`simulate_exit` is handed a ``price_path`` the caller GUARANTEES is in
    chronological order and consists ONLY of prices observed strictly AFTER entry
    (the collector builds it from post-entry trades only). The walk makes its exit
    decision from the path prices alone — it NEVER consults ``final_price`` (the
    resolved outcome) until the path is exhausted with no trigger. So the overlay can
    only "know" what was observable at each step, never the future outcome. The same
    return-on-cost math and 200 bps fee as :mod:`exit_rules` / :mod:`shadow_settlement`
    are used so the numbers are comparable to the shadow ledger.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple


__all__ = [
    "DEFAULT_FEE_BPS",
    "RISK_ADJUSTED_WORST_FLOOR",
    "simulate_exit",
    "grid_search",
]

# Polymarket-style proceeds haircut on an exit/redemption: ~2% (200 bps). Kept in
# sync with exit_rules.DEFAULT_FEE_BPS / shadow_settlement.DEFAULT_FEE_BPS so the
# backtest's realized returns are directly comparable to the shadow ledger's.
DEFAULT_FEE_BPS = 200.0

# The "risk_adjusted" pick caps a single trade's worst loss at 60% of cost: a combo
# is only eligible if EVERY trade's return_on_cost stayed >= this floor.
RISK_ADJUSTED_WORST_FLOOR = -0.6


def simulate_exit(
    entry_price: float,
    price_path: Sequence[float],
    final_price: float,
    *,
    stop_loss_pct: Optional[float],
    take_profit_pct: Optional[float],
    take_profit_price: Optional[float],
    fee_bps: float = DEFAULT_FEE_BPS,
) -> Tuple[float, str]:
    """Simulate one position under a stop-loss / take-profit overlay.

    Walks ``price_path`` in chronological order — the caller GUARANTEES every price
    is observed strictly AFTER entry, so there is no look-ahead — and exits at the
    FIRST price that trips a rule (first trigger in TIME wins). At each price ``P``
    (validated to lie in ``[0, 1]``; out-of-range prices are skipped, never acted
    on) the return-on-cost is ``r = P / entry_price - 1`` and:

      * stop-loss is checked FIRST: if ``stop_loss_pct`` is set and
        ``r <= -abs(stop_loss_pct)`` -> EXIT at ``P``, reason ``'stop_loss'`` (cut
        the loss before considering any upside lock — the capital-preserving
        choice, matching :func:`exit_rules.evaluate_exit`);
      * else take-profit: an absolute price target wins over the relative one —
        if ``take_profit_price`` is set and ``P >= take_profit_price`` -> EXIT
        ``'take_profit'``; elif ``take_profit_pct`` is set and
        ``r >= take_profit_pct`` -> EXIT ``'take_profit'``.

    If no rule fires across the WHOLE path, the position is held to resolution and
    exits at ``final_price`` (``1.0`` won / ``0.0`` lost), reason ``'resolution'``.
    ``final_price`` is consulted ONLY at this point — never during the walk.

    The realized return PER $1 of cost (notional-agnostic) is::

        return_on_cost = (1.0 / entry_price) * exit_price * (1 - fee_bps / 10_000) - 1.0

    i.e. $1 of cost buys ``1/entry_price`` shares, each sold/redeemed at
    ``exit_price`` net of the ``fee_bps`` haircut. An unsizable entry
    (``entry_price <= 0``) returns ``(0.0, 'skip')`` so the caller can drop it
    rather than fabricate a number.

    Args:
        entry_price: Dollars per outcome share paid at entry, expected in ``(0, 1]``.
        price_path: Held outcome's prices in chronological order, ALL strictly
            after entry (caller-guaranteed; no future leak).
        final_price: Resolved price of the held outcome — ``1.0`` (won) or ``0.0``
            (lost).
        stop_loss_pct: Cut when ``r <= -abs(stop_loss_pct)`` (e.g. ``0.4`` = down
            40%). ``None`` disables the stop.
        take_profit_pct: Lock when ``r >= take_profit_pct`` (e.g. ``0.5`` = up 50%).
            ``None`` disables this leg.
        take_profit_price: Lock when ``P >= take_profit_price`` (e.g. ``0.90``).
            Checked before ``take_profit_pct``. ``None`` disables.
        fee_bps: Proceeds haircut in basis points (default 200 = ~2%).

    Returns:
        ``(return_on_cost, reason)`` where reason is one of ``'stop_loss'``,
        ``'take_profit'``, ``'resolution'``, ``'skip'``.
    """
    if entry_price is None or entry_price <= 0:
        return (0.0, "skip")

    entry = float(entry_price)
    sl = None if stop_loss_pct is None else abs(float(stop_loss_pct))

    for raw in price_path:
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            continue
        price = float(raw)
        if not (0.0 <= price <= 1.0):
            continue

        r = price / entry - 1.0

        # Stop-loss FIRST: cap the downside before any upside lock.
        if sl is not None and r <= -sl:
            return (_return_on_cost(entry, price, fee_bps), "stop_loss")

        # Take-profit: absolute price target takes precedence over the relative one.
        if take_profit_price is not None and price >= float(take_profit_price):
            return (_return_on_cost(entry, price, fee_bps), "take_profit")
        if take_profit_pct is not None and r >= float(take_profit_pct):
            return (_return_on_cost(entry, price, fee_bps), "take_profit")

    # No trigger across the whole path -> hold to resolution.
    return (_return_on_cost(entry, float(final_price), fee_bps), "resolution")


def _return_on_cost(entry_price: float, exit_price: float, fee_bps: float) -> float:
    """Realized return per $1 of cost: ``(1/entry)*exit*(1 - fee) - 1``."""
    return (1.0 / entry_price) * exit_price * (1.0 - float(fee_bps) / 10_000.0) - 1.0


def _aggregate_combo(
    samples: Sequence[Dict[str, Any]],
    *,
    stop_loss_pct: Optional[float],
    take_profit_pct: Optional[float],
    take_profit_price: Optional[float],
    fee_bps: float,
) -> Dict[str, Any]:
    """Run one (sl, tp_pct, tp_price) combo over all samples and aggregate.

    Skips samples whose entry is unsizable (``simulate_exit`` returns reason
    ``'skip'``) so the aggregates reflect only positions that could be sized. The
    ``exit_mix`` counts cover only the included samples and therefore sum to ``n``.
    """
    returns: List[float] = []
    exit_mix: Dict[str, int] = {}
    for sample in samples:
        ret, reason = simulate_exit(
            float(sample.get("entry_price", 0.0)),
            sample.get("price_path") or [],
            float(sample.get("final_price", 0.0)),
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            take_profit_price=take_profit_price,
            fee_bps=fee_bps,
        )
        if reason == "skip":
            continue
        returns.append(ret)
        exit_mix[reason] = exit_mix.get(reason, 0) + 1

    n = len(returns)
    total = sum(returns)
    wins = sum(1 for r in returns if r > 0)
    return {
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "take_profit_price": take_profit_price,
        "n": n,
        "mean_return": (total / n) if n else 0.0,
        "total_return": total,
        "win_rate": (wins / n) if n else 0.0,
        "worst_return": min(returns) if returns else 0.0,
        "exit_mix": exit_mix,
    }


def grid_search(
    samples: Sequence[Dict[str, Any]],
    *,
    sl_values: Sequence[Optional[float]],
    tp_pct_values: Sequence[Optional[float]],
    tp_price_values: Sequence[Optional[float]],
    fee_bps: float = DEFAULT_FEE_BPS,
) -> Dict[str, Any]:
    """Backtest every (stop_loss, take_profit_pct, take_profit_price) combo.

    For EVERY combo in the cartesian product of ``sl_values`` x ``tp_pct_values`` x
    ``tp_price_values`` (include ``None`` in each list to test "no stop" / "no TP"),
    :func:`simulate_exit` is run over all ``samples`` and the results aggregated.
    The BASELINE combo — all three ``None``, i.e. hold every position to resolution
    — is always included (added explicitly if the caller's lists omit ``None``) so
    every overlay can be compared against doing nothing.

    ``samples`` is a list of ``{'entry_price', 'price_path', 'final_price'}`` dicts
    (extra keys ignored). Each combo aggregate carries ``stop_loss_pct``,
    ``take_profit_pct``, ``take_profit_price``, ``n``, ``mean_return``,
    ``total_return``, ``win_rate`` (share with ``return_on_cost > 0``),
    ``worst_return`` (the single worst trade), and ``exit_mix`` (counts by reason,
    summing to ``n``).

    Returns a dict::

        {
          "combos": [...],            # every combo, sorted by mean_return desc
          "baseline": {...},          # the all-None hold-to-resolution combo
          "best_by_meanret": {...},   # combos[0] (highest mean_return)
          "risk_adjusted": {...},     # max mean_return among combos whose
                                      # worst_return >= RISK_ADJUSTED_WORST_FLOOR
        }

    ``risk_adjusted`` caps the single-trade loss at 60% of cost
    (:data:`RISK_ADJUSTED_WORST_FLOOR`); if no combo clears that floor it is
    ``None``.
    """
    seen: set = set()
    combos: List[Dict[str, Any]] = []

    def _key(sl: Optional[float], tp_pct: Optional[float], tp_price: Optional[float]):
        return (sl, tp_pct, tp_price)

    for sl, tp_pct, tp_price in product(sl_values, tp_pct_values, tp_price_values):
        key = _key(sl, tp_pct, tp_price)
        if key in seen:
            continue
        seen.add(key)
        combos.append(
            _aggregate_combo(
                samples,
                stop_loss_pct=sl,
                take_profit_pct=tp_pct,
                take_profit_price=tp_price,
                fee_bps=fee_bps,
            )
        )

    # Guarantee the baseline (hold-to-resolution) is present for comparison even if
    # the caller's grids omitted None on some axis.
    baseline_key = _key(None, None, None)
    if baseline_key not in seen:
        seen.add(baseline_key)
        combos.append(
            _aggregate_combo(
                samples,
                stop_loss_pct=None,
                take_profit_pct=None,
                take_profit_price=None,
                fee_bps=fee_bps,
            )
        )

    combos.sort(key=lambda c: c["mean_return"], reverse=True)

    baseline = next(
        c
        for c in combos
        if c["stop_loss_pct"] is None
        and c["take_profit_pct"] is None
        and c["take_profit_price"] is None
    )

    best_by_meanret = combos[0] if combos else None

    eligible = [c for c in combos if c["worst_return"] >= RISK_ADJUSTED_WORST_FLOOR]
    # combos is already sorted by mean_return desc, so the first eligible one is the
    # max-mean_return combo that respects the worst-loss floor.
    risk_adjusted = eligible[0] if eligible else None

    return {
        "combos": combos,
        "baseline": baseline,
        "best_by_meanret": best_by_meanret,
        "risk_adjusted": risk_adjusted,
    }
