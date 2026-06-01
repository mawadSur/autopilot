"""Model-free arbitrage detection for binary prediction markets (SHADOW-ONLY).

This module is the first *real* profit path for the prediction-market stack. It
does NOT depend on a forecaster — that is the whole point. The scanner's
``research_priority`` (``src/ranker.py``) ranks markets by how *interesting* they
are to research; it never computes mispricing or expected value. The calibration
"edge" is currently a MOCK (``calibration_agent/ml_service.py`` returns
``implied_prob + uniform(+/-0.02)``), so any LLM-vs-market directional edge today
is noise. Intra-market arbitrage sidesteps all of that: it is a pure accounting
identity on a binary market and needs no probability estimate at all.

SAFETY (Constitution: safety first for anything that moves money)
----------------------------------------------------------------
This module is **SHADOW-ONLY**. It *detects* and *logs* opportunities. It places
NO orders, signs nothing, and adds no execution path. When a :class:`PnlLedger`
is passed to :func:`scan_intramarket_arbs`, opportunities are appended as
``status='open'`` shadow :class:`TradeRecord`s so the closed loop (detect ->
record -> later settle) can be audited with zero real exposure. Wiring real
execution is explicitly out of scope here.

The intra-market arbitrage identity
------------------------------------
On a Polymarket binary, the YES and NO tokens of the same market together pay
out exactly ``$1`` at resolution: if the event happens YES pays $1 and NO pays
$0; if it doesn't, NO pays $1 and YES pays $0. Either way the *pair* pays $1.

So if you can BUY 1 YES at ``yes_ask`` and BUY 1 NO at ``no_ask`` for a combined
cost of ``yes_ask + no_ask`` and you are guaranteed $1 back, then whenever::

    yes_ask + no_ask < 1

you have locked in a risk-free gross edge of ``1 - (yes_ask + no_ask)`` per $1 of
payout, *before costs*. Costs that must be honestly netted out (Constitution:
honest costs):

* **Settlement fee** — Polymarket charges a fee on the winning payout at
  resolution (~200 bps here). It applies to the $1 that comes back, so the fee in
  dollars per pair is ``(settlement_fee_bps / 10_000) * 1.0``.
* **Gas** — fixed on-chain cost per pair (USD), passed in by the caller.

Net edge per pair is therefore::

    net_edge_per_pair = gross_edge - settlement_fee_dollars - gas_usd

and only a strictly-positive net edge is an arb.

Costs are charged once *per pair*, and we report a per-pair cost basis of
``yes_ask + no_ask`` (what you actually pay up front to acquire the pair). The
gas term is *not* folded into ``cost_basis`` because cost_basis is the tradeable
acquisition price of the pair; gas/fees are netted separately into the edge so
the economics stay legible.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

try:  # Flat import under PYTHONPATH=src (matches the rest of the stack).
    from state.pnl_ledger import PnlLedger, TradeRecord
except Exception:  # pragma: no cover - only hit if ledger module is unavailable.
    PnlLedger = Any  # type: ignore
    TradeRecord = Any  # type: ignore


logger = logging.getLogger(__name__)

# Polymarket's resolution/settlement fee, in basis points, charged on the $1
# winning payout. Kept as a module default so callers (and tests) can override.
DEFAULT_SETTLEMENT_FEE_BPS = 200.0

# A binary YES/NO pair always redeems for exactly $1 at resolution.
GUARANTEED_PAYOUT_USD = 1.0

# Field-name variants tolerated when reading a market dict. The scanner export
# rows (see ``main.py`` EXPORT_FIELDS) and various venue adapters spell these
# differently; map them all to the canonical (yes_ask, no_ask, id) here.
_YES_ASK_KEYS = ("yes_ask", "yes_ask_price", "yesAsk", "ask_yes", "yes_price")
_NO_ASK_KEYS = ("no_ask", "no_ask_price", "noAsk", "ask_no", "no_price")
_ID_KEYS = ("market_id", "id", "ticker", "condition_id", "conditionId", "slug")


def _settlement_fee_dollars(settlement_fee_bps: float) -> float:
    """Dollar fee charged on the $1 winning payout for one YES+NO pair."""
    return (float(settlement_fee_bps) / 10_000.0) * GUARANTEED_PAYOUT_USD


def _validate_price(value: Any, label: str) -> float:
    """Coerce ``value`` to a float price strictly inside ``(0, 1)``.

    Prices at or beyond the bounds are rejected: ``<= 0`` is free money / a data
    error, ``>= 1`` means the leg costs at least the whole guaranteed payout, so
    neither is a tradeable ask on a binary market.
    """
    try:
        price = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not a number: {value!r}") from exc
    if not (0.0 < price < 1.0):
        raise ValueError(f"{label} must be in the open interval (0, 1); got {price}")
    return price


def intramarket_arb(
    yes_ask: float,
    no_ask: float,
    *,
    settlement_fee_bps: float = DEFAULT_SETTLEMENT_FEE_BPS,
    gas_usd: float = 0.0,
    size_usd: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """Detect a model-free intra-market arbitrage on one binary market.

    Buying 1 YES + 1 NO on the same binary market guarantees a ``$1`` payout at
    resolution. If the combined ask ``yes_ask + no_ask`` is below $1, the gap is
    a locked gross edge; this nets it for the settlement fee (charged on the $1
    payout) and gas, and returns the opportunity *only* if the net edge is
    strictly positive.

    Parameters
    ----------
    yes_ask, no_ask:
        Ask (buy) prices of the YES and NO legs, each in the open interval
        ``(0, 1)``. Validated; out-of-range or non-numeric inputs raise
        :class:`ValueError`.
    settlement_fee_bps:
        Fee in basis points charged on the $1 winning payout at resolution
        (default 200 bps). Converted to dollars-per-pair internally.
    gas_usd:
        Fixed on-chain execution cost per pair, in USD (default 0).
    size_usd:
        If > 0, treated as the dollar amount of guaranteed *payout* to acquire
        (i.e. the number of YES+NO pairs, since each pair pays $1). Used only to
        compute ``est_profit_usd``; it does not affect whether an arb exists.

    Returns
    -------
    dict | None
        ``None`` if there is no net-positive edge after costs. Otherwise a dict::

            {
                "gross_edge":        1 - (yes_ask + no_ask),
                "net_edge_per_pair": gross_edge - fee$ - gas,
                "net_edge_pct":      net_edge_per_pair / cost_basis,
                "cost_basis":        yes_ask + no_ask,
                "settlement_fee_usd": fee charged on the $1 payout per pair,
                "gas_usd":           gas cost per pair,
                "est_profit_usd":    net_edge_per_pair * n_pairs (if size_usd>0),
                "legs": [{"side": "YES", "price": yes_ask},
                         {"side": "NO",  "price": no_ask}],
            }
    """
    yes_ask = _validate_price(yes_ask, "yes_ask")
    no_ask = _validate_price(no_ask, "no_ask")

    cost_basis = yes_ask + no_ask
    gross_edge = GUARANTEED_PAYOUT_USD - cost_basis
    if gross_edge <= 0.0:
        # No gross edge: the pair costs at least the guaranteed payout.
        return None

    fee_usd = _settlement_fee_dollars(settlement_fee_bps)
    gas = max(0.0, float(gas_usd))
    net_edge_per_pair = gross_edge - fee_usd - gas
    if net_edge_per_pair <= 0.0:
        # Gross edge fully eaten by settlement fee + gas; not tradeable.
        return None

    # net_edge_pct is return-on-capital: profit per $1 of capital deployed to
    # acquire the pair (cost_basis is what you pay up front).
    net_edge_pct = net_edge_per_pair / cost_basis

    result: Dict[str, Any] = {
        "gross_edge": gross_edge,
        "net_edge_per_pair": net_edge_per_pair,
        "net_edge_pct": net_edge_pct,
        "cost_basis": cost_basis,
        "settlement_fee_usd": fee_usd,
        "gas_usd": gas,
        "est_profit_usd": 0.0,
        "legs": [
            {"side": "YES", "price": yes_ask},
            {"side": "NO", "price": no_ask},
        ],
    }

    if size_usd and float(size_usd) > 0.0:
        # size_usd is the dollar amount of guaranteed $1 payout to acquire, i.e.
        # the pair count (each pair redeems for exactly $1). Profit scales
        # linearly in the number of pairs.
        n_pairs = float(size_usd) / GUARANTEED_PAYOUT_USD
        result["est_profit_usd"] = net_edge_per_pair * n_pairs

    return result


def _first_present(market: Dict[str, Any], keys: Iterable[str]) -> Any:
    """Return the first value for any of ``keys`` present (not None) in ``market``."""
    for key in keys:
        if key in market and market[key] is not None:
            return market[key]
    return None


def _market_identifier(market: Dict[str, Any]) -> str:
    """Best-effort stable identifier for a market dict (tolerant of key drift)."""
    ident = _first_present(market, _ID_KEYS)
    return str(ident) if ident is not None else "unknown"


def scan_intramarket_arbs(
    markets: Iterable[Dict[str, Any]],
    *,
    ledger: "Optional[PnlLedger]" = None,
    min_net_edge_pct: float = 0.005,
    size_usd: float = 0.0,
    strategy: str = "intramarket_arb",
) -> List[Dict[str, Any]]:
    """Scan an iterable of market dicts for intra-market arbitrage opportunities.

    Each market dict must carry a YES ask and a NO ask. The reader is tolerant of
    field-name variants:

        * YES ask: one of ``{yes_ask, yes_ask_price, yesAsk, ask_yes, yes_price}``
        * NO ask:  one of ``{no_ask, no_ask_price, noAsk, ask_no, no_price}``
        * id:      one of ``{market_id, id, ticker, condition_id,
                   conditionId, slug}`` (used only for labelling/logging)

    Markets missing either ask, or whose asks fail price validation, are skipped
    (logged at DEBUG) rather than raising — a scan over a noisy export should not
    abort on one bad row.

    Parameters
    ----------
    markets:
        Iterable of market dicts (e.g. scanner export rows + explicit asks).
    ledger:
        Optional :class:`PnlLedger`. When provided, each detected opportunity is
        appended as a SHADOW ``status='open'`` :class:`TradeRecord`
        (``venue='polymarket'``, ``strategy=strategy``, ``side='YES+NO'``,
        ``entry_price=cost_basis``, ``fees_usd=settlement_fee_usd + gas``). This
        demonstrates the closed loop with **zero execution** — no order is placed.
    min_net_edge_pct:
        Minimum net edge (as a fraction of cost basis) to report. Default 0.5%.
    size_usd:
        Forwarded to :func:`intramarket_arb` for ``est_profit_usd`` and used as
        the shadow record ``size`` (USD notional). 0 means size-agnostic.
    strategy:
        Strategy label stamped onto detected arbs and shadow records.

    Returns
    -------
    list[dict]
        Detected opportunities (each the :func:`intramarket_arb` dict plus
        ``market_id`` and ``strategy``), sorted by ``net_edge_pct`` descending.
    """
    opportunities: List[Dict[str, Any]] = []

    for market in markets:
        if not isinstance(market, dict):
            logger.debug("Skipping non-dict market row: %r", market)
            continue

        yes_raw = _first_present(market, _YES_ASK_KEYS)
        no_raw = _first_present(market, _NO_ASK_KEYS)
        if yes_raw is None or no_raw is None:
            logger.debug(
                "Skipping market %s: missing yes_ask/no_ask",
                _market_identifier(market),
            )
            continue

        try:
            arb = intramarket_arb(
                yes_raw,
                no_raw,
                settlement_fee_bps=float(market.get("settlement_fee_bps", DEFAULT_SETTLEMENT_FEE_BPS)),
                gas_usd=float(market.get("gas_usd", 0.0)),
                size_usd=size_usd,
            )
        except ValueError as exc:
            logger.debug(
                "Skipping market %s: invalid arb inputs (%s)",
                _market_identifier(market),
                exc,
            )
            continue

        if arb is None or arb["net_edge_pct"] < float(min_net_edge_pct):
            continue

        market_id = _market_identifier(market)
        arb = {**arb, "market_id": market_id, "strategy": strategy}
        opportunities.append(arb)

        if ledger is not None:
            _append_shadow_record(
                ledger,
                arb,
                market_id=market_id,
                strategy=strategy,
                size_usd=size_usd,
            )

    opportunities.sort(key=lambda a: a["net_edge_pct"], reverse=True)
    return opportunities


def _append_shadow_record(
    ledger: "PnlLedger",
    arb: Dict[str, Any],
    *,
    market_id: str,
    strategy: str,
    size_usd: float,
) -> None:
    """Append one SHADOW (status='open') TradeRecord for a detected arb.

    SHADOW-ONLY: this records the opportunity in the audit ledger; it does not
    and must not place any order. ``entry_price`` is the per-pair cost basis,
    ``fees_usd`` is the per-pair settlement fee + gas, and the note flags this as
    a shadow arb so any reader can see no capital was actually committed.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    fees_usd = float(arb.get("settlement_fee_usd", 0.0)) + float(arb.get("gas_usd", 0.0))
    trade_id = f"arb-{market_id}-{now_iso}"

    record = TradeRecord(
        trade_id=trade_id,
        ts_utc=now_iso,
        venue="polymarket",
        market_id=market_id,
        side="YES+NO",
        entry_price=float(arb["cost_basis"]),
        size=float(size_usd),
        fees_usd=fees_usd,
        slippage_bps=0.0,
        strategy=strategy,
        status="open",
        notes=(
            "SHADOW intra-market arb (NO order placed): "
            f"gross={arb['gross_edge']:.4f} "
            f"net_per_pair={arb['net_edge_per_pair']:.4f} "
            f"net_pct={arb['net_edge_pct']:.4%}"
        ),
    )
    ledger.append(record)


def net_directional_edge(
    fair_prob: float,
    market_ask: float,
    *,
    settlement_fee_bps: float = DEFAULT_SETTLEMENT_FEE_BPS,
    gas_usd: float = 0.0,
) -> float:
    """Expected value per $1 of buying YES at ``market_ask`` given ``fair_prob``.

    WARNING (read before using this for anything):
        ``fair_prob`` MUST come from a VALIDATED, non-mock forecaster. Today the
        only probability source in this repo
        (``calibration_agent/ml_service.get_xgboost_probability``) is a MOCK that
        returns ``implied_prob + uniform(+/-0.02)`` — pure noise. Feeding that
        here produces a number that *looks* like an edge but is not one. This
        directional path is therefore NOT yet a real edge and must NOT be traded
        until a shadow track record demonstrates it beats the market price net of
        fees. The model-free :func:`intramarket_arb` is the only path that is
        safe to act on today (and even that is shadow-only here).

    Economics
    ---------
    Buy 1 YES at ``market_ask``. With probability ``fair_prob`` the event happens
    and YES redeems for $1 (minus the settlement fee on that $1 payout); with
    probability ``1 - fair_prob`` it redeems for $0. Expected value per unit::

        EV = fair_prob * (1 - market_ask - settlement_fee$) + (1 - fair_prob) * (0 - market_ask) - gas
           = fair_prob * (1 - settlement_fee$) - market_ask - gas

    The settlement fee is only paid when YES wins (the payout case), so it is
    weighted by ``fair_prob``. ``gas_usd`` is a fixed per-unit cost paid in both
    outcomes.

    Parameters
    ----------
    fair_prob:
        Probability the YES outcome resolves true, in ``[0, 1]``.
    market_ask:
        Ask price to buy YES, in the open interval ``(0, 1)``.
    settlement_fee_bps:
        Fee in basis points on the $1 winning payout (default 200 bps).
    gas_usd:
        Fixed per-unit execution cost in USD (default 0).

    Returns
    -------
    float
        Net expected value per $1 of YES bought. Positive means a (claimed) edge;
        negative means a losing buy after costs. May be negative.
    """
    if not (0.0 <= float(fair_prob) <= 1.0):
        raise ValueError(f"fair_prob must be in [0, 1]; got {fair_prob}")
    ask = _validate_price(market_ask, "market_ask")

    fee_usd = _settlement_fee_dollars(settlement_fee_bps)
    gas = max(0.0, float(gas_usd))
    return float(fair_prob) * (GUARANTEED_PAYOUT_USD - fee_usd) - ask - gas


def market_row_to_arb_input(
    row: Dict[str, Any],
    *,
    no_ask: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Additive helper: map a scanner export row -> an arb-input dict.

    Scanner export rows (``main.py`` EXPORT_FIELDS) carry ``implied_prob`` and
    ``spread`` but NOT separate YES/NO asks, so they cannot by themselves prove an
    intra-market arb. This helper is therefore *opt-in* and *additive*: it returns
    ``None`` unless a NO ask is available (either already on the row under a
    tolerated key, or passed explicitly via ``no_ask``).

    When asks are not directly present, the YES ask is approximated as
    ``implied_prob`` (mid) — callers should prefer feeding true book asks. The NO
    leg must be supplied; we deliberately do NOT fabricate ``no_ask = 1 - yes_bid``
    here, to avoid manufacturing phantom arbs from a single mid price.

    Returns a dict with ``yes_ask``, ``no_ask`` and the row's id, or ``None`` if a
    NO ask cannot be determined.
    """
    yes_ask = _first_present(row, _YES_ASK_KEYS)
    if yes_ask is None:
        implied = row.get("implied_prob")
        if implied is not None:
            yes_ask = implied
    resolved_no = no_ask if no_ask is not None else _first_present(row, _NO_ASK_KEYS)
    if yes_ask is None or resolved_no is None:
        return None
    return {
        "market_id": _market_identifier(row),
        "yes_ask": float(yes_ask),
        "no_ask": float(resolved_no),
    }


__all__ = [
    "DEFAULT_SETTLEMENT_FEE_BPS",
    "GUARANTEED_PAYOUT_USD",
    "intramarket_arb",
    "scan_intramarket_arbs",
    "net_directional_edge",
    "market_row_to_arb_input",
]
