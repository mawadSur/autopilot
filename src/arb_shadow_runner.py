"""SHADOW intra-market arbitrage loop for Polymarket (NO execution).

This is the runnable glue that turns the read-only CLOB order-book client
(``src/exchanges/polymarket_market_data.py``) and the model-free arb detector
(``src/arb_detector.py``) into a repeatable scan: fetch active markets, read
each market's live YES/NO best asks, and log any ``yes_ask + no_ask < 1``
(net of costs) opportunity to the append-only PnL ledger as a SHADOW
``status='open'`` record.

SAFETY (Constitution: safety first for anything that moves money):
    SHADOW-ONLY. This module DETECTS and LOGS opportunities. It places NO
    orders, signs nothing, touches no wallet, and adds no execution path. The
    only side effect is appending shadow records to the ledger JSONL. Every
    scan prints under a 'SHADOW MODE — NO ORDERS PLACED' banner so an operator
    can never mistake a shadow run for live trading.

Dependency injection:
    ``fetch_markets_fn`` and ``market_data_client`` are injectable so the whole
    loop runs offline in tests with fakes — no network, no real CLOB calls.
    The defaults wire ``fetcher.fetch_active_markets`` and the read-only
    Polymarket market-data module.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

try:  # Flat imports under PYTHONPATH=src (matches the rest of the stack).
    import fetcher
    from exchanges import polymarket_market_data
    from arb_detector import scan_intramarket_arbs
    from state.pnl_ledger import DEFAULT_LEDGER_PATH, PnlLedger
except Exception:  # pragma: no cover - import-time fallback for tooling.
    fetcher = None  # type: ignore
    polymarket_market_data = None  # type: ignore
    from arb_detector import scan_intramarket_arbs  # type: ignore
    from state.pnl_ledger import DEFAULT_LEDGER_PATH, PnlLedger  # type: ignore


logger = logging.getLogger(__name__)

_SHADOW_BANNER = "SHADOW MODE — NO ORDERS PLACED"

# Tolerated identifier / title field-name variants, so we can read either a
# Gamma market dict or a ``models.Market`` dataclass instance.
_ID_KEYS = ("market_id", "id", "condition_id", "conditionId", "slug", "ticker")
_TITLE_KEYS = ("title", "question", "name")


def _field(market: Any, keys: Sequence[str]) -> Optional[Any]:
    """First present (non-None) value across ``keys`` on a dict OR an object."""
    for key in keys:
        if isinstance(market, dict):
            if key in market and market[key] is not None:
                return market[key]
        else:
            value = getattr(market, key, None)
            if value is not None:
                return value
    return None


def build_market_rows(
    markets: Sequence[Any],
    market_data_client: Any = None,
    session: Any = None,
    max_markets: int = 50,
) -> List[Dict[str, Any]]:
    """Build ``{id, title, yes_ask, no_ask}`` rows from active markets.

    For each market (up to ``max_markets``) the live YES/NO best asks are read
    via ``market_data_client.get_yes_no_best_asks(market, session=...)``.
    Markets whose asks are unavailable — missing/!=2 clobTokenIds, an empty
    asks side, or a per-market read error — are SKIPPED (logged) so one bad
    market never kills the scan.

    Args:
        markets: Active market objects/dicts (e.g. from
            ``fetcher.fetch_active_markets``). Each must expose ``clobTokenIds``
            for the asks to resolve.
        market_data_client: Object exposing ``get_yes_no_best_asks``. Defaults
            to the read-only ``exchanges.polymarket_market_data`` module. Inject
            a fake in tests.
        session: Optional shared ``requests.Session`` forwarded to the client.
        max_markets: Cap on markets scanned per pass (default 50).

    Returns:
        A list of arb-input rows; each row is consumed directly by
        :func:`arb_detector.scan_intramarket_arbs`.
    """
    client = market_data_client if market_data_client is not None else polymarket_market_data
    if client is None:  # pragma: no cover - only if the module failed to import.
        raise RuntimeError("no market_data_client available (polymarket_market_data import failed)")

    rows: List[Dict[str, Any]] = []
    for market in list(markets)[: max(0, int(max_markets))]:
        market_id = _field(market, _ID_KEYS)
        market_id = str(market_id) if market_id is not None else "unknown"
        title = _field(market, _TITLE_KEYS)
        title = str(title) if title is not None else ""

        try:
            asks = client.get_yes_no_best_asks(market, session=session)
        except Exception as exc:  # noqa: BLE001 - resilience: skip + continue.
            logger.warning("Skipping market %s: order-book read failed (%s)", market_id, exc)
            continue

        if asks is None:
            logger.debug("Skipping market %s: YES/NO asks unavailable", market_id)
            continue

        yes_ask, no_ask = asks
        rows.append(
            {
                "id": market_id,
                "title": title,
                "yes_ask": yes_ask,
                "no_ask": no_ask,
            }
        )

    return rows


def run_once(
    *,
    ledger: PnlLedger,
    fetch_markets_fn: Optional[Callable[..., Sequence[Any]]] = None,
    market_data_client: Any = None,
    max_markets: int = 50,
    min_net_edge_pct: float = 0.005,
    size_usd: float = 0.0,
    session: Any = None,
) -> List[Dict[str, Any]]:
    """Run one SHADOW arb scan and return the detected opportunities.

    Fetches active markets, builds YES/NO ask rows, scans for intra-market
    arbs (logging each as a shadow ``status='open'`` ledger record), and
    prints a banner-fronted summary. Places NO orders.

    Args:
        ledger: Append-only :class:`PnlLedger` to log shadow opportunities to.
        fetch_markets_fn: Callable returning active markets. Defaults to
            ``fetcher.fetch_active_markets``. Inject a fake in tests.
        market_data_client: Order-book client (see :func:`build_market_rows`).
        max_markets: Cap on markets scanned this pass.
        min_net_edge_pct: Minimum net edge (fraction of cost basis) to report.
        size_usd: USD payout notional for ``est_profit_usd`` / shadow size.
        session: Optional shared ``requests.Session`` for connection reuse.

    Returns:
        The list of detected arbitrage opportunity dicts.
    """
    if fetch_markets_fn is not None:
        markets = list(fetch_markets_fn() or [])
    else:
        if fetcher is None:  # pragma: no cover - only if fetcher import failed.
            raise RuntimeError("no fetch_markets_fn provided and fetcher import failed")
        # Gamma returns markets ordered by 24h volume (desc), so the top pages
        # are the most liquid markets -- exactly where arb depth lives. Bound
        # the page count to roughly ``max_markets`` so we don't paginate into
        # Gamma's offset ceiling (it 422s past ~10k markets) when we only want
        # the top few.
        pages = max(1, (max_markets // 100) + 2)
        markets = list(fetcher.fetch_active_markets(max_pages=pages) or [])
    rows = build_market_rows(
        markets,
        market_data_client=market_data_client,
        session=session,
        max_markets=max_markets,
    )

    opportunities = scan_intramarket_arbs(
        rows,
        ledger=ledger,
        min_net_edge_pct=min_net_edge_pct,
        size_usd=size_usd,
    )

    est_profit = sum(float(opp.get("est_profit_usd", 0.0)) for opp in opportunities)
    print("=" * 60)
    print(_SHADOW_BANNER)
    print(
        f"  markets fetched : {len(markets)}\n"
        f"  rows scanned    : {len(rows)}\n"
        f"  arbs found      : {len(opportunities)}\n"
        f"  est shadow profit (USD): {est_profit:.4f}"
    )
    for opp in opportunities:
        print(
            f"  [arb] {opp.get('market_id', 'unknown')}: "
            f"net_pct={opp.get('net_edge_pct', 0.0):.4%} "
            f"cost_basis={opp.get('cost_basis', 0.0):.4f}"
        )
    print("=" * 60)

    return opportunities


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for the SHADOW arb runner.

    ``--once`` (default) runs a single scan; ``--interval`` loops, sleeping
    between scans until interrupted. NO orders are ever placed.
    """
    parser = argparse.ArgumentParser(
        description="SHADOW Polymarket intra-market arb scanner (logs opportunities, places NO orders)."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan and exit (the default when --interval is unset).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Loop, sleeping this many SECONDS between scans (graceful Ctrl-C).",
    )
    parser.add_argument("--max-markets", type=int, default=50, help="Max markets scanned per pass.")
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.005,
        help="Minimum net edge (fraction of cost basis) to report (default 0.005 = 0.5%%).",
    )
    parser.add_argument(
        "--size",
        type=float,
        default=0.0,
        help="USD payout notional for est_profit_usd / shadow record size (0 = size-agnostic).",
    )
    parser.add_argument(
        "--ledger-path",
        type=str,
        default=DEFAULT_LEDGER_PATH,
        help=f"Path to the JSONL PnL ledger (default {DEFAULT_LEDGER_PATH}).",
    )
    parser.add_argument(
        "--discord",
        action="store_true",
        help="After each scan, post per-trade P/L + portfolio value to Discord "
        "(reads DISCORD_WEBHOOK_URL; no-op if unset). Read-only reporting.",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Paper bankroll (USD) used as the portfolio-equity baseline in reports.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ledger = PnlLedger(args.ledger_path)

    notifier = None
    report_fn = None
    if args.discord:
        try:
            from portfolio_reporter import load_env_files
            from portfolio_reporter import report_to_discord as report_fn  # noqa: F811
            from alerts.notifier import Notifier
            load_env_files()  # pick up DISCORD_WEBHOOK_URL from .env for CLI runs
            notifier = Notifier()
        except ImportError:
            logging.warning(
                "Discord reporting unavailable (alerts/portfolio_reporter import "
                "failed); continuing without it."
            )

    # An intra-market arb pair redeems for $1 at resolution, so its locked mark
    # is 1.0; other strategies are left unpriced (reported as 'pending').
    def _arb_price_fn(record: Any) -> Optional[float]:
        return 1.0 if getattr(record, "side", None) == "YES+NO" else None

    def _scan() -> None:
        run_once(
            ledger=ledger,
            max_markets=args.max_markets,
            min_net_edge_pct=args.min_edge,
            size_usd=args.size,
        )
        if notifier is not None and report_fn is not None:
            report_fn(
                ledger,
                notifier,
                price_fn=_arb_price_fn,
                bankroll_usd=args.bankroll,
                label="Shadow arb",
            )

    if args.interval is not None and args.interval > 0:
        print(f"{_SHADOW_BANNER}: looping every {args.interval:.1f}s (Ctrl-C to stop).")
        try:
            while True:
                _scan()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped (KeyboardInterrupt). No orders were ever placed.")
    else:
        _scan()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
