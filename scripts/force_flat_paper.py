"""Manually force-flat all paper positions when the supervisor's
kill_switch latch can't retry (e.g. DNS failed during the auto-trip).

Mirrors LiveSupervisor._paper_force_flat: 5 bps slippage against the
current Coinbase ticker mid, in the close direction. Records realized
PnL via position_store.record_close so the closed positions show up in
list_closed_today + daily_realized_pnl_usd.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
for p in (SRC, REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exchanges.coinbase import CoinbaseExchange  # noqa: E402
from state.position_store import PositionStore  # noqa: E402

PAPER_SLIPPAGE_BPS = 5.0


def main() -> int:
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    ps = PositionStore(redis_url=redis_url)
    opens = ps.list_open()
    if not opens:
        print("no open positions; nothing to do")
        return 0
    print(f"closing {len(opens)} paper position(s)")

    # Cache ticker per symbol to avoid 100 API calls.
    exchange = CoinbaseExchange()
    tickers: dict[str, float] = {}
    closed = 0
    failed = 0
    pnl_total = 0.0
    for position in opens:
        try:
            if position.exchange != "coinbase-paper":
                print(f"  SKIP {position.position_id[:8]}: exchange={position.exchange!r} (not paper)")
                continue
            if position.symbol not in tickers:
                t = exchange.get_ticker(position.symbol)
                tickers[position.symbol] = float(t.mid)
            mid = tickers[position.symbol]
            slip = PAPER_SLIPPAGE_BPS / 10_000.0
            # long position closes via sell -> price * (1 - slip)
            exit_price = mid * (1.0 - slip) if position.side == "long" else mid * (1.0 + slip)
            if exit_price <= 0:
                exit_price = float(position.entry_price)
            exit_quote = exit_price * float(position.base_size)
            updated = ps.record_close(
                position.position_id,
                exit_price=exit_price,
                exit_quote_usd=exit_quote,
            )
            closed += 1
            pnl_total += float(updated.realized_pnl_usd or 0.0)
            if closed <= 3 or closed == len(opens):
                print(
                    f"  closed {closed}/{len(opens)}: id={position.position_id[:8]} "
                    f"entry={position.entry_price:.2f} exit={exit_price:.2f} "
                    f"pnl={float(updated.realized_pnl_usd or 0.0):+.4f}"
                )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {position.position_id[:8]}: {exc}")

    print(
        f"\ndone: closed={closed} failed={failed} "
        f"realized_pnl_total=${pnl_total:+.2f}"
    )
    remaining = ps.list_open()
    print(f"open after: {len(remaining)}  open_notional_usd: ${ps.open_notional_usd():.2f}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
