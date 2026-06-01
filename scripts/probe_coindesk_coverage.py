"""Quick probe: does CoinDesk have L2 + trades for the dates we need?

Checks ETH-USD and ETH-USDT on coinbase for coverage at:
- today
- 30 days ago
- 60 days ago
- 90 days ago (start of our training data, 2026-01-29)
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv

load_dotenv(REPO / "src" / ".env")
load_dotenv(REPO / ".env")

from coindesk_client import build_client_from_env, CoinDeskClient


def _probe(client: CoinDeskClient, market: str, instrument: str, dt: datetime):
    to_ts = int(dt.timestamp())
    label = dt.strftime("%Y-%m-%d")
    out = f"  {market:<10} {instrument:<10} @ {label} (ts={to_ts})"
    try:
        snap = client.get_l2_snapshots(market=market, instrument=instrument, to_ts=to_ts, limit=10)
        df_snap = CoinDeskClient.normalize_l2_snapshots(snap)
        n_snap = len(df_snap)
        snap_status = f"L2 snapshots: {n_snap}"
    except Exception as e:
        snap_status = f"L2 snapshots: ERROR {type(e).__name__}: {str(e)[:80]}"

    try:
        trades = client.get_trades(market=market, instrument=instrument, to_ts=to_ts, limit=10)
        df_t = CoinDeskClient.normalize_trades(trades)
        n_t = len(df_t)
        trade_status = f"trades: {n_t}"
    except Exception as e:
        trade_status = f"trades: ERROR {type(e).__name__}: {str(e)[:80]}"

    try:
        m = client.get_l2_metrics(market=market, instrument=instrument, to_ts=to_ts, limit=10)
        df_m = CoinDeskClient.normalize_l2_metrics(m)
        n_m = len(df_m)
        m_status = f"l2_metrics: {n_m}"
    except Exception as e:
        m_status = f"l2_metrics: ERROR {type(e).__name__}: {str(e)[:80]}"

    print(f"{out}\n    {snap_status}\n    {trade_status}\n    {m_status}")


def main():
    client = build_client_from_env()
    now = datetime.now(timezone.utc)
    checkpoints = [
        ("today", now - timedelta(hours=1)),
        ("7d ago", now - timedelta(days=7)),
        ("30d ago", now - timedelta(days=30)),
        ("60d ago", now - timedelta(days=60)),
        ("90d ago", now - timedelta(days=90)),
        ("training start 2026-01-29", datetime(2026, 1, 29, 12, 0, tzinfo=timezone.utc)),
    ]
    instruments = [
        ("coinbase", "ETH-USD"),
        ("coinbase", "ETH-USDT"),
    ]
    for label, dt in checkpoints:
        print(f"\n=== {label} ===")
        for market, instr in instruments:
            _probe(client, market, instr, dt)


if __name__ == "__main__":
    main()
