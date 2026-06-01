"""Coinbase WS L2 + trade-tape collector for crypto microstructure features.

Subscribes to ``level2`` and ``market_trades`` channels on
``wss://advanced-trade-ws.coinbase.com``, maintains an in-memory order
book per symbol, and on each UTC minute boundary writes a single row per
symbol to a daily parquet. The schema matches
``history_coindesk.L2_FEATURE_COLUMNS`` + ``TRADE_AGG_COLUMNS`` so the
existing ``build_dataset.py`` consumes the data with no code change.

This solves the bucket-2 feature gap: the existing OHLCV CSVs under
``data/crypto/<sym>/1m/`` were never enriched with microstructure (the
CoinDesk historical-orderbook endpoint doesn't exist on the free tier).
This collector captures it going forward.

Run as a long-lived daemon under launchd (see
``ops/launchd/com.autopilot.microstructure-collector.plist``). Use::

    ./.venv/bin/python -m microstructure_collector --symbols ETH-USD,BTC-USD

CLI is the production-mode entry point; the module also exposes
``MicrostructureCollector`` for use in tests with a mock WS.

Output layout::

    data/crypto/<SYM>/microstructure_1m/<YYYY-MM-DD>.parquet

One file per UTC day, one row per minute, columns:
``timestamp`` (UTC, minute-floored) plus the 19 L2 columns + 7 trade
columns. ``timestamp`` is the **start** of the minute (e.g. 12:34:00 covers
trades arriving in [12:34:00, 12:35:00)). The book snapshot is taken at
the minute boundary.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# sys.path shim so `python -m microstructure_collector` works without
# PYTHONPATH=src.
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
import websockets

LOGGER = logging.getLogger("microstructure_collector")

WS_URL = "wss://advanced-trade-ws.coinbase.com"

# Match history_coindesk.L2_FEATURE_COLUMNS + TRADE_AGG_COLUMNS so
# build_dataset.py picks this up without schema changes.
L2_FEATURE_COLUMNS: List[str] = [
    "best_bid", "best_ask", "bid_size_l1", "ask_size_l1",
    "bid_depth_5", "ask_depth_5", "vwap_bid_5", "vwap_ask_5",
    "bid_depth_10", "ask_depth_10", "vwap_bid_10", "vwap_ask_10",
    "bid_depth_20", "ask_depth_20", "vwap_bid_20", "vwap_ask_20",
    "book_poc", "book_va_low", "book_va_high",
]
TRADE_AGG_COLUMNS: List[str] = [
    "trade_count",
    "buy_count",
    "sell_count",
    "taker_buy_volume_base",
    "taker_sell_volume_base",
    "taker_buy_volume_quote",
    "taker_sell_volume_quote",
]
ALL_FEATURE_COLUMNS = ["timestamp", *L2_FEATURE_COLUMNS, *TRADE_AGG_COLUMNS]


@dataclass
class SymbolState:
    """Per-symbol in-memory state.

    The book is two dicts keyed by price; we never trust incoming
    ``offer`` quantities = 0 except to delete a level.
    """

    symbol: str
    bids: Dict[float, float] = field(default_factory=dict)
    asks: Dict[float, float] = field(default_factory=dict)
    book_seeded: bool = False
    # Trade-tape accumulators for the *current minute*; flushed on
    # minute boundary.
    trade_count: int = 0
    buy_count: int = 0
    sell_count: int = 0
    taker_buy_volume_base: float = 0.0
    taker_sell_volume_base: float = 0.0
    taker_buy_volume_quote: float = 0.0
    taker_sell_volume_quote: float = 0.0
    last_trade_ts: Optional[datetime] = None

    def reset_trade_bucket(self) -> None:
        self.trade_count = 0
        self.buy_count = 0
        self.sell_count = 0
        self.taker_buy_volume_base = 0.0
        self.taker_sell_volume_base = 0.0
        self.taker_buy_volume_quote = 0.0
        self.taker_sell_volume_quote = 0.0


def _top_levels(
    book: Dict[float, float], n: int, *, reverse: bool
) -> List[Tuple[float, float]]:
    """Top-N price levels by side. ``reverse=True`` for bids (high→low)."""
    items = sorted(((p, s) for p, s in book.items() if s > 0),
                   key=lambda kv: kv[0], reverse=reverse)
    return items[:n]


def _vwap(levels: List[Tuple[float, float]]) -> float:
    depth = sum(s for _, s in levels)
    if depth <= 0:
        return float("nan")
    return sum(p * s for p, s in levels) / depth


def _book_profile(
    bids: Dict[float, float], asks: Dict[float, float],
) -> Tuple[float, float, float]:
    """Volume profile: point-of-control + 70% value area boundaries.

    Mirrors ``history_coindesk._book_profile`` so live + historical
    parquets carry the same definition.
    """
    book: Dict[float, float] = {}
    for p, s in bids.items():
        if s > 0:
            book[p] = book.get(p, 0.0) + s
    for p, s in asks.items():
        if s > 0:
            book[p] = book.get(p, 0.0) + s
    if not book:
        return float("nan"), float("nan"), float("nan")

    prices = np.array(sorted(book.keys()), dtype=float)
    sizes = np.array([book[p] for p in prices], dtype=float)
    total = float(sizes.sum())
    if total <= 0.0:
        return float("nan"), float("nan"), float("nan")

    poc_idx = int(np.argmax(sizes))
    left = right = poc_idx
    covered = float(sizes[poc_idx])
    target = total * 0.70

    while covered < target and (left > 0 or right < len(prices) - 1):
        left_size = sizes[left - 1] if left > 0 else -1.0
        right_size = sizes[right + 1] if right < len(prices) - 1 else -1.0
        if right_size > left_size and right < len(prices) - 1:
            right += 1
            covered += float(sizes[right])
        elif left > 0:
            left -= 1
            covered += float(sizes[left])
        elif right < len(prices) - 1:
            right += 1
            covered += float(sizes[right])
        else:
            break
    return float(prices[poc_idx]), float(prices[left]), float(prices[right])


def snapshot_features(state: SymbolState, minute_start: datetime) -> Dict[str, float]:
    """Compute the 26-feature row for ``minute_start`` from current state.

    Trade fields read the accumulated bucket since the last flush. L2
    fields read the live book.
    """
    bid_top1 = _top_levels(state.bids, 1, reverse=True)
    ask_top1 = _top_levels(state.asks, 1, reverse=False)
    best_bid = bid_top1[0][0] if bid_top1 else float("nan")
    best_ask = ask_top1[0][0] if ask_top1 else float("nan")
    bid_size_l1 = bid_top1[0][1] if bid_top1 else 0.0
    ask_size_l1 = ask_top1[0][1] if ask_top1 else 0.0

    row: Dict[str, float] = {
        "timestamp": minute_start,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_size_l1": bid_size_l1,
        "ask_size_l1": ask_size_l1,
    }
    for n in (5, 10, 20):
        b = _top_levels(state.bids, n, reverse=True)
        a = _top_levels(state.asks, n, reverse=False)
        row[f"bid_depth_{n}"] = float(sum(s for _, s in b))
        row[f"ask_depth_{n}"] = float(sum(s for _, s in a))
        row[f"vwap_bid_{n}"] = _vwap(b)
        row[f"vwap_ask_{n}"] = _vwap(a)

    poc, va_low, va_high = _book_profile(state.bids, state.asks)
    row["book_poc"] = poc
    row["book_va_low"] = va_low
    row["book_va_high"] = va_high

    row["trade_count"] = float(state.trade_count)
    row["buy_count"] = float(state.buy_count)
    row["sell_count"] = float(state.sell_count)
    row["taker_buy_volume_base"] = float(state.taker_buy_volume_base)
    row["taker_sell_volume_base"] = float(state.taker_sell_volume_base)
    row["taker_buy_volume_quote"] = float(state.taker_buy_volume_quote)
    row["taker_sell_volume_quote"] = float(state.taker_sell_volume_quote)
    return row


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "-")


def daily_parquet_path(out_root: Path, symbol: str, day: datetime) -> Path:
    return (out_root / _safe_symbol(symbol) / "microstructure_1m"
            / f"{day.strftime('%Y-%m-%d')}.parquet")


def append_row_to_daily_parquet(
    out_root: Path, symbol: str, row: Dict[str, float],
) -> Path:
    """Atomic append: read existing day file, concat, write to .tmp, rename.

    Single-writer assumption (one collector process per host). On a
    fresh day, creates the file.
    """
    ts = row["timestamp"]
    if not isinstance(ts, datetime):
        ts = pd.Timestamp(ts).to_pydatetime()
    day = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    path = daily_parquet_path(out_root, symbol, day)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame([row])[ALL_FEATURE_COLUMNS]
    new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], utc=True)

    if path.exists():
        existing = pd.read_parquet(path)
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
        merged = merged.sort_values("timestamp").reset_index(drop=True)
    else:
        merged = new_df

    tmp = path.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def _floor_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


def _next_minute_boundary(now: datetime) -> datetime:
    return _floor_minute(now) + timedelta(minutes=1)


class MicrostructureCollector:
    """asyncio-friendly Coinbase WS collector.

    Long-lived; ``run()`` connects, subscribes, and routes messages
    forever (with auto-reconnect). ``handle_message`` is split out so
    tests can drive it with synthesized payloads.
    """

    def __init__(
        self,
        *,
        symbols: Iterable[str],
        out_root: Path,
        ws_url: str = WS_URL,
        reconnect_delay_s: float = 5.0,
    ) -> None:
        self.symbols: List[str] = [_safe_symbol(s) for s in symbols]
        if not self.symbols:
            raise ValueError("MicrostructureCollector: at least one symbol required")
        self.out_root = Path(out_root).expanduser().resolve()
        self.ws_url = ws_url
        self.reconnect_delay_s = reconnect_delay_s
        self.state: Dict[str, SymbolState] = {
            s: SymbolState(symbol=s) for s in self.symbols
        }
        self._current_minute: Dict[str, datetime] = {}
        self._stop = asyncio.Event()

    # ---------------------------------------------------------------
    # Message handling
    # ---------------------------------------------------------------
    def handle_message(self, msg: dict, *, now: Optional[datetime] = None) -> None:
        """Route one parsed WS message into the state.

        ``now`` is injected for testing; production passes ``None`` and
        we read wall clock. The minute-boundary flush happens here so a
        single test can drive the full lifecycle.

        Ordering matters: we flush the previous minute *before* routing
        the message into state, so a trade that crosses the boundary
        (arrives at 12:01:01 after the last 12:00:xx event) lands in
        the new minute's bucket, not the old one.
        """
        wall = now or datetime.now(timezone.utc)
        self._maybe_flush(wall)

        channel = msg.get("channel") or msg.get("type")
        if channel == "l2_data":
            self._handle_level2(msg)
        elif channel == "market_trades":
            self._handle_market_trades(msg)
        elif channel == "subscriptions":
            LOGGER.info("subscription ack: %s", msg.get("events"))
        elif channel == "heartbeats":
            pass
        else:
            LOGGER.debug("ignoring channel=%s", channel)

    def _maybe_flush(self, wall: datetime) -> None:
        """Flush any per-symbol bucket whose minute has just closed."""
        minute = _floor_minute(wall)
        for sym, st in self.state.items():
            last_seen = self._current_minute.get(sym)
            if last_seen is None:
                self._current_minute[sym] = minute
                continue
            if minute > last_seen and st.book_seeded:
                try:
                    row = snapshot_features(st, last_seen)
                    append_row_to_daily_parquet(self.out_root, sym, row)
                    LOGGER.info(
                        "flushed %s @ %s: best_bid=%s best_ask=%s trades=%d",
                        sym, last_seen.isoformat(timespec="seconds"),
                        row.get("best_bid"), row.get("best_ask"), st.trade_count,
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("flush failed for %s @ %s: %r",
                                     sym, last_seen, exc)
                st.reset_trade_bucket()
                self._current_minute[sym] = minute

    def _handle_level2(self, msg: dict) -> None:
        events = msg.get("events") or []
        for ev in events:
            sym = ev.get("product_id")
            if not sym or sym not in self.state:
                continue
            st = self.state[sym]
            ev_type = ev.get("type")
            updates = ev.get("updates") or []
            if ev_type == "snapshot":
                st.bids.clear()
                st.asks.clear()
            for u in updates:
                try:
                    price = float(u.get("price_level"))
                    size = float(u.get("new_quantity", 0.0))
                except (TypeError, ValueError):
                    continue
                side = (u.get("side") or "").lower()
                target = st.bids if side == "bid" else st.asks if side == "offer" else None
                if target is None:
                    continue
                if size <= 0:
                    target.pop(price, None)
                else:
                    target[price] = size
            if ev_type == "snapshot":
                st.book_seeded = True

    def _handle_market_trades(self, msg: dict) -> None:
        events = msg.get("events") or []
        for ev in events:
            trades = ev.get("trades") or []
            for tr in trades:
                sym = tr.get("product_id")
                if not sym or sym not in self.state:
                    continue
                st = self.state[sym]
                try:
                    price = float(tr.get("price"))
                    size = float(tr.get("size"))
                except (TypeError, ValueError):
                    continue
                side = (tr.get("side") or "").lower()
                qty_quote = price * size
                st.trade_count += 1
                # Coinbase: "side" describes the taker side (BUY = taker
                # bought i.e. lifted the ask).
                if side == "buy":
                    st.buy_count += 1
                    st.taker_buy_volume_base += size
                    st.taker_buy_volume_quote += qty_quote
                elif side == "sell":
                    st.sell_count += 1
                    st.taker_sell_volume_base += size
                    st.taker_sell_volume_quote += qty_quote
                # tr["time"] is ISO8601; we don't need it because the
                # bucket is keyed on the wall-clock minute we flush at.

    # ---------------------------------------------------------------
    # WS lifecycle
    # ---------------------------------------------------------------
    async def _subscribe(self, ws) -> None:
        """Send subscribe frames for both channels for all symbols."""
        for channel in ("level2", "market_trades"):
            payload = {
                "type": "subscribe",
                "product_ids": list(self.symbols),
                "channel": channel,
            }
            await ws.send(json.dumps(payload))
            LOGGER.info("subscribed channel=%s products=%s", channel, self.symbols)

    async def _run_once(self) -> None:
        """One WS session. Returns on disconnect; caller reconnects."""
        async with websockets.connect(self.ws_url, max_size=None) as ws:
            LOGGER.info("WS connected: %s", self.ws_url)
            await self._subscribe(ws)
            async for raw in ws:
                if self._stop.is_set():
                    return
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    LOGGER.warning("dropped non-JSON frame")
                    continue
                try:
                    self.handle_message(msg)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("handle_message raised: %r", exc)

    async def run(self) -> None:
        """Connect-forever loop with bounded backoff."""
        while not self._stop.is_set():
            try:
                await self._run_once()
            except (websockets.ConnectionClosed, OSError) as exc:
                LOGGER.warning("WS disconnected: %r; reconnecting in %.1fs",
                               exc, self.reconnect_delay_s)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("WS session raised: %r; reconnecting in %.1fs",
                                 exc, self.reconnect_delay_s)
            if self._stop.is_set():
                break
            try:
                await asyncio.wait_for(self._stop.wait(),
                                       timeout=self.reconnect_delay_s)
            except asyncio.TimeoutError:
                pass

    async def run_for(self, duration_s: float) -> None:
        """Run ``run()`` for at most ``duration_s`` seconds then stop cleanly.

        Used by smoke tests / one-shot CLI invocations. Production launchd
        invocation uses ``run()`` directly (no timeout).
        """
        run_task = asyncio.create_task(self.run())
        try:
            await asyncio.wait_for(asyncio.shield(self._sleep_until_stop(duration_s)),
                                   timeout=duration_s + 1.0)
        except asyncio.TimeoutError:
            pass
        finally:
            self.stop()
            try:
                await asyncio.wait_for(run_task, timeout=5.0)
            except asyncio.TimeoutError:
                run_task.cancel()
                try:
                    await run_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass

    async def _sleep_until_stop(self, duration_s: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=duration_s)
        except asyncio.TimeoutError:
            pass

    def stop(self) -> None:
        self._stop.set()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--symbols", default="ETH-USD",
                   help="Comma-separated Coinbase product IDs (e.g. ETH-USD,BTC-USD)")
    p.add_argument("--out-root", default="data/crypto",
                   help="Parent directory for <SYM>/microstructure_1m/<date>.parquet")
    p.add_argument("--log-level", default="INFO")
    p.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help=(
            "If set, exit cleanly after this many seconds (smoke-test mode). "
            "Default: run forever (production / launchd)."
        ),
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    collector = MicrostructureCollector(
        symbols=symbols,
        out_root=Path(args.out_root),
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, collector.stop)
        except NotImplementedError:
            pass
    try:
        if args.duration_s is not None:
            loop.run_until_complete(collector.run_for(float(args.duration_s)))
        else:
            loop.run_until_complete(collector.run())
    finally:
        loop.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
