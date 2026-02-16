#!/usr/bin/env python3
"""
live_coindesk_collector.py — live CoinDesk WebSocket collector + inference + (paper) execution.

Aggregates trades + top-of-book into 1-minute buckets, builds the same feature
schema as training, runs inference, and optionally sends orders via ccxt.

Env:
  COINDESK_API_KEY (required)
  COINDESK_WS_URL (optional; override websocket URL)
  COINDESK_WS_MARKET / COINDESK_WS_INSTRUMENT (optional defaults)

Execution venue (ccxt coinbase):
  COINBASE_API_KEY / COINBASE_API_SECRET / COINBASE_PASSPHRASE
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

import aiohttp
import numpy as np
import pandas as pd

import ccxt  # type: ignore

from utils import compute_features, load_model_bundle, FEATURE_COLUMNS


DEFAULT_WS_URL = "wss://data-api.coindesk.com/streaming/spot"


def _ts_to_seconds(ts_val: Any) -> Optional[int]:
    if ts_val is None:
        return None
    try:
        ts_int = int(ts_val)
    except Exception:
        return None
    if ts_int > 10_000_000_000:
        return ts_int // 1000
    return ts_int


def _parse_trade(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Try common shapes
    data = msg.get("data") or msg.get("Data") or msg
    if isinstance(data, list):
        # take first trade
        if not data:
            return None
        data = data[0]
    if not isinstance(data, dict):
        return None
    price = data.get("price") or data.get("PRICE")
    qty = data.get("size") or data.get("qty") or data.get("quantity") or data.get("amount") or data.get("QUANTITY")
    side = data.get("side") or data.get("SIDE")
    ts = _ts_to_seconds(data.get("timestamp") or data.get("time") or data.get("TIMESTAMP"))
    if price is None or qty is None or ts is None:
        return None
    side_norm = None
    if side is not None:
        s = str(side).lower()
        if s in {"buy", "b"}:
            side_norm = "buy"
        elif s in {"sell", "s"}:
            side_norm = "sell"
    return {"timestamp": ts, "price": float(price), "qty": float(qty), "side": side_norm}


def _parse_book(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    data = msg.get("data") or msg.get("Data") or msg
    if not isinstance(data, dict):
        return None
    bid = data.get("best_bid") or data.get("bid") or data.get("bid_price")
    ask = data.get("best_ask") or data.get("ask") or data.get("ask_price")
    bid_sz = data.get("bid_size") or data.get("bid_qty") or data.get("bid_size_l1")
    ask_sz = data.get("ask_size") or data.get("ask_qty") or data.get("ask_size_l1")
    ts = _ts_to_seconds(data.get("timestamp") or data.get("time") or data.get("TIMESTAMP"))
    if bid is None or ask is None or ts is None:
        return None
    return {
        "timestamp": ts,
        "best_bid": float(bid),
        "best_ask": float(ask),
        "bid_size_l1": float(bid_sz) if bid_sz is not None else math.nan,
        "ask_size_l1": float(ask_sz) if ask_sz is not None else math.nan,
    }


@dataclass
class MinuteBucket:
    ts: int
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume_base: float = 0.0
    volume_quote: float = 0.0
    trade_count: int = 0
    buy_count: int = 0
    sell_count: int = 0
    taker_buy_volume_base: float = 0.0
    taker_sell_volume_base: float = 0.0
    taker_buy_volume_quote: float = 0.0
    taker_sell_volume_quote: float = 0.0
    best_bid: float = math.nan
    best_ask: float = math.nan
    bid_size_l1: float = math.nan
    ask_size_l1: float = math.nan

    def add_trade(self, price: float, qty: float, side: Optional[str]):
        if self.open is None:
            self.open = price
            self.high = price
            self.low = price
        self.high = max(self.high, price) if self.high is not None else price
        self.low = min(self.low, price) if self.low is not None else price
        self.close = price
        self.volume_base += qty
        self.volume_quote += price * qty
        self.trade_count += 1
        if side == "buy":
            self.buy_count += 1
            self.taker_buy_volume_base += qty
            self.taker_buy_volume_quote += price * qty
        elif side == "sell":
            self.sell_count += 1
            self.taker_sell_volume_base += qty
            self.taker_sell_volume_quote += price * qty

    def update_book(self, best_bid, best_ask, bid_size_l1, ask_size_l1):
        self.best_bid = best_bid
        self.best_ask = best_ask
        self.bid_size_l1 = bid_size_l1
        self.ask_size_l1 = ask_size_l1

    def finalize(self, last_trade_price: Optional[float]) -> Optional[Dict[str, Any]]:
        if self.open is None and last_trade_price is None:
            return None
        if self.open is None:
            self.open = last_trade_price
            self.high = last_trade_price
            self.low = last_trade_price
            self.close = last_trade_price
        if self.close is None:
            self.close = self.open
        return {
            "timestamp": self.ts,
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": float(self.volume_base),
            "volume_base": float(self.volume_base),
            "volume_quote": float(self.volume_quote),
            "trade_count": int(self.trade_count),
            "buy_count": int(self.buy_count),
            "sell_count": int(self.sell_count),
            "taker_buy_volume_base": float(self.taker_buy_volume_base),
            "taker_sell_volume_base": float(self.taker_sell_volume_base),
            "taker_buy_volume_quote": float(self.taker_buy_volume_quote),
            "taker_sell_volume_quote": float(self.taker_sell_volume_quote),
            "best_bid": float(self.best_bid) if self.best_bid == self.best_bid else math.nan,
            "best_ask": float(self.best_ask) if self.best_ask == self.best_ask else math.nan,
            "bid_size_l1": float(self.bid_size_l1) if self.bid_size_l1 == self.bid_size_l1 else math.nan,
            "ask_size_l1": float(self.ask_size_l1) if self.ask_size_l1 == self.ask_size_l1 else math.nan,
        }


class LiveAggregator:
    def __init__(self):
        self.current_ts: Optional[int] = None
        self.bucket: Optional[MinuteBucket] = None
        self.last_trade_price: Optional[float] = None
        self.last_book: Dict[str, float] = {}
        self.rows: List[Dict[str, Any]] = []

    def _roll_to(self, new_minute: int):
        # finalize current bucket
        if self.bucket:
            row = self.bucket.finalize(self.last_trade_price)
            if row:
                self.rows.append(row)
        # create new bucket
        self.current_ts = new_minute
        self.bucket = MinuteBucket(ts=new_minute)
        if self.last_book:
            self.bucket.update_book(
                self.last_book.get("best_bid", math.nan),
                self.last_book.get("best_ask", math.nan),
                self.last_book.get("bid_size_l1", math.nan),
                self.last_book.get("ask_size_l1", math.nan),
            )

    def on_trade(self, ts: int, price: float, qty: float, side: Optional[str]):
        minute_ts = (ts // 60) * 60
        if self.current_ts is None:
            self._roll_to(minute_ts)
        elif minute_ts > self.current_ts:
            # roll through any gaps
            while self.current_ts is not None and minute_ts > self.current_ts:
                self._roll_to(self.current_ts + 60)
        # update
        self.last_trade_price = price
        self.bucket.add_trade(price, qty, side)

    def on_book(self, ts: int, best_bid: float, best_ask: float, bid_size_l1: float, ask_size_l1: float):
        minute_ts = (ts // 60) * 60
        if self.current_ts is None:
            self._roll_to(minute_ts)
        elif minute_ts > self.current_ts:
            while self.current_ts is not None and minute_ts > self.current_ts:
                self._roll_to(self.current_ts + 60)
        self.last_book = {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bid_size_l1": bid_size_l1,
            "ask_size_l1": ask_size_l1,
        }
        if self.bucket:
            self.bucket.update_book(best_bid, best_ask, bid_size_l1, ask_size_l1)

    def flush_ready(self) -> List[Dict[str, Any]]:
        out = self.rows
        self.rows = []
        return out


class Trader:
    def __init__(self, *, paper: bool, symbol: str, quote_qty: float, allow_shorts: bool):
        self.paper = paper
        self.symbol = symbol
        self.quote_qty = float(quote_qty)
        self.allow_shorts = allow_shorts
        self.position = 0  # -1 short, 0 flat, 1 long
        self.exchange = None
        if not paper:
            key = os.getenv("COINBASE_API_KEY")
            secret = os.getenv("COINBASE_API_SECRET")
            password = os.getenv("COINBASE_PASSPHRASE")
            if not key or not secret or not password:
                raise RuntimeError("Missing Coinbase API credentials")
            self.exchange = ccxt.coinbase({
                "apiKey": key,
                "secret": secret,
                "password": password,
            })

    def _market_buy(self):
        if self.paper:
            print(f"[paper] BUY {self.symbol} for {self.quote_qty}")
            self.position = 1
            return
        self.exchange.create_market_buy_order(self.symbol, None, {"quoteOrderQty": self.quote_qty})
        self.position = 1

    def _market_sell(self):
        if self.paper:
            print(f"[paper] SELL {self.symbol} for {self.quote_qty}")
            self.position = 0
            return
        self.exchange.create_market_sell_order(self.symbol, None, {"quoteOrderQty": self.quote_qty})
        self.position = 0

    def on_signal(self, signal: int):
        if signal == 1 and self.position == 0:
            self._market_buy()
        elif signal == -1 and self.position == 1:
            self._market_sell()


def decide_signal(meta: Dict[str, Any], probs: np.ndarray, last_row: pd.Series) -> int:
    p_short, p_hold, p_long = probs[0], probs[1], probs[2]
    thr_long = float(meta.get("thr_long", meta.get("buy_threshold", 0.75)))
    thr_short = float(meta.get("thr_short", meta.get("sell_threshold", 0.75)))
    fee_pct = float(meta.get("fee_pct", 0.0001))
    slippage_pct = float(meta.get("slippage_pct", 0.0001))
    round_trip_cost = float(meta.get("round_trip_cost", 2.0 * (fee_pct + slippage_pct)))
    cost_mult = float(meta.get("cost_mult", 1.5))
    k_tp = float(meta.get("k_tp", 1.2))
    k_sl = float(meta.get("k_sl", 1.0))

    atrp = float(last_row.get("atrp_14", np.nan))
    if not np.isfinite(atrp):
        tp = max(cost_mult * round_trip_cost, float(meta.get("tp_pct", 0.01)))
        sl = max(cost_mult * round_trip_cost, float(meta.get("sl_pct", 0.005)))
    else:
        tp = max(cost_mult * round_trip_cost, k_tp * atrp)
        sl = max(cost_mult * round_trip_cost, k_sl * atrp)

    ev_long = p_long * tp - p_short * sl - round_trip_cost
    ev_short = p_short * tp - p_long * sl - round_trip_cost

    if p_long >= thr_long and ev_long > 0:
        return 1
    if p_short >= thr_short and ev_short > 0:
        return -1
    return 0


async def run(args):
    api_key = os.getenv("COINDESK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing COINDESK_API_KEY")

    model, scaler, meta = load_model_bundle(args.model_dir)
    feature_cols = list(meta.get("feature_cols", FEATURE_COLUMNS))
    if feature_cols != FEATURE_COLUMNS:
        raise ValueError("Feature list mismatch between meta and utils.FEATURE_COLUMNS")
    window_size = int(meta.get("window_size", 90))

    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        assert list(scaler.feature_names_in_) == feature_cols, "Scaler feature order mismatch"

    model.eval()
    device = "cuda" if args.device == "cuda" and hasattr(model, "to") else "cpu"
    model = model.to(device)

    trader = Trader(paper=args.paper, symbol=args.exec_symbol, quote_qty=args.quote_qty, allow_shorts=args.allow_shorts)
    agg = LiveAggregator()
    history: List[Dict[str, Any]] = []

    ws_url = args.ws_url or os.getenv("COINDESK_WS_URL", DEFAULT_WS_URL)
    market = args.market
    instrument = args.instrument

    sub_msg = {
        "type": "subscribe",
        "channels": [
            {"name": "trades", "market": market, "instrument": instrument},
            {"name": "book", "market": market, "instrument": instrument},
        ],
        "api_key": api_key,
    }

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url) as ws:
            await ws.send_json(sub_msg)

            async def flush_loop():
                while True:
                    await asyncio.sleep(1)
                    now_min = int(time.time() // 60) * 60
                    if agg.current_ts is None:
                        continue
                    if now_min > agg.current_ts:
                        while agg.current_ts < now_min:
                            agg._roll_to(agg.current_ts + 60)
                    rows = agg.flush_ready()
                    for row in rows:
                        history.append(row)
                        df = pd.DataFrame(history)
                        feats = compute_features(df)
                        window = feats[feature_cols].tail(window_size)
                        if len(window) < window_size:
                            continue
                        X = window.to_numpy(dtype=np.float32, copy=False).reshape(1, window_size, -1)
                        if scaler is not None:
                            B, T, F = X.shape
                            X = scaler.transform(X.reshape(B * T, F)).reshape(B, T, F)
                        import torch
                        xb = torch.from_numpy(X).to(device)
                        with torch.no_grad():
                            logits = model(xb)
                            probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
                        sig = decide_signal(meta, probs, feats.iloc[-1])
                        if sig != 0:
                            trader.on_signal(sig)

            asyncio.create_task(flush_loop())

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except Exception:
                        continue
                    trade = _parse_trade(data)
                    if trade:
                        agg.on_trade(trade["timestamp"], trade["price"], trade["qty"], trade["side"])
                        continue
                    book = _parse_book(data)
                    if book:
                        agg.on_book(book["timestamp"], book["best_bid"], book["best_ask"], book["bid_size_l1"], book["ask_size_l1"])
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break


def main():
    ap = argparse.ArgumentParser(description="Live CoinDesk collector + inference")
    ap.add_argument("--market", default=os.getenv("COINDESK_WS_MARKET", "coinbase"))
    ap.add_argument("--instrument", default=os.getenv("COINDESK_WS_INSTRUMENT", "ETH-USDT"))
    ap.add_argument("--ws-url", default=None)
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--paper", action="store_true", default=True)
    ap.add_argument("--allow-shorts", action="store_true", default=False)
    ap.add_argument("--quote-qty", type=float, default=15.0)
    ap.add_argument("--exec-symbol", default="ETH/USD")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
