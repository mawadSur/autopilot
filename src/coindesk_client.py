#!/usr/bin/env python3
"""
coindesk_client.py — reusable CoinDesk Data API client helpers.

Endpoints can be overridden via constructor args or environment variables:
  COINDESK_BASE_URL
  COINDESK_OHLCV_ENDPOINT
  COINDESK_TRADES_ENDPOINT
  COINDESK_L2_METRICS_ENDPOINT
  COINDESK_L2_SNAPSHOTS_ENDPOINT
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


DEFAULT_BASE_URL = "https://data-api.coindesk.com"


@dataclass
class CoinDeskEndpoints:
    ohlcv_minutes: str = "/spot/v1/historical/minutes"
    trades: str = "/spot/v1/historical/trades"
    l2_metrics: str = "/spot/v1/historical/orderbook/metrics"
    l2_snapshots: str = "/spot/v1/historical/orderbook/snapshots"


@dataclass
class CoinDeskConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout_s: int = 30
    max_retries: int = 6
    retry_backoff_s: float = 1.0
    endpoints: CoinDeskEndpoints = field(default_factory=CoinDeskEndpoints)


def _pick(d: Dict[str, Any], *keys: str, default=None):
    for k in keys:
        if k in d:
            return d[k]
        # try case-insensitive
        for kk in d.keys():
            if str(kk).lower() == str(k).lower():
                return d[kk]
    return default


def _ts_to_seconds(ts_val: Any) -> Optional[int]:
    if ts_val is None:
        return None
    try:
        ts_int = int(ts_val)
    except Exception:
        return None
    # Heuristic: milliseconds vs seconds
    if ts_int > 10_000_000_000:  # ms
        return ts_int // 1000
    return ts_int


def _extract_data_list(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Handle both flat and nested CoinDesk/CCC payload shapes.

    Examples:
    - {"Data": [...]}                 -> [...]
    - {"data": [...]}                 -> [...]
    - {"Data": {"Data": [...]}}       -> [...]
    - {"data": {"data": [...]}}       -> [...]
    """
    root = payload.get("Data", payload.get("data", []))
    if isinstance(root, dict):
        nested = root.get("Data", root.get("data", []))
        if isinstance(nested, list):
            return nested
        return []
    if isinstance(root, list):
        return root
    return []


class CoinDeskClient:
    def __init__(self, cfg: CoinDeskConfig):
        self.cfg = cfg
        self.session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Apikey {self.cfg.api_key}",
            "Accept": "application/json",
            "User-Agent": "coindesk-dataset-builder/1.0",
        }

    def _request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = self.cfg.base_url.rstrip("/") + endpoint
        params = dict(params)
        params["api_key"] = self.cfg.api_key
        last_err: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                r = self.session.get(url, headers=self._headers(), params=params, timeout=self.cfg.timeout_s)
                if r.status_code == 429:
                    time.sleep(self.cfg.retry_backoff_s * attempt)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                time.sleep(self.cfg.retry_backoff_s * attempt)
        raise RuntimeError(f"CoinDesk request failed after retries: {last_err}")

    # ---- OHLCV minutes ----
    def get_ohlcv_minutes(self, *, market: str, instrument: str, to_ts: int, limit: int = 2000,
                          aggregate: int = 1, fill: bool = True, groups: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "market": market,
            "instrument": instrument,
            "to_ts": int(to_ts),
            "limit": int(limit),
            "aggregate": int(aggregate),
            "fill": "true" if fill else "false",
            "response_format": "JSON",
        }
        if groups:
            params["groups"] = groups
        return self._request(self.cfg.endpoints.ohlcv_minutes, params)

    @staticmethod
    def normalize_ohlcv_minutes(payload: Dict[str, Any]) -> pd.DataFrame:
        data = _extract_data_list(payload)
        if not data:
            return pd.DataFrame(columns=[
                "timestamp", "open", "high", "low", "close",
                "volume_base", "volume_quote", "trade_count"
            ])
        rows: List[Dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            ts = _ts_to_seconds(_pick(item, "TIMESTAMP", "timestamp"))
            if ts is None:
                continue
            o = _pick(item, "OPEN", "open")
            h = _pick(item, "HIGH", "high")
            l = _pick(item, "LOW", "low")
            c = _pick(item, "CLOSE", "close")
            vb = _pick(item, "VOLUME", "volume", "BASE_VOLUME", "VOLUME_BASE")
            vq = _pick(item, "QUOTE_VOLUME", "VOLUME_QUOTE")
            tc = _pick(item, "TRADE_COUNT", "TRADES", "NUM_TRADES")
            if any(x is None for x in (o, h, l, c, vb)):
                continue
            rows.append({
                "timestamp": int(ts),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume_base": float(vb),
                "volume_quote": float(vq) if vq is not None else None,
                "trade_count": int(tc) if tc is not None else None,
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # ---- Trades ----
    def get_trades(self, *, market: str, instrument: str, to_ts: int, limit: int = 2000) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "market": market,
            "instrument": instrument,
            "to_ts": int(to_ts),
            "limit": int(limit),
            "response_format": "JSON",
        }
        return self._request(self.cfg.endpoints.trades, params)

    @staticmethod
    def normalize_trades(payload: Dict[str, Any]) -> pd.DataFrame:
        data = payload.get("Data", payload.get("data", []))
        if not isinstance(data, list):
            data = []
        if not data:
            return pd.DataFrame(columns=["timestamp", "price", "qty_base", "qty_quote", "side"])

        rows: List[Dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            ts = _ts_to_seconds(_pick(item, "TIMESTAMP", "timestamp", "time"))
            if ts is None:
                continue
            price = _pick(item, "PRICE", "price")
            qty_base = _pick(item, "QUANTITY", "SIZE", "AMOUNT", "qty", "BASE_VOLUME")
            qty_quote = _pick(item, "QUOTE_VOLUME", "QUOTE_AMOUNT")
            side = _pick(item, "SIDE", "side", "TYPE", "type")
            is_buyer_maker = _pick(item, "IS_BUYER_MAKER", "isBuyerMaker")

            if price is None or qty_base is None:
                continue

            side_norm = None
            if side is not None:
                s = str(side).lower()
                if s in {"buy", "bid", "b"}:
                    side_norm = "buy"
                elif s in {"sell", "ask", "s"}:
                    side_norm = "sell"
            if side_norm is None and is_buyer_maker is not None:
                # Binance-style: isBuyerMaker True => taker is sell
                side_norm = "sell" if bool(is_buyer_maker) else "buy"

            rows.append({
                "timestamp": int(ts),
                "price": float(price),
                "qty_base": float(qty_base),
                "qty_quote": float(qty_quote) if qty_quote is not None else float(price) * float(qty_base),
                "side": side_norm,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.drop_duplicates().sort_values("timestamp").reset_index(drop=True)

    # ---- L2 metrics ----
    def get_l2_metrics(self, *, market: str, instrument: str, to_ts: int, limit: int = 2000) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "market": market,
            "instrument": instrument,
            "to_ts": int(to_ts),
            "limit": int(limit),
            "response_format": "JSON",
        }
        return self._request(self.cfg.endpoints.l2_metrics, params)

    @staticmethod
    def normalize_l2_metrics(payload: Dict[str, Any]) -> pd.DataFrame:
        data = payload.get("Data", payload.get("data", []))
        if not isinstance(data, list):
            data = []
        if not data:
            return pd.DataFrame()
        rows = []
        for item in data:
            if not isinstance(item, dict):
                continue
            ts = _ts_to_seconds(_pick(item, "TIMESTAMP", "timestamp", "time"))
            if ts is None:
                continue
            row = {"timestamp": int(ts)}
            for k, v in item.items():
                if str(k).lower() in {"timestamp", "time"}:
                    continue
                row[str(k).lower()] = v
            rows.append(row)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # ---- L2 snapshots ----
    def get_l2_snapshots(self, *, market: str, instrument: str, to_ts: int, limit: int = 2000) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "market": market,
            "instrument": instrument,
            "to_ts": int(to_ts),
            "limit": int(limit),
            "response_format": "JSON",
        }
        return self._request(self.cfg.endpoints.l2_snapshots, params)

    @staticmethod
    def normalize_l2_snapshots(payload: Dict[str, Any]) -> pd.DataFrame:
        data = payload.get("Data", payload.get("data", []))
        if not isinstance(data, list):
            data = []
        if not data:
            return pd.DataFrame(columns=["timestamp", "bids", "asks"])
        rows = []
        for item in data:
            if not isinstance(item, dict):
                continue
            ts = _ts_to_seconds(_pick(item, "TIMESTAMP", "timestamp", "time"))
            if ts is None:
                continue
            bids = _pick(item, "BIDS", "bids", default=[])
            asks = _pick(item, "ASKS", "asks", default=[])
            rows.append({"timestamp": int(ts), "bids": bids, "asks": asks})
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def build_client_from_env() -> CoinDeskClient:
    api_key = os.getenv("COINDESK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing COINDESK_API_KEY")
    endpoints = CoinDeskEndpoints(
        ohlcv_minutes=os.getenv("COINDESK_OHLCV_ENDPOINT", CoinDeskEndpoints.ohlcv_minutes),
        trades=os.getenv("COINDESK_TRADES_ENDPOINT", CoinDeskEndpoints.trades),
        l2_metrics=os.getenv("COINDESK_L2_METRICS_ENDPOINT", CoinDeskEndpoints.l2_metrics),
        l2_snapshots=os.getenv("COINDESK_L2_SNAPSHOTS_ENDPOINT", CoinDeskEndpoints.l2_snapshots),
    )
    cfg = CoinDeskConfig(
        api_key=api_key,
        base_url=os.getenv("COINDESK_BASE_URL", DEFAULT_BASE_URL),
        endpoints=endpoints,
    )
    return CoinDeskClient(cfg)
