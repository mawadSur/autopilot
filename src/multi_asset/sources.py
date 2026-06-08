"""Unified market-data sources for the multi-asset pipeline.

Every source returns the SAME normalized OHLCV schema so the rest of the
pipeline (``compute_features`` -> dataset -> LSTM) is asset-class agnostic::

    columns = ["timestamp", "open", "high", "low", "close", "volume"]
      * timestamp -- ISO-8601 UTC string ("2026-06-08T13:30:00+00:00"),
                     ascending, de-duplicated. (Matches what
                     ``utils.compute_features`` and ``build_dataset.load_ohlcv``
                     already expect.)
      * open/high/low/close -- float price
      * volume -- float (base units for crypto, shares for equities)

The third-party clients (``ccxt`` for crypto, ``yfinance`` for equities) are
imported lazily *inside* the fetch methods so this module imports — and the
unit tests run — with neither dependency installed. Tests exercise the pure
normalizers (:func:`normalize_ccxt_ohlcv`, :func:`normalize_yfinance`) by
injecting raw frames/rows directly.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

OHLCV_COLUMNS: List[str] = ["timestamp", "open", "high", "low", "close", "volume"]

# Canonical bar-step in seconds per granularity label. Used downstream for the
# time-gap guard when building sequences and for history sizing.
GRANULARITY_SECONDS: Dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "1d": 86_400,
}


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to the canonical schema: typed, UTC-ISO ts, sorted, de-duped."""
    if df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)
    # Normalize to a stable ISO string so CSV round-trips are byte-identical.
    out["timestamp"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%S%z").str.replace(
        r"(\d{2})(\d{2})$", r"\1:\2", regex=True
    )
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    out = out[OHLCV_COLUMNS]
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.drop_duplicates(subset="timestamp", keep="last")
    # Sort by the underlying instant, not the string, then reset.
    out = out.assign(_k=pd.to_datetime(out["timestamp"], utc=True))
    out = out.sort_values("_k").drop(columns="_k").reset_index(drop=True)
    return out


def normalize_ccxt_ohlcv(rows: Sequence[Sequence[float]]) -> pd.DataFrame:
    """Normalize ccxt's ``fetch_ohlcv`` output ([ms, o, h, l, c, v] rows)."""
    if rows is None or len(rows) == 0:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    arr = np.asarray(rows, dtype="float64")
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(arr[:, 0], unit="ms", utc=True),
            "open": arr[:, 1],
            "high": arr[:, 2],
            "low": arr[:, 3],
            "close": arr[:, 4],
            "volume": arr[:, 5],
        }
    )
    return _finalize(df)


def normalize_yfinance(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a yfinance ``download``/``history`` frame to the schema.

    yfinance returns a DatetimeIndex and columns ``Open High Low Close Volume``
    (sometimes a MultiIndex when multiple tickers are requested). This handles
    the single-ticker shape; the source fetches one ticker at a time.
    """
    if frame is None or len(frame) == 0:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    df = frame.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # Collapse ("Open","AAPL") -> "Open"; keep the price level.
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    # The index column is "Date" (daily) or "Datetime" (intraday).
    ts_col = next((c for c in ("Datetime", "Date", "index") if c in df.columns), df.columns[0])
    rename = {
        ts_col: "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename)
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"yfinance frame missing columns {missing}; got {list(df.columns)}")
    return _finalize(df)


class DataSource(abc.ABC):
    """A source that yields normalized OHLCV for a symbol + granularity."""

    asset_class: str = "unknown"

    @abc.abstractmethod
    def fetch_ohlcv(
        self, symbol: str, *, granularity: str = "1m", days: float = 30.0
    ) -> pd.DataFrame:
        """Return the canonical OHLCV frame for the trailing ``days``."""
        raise NotImplementedError


class CryptoSource(DataSource):
    """Crypto OHLCV via ccxt (default exchange: coinbase)."""

    asset_class = "crypto"

    def __init__(self, exchange_id: str = "coinbase") -> None:
        self.exchange_id = exchange_id

    def fetch_ohlcv(
        self, symbol: str, *, granularity: str = "1m", days: float = 30.0
    ) -> pd.DataFrame:
        import ccxt  # lazy: keep module importable without ccxt

        ex = getattr(ccxt, self.exchange_id)({"enableRateLimit": True, "timeout": 20000})
        timeframe = granularity if granularity != "1d" else "1d"
        step_ms = GRANULARITY_SECONDS[granularity] * 1000
        # ccxt caps page size; page backward from now.
        now_ms = ex.milliseconds()
        since = int(now_ms - days * 86_400_000.0)
        all_rows: List[Sequence[float]] = []
        cursor = since
        while cursor < now_ms:
            batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1000)
            if not batch:
                break
            all_rows.extend(batch)
            cursor = batch[-1][0] + step_ms
            if len(batch) < 1000:
                break
        return normalize_ccxt_ohlcv(all_rows)


class StockSource(DataSource):
    """Equity OHLCV via yfinance (free, no API key).

    Note yfinance intraday history is limited (~60 days for <1d bars); daily
    bars ("1d") go back years. The pipeline works at any granularity as long as
    crypto and stocks share the SAME granularity in a given pooled dataset.
    """

    asset_class = "stock"

    # yfinance interval strings differ from our labels for the daily case.
    _INTERVAL = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m", "1d": "1d"}

    def fetch_ohlcv(
        self, symbol: str, *, granularity: str = "1d", days: float = 365.0
    ) -> pd.DataFrame:
        import yfinance as yf  # lazy: keep module importable without yfinance

        interval = self._INTERVAL[granularity]
        period = f"{int(max(days, 1))}d"
        raw = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        return normalize_yfinance(raw)


def get_source(asset_class: str, **kwargs: Any) -> DataSource:
    """Factory: ``"crypto"`` -> :class:`CryptoSource`, ``"stock"`` -> StockSource."""
    ac = asset_class.lower()
    if ac == "crypto":
        return CryptoSource(**kwargs)
    if ac in ("stock", "equity"):
        return StockSource()
    raise ValueError(f"Unknown asset_class {asset_class!r}; expected 'crypto' or 'stock'.")
