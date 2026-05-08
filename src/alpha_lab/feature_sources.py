"""FeatureSource adapters for the alpha-lab miner.

Phase 4 / E2 commit 4. Two adapters:

* :class:`CryptoFeatureSource` — wraps a per-symbol feature parquet on disk
  (the same shape produced by ``crypto_training/build_dataset.py`` and used
  by :mod:`crypto_training.train_xgboost`). Defaults to reading from
  ``data/crypto/datasets/<symbol>.parquet``. The skeleton supports a
  parquet-only path because the live Coinbase REST connector lives in
  :mod:`exchanges.coinbase` and pulling rolling-window features in real
  time is a separate concern (the regime-memory backfill solves the same
  problem and is the right model to follow when the live wire-up lands).

* :class:`PolymarketFeatureSource` — wraps a Polymarket "macro" market
  (e.g. CPI / FOMC / election forecasts) and exposes its midpoint price as
  a single feature column over time. The skeleton accepts an injected
  fetcher callable so unit tests can stub the network. The default fetcher
  uses :func:`fetcher.fetch_active_markets`, but the actual conversion of
  point-in-time market snapshots into a time series requires a Polymarket
  history API call that this PR explicitly leaves as a TODO.

Both classes implement the :class:`alpha_lab.correlation_miner.FeatureSource`
Protocol. ``build_default_feature_sources`` is the factory function the
nightly runner's CLI calls — it returns an empty list today (no defaults
wired) so production runs are explicit about which sources to enable.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)


__all__ = [
    "CryptoFeatureSource",
    "PolymarketFeatureSource",
    "build_default_feature_sources",
]


# Default columns excluded from the rank-IC mine even when present in the
# parquet. ``label`` is the supervised target (would leak), ``timestamp`` is
# the index (already handled), and ``symbol`` is a metadata column on
# multi-symbol parquets.
_CRYPTO_FEATURE_BLACKLIST = frozenset({"label", "timestamp", "symbol"})


class CryptoFeatureSource:
    """Per-symbol crypto feature source backed by a parquet on disk.

    The parquet schema matches what :mod:`crypto_training.build_dataset`
    emits: ~136 columns of indicators (returns, EMAs, MACD, ATR, etc.) plus a
    ``timestamp`` column and a supervised ``label`` column. The label is
    excluded from the rank-IC mine (it would trivially correlate with future
    return, defeating the purpose).

    For symbols that aren't on disk yet, the source emits a warning at
    construction and ``fetch_window`` returns an empty DataFrame — callers
    should treat absence as "skip this source for tonight's run".

    Args:
        symbol: identifier (e.g. ``"BTC-USD"``). Used as ``self.name``.
        parquet_path: explicit override. When None, defaults to
            ``data/crypto/datasets/<symbol>.parquet`` relative to ``cwd``.
        feature_columns: optional whitelist. Defaults to "all numeric
            columns minus the blacklist".
        timestamp_column: column to promote to a UTC ``DatetimeIndex``.
            Defaults to ``"timestamp"``.
    """

    def __init__(
        self,
        symbol: str,
        *,
        parquet_path: Optional[Path] = None,
        feature_columns: Optional[Sequence[str]] = None,
        timestamp_column: str = "timestamp",
    ) -> None:
        if not symbol:
            raise ValueError("symbol must be a non-empty string")
        self._symbol = str(symbol)
        # Snake-case the symbol for filename lookup (BTC-USD -> btc_usd).
        slug = self._symbol.replace("-", "_").replace("/", "_").lower()
        self._parquet_path = (
            Path(parquet_path)
            if parquet_path is not None
            else Path("data/crypto/datasets") / f"{slug}_v1.parquet"
        )
        self._feature_columns = list(feature_columns) if feature_columns else None
        self._timestamp_column = timestamp_column

        if not self._parquet_path.exists():
            LOGGER.warning(
                "alpha_lab CryptoFeatureSource: parquet missing for %s at %s — "
                "fetch_window will return empty",
                self._symbol,
                self._parquet_path,
            )

    # ------------------------------------------------------------------
    # Protocol surface
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return f"crypto:{self._symbol}"

    @property
    def asset_class(self) -> Any:
        # Lazy import to keep the alpha_lab package import-light when the
        # protocols module isn't on the path (e.g. during a hermetic test).
        try:
            from protocols.tradeable import AssetClass

            return AssetClass.SPOT_CRYPTO
        except Exception:  # noqa: BLE001 - fallback string is fine
            return "spot_crypto"

    @property
    def parquet_path(self) -> Path:
        return self._parquet_path

    @property
    def symbol(self) -> str:
        return self._symbol

    def fetch_window(
        self, start_utc: datetime, end_utc: datetime
    ) -> pd.DataFrame:
        """Read the parquet, slice on the requested window, return numeric features.

        Returns an empty DataFrame on any IO / shape error. The miner already
        treats empty frames as "skip" so a degraded source can't crash the
        nightly run.
        """
        if not self._parquet_path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(self._parquet_path)
        except Exception as exc:  # noqa: BLE001 - hermetic best-effort
            LOGGER.warning(
                "alpha_lab CryptoFeatureSource: failed to read %s: %s",
                self._parquet_path,
                exc,
            )
            return pd.DataFrame()

        if df.empty:
            return df

        # Promote the timestamp column to a UTC DatetimeIndex.
        if self._timestamp_column not in df.columns:
            LOGGER.warning(
                "alpha_lab CryptoFeatureSource: timestamp column %r missing in %s",
                self._timestamp_column,
                self._parquet_path,
            )
            return pd.DataFrame()
        df.index = pd.to_datetime(df[self._timestamp_column], utc=True, errors="coerce")
        df = df.drop(columns=[self._timestamp_column])

        # Time-window filter.
        df = df.loc[(df.index >= pd.Timestamp(start_utc)) & (df.index <= pd.Timestamp(end_utc))]
        if df.empty:
            return df

        # Column selection.
        if self._feature_columns is not None:
            keep = [c for c in self._feature_columns if c in df.columns]
            df = df[keep]
        else:
            df = df.drop(
                columns=[c for c in _CRYPTO_FEATURE_BLACKLIST if c in df.columns],
                errors="ignore",
            )
            df = df.select_dtypes(include="number")
        return df


class PolymarketFeatureSource:
    """Polymarket macro-market feature source.

    Skeleton: exposes a single ``midpoint`` feature derived from a Polymarket
    market's current price. A true time-series view requires Polymarket's
    history endpoint (or a separately-maintained snapshot DB) — this PR
    leaves that wire-up as a TODO and ships a sample-injection seam so unit
    tests can drive the source with synthetic data.

    Args:
        market_id: Polymarket condition_id / market identifier.
        fetcher: optional callable ``(market_id, start, end) -> DataFrame``
            returning a UTC-indexed frame with at least a ``midpoint`` column.
            When None, ``fetch_window`` returns an empty DataFrame and logs a
            "wire-up needed" warning.
    """

    def __init__(
        self,
        market_id: str,
        *,
        fetcher: Optional[
            Callable[[str, datetime, datetime], pd.DataFrame]
        ] = None,
        question: Optional[str] = None,
    ) -> None:
        if not market_id:
            raise ValueError("market_id must be a non-empty string")
        self._market_id = str(market_id)
        self._fetcher = fetcher
        self._question = question or ""

    # ------------------------------------------------------------------
    # Protocol surface
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        if self._question:
            return f"polymarket:{self._market_id}:{self._question[:40]}"
        return f"polymarket:{self._market_id}"

    @property
    def asset_class(self) -> Any:
        try:
            from protocols.tradeable import AssetClass

            return AssetClass.PREDICTION_BINARY
        except Exception:  # noqa: BLE001
            return "prediction_binary"

    @property
    def market_id(self) -> str:
        return self._market_id

    def fetch_window(
        self, start_utc: datetime, end_utc: datetime
    ) -> pd.DataFrame:
        """Return a UTC-indexed DataFrame with at least a ``midpoint`` column.

        Empty DataFrame when no fetcher is wired (the skeleton state) so the
        miner skips the source cleanly.
        """
        if self._fetcher is None:
            LOGGER.warning(
                "alpha_lab PolymarketFeatureSource(%s): no fetcher wired — "
                "see INTEGRATION.md",
                self._market_id,
            )
            return pd.DataFrame()
        try:
            df = self._fetcher(self._market_id, start_utc, end_utc)
        except Exception as exc:  # noqa: BLE001 - hermetic best-effort
            LOGGER.warning(
                "alpha_lab PolymarketFeatureSource(%s): fetcher raised %s",
                self._market_id,
                exc,
            )
            return pd.DataFrame()
        if df is None or not isinstance(df, pd.DataFrame):
            return pd.DataFrame()
        if df.empty:
            return df
        # Coerce to UTC DatetimeIndex if the fetcher emitted a tz-naive index.
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
            df = df.copy()
            df.index = df.index.tz_localize(timezone.utc)
        return df


# ---------------------------------------------------------------------------
# Factory wiring (called by nightly_runner.main)
# ---------------------------------------------------------------------------
def build_default_feature_sources() -> List[Any]:
    """Return the list of sources the production nightly runner should use.

    Today: empty. The nightly runner logs a "no sources wired" warning and
    exits 0 when this returns []. Operators wire production sources by
    overriding this factory in a private deploy module — keeping the
    skeleton repo deploy-config-free.

    Future PR: read ``ALPHA_LAB_CRYPTO_SYMBOLS=BTC-USD,ETH-USD`` and
    ``ALPHA_LAB_POLYMARKET_MARKET_IDS=...`` from env, instantiate one
    source per entry, return the list.
    """
    return []
