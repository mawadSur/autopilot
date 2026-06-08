"""Declarative multi-asset universe.

Replaces the hardcoded ``SYMBOLS = [...]`` lists scattered across the crypto
scripts with one place that lists every instrument the pooled model trains on,
its asset class, and its data source. Persisted as JSON so a run is reproducible
and the universe can be edited without touching code.

On-disk data layout (one tree per asset class, mirrors the existing
``data/crypto/<SYM>/<gran>/`` convention)::

    data/crypto/BTC-USD/1d/ohlcv.csv
    data/crypto/ETH-USD/1d/ohlcv.csv
    data/stocks/AAPL/1d/ohlcv.csv
    data/stocks/SPY/1d/ohlcv.csv
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List


def safe_symbol(symbol: str) -> str:
    """Filesystem-safe symbol: ``ETH/USD`` -> ``ETH-USD``, ``AAPL`` -> ``AAPL``."""
    return symbol.replace("/", "-").replace(":", "-")


@dataclass(frozen=True)
class Instrument:
    symbol: str            # source-native symbol, e.g. "BTC/USD" or "AAPL"
    asset_class: str       # "crypto" | "stock"
    exchange: str = ""     # ccxt exchange id for crypto; "" for stocks (yfinance)

    @property
    def asset_id(self) -> str:
        """Stable pooled-model identity: ``crypto:BTC-USD`` / ``stock:AAPL``."""
        return f"{self.asset_class}:{safe_symbol(self.symbol)}"

    def data_dir(self, data_root: Path, granularity: str) -> Path:
        tree = "crypto" if self.asset_class == "crypto" else "stocks"
        return Path(data_root) / tree / safe_symbol(self.symbol) / granularity


@dataclass
class Universe:
    granularity: str
    instruments: List[Instrument] = field(default_factory=list)

    def asset_ids(self) -> List[str]:
        return [i.asset_id for i in self.instruments]

    def to_json(self, path: Path) -> None:
        payload = {
            "granularity": self.granularity,
            "instruments": [asdict(i) for i in self.instruments],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def from_json(path: Path) -> "Universe":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return Universe(
            granularity=str(data["granularity"]),
            instruments=[Instrument(**d) for d in data["instruments"]],
        )


# A modest, key-free default: liquid crypto on Coinbase + liquid US equities via
# yfinance. Daily bars so stocks aren't capped by yfinance's ~60-day intraday
# window — crypto and stocks must share one granularity per pooled dataset.
DEFAULT_UNIVERSE = Universe(
    granularity="1d",
    instruments=[
        Instrument("BTC/USD", "crypto", "coinbase"),
        Instrument("ETH/USD", "crypto", "coinbase"),
        Instrument("SOL/USD", "crypto", "coinbase"),
        Instrument("AAPL", "stock"),
        Instrument("MSFT", "stock"),
        Instrument("SPY", "stock"),
        Instrument("NVDA", "stock"),
    ],
)
