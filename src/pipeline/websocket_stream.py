#!/usr/bin/env python3
"""
WebSocket streaming for real-time market data and trading signals.

Provides:
- Real-time Binance klines streaming via WebSocket (aiohttp)
- Connection management with auto-reconnect
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional

import aiohttp

from dotenv import load_dotenv

load_dotenv()

BINANCE_WS_BASE = os.getenv("BINANCE_WS_BASE", "wss://stream.binance.com:9443")


class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class TradeSignal:
    timestamp: int
    symbol: str
    signal_type: SignalType
    price: float
    confidence: float
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "signal": self.signal_type.value,
            "price": self.price,
            "confidence": self.confidence,
            "threshold": self.threshold,
        }


@dataclass
class KlineData:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool

    @property
    def as_dict(self) -> Dict[str, Any]:
        return {
            "open_time": self.open_time,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "is_closed": self.is_closed,
        }


class BinanceWebSocket:
    """
    Async WebSocket client for Binance streams using aiohttp.
    """

    def __init__(
        self,
        symbol: str = "ethusdt",
        interval: str = "1m",
        testnet: bool = False,
    ):
        self.symbol = symbol.lower()
        self.interval = interval
        self.testnet = testnet

        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._handlers: List[Callable[[KlineData], Any]] = []
        self._signal_handlers: List[Callable[[TradeSignal], Any]] = []

        self._base_url = (
            "wss://testnet.binance.vision/ws" if testnet else f"{BINANCE_WS_BASE}/ws"
        )

        self._last_kline: Optional[KlineData] = None
        self._kline_history: Deque[KlineData] = deque(maxlen=200)

    def add_kline_handler(self, handler: Callable[[KlineData], Any]) -> None:
        self._handlers.append(handler)

    def add_signal_handler(self, handler: Callable[[TradeSignal], Any]) -> None:
        self._signal_handlers.append(handler)

    async def connect(self) -> None:
        stream_name = f"{self.symbol}@kline_{self.interval}"
        uri = f"{self._base_url}/{stream_name}"

        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(uri)
        self._running = True
        print(f"[ws] Connected to {stream_name}")

    async def disconnect(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()
        print("[ws] Disconnected")

    async def stream(self) -> None:
        if not self._ws:
            await self.connect()

        try:
            async for msg in self._ws:
                if not self._running:
                    break
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(json.loads(msg.data))
                elif msg.type in (aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED):
                    print("[ws] Connection closed, reconnecting...")
                    await self._reconnect()
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"[ws] Error: {self._ws.exception()}")
                    await self._reconnect()
                    break
        except Exception as e:
            print(f"[ws] Unexpected error: {e}")
            await self._reconnect()

    async def _reconnect(self) -> None:
        await asyncio.sleep(1)
        try:
            await self.connect()
            await self.stream()
        except Exception as e:
            print(f"[ws] Reconnect failed: {e}")

    async def _handle_message(self, data: Dict) -> None:
        if data.get("e") == "kline":
            kline = data["k"]
            kline_data = KlineData(
                open_time=int(kline["t"]),
                open=float(kline["o"]),
                high=float(kline["h"]),
                low=float(kline["l"]),
                close=float(kline["c"]),
                volume=float(kline["v"]),
                is_closed=bool(kline["x"]),
            )

            self._last_kline = kline_data
            self._kline_history.append(kline_data)

            for handler in self._handlers:
                if asyncio.iscoroutinefunction(handler):
                    await handler(kline_data)
                else:
                    handler(kline_data)

            if kline_data.is_closed:
                await self._emit_signals(kline_data)

    async def _emit_signals(self, kline: KlineData) -> None:
        signal = TradeSignal(
            timestamp=kline.open_time,
            symbol=self.symbol.upper(),
            signal_type=SignalType.HOLD,
            price=kline.close,
            confidence=0.0,
            threshold=0.0,
        )

        for handler in self._signal_handlers:
            if asyncio.iscoroutinefunction(handler):
                await handler(signal)
            else:
                handler(signal)

    @property
    def last_kline(self) -> Optional[KlineData]:
        return self._last_kline

    @property
    def kline_history(self) -> List[KlineData]:
        return self._kline_history.copy()


async def run_websocket_stream(
    symbol: str = "ethusdt",
    interval: str = "1m",
    on_signal: Optional[Callable[[TradeSignal], None]] = None,
) -> None:
    ws = BinanceWebSocket(symbol=symbol, interval=interval)

    if on_signal:
        ws.add_signal_handler(on_signal)

    await ws.connect()

    try:
        await ws.stream()
    except KeyboardInterrupt:
        pass
    finally:
        await ws.disconnect()


__all__ = [
    "SignalType",
    "TradeSignal",
    "KlineData",
    "BinanceWebSocket",
    "run_websocket_stream",
]
