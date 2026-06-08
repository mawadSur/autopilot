"""Coinbase Advanced Trade WebSocket market-data stream.

Subscribes to ``ticker`` + ``candles`` channels on
``wss://advanced-trade-ws.coinbase.com`` and maintains thread-safe
in-memory caches so synchronous callers (``CoinbaseExchange.get_ticker``,
``CoinbaseExchange.fetch_recent_candles``) can serve hot data without
hitting REST on every call.

Architecture
------------
* A background daemon thread owns an ``asyncio`` event loop that runs
  the WS connection.
* Sync callers read from caches protected by ``threading.Lock``.
* Subscriptions are lazy: the first read for a new symbol fires a
  ``subscribe`` frame; reconnects re-subscribe everything from the
  tracked set.
* No-op when ``websockets`` isn't importable (treat WS as soft
  dependency; REST always works).

Cache semantics
---------------
* Ticker: latest ``Ticker`` plus a monotonic timestamp; ``get_ticker``
  treats anything older than ``ticker_freshness_s`` (default 10s) as
  stale and signals fallback.
* Candles: per-symbol ``OrderedDict[unix_minute, dict]`` keyed by bar
  start; pruned to ``max_candles`` (default 400 bars ~ 6.7h of 1m).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

LOGGER = logging.getLogger(__name__)

WS_URL = "wss://advanced-trade-ws.coinbase.com"
DEFAULT_TICKER_FRESHNESS_S = 10.0
DEFAULT_CANDLE_FRESHNESS_S = 90.0
DEFAULT_MAX_CANDLES = 400


def _to_product_id(symbol: str) -> str:
    """``ETH/USD`` or ``ETH-USD`` -> ``ETH-USD`` (Coinbase product id)."""
    if "/" in symbol:
        return symbol.replace("/", "-").upper()
    return symbol.upper()


def _from_product_id(product_id: str) -> str:
    """``ETH-USD`` -> ``ETH/USD`` (ccxt-style symbol used elsewhere)."""
    if "-" in product_id:
        base, quote = product_id.split("-", 1)
        return f"{base.upper()}/{quote.upper()}"
    return product_id.upper()


class CoinbaseMarketStream:
    """Coinbase Advanced Trade WS client with ticker + candles caches.

    Lifecycle:

        stream = CoinbaseMarketStream()
        stream.start()                       # spawn thread + loop
        stream.ensure_subscribed("ETH-USD")  # idempotent
        snapshot = stream.get_ticker("ETH/USD")
        bars = stream.get_candles("ETH/USD", limit=350)
        stream.stop()                        # graceful shutdown
    """

    def __init__(
        self,
        *,
        ws_url: str = WS_URL,
        reconnect_delay_s: float = 5.0,
        ticker_freshness_s: float = DEFAULT_TICKER_FRESHNESS_S,
        candle_freshness_s: float = DEFAULT_CANDLE_FRESHNESS_S,
        max_candles: int = DEFAULT_MAX_CANDLES,
    ) -> None:
        self.ws_url = ws_url
        self.reconnect_delay_s = float(reconnect_delay_s)
        self.ticker_freshness_s = float(ticker_freshness_s)
        self.candle_freshness_s = float(candle_freshness_s)
        self.max_candles = int(max_candles)

        # Caches. ``_lock`` guards both. Cheap to hold; reads are O(1)
        # and writes happen at WS rate (<10 Hz for ticker, ~1/min for
        # candle closes), so contention is negligible.
        self._lock = threading.Lock()
        self._tickers: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self._candles: Dict[str, "OrderedDict[int, Dict[str, Any]]"] = {}

        # Subscriptions tracked here so reconnects can replay them.
        self._tracked_products: Set[str] = set()
        # Pending subscribe requests the loop has not yet flushed onto
        # the socket. Drained at connect time and after each new add.
        self._pending_subscribes: "asyncio.Queue[str]" = None  # type: ignore[assignment]

        # Thread / loop management.
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop = threading.Event()
        self._connected = threading.Event()

    # ------------------------------------------------------------------
    # Public sync API
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Spawn the background loop. Idempotent."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._connected.clear()
        self._thread = threading.Thread(
            target=self._thread_main,
            name="coinbase-ws-stream",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, timeout_s: float = 5.0) -> None:
        """Signal the loop to exit and join the thread."""
        self._stop.set()
        loop = self._loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass
        t = self._thread
        if t is not None:
            t.join(timeout=timeout_s)

    def ensure_subscribed(self, symbol: str) -> None:
        """Add ``symbol`` to the tracked set; subscribe on next reconnect or now."""
        product_id = _to_product_id(symbol)
        with self._lock:
            if product_id in self._tracked_products:
                return
            self._tracked_products.add(product_id)
        # If the loop is already running, hand the new subscribe to it.
        loop = self._loop
        if loop is not None and self._pending_subscribes is not None:
            try:
                loop.call_soon_threadsafe(
                    self._pending_subscribes.put_nowait, product_id
                )
            except RuntimeError:
                # Loop closed -- reconnect will pick it up from the set.
                pass

    def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return the cached ticker dict if fresh, else None.

        Caller is expected to fall back to REST on ``None``. Dict shape
        matches the parsed fields the REST path already exposes:
        ``bid``, ``ask``, ``last``, ``volume_24h_base``, ``as_of_utc``.
        """
        product_id = _to_product_id(symbol)
        now = time.monotonic()
        with self._lock:
            entry = self._tickers.get(product_id)
        if entry is None:
            return None
        ticker_dict, ts = entry
        if now - ts > self.ticker_freshness_s:
            return None
        return dict(ticker_dict)

    def get_candles(self, symbol: str, *, limit: int = 350) -> Optional[List[Dict[str, Any]]]:
        """Return cached candles sorted oldest-first if we have ``>= limit`` fresh.

        Returns ``None`` when cache is empty, too short, or stale. ``stale``
        means the newest bar is older than ``candle_freshness_s`` (default
        90s) — a healthy 1m feed pushes the in-progress bar every few
        seconds.
        """
        product_id = _to_product_id(symbol)
        with self._lock:
            bucket = self._candles.get(product_id)
            if not bucket or len(bucket) < limit:
                return None
            # OrderedDict insertion order tracks bar start.
            bars = list(bucket.values())[-limit:]
        if not bars:
            return None
        # Freshness: newest bar's start + 60 must be within freshness window.
        try:
            newest_start = int(bars[-1].get("_unix", 0))
        except (TypeError, ValueError):
            newest_start = 0
        wall = int(datetime.now(timezone.utc).timestamp())
        if wall - (newest_start + 60) > self.candle_freshness_s:
            return None
        return [dict(b) for b in bars]

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    # ------------------------------------------------------------------
    # Background thread + asyncio loop
    # ------------------------------------------------------------------
    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            self._pending_subscribes = asyncio.Queue()
            loop.run_until_complete(self._run_forever())
        except Exception:  # noqa: BLE001
            LOGGER.exception("CoinbaseMarketStream thread crashed")
        finally:
            try:
                loop.close()
            except Exception:  # noqa: BLE001
                pass
            self._loop = None

    async def _run_forever(self) -> None:
        # Lazy import so the module imports cleanly even when websockets
        # is missing (tests can monkeypatch).
        try:
            import websockets  # type: ignore[import-not-found]
        except ImportError:
            LOGGER.warning(
                "CoinbaseMarketStream: 'websockets' not installed; stream disabled"
            )
            return
        backoff = self.reconnect_delay_s
        while not self._stop.is_set():
            try:
                await self._run_once(websockets)
                backoff = self.reconnect_delay_s  # reset after clean session
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "WS session ended: %r; reconnecting in %.1fs", exc, backoff
                )
            if self._stop.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.5, 60.0)

    async def _run_once(self, websockets_mod: Any) -> None:
        async with websockets_mod.connect(self.ws_url, max_size=None) as ws:
            LOGGER.info("coinbase WS connected: %s", self.ws_url)
            self._connected.set()
            # Replay current tracked subscriptions (fresh socket means
            # the server has no record of them).
            with self._lock:
                to_subscribe = list(self._tracked_products)
            for product_id in to_subscribe:
                await self._send_subscribe(ws, product_id)
            # Then prime-pump the pending queue with any new symbols
            # registered between sessions.
            while not self._pending_subscribes.empty():
                pid = self._pending_subscribes.get_nowait()
                if pid not in to_subscribe:
                    await self._send_subscribe(ws, pid)

            recv_task = asyncio.create_task(self._recv_loop(ws))
            sub_task = asyncio.create_task(self._subscribe_loop(ws))
            try:
                done, pending = await asyncio.wait(
                    {recv_task, sub_task}, return_when=asyncio.FIRST_COMPLETED
                )
                for p in pending:
                    p.cancel()
                # Surface any exception from the completed task.
                for d in done:
                    d.result()
            finally:
                self._connected.clear()

    async def _send_subscribe(self, ws: Any, product_id: str) -> None:
        for channel in ("ticker", "candles"):
            payload = {
                "type": "subscribe",
                "product_ids": [product_id],
                "channel": channel,
            }
            await ws.send(json.dumps(payload))
        LOGGER.info("coinbase WS subscribed: %s (ticker + candles)", product_id)

    async def _subscribe_loop(self, ws: Any) -> None:
        """Drain new subscribe requests onto the live socket."""
        while True:
            product_id = await self._pending_subscribes.get()
            await self._send_subscribe(ws, product_id)

    async def _recv_loop(self, ws: Any) -> None:
        async for raw in ws:
            if self._stop.is_set():
                return
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                LOGGER.debug("coinbase WS: dropped non-JSON frame")
                continue
            try:
                self.handle_message(msg)
            except Exception:  # noqa: BLE001
                LOGGER.exception("coinbase WS handle_message raised")

    # ------------------------------------------------------------------
    # Message routing (split out so tests can call directly)
    # ------------------------------------------------------------------
    def handle_message(self, msg: Dict[str, Any]) -> None:
        channel = msg.get("channel")
        if channel == "ticker":
            self._handle_ticker(msg)
        elif channel == "candles":
            self._handle_candles(msg)
        # else: subscriptions / heartbeats / etc. -- ignore.

    def _handle_ticker(self, msg: Dict[str, Any]) -> None:
        events = msg.get("events") or []
        wall_iso = msg.get("timestamp") or datetime.now(timezone.utc).isoformat()
        ts_mono = time.monotonic()
        for ev in events:
            for t in ev.get("tickers") or []:
                product_id = t.get("product_id")
                if not product_id:
                    continue
                bid = _coerce_f(t.get("best_bid"))
                ask = _coerce_f(t.get("best_ask"))
                last = _coerce_f(t.get("price"))
                volume = _coerce_f(t.get("volume_24_h") or t.get("volume_24h"))
                # Fill in missing bid/ask with last to mirror REST behavior.
                if (bid <= 0 or ask <= 0) and last > 0:
                    if bid <= 0:
                        bid = last
                    if ask <= 0:
                        ask = last
                if bid <= 0 or ask <= 0:
                    continue
                ticker_dict = {
                    "symbol": _from_product_id(product_id),
                    "bid": bid,
                    "ask": ask,
                    "last": last,
                    "volume_24h_base": volume,
                    "as_of_utc": str(wall_iso),
                }
                with self._lock:
                    self._tickers[product_id] = (ticker_dict, ts_mono)

    def _handle_candles(self, msg: Dict[str, Any]) -> None:
        events = msg.get("events") or []
        for ev in events:
            for c in ev.get("candles") or []:
                product_id = c.get("product_id")
                if not product_id:
                    continue
                try:
                    start_unix = int(c["start"])
                except (KeyError, TypeError, ValueError):
                    continue
                row = {
                    "timestamp": datetime.fromtimestamp(
                        start_unix, tz=timezone.utc
                    ).isoformat(),
                    "_unix": start_unix,
                    "open": _coerce_f(c.get("open")),
                    "high": _coerce_f(c.get("high")),
                    "low": _coerce_f(c.get("low")),
                    "close": _coerce_f(c.get("close")),
                    "volume": _coerce_f(c.get("volume")),
                }
                with self._lock:
                    bucket = self._candles.setdefault(product_id, OrderedDict())
                    # Overwrite the existing bar for that minute (the
                    # in-progress bar pushes many updates with the same
                    # start). OrderedDict preserves insertion order, so
                    # we re-insert at the end if missing.
                    if start_unix in bucket:
                        bucket[start_unix] = row
                    else:
                        bucket[start_unix] = row
                        # Re-sort by start (in case out-of-order arrival).
                        if len(bucket) > 1:
                            sorted_items = sorted(bucket.items(), key=lambda kv: kv[0])
                            bucket.clear()
                            for k, v in sorted_items:
                                bucket[k] = v
                    while len(bucket) > self.max_candles:
                        bucket.popitem(last=False)


def _coerce_f(value: Any, default: float = 0.0) -> float:
    """Coinbase ships numerics as strings; coerce safely."""
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
