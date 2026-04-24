#!/usr/bin/env python3
"""
Real-time market data consumer via Binance WebSocket.

Connects to Binance 1-minute kline stream and:
- Receives completed candles
- Caches in Redis for feature computation
- Maintains price/volume history
"""

import asyncio
import json
import os
import argparse
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp
import redis
from dotenv import load_dotenv

load_dotenv()

# Redis connection
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

BINANCE_WS_BASE = os.getenv("BINANCE_WS_BASE", "wss://stream.binance.com:9443")

class BinanceWebSocketStream:
    """Binance WebSocket 1-minute kline stream consumer."""

    def __init__(self, symbol: str = "ETHUSDT", redis_host: str = REDIS_HOST, redis_port: int = REDIS_PORT):
        self.symbol = symbol.upper()
        self.candle_stream = f"stream:{self.symbol}:1m"  # Use stream for live data
        self.cache_key = f"cache:{self.symbol}:latest_candle"
        
        # Redis connection
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # WebSocket
        self.ws_url = f"{BINANCE_WS_BASE}/ws/{self.symbol.lower()}@kline_1m"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # State
        self.candle_count = 0
        self.last_close_time = None
        self.running = False

    async def connect(self):
        """Connect to Binance WebSocket stream."""
        self.session = aiohttp.ClientSession()
        self.running = True
        
        print(f"🔗 Connecting to Binance stream: {self.symbol} 1m...")
        
        try:
            async with self.session.ws_connect(self.ws_url) as ws:
                print(f"✓ Connected to {self.ws_url}")
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await self._handle_message(json.loads(msg.data))
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        print(f"✗ WebSocket error: {ws.exception()}")
                        break
        except Exception as e:
            print(f"✗ Connection error: {e}")
        finally:
            self.running = False
            if self.session:
                await self.session.close()

    async def _handle_message(self, data: Dict):
        """Process incoming kline data."""
        try:
            event_data = data.get("k", {})
            
            # Extract candle data
            candle = {
                "symbol": self.symbol,
                "time": event_data.get("t"),
                "close_time": event_data.get("T"),
                "open": float(event_data.get("o")),
                "high": float(event_data.get("h")),
                "low": float(event_data.get("l")),
                "close": float(event_data.get("c")),
                "volume": float(event_data.get("v")),
                "quote_asset_volume": float(event_data.get("q", 0)),
                "trades": int(event_data.get("n", 0)),
                "timestamp": datetime.now().isoformat(),
            }
            
            is_closed = event_data.get("x", False)
            
            if is_closed:
                # Candle is complete, store in Redis
                await self._store_candle(candle)
                self.candle_count += 1
                self.last_close_time = candle["close_time"]
                
                print(
                    f"✓ [{self.candle_count:05d}] {self.symbol} | "
                    f"O:{candle['open']:.2f} H:{candle['high']:.2f} "
                    f"L:{candle['low']:.2f} C:{candle['close']:.2f} V:{candle['volume']:.0f}"
                )
            else:
                # Candle still building (preview)
                pass
                
        except Exception as e:
            print(f"✗ Error processing message: {e}")

    async def _store_candle(self, candle: Dict):
        """Store completed candle in Redis list and cache."""
        try:
            # Store in Redis list
            candle_data = {
                "time": str(candle["time"]),
                "close_time": str(candle["close_time"]),
                "open": str(candle["open"]),
                "high": str(candle["high"]),
                "low": str(candle["low"]),
                "close": str(candle["close"]),
                "volume": str(candle["volume"]),
            }
            # Use LPUSH + LTRIM to simulate a capped stream/list
            self.redis_client.lpush(self.candle_stream, json.dumps(candle_data))
            self.redis_client.ltrim(self.candle_stream, 0, 1000)
            
            # Update latest candle cache
            self.redis_client.hset(
                self.cache_key,
                mapping={
                    "time": candle["time"],
                    "open": candle["open"],
                    "high": candle["high"],
                    "low": candle["low"],
                    "close": candle["close"],
                    "volume": candle["volume"],
                    "timestamp": candle["timestamp"],
                }
            )
            
            # Publish event
            self.redis_client.publish(
                f"candle:{self.symbol}:complete",
                json.dumps(candle)
            )
            
        except Exception as e:
            print(f"✗ Error storing candle: {e}")

    def get_latest_candles(self, count: int = 100) -> List[Dict]:
        """Get latest N candles from list."""
        try:
            entries = self.redis_client.lrange(self.candle_stream, 0, count - 1)
            candles = []
            for entry_data in entries:
                data = json.loads(entry_data)
                candles.append({
                    **{k: float(v) for k, v in data.items()}
                })
            return candles
        except Exception as e:
            print(f"✗ Error fetching candles: {e}")
            return []

    async def disconnect(self):
        """Gracefully disconnect."""
        self.running = False
        if self.session:
            await self.session.close()


async def main():
    """Run the WebSocket stream consumer."""
    parser = argparse.ArgumentParser(description="Binance WebSocket Stream Consumer")
    parser.add_argument("--symbol", type=str, default=os.getenv("TRADE_SYMBOL", "ETHUSDT"), help="Trading symbol (e.g. ETHUSDT)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("BINANCE WEBSOCKET 1M KLINE STREAM CONSUMER")
    print("=" * 60)
    print(f"Streaming: {args.symbol.upper()} 1m candles → Redis")
    print("Press Ctrl+C to stop\n")

    stream = BinanceWebSocketStream(symbol=args.symbol)
    
    try:
        await stream.connect()
    except KeyboardInterrupt:
        print("\n⏹ Shutting down...")
        await stream.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
