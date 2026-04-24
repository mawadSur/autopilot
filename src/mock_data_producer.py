import redis
import json
import time
import random
from datetime import datetime

def mock_producer():
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    symbol = "ETHUSDT"
    list_key = f"stream:{symbol}:1m"
    price = 2500.0
    
    print(f"🚀 Starting mock data producer for {symbol}...")
    
    while True:
        try:
            # Generate a mock candle
            change = random.uniform(-2.0, 2.0)
            new_price = price + change
            candle = {
                "time": int(time.time() * 1000),
                "close_time": int(time.time() * 1000) + 60000,
                "open": str(price),
                "high": str(max(price, new_price) + random.uniform(0, 1)),
                "low": str(min(price, new_price) - random.uniform(0, 1)),
                "close": str(new_price),
                "volume": str(random.uniform(10, 100))
            }
            price = new_price
            
            # LPUSH to Redis
            r.lpush(list_key, json.dumps(candle))
            r.ltrim(list_key, 0, 1000)
            
            # Also push a mock signal occasionally
            if random.random() > 0.7:
                signal = {
                    "id": random.randint(1, 1000),
                    "symbol": symbol,
                    "action": "BUY" if random.random() > 0.5 else "SELL",
                    "confidence": random.uniform(0.6, 0.95),
                    "threshold": 0.6,
                    "timestamp": datetime.now().isoformat(),
                    "candle_time": str(candle["time"]),
                    "features": [0.0] * 17
                }
                r.lpush(f"list:{symbol}:signals", json.dumps(signal))
            
            print(f"✓ Mock candle: {candle['close']}")
            time.sleep(5)
        except Exception as e:
            print(f"✗ Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    mock_producer()
