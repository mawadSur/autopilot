#!/usr/bin/env python3
"""
Live trading engine that consumes signals from Redis and executes trades.

Integrates with Binance API for real-time order execution.
Monitors positions, manages risk, and logs all activity.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import redis
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

load_dotenv()

# Redis connection
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Trading config
SYMBOL = os.getenv("TRADE_SYMBOL", "ETHUSDT")
QUOTE_USDT = float(os.getenv("TRADE_QUANTITY_USDT", "15"))


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


DRY_RUN = _env_bool("DRY_RUN", True)
TESTNET = _env_bool("TESTNET", False)

# API credentials
API_KEY = os.getenv("BINANCE_KEY") or os.getenv("BINANCE_TESTNET_KEY")
API_SECRET = os.getenv("BINANCE_SECRET") or os.getenv("BINANCE_TESTNET_SECRET")


class LiveTradingEngine:
    """Real-time trading engine consuming signals from Redis."""

    def __init__(self, symbol: str = SYMBOL):
        self.symbol = symbol.upper()
        self.state_file = f"state_{self.symbol}.json"

        # Redis (using lists instead of streams for compatibility)
        self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.signal_list = f"list:{self.symbol}:signals"
        self.trade_list = f"list:{self.symbol}:trades"
        self.last_id = "0"

        # Binance client
        self.client = self._get_client()

        # Position tracking
        self.position_open = False
        self.position_side = None  # "BUY" or "SELL"
        self.entry_price = 0.0
        self.entry_time = None
        self.trade_count = 0

        # Load persisted state
        self._load_state()

        # Risk management
        self.max_position_time = int(os.getenv("MAX_POSITION_TIME_MINUTES", "60")) * 60  # seconds
        self.stop_loss_pct = float(os.getenv("STOP_LOSS_PCT", "0.02"))  # 2%
        self.take_profit_pct = float(os.getenv("TAKE_PROFIT_PCT", "0.05"))  # 5%

    def _get_client(self) -> Client:
        """Initialize Binance client."""
        if not DRY_RUN and (not API_KEY or not API_SECRET):
            raise RuntimeError("Missing BINANCE credentials in environment")

        if DRY_RUN:
            # Return None for dry run mode
            return None

        return Client(API_KEY, API_SECRET, testnet=TESTNET)

    def _load_state(self):
        """Load state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                    self.position_open = state.get("position_open", False)
                    self.position_side = state.get("position_side")
                    self.entry_price = state.get("entry_price", 0.0)
                    entry_time_str = state.get("entry_time")
                    self.entry_time = datetime.fromisoformat(entry_time_str) if entry_time_str else None
                    self.trade_count = state.get("trade_count", 0)
                print(f"✓ State loaded from {self.state_file}")
            except Exception as e:
                print(f"✗ Failed to load state: {e}")

    def _save_state(self):
        """Save state to file."""
        try:
            state = {
                "position_open": self.position_open,
                "position_side": self.position_side,
                "entry_price": self.entry_price,
                "entry_time": self.entry_time.isoformat() if self.entry_time else None,
                "trade_count": self.trade_count
            }
            with open(self.state_file, "w") as f:
                json.dump(state, f)
        except Exception as e:
            print(f"✗ Failed to save state: {e}")

    def start(self):
        """Start consuming signals and trading."""
        print(f"💰 Starting live trading engine for {self.symbol}")
        print(f"   DRY_RUN: {DRY_RUN}")
        print(f"   TESTNET: {TESTNET}")
        print(f"   QUOTE_QTY: {QUOTE_USDT} USDT")
        print(f"   STOP_LOSS: {self.stop_loss_pct*100}%")
        print(f"   TAKE_PROFIT: {self.take_profit_pct*100}%")

        if self.position_open:
            print(f"   ⚠️ RESUMING OPEN POSITION: {self.position_side} at {self.entry_price:.2f}")

        while True:
            try:
                # Poll for new signals from list
                signal_data = self.redis.lpop(self.signal_list)

                if signal_data:
                    # Parse signal
                    signal = json.loads(signal_data)
                    self._process_signal(signal)
                else:
                    # Check for position management even without new signals
                    self._check_position_management()
                    # No new signals, wait before checking again
                    time.sleep(0.1)

            except KeyboardInterrupt:
                print("🛑 Trading engine stopped")
                # Removed _close_position_if_open() to allow persistent positions across restarts
                break
            except Exception as e:
                print(f"✗ Trading engine error: {e}")
                time.sleep(5)

    def _process_signal(self, signal: Dict):
        """Process incoming trading signal."""
        action = signal["action"]
        confidence = signal["confidence"]
        timestamp = signal["timestamp"]

        print(f"📡 Signal received: {action} | Conf: {confidence:.3f} | {timestamp} | {self.symbol}")

        # Check current position
        if self.position_open:
            if self.position_side == "BUY" and action == "SELL":
                # Close long position
                self._close_position("SIGNAL_SELL", signal)
            elif self.position_side == "SELL" and action == "BUY":
                # Close short position (if we had short selling)
                self._close_position("SIGNAL_BUY", signal)
            else:
                # print(f"   Ignoring {action} signal - already in {self.position_side} position")
                pass
        else:
            # Open new position
            if action in ["BUY", "SELL"]:
                self._open_position(action, signal)

        # Always check risk management
        self._check_position_management()

    def _open_position(self, side: str, signal: Dict):
        """Open a new trading position."""
        try:
            executed_price = 0.0
            if DRY_RUN:
                print(f"[DRY RUN] {side} {self.symbol} for {QUOTE_USDT} USDT")
                executed_price = float(signal.get("price", 0))
                if executed_price == 0:
                    # Try to get current price
                    try:
                        ticker = self.client.get_symbol_ticker(symbol=self.symbol) if self.client else {"price": 0}
                        executed_price = float(ticker["price"])
                    except Exception:
                        executed_price = 0.0
            else:
                # Get current price for logging
                ticker = self.client.get_symbol_ticker(symbol=self.symbol)
                current_price = float(ticker["price"])

                # Place market order
                order = self.client.create_order(
                    symbol=self.symbol,
                    side=side,
                    type=ORDER_TYPE_MARKET,
                    quoteOrderQty=str(QUOTE_USDT),
                )

                # Extract execution price
                if "fills" in order and order["fills"]:
                    executed_price = float(order["fills"][0]["price"])
                else:
                    executed_price = current_price

            # Update position state
            self.position_open = True
            self.position_side = side
            self.entry_price = executed_price
            self.entry_time = datetime.now()
            self.trade_count += 1

            self._save_state()

            print(f"🚀 {side} POSITION OPENED | {self.symbol} | Price: {self.entry_price:.2f} | Conf: {signal['confidence']:.3f}")
            print(f"   Trade #{self.trade_count} | {signal['timestamp']}")

            # Log trade
            self._log_trade("OPEN", side, self.entry_price, signal)

        except Exception as e:
            print(f"✗ Failed to open {side} position: {e}")

    def _close_position(self, reason: str, signal: Optional[Dict] = None):
        """Close current position."""
        if not self.position_open:
            return

        try:
            side = SIDE_SELL if self.position_side == "BUY" else SIDE_BUY
            exit_price = 0.0

            if DRY_RUN:
                print(f"[DRY RUN] {side} {self.symbol} (Close position)")
                try:
                    ticker = self.client.get_symbol_ticker(symbol=self.symbol) if self.client else {"price": 0}
                    exit_price = float(ticker["price"])
                except Exception:
                    exit_price = self.entry_price  # Fallback
            else:
                # Get current price
                ticker = self.client.get_symbol_ticker(symbol=self.symbol)
                current_price = float(ticker["price"])

                # Place closing order
                order = self.client.create_order(
                    symbol=self.symbol,
                    side=side,
                    type=ORDER_TYPE_MARKET,
                    quoteOrderQty=str(QUOTE_USDT),
                )

                # Extract execution price
                if "fills" in order and order["fills"]:
                    exit_price = float(order["fills"][0]["price"])
                else:
                    exit_price = current_price

            # Calculate P&L
            if self.position_side == "BUY":
                pnl_pct = (exit_price - self.entry_price) / self.entry_price
            else:
                pnl_pct = (self.entry_price - exit_price) / self.entry_price

            # Reset position state
            self.position_open = False
            entry_price = self.entry_price
            old_side = self.position_side
            self.entry_price = 0.0
            self.entry_time = None
            self.position_side = None

            self._save_state()

            print(f"🏁 POSITION CLOSED | {self.symbol} | {reason} | P&L: {pnl_pct*100:.2f}%")
            print(f"   Entry: {entry_price:.2f} | Exit: {exit_price:.2f}")

            # Log trade
            self._log_trade("CLOSE", old_side, exit_price, signal, pnl_pct)

        except Exception as e:
            print(f"✗ Failed to close position: {e}")

    def _check_position_management(self):
        """Check for stop-loss, take-profit, or time-based exits."""
        if not self.position_open:
            return

        try:
            if self.client is not None:
                ticker = self.client.get_symbol_ticker(symbol=self.symbol)
                current_price = float(ticker["price"])
            elif self.entry_price > 0:
                current_price = self.entry_price  # DRY_RUN: no live feed; stops won't fire on price
            else:
                return

            # Check time limit
            if self.entry_time:
                position_duration = (datetime.now() - self.entry_time).total_seconds()
                if position_duration > self.max_position_time:
                    print(f"⏰ Time limit reached ({position_duration/60:.1f} min)")
                    self._close_position("TIME_LIMIT")
                    return

            # Check stop loss and take profit
            if self.position_side == "BUY":
                loss_pct = (self.entry_price - current_price) / self.entry_price
                profit_pct = (current_price - self.entry_price) / self.entry_price

                if loss_pct >= self.stop_loss_pct:
                    print(f"🛑 STOP LOSS triggered | Loss: {loss_pct*100:.2f}%")
                    self._close_position("STOP_LOSS")
                elif profit_pct >= self.take_profit_pct:
                    print(f"💰 TAKE PROFIT triggered | Profit: {profit_pct*100:.2f}%")
                    self._close_position("TAKE_PROFIT")

            elif self.position_side == "SELL":
                # For short positions (if implemented)
                loss_pct = (current_price - self.entry_price) / self.entry_price
                profit_pct = (self.entry_price - current_price) / self.entry_price

                if loss_pct >= self.stop_loss_pct:
                    print(f"🛑 STOP LOSS triggered | Loss: {loss_pct*100:.2f}%")
                    self._close_position("STOP_LOSS")
                elif profit_pct >= self.take_profit_pct:
                    print(f"💰 TAKE PROFIT triggered | Profit: {profit_pct*100:.2f}%")
                    self._close_position("TAKE_PROFIT")

        except Exception as e:
            print(f"✗ Position management error: {e}")

    def _close_position_if_open(self):
        """Close any open position on shutdown."""
        if self.position_open:
            print("🔄 Closing position on shutdown...")
            self._close_position("SHUTDOWN")

    def _log_trade(self, action: str, side: str, price: float, signal: Optional[Dict] = None, pnl: Optional[float] = None):
        """Log trade to Redis for analysis."""
        try:
            trade_data = {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "side": side,
                "price": price,
                "symbol": self.symbol,
                "pnl_pct": pnl,
                "signal_confidence": signal.get("confidence") if signal else None,
                "dry_run": DRY_RUN
            }

            self.redis.lpush(self.trade_list, json.dumps(trade_data))

        except Exception as e:
            print(f"✗ Trade logging error: {e}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Live trading engine")
    parser.add_argument("--symbol", default=SYMBOL, help="Trading symbol")

    args = parser.parse_args()

    engine = LiveTradingEngine(symbol=args.symbol)
    engine.start()


if __name__ == "__main__":
    main()
