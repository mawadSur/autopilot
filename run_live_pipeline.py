#!/usr/bin/env python3
"""
Complete live trading pipeline orchestrator.

Starts all components:
1. WebSocket data consumer
2. Feature engineering service
3. Inference engine
4. Trading engine
5. Data logger

Manages process lifecycle and provides monitoring.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

# Service configurations
SERVICES = {
    "websocket": {
        "script": "live_data_stream.py",
        "description": "Binance WebSocket data consumer"
    },
    "features": {
        "script": "feature_engine_live.py",
        "description": "Real-time feature computation"
    },
    "inference": {
        "script": "inference_live.py",
        "description": "Model inference engine"
    },
    "trading": {
        "script": "live_trading_engine.py",
        "description": "Live trading execution"
    },
    "logger": {
        "script": "data_logger_live.py",
        "description": "Data persistence logger"
    }
}


class PipelineOrchestrator:
    """Manages the complete live trading pipeline for multiple symbols."""

    def __init__(self, services_to_run: List[str] = None, symbols: List[str] = None):
        self.services_to_run = services_to_run or list(SERVICES.keys())
        self.symbols = [s.upper() for s in symbols] if symbols else ["ETHUSDT"]
        self.processes: Dict[str, subprocess.Popen] = {}  # key: "symbol:service"
        self.running = False

    def start_pipeline(self):
        """Start all pipeline services for all symbols."""
        print("🚀 Starting Live Trading Pipeline...")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Services: {', '.join(self.services_to_run)}")
        print("-" * 50)

        self.running = True

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            # Start services in order
            startup_order = ["websocket", "features", "inference", "logger", "trading"]

            for symbol in self.symbols:
                for service_name in startup_order:
                    if service_name in self.services_to_run:
                        self._start_service(service_name, symbol)

            # Monitor services
            self._monitor_pipeline()

        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested by user")
        except Exception as e:
            print(f"✗ Pipeline error: {e}")
        finally:
            self._shutdown_pipeline()

    def _start_service(self, service_name: str, symbol: str):
        """Start a single service for a specific symbol."""
        if service_name not in SERVICES:
            print(f"✗ Unknown service: {service_name}")
            return

        config = SERVICES[service_name]
        script_path = os.path.join("src", config["script"])

        if not os.path.exists(script_path):
            print(f"✗ Script not found: {script_path}")
            return

        proc_key = f"{symbol}:{service_name}"
        print(f"▶️  Starting {service_name} for {symbol}: {config['description']}")

        try:
            # Set environment for UTF-8 encoding
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            # Start process with --symbol argument
            process = subprocess.Popen(
                [sys.executable, script_path, "--symbol", symbol],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )

            self.processes[proc_key] = process

            # Give it a moment to start
            time.sleep(1)

            # Check if still running
            if process.poll() is None:
                print(f"✓ {service_name} [{symbol}] started (PID: {process.pid})")
            else:
                # Process exited immediately
                stdout, _ = process.communicate()
                print(f"✗ {service_name} [{symbol}] failed to start:")
                print(stdout)

        except Exception as e:
            print(f"✗ Failed to start {service_name} for {symbol}: {e}")

    def _monitor_pipeline(self):
        """Monitor running services and handle failures."""
        while self.running:
            for proc_key, process in list(self.processes.items()):
                if process.poll() is not None:
                    # Process has exited
                    symbol, service_name = proc_key.split(":", 1)
                    exit_code = process.returncode
                    stdout, _ = process.communicate()

                    if exit_code == 0:
                        print(f"✓ {service_name} [{symbol}] exited normally")
                    else:
                        print(f"✗ {service_name} [{symbol}] crashed (exit code: {exit_code})")
                        print("Last output:")
                        print(stdout[-500:])  # Last 500 chars

                    # Remove from processes
                    del self.processes[proc_key]

                    # Decide whether to restart
                    if self.running and service_name in ["websocket", "features", "inference", "logger"]:
                        print(f"🔄 Restarting {service_name} for {symbol}...")
                        self._start_service(service_name, symbol)

            # Check if all critical services are still running for at least one symbol
            critical_services = ["websocket", "features", "inference"]
            running_proc_keys = self.processes.keys()
            
            any_running = False
            for proc_key in running_proc_keys:
                _, service_name = proc_key.split(":", 1)
                if service_name in critical_services:
                    any_running = True
                    break

            if not any_running and self.processes:
                print("❌ All critical services have stopped. Shutting down pipeline.")
                self.running = False
                break

            time.sleep(5)

    def _shutdown_pipeline(self):
        """Gracefully shutdown all services."""
        print("\n🔄 Shutting down pipeline...")

        # Stop in reverse order
        shutdown_order = ["trading", "inference", "features", "logger", "websocket"]

        for service_name in shutdown_order:
            for symbol in self.symbols:
                proc_key = f"{symbol}:{service_name}"
                if proc_key in self.processes:
                    process = self.processes[proc_key]
                    print(f"🛑 Stopping {service_name} for {symbol}...")

                    try:
                        # Try graceful shutdown first
                        process.terminate()
                        process.wait(timeout=5)
                        print(f"✓ {service_name} [{symbol}] stopped")
                    except subprocess.TimeoutExpired:
                        # Force kill if it doesn't respond
                        process.kill()
                        process.wait()
                        print(f"⚠️ {service_name} [{symbol}] force-killed")

        self.processes.clear()
        print("✓ Pipeline shutdown complete")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n📡 Received signal {signum}")
        self.running = False


def check_dependencies():
    """Check if required dependencies are available."""
    missing = []

    # Check Redis
    try:
        import redis
        r = redis.Redis()
        r.ping()
    except Exception:
        missing.append("Redis server (redis-server)")

    # Check Python packages
    required_packages = ["aiohttp", "h5py", "torch", "joblib", "pandas", "numpy"]
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(f"Python package: {package}")

    if missing:
        print("❌ Missing dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print("\nPlease install missing dependencies and try again.")
        return False

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Live Trading Pipeline Orchestrator")
    parser.add_argument(
        "--symbols",
        type=str,
        default=os.getenv("SYMBOLS", os.getenv("TRADE_SYMBOL", "ETHUSDT")),
        help="Comma-separated list of symbols (e.g. ETHUSDT,BTCUSDT)"
    )
    parser.add_argument(
        "--services",
        nargs="*",
        choices=list(SERVICES.keys()),
        default=list(SERVICES.keys()),
        help="Services to run (default: all)"
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies and exit"
    )

    args = parser.parse_args()

    if args.check_deps:
        if check_dependencies():
            print("✅ All dependencies available")
        else:
            sys.exit(1)
        return

    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Start pipeline
    orchestrator = PipelineOrchestrator(args.services, symbols=symbols)
    orchestrator.start_pipeline()


if __name__ == "__main__":
    main()