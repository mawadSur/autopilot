"""Stdlib HTTP server for the read-only SHADOW dashboard.

OBSERVABILITY ONLY. This server exposes exactly two GET endpoints and nothing
else:

* ``GET /``           -> the self-contained dashboard HTML page.
* ``GET /api/state``  -> ``build_state(PnlLedger(ledger_path))`` as JSON.

There is no order, trade, settle, or any other write surface here. The ledger is
opened read-only and re-read on every ``/api/state`` request (a cheap JSONL fold),
so the page always reflects the latest appended events without holding the file
open or caching stale state. Built on :class:`http.server.ThreadingHTTPServer` so
the 4s poll from a couple of browser tabs never blocks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

# Deliberate sys.path bootstrap (matches main.py / src/orchestrator.py): ensure
# the repo's ``src/`` is importable so the flat imports below resolve even when
# this file is launched as a script path (``python src/dashboard/server.py``),
# where sys.path[0] is ``src/dashboard`` rather than ``src``. Read-only.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dashboard.state import build_state  # noqa: E402
from state.pnl_ledger import PnlLedger  # noqa: E402

# The dashboard's single static asset lives next to this module.
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
_INDEX_HTML = os.path.join(_STATIC_DIR, "index.html")

# Default ledger for the whale-follow shadow loop (overridable via --ledger-path).
DEFAULT_WHALE_LEDGER = "runs/whale_optimized_ledger.jsonl"


def make_handler(ledger_path: str, bankroll_usd: float):
    """Build a request-handler class bound to a ledger path + bankroll.

    Returning a closure-bound subclass keeps the server config off of mutable
    globals while still fitting the stdlib ``BaseHTTPRequestHandler`` contract.
    """

    class DashboardHandler(BaseHTTPRequestHandler):
        # Identify ourselves plainly; this is a local observability tool.
        server_version = "ShadowDashboard/1.0"

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            # Always serve fresh state; the page polls on an interval.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler naming.
            # Strip any query string; we have no query-driven endpoints.
            path = self.path.split("?", 1)[0]

            if path == "/":
                try:
                    with open(_INDEX_HTML, "rb") as handle:
                        body = handle.read()
                except OSError:
                    self._send_json(500, {"error": "dashboard page missing"})
                    return
                self._send(200, body, "text/html; charset=utf-8")
                return

            if path == "/api/state":
                try:
                    ledger = PnlLedger(ledger_path)
                    state = build_state(ledger, bankroll_usd=bankroll_usd)
                except Exception as exc:  # noqa: BLE001 - report, never crash the server.
                    self._send_json(500, {"error": f"failed to build state: {exc}"})
                    return
                self._send_json(200, state)
                return

            # Everything else is a hard 404 — there is no other surface.
            self._send_json(404, {"error": "not found"})

        # POST/PUT/etc. are simply not implemented: BaseHTTPRequestHandler
        # returns 501 for any verb without a do_<VERB>, so there is provably no
        # write path. We intentionally do NOT define do_POST or any mutator.

        def log_message(self, fmt: str, *args: Any) -> None:
            # Keep the console quiet but informative: method + path + status.
            try:
                print(f"[dashboard] {self.address_string()} {fmt % args}")
            except Exception:  # pragma: no cover - logging must never crash.
                pass

    return DashboardHandler


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only web dashboard for the whale-follow SHADOW loop."
    )
    parser.add_argument(
        "--port", type=int, default=8888, help="HTTP port to listen on (default 8888)."
    )
    parser.add_argument(
        "--ledger-path",
        default=DEFAULT_WHALE_LEDGER,
        help=f"PnL ledger JSONL to read (default {DEFAULT_WHALE_LEDGER}).",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Starting bankroll in USD for equity math (default 1000).",
    )
    args = parser.parse_args(argv)

    handler = make_handler(args.ledger_path, args.bankroll)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)

    url = f"http://127.0.0.1:{args.port}/"
    abs_ledger = os.path.abspath(args.ledger_path)
    print(f"SHADOW dashboard (read-only) serving {url}")
    print(f"  ledger:   {abs_ledger}")
    print(f"  bankroll: ${args.bankroll:,.2f}")
    if not os.path.exists(args.ledger_path):
        print("  note: ledger file does not exist yet — page will show an empty book.")
    print("  Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
