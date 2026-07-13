"""Dashboard exporter — publishes a live snapshot of the crypto bot's trading
state to the Vercel dashboard.

Why this exists
---------------
The trading bot (``src/live_supervisor.py``) runs locally and writes every
position to **local Redis** via ``PositionStore``. A Vercel serverless
function cannot reach that local Redis. This exporter bridges the gap: it
reads ``PositionStore`` on an interval, computes the dashboard view-state
(equity curve, PnL, win rate, open/closed positions, per-symbol breakdown),
and POSTs a compact JSON snapshot to the dashboard's ``/api/ingest`` endpoint
(authenticated with a shared secret). The bot is never touched.

Data flow::

    live_supervisor -> local Redis (PositionStore)   [unchanged]
    dashboard_exporter (this) --HTTPS POST--> Vercel /api/ingest -> Vercel KV
    Vercel dashboard <-- reads KV snapshot behind login

Usage::

    # one-shot, print the snapshot JSON to stdout (no network) — great for dev
    PYTHONPATH=src ./.venv/bin/python scripts/dashboard_exporter.py --dry-run --once

    # run the publish loop against a deployed dashboard
    DASHBOARD_INGEST_URL="https://your-app.vercel.app/api/ingest" \
    DASHBOARD_EXPORTER_SECRET="<the same secret set in Vercel>" \
    PYTHONPATH=src ./.venv/bin/python scripts/dashboard_exporter.py --interval 30

Config (CLI flag overrides env):
    --ingest-url   / DASHBOARD_INGEST_URL      target /api/ingest URL
    --secret       / DASHBOARD_EXPORTER_SECRET shared secret (x-exporter-secret header)
    --bankroll     / DASHBOARD_BANKROLL_USD    starting equity baseline (default 10000)
    --interval                                  seconds between publishes (default 30)
    --days                                      closed-history window in days (default 14)
    --once                                      publish (or print) once and exit
    --dry-run                                   print JSON to stdout, never POST
    --no-marks                                  skip live mark-price fetch (unrealized=0)

Exit codes: 0 normal; 2 config error (missing ingest URL/secret when publishing).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- make ``src`` importable + load .env the same way the supervisor does ----
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from dotenv import load_dotenv

    load_dotenv(_REPO / ".env", override=False)
    load_dotenv(_SRC / ".env", override=False)
except Exception:  # noqa: BLE001 - dotenv is best-effort
    pass

from state.position_store import Position, PositionStore  # noqa: E402

SCHEMA_VERSION = 1
CLOSED_FEED_CAP = 120  # most-recent closed trades sent to the feed
EQUITY_CURVE_CAP = 600  # cap curve points so the payload stays small


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _confidence_obj(score: Optional[float]) -> Optional[Dict[str, Any]]:
    """Map a raw confidence (0..1) to ``{score, label}`` for the UI bar."""
    if score is None:
        return None
    try:
        s = float(score)
    except (TypeError, ValueError):
        return None
    if s != s:  # NaN
        return None
    if s >= 0.66:
        label = "high"
    elif s >= 0.50:
        label = "medium"
    else:
        label = "low"
    return {"score": round(s, 4), "label": label}


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if f == f else default  # drop NaN
    except (TypeError, ValueError):
        return default


def _unrealized(position: Position, mark: Optional[float]) -> float:
    """Mark-to-market PnL for an open position, 0 when no mark is available."""
    if mark is None:
        return 0.0
    size = _safe_float(position.base_size)
    entry = _safe_float(position.entry_price)
    if position.side == "long":
        return (mark - entry) * size
    return (entry - mark) * size


# --------------------------------------------------------------------------
# closed-history walk (multi-day) — mirrors diagnose_calibration_drift's pattern
# --------------------------------------------------------------------------
def _closed_positions_in_window(store: PositionStore, days: int) -> List[Position]:
    """Collect closed positions from the last ``days`` UTC-date closed-sets.

    ``PositionStore`` records each closed position id into a per-UTC-date set
    (``{ns}:closed:{YYYY-MM-DD}``). There's no built-in cross-day query, so we
    walk the last ``days`` date-sets and dedupe (a position closes on exactly
    one date, but we guard anyway). Sorted by ``closed_at_utc`` ascending.
    """
    now = _utcnow()
    seen: set[str] = set()
    out: List[Position] = []
    for d in range(days):
        day = now - timedelta(days=d)
        key = store._closed_set_key(day)  # noqa: SLF001 - documented reuse
        try:
            ids = store._redis.smembers(key)  # noqa: SLF001
        except Exception:  # noqa: BLE001 - a bad day-key shouldn't kill the run
            continue
        for pid in ids or []:
            if pid in seen:
                continue
            seen.add(pid)
            pos = store.get(pid)
            if pos is not None and pos.status == "closed":
                out.append(pos)
    out.sort(key=lambda p: p.closed_at_utc or "")
    return out


# --------------------------------------------------------------------------
# best-effort live marks (unrealized PnL on open positions)
# --------------------------------------------------------------------------
def _fetch_marks(symbols: List[str]) -> Dict[str, float]:
    """Best-effort current mid per unique symbol via the public Coinbase feed.

    Returns ``{}`` on any failure (network, missing creds, geo-block). The
    caller degrades gracefully to unrealized=0 / mark=entry, so this must
    never raise.
    """
    marks: Dict[str, float] = {}
    if not symbols:
        return marks
    try:
        from exchanges.coinbase import CoinbaseExchange
    except Exception:  # noqa: BLE001
        return marks
    try:
        ex = CoinbaseExchange()
    except Exception:  # noqa: BLE001
        return marks
    for sym in set(symbols):
        try:
            ticker = ex.get_ticker(sym)
            mid = _safe_float(getattr(ticker, "mid", None), default=0.0)
            if mid > 0:
                marks[sym] = mid
        except Exception:  # noqa: BLE001 - skip this symbol, keep the rest
            continue
    return marks


# --------------------------------------------------------------------------
# snapshot builder
# --------------------------------------------------------------------------
def build_snapshot(
    store: PositionStore,
    *,
    bankroll_usd: float,
    days: int,
    fetch_marks: bool,
    bot_meta: Dict[str, Any],
) -> Dict[str, Any]:
    now = _utcnow()
    open_positions = store.list_open()
    closed = _closed_positions_in_window(store, days)

    marks = _fetch_marks([p.symbol for p in open_positions]) if fetch_marks else {}

    # ---- open positions view + unrealized ----
    open_view: List[Dict[str, Any]] = []
    unrealized_total = 0.0
    for p in open_positions:
        mark = marks.get(p.symbol)
        upnl = _unrealized(p, mark)
        unrealized_total += upnl
        open_view.append(
            {
                "position_id": p.position_id,
                "symbol": p.symbol,
                "side": p.side,
                "entry_price": _safe_float(p.entry_price),
                "mark_price": _safe_float(mark) if mark is not None else None,
                "base_size": _safe_float(p.base_size),
                "notional_usd": _safe_float(p.entry_quote_usd),
                "unrealized_pnl_usd": round(upnl, 4),
                "opened_at_utc": p.opened_at_utc,
                "confidence": _confidence_obj(p.entry_confidence),
                "regime_label": p.regime_label,
            }
        )
    # biggest exposure first
    open_view.sort(key=lambda r: r["notional_usd"], reverse=True)

    # ---- closed view + aggregates ----
    realized_total = 0.0
    fees_total = 0.0
    wins = 0
    losses = 0
    win_pnls: List[float] = []
    loss_pnls: List[float] = []
    per_symbol: Dict[str, Dict[str, Any]] = {}
    exit_mix: Dict[str, int] = {"won": 0, "lost": 0}
    equity_curve: List[Dict[str, Any]] = []
    equity = float(bankroll_usd)
    # curve anchor at window start
    curve_anchor_t = closed[0].opened_at_utc if closed else _iso(now)
    equity_curve.append({"t": curve_anchor_t, "equity": round(equity, 2)})

    closed_view: List[Dict[str, Any]] = []
    for p in closed:
        pnl = _safe_float(p.realized_pnl_usd)
        realized_total += pnl
        fees_total += _safe_float(p.fees_usd)
        equity += pnl
        equity_curve.append({"t": p.closed_at_utc or p.opened_at_utc, "equity": round(equity, 2)})
        reason = "won" if pnl > 0 else "lost"
        if pnl > 0:
            wins += 1
            win_pnls.append(pnl)
        else:
            losses += 1
            loss_pnls.append(pnl)
        exit_mix[reason] = exit_mix.get(reason, 0) + 1

        sym = per_symbol.setdefault(
            p.symbol,
            {"symbol": p.symbol, "n_open": 0, "n_closed": 0, "realized_pnl_usd": 0.0, "wins": 0},
        )
        sym["n_closed"] += 1
        sym["realized_pnl_usd"] += pnl
        if pnl > 0:
            sym["wins"] += 1

        closed_view.append(
            {
                "position_id": p.position_id,
                "symbol": p.symbol,
                "side": p.side,
                "entry_price": _safe_float(p.entry_price),
                "exit_price": _safe_float(p.exit_price),
                "realized_pnl_usd": round(pnl, 4),
                "fees_usd": _safe_float(p.fees_usd),
                "reason": reason,
                "opened_at_utc": p.opened_at_utc,
                "closed_at_utc": p.closed_at_utc,
                "confidence": _confidence_obj(p.entry_confidence),
                "regime_label": p.regime_label,
            }
        )

    # fold open counts into per_symbol
    for p in open_positions:
        sym = per_symbol.setdefault(
            p.symbol,
            {"symbol": p.symbol, "n_open": 0, "n_closed": 0, "realized_pnl_usd": 0.0, "wins": 0},
        )
        sym["n_open"] += 1

    n_settled = len(closed)
    win_rate = (wins / n_settled) if n_settled else 0.0
    avg_win = (sum(win_pnls) / len(win_pnls)) if win_pnls else 0.0
    avg_loss = (sum(loss_pnls) / len(loss_pnls)) if loss_pnls else 0.0
    total_fees = fees_total + sum(_safe_float(p.fees_usd) for p in open_positions)
    equity_now = bankroll_usd + realized_total + unrealized_total

    # per-symbol finalize (win_rate + open notional)
    open_notional_by_sym: Dict[str, float] = {}
    for p in open_positions:
        open_notional_by_sym[p.symbol] = open_notional_by_sym.get(p.symbol, 0.0) + _safe_float(
            p.entry_quote_usd
        )
    per_symbol_list = []
    for sym_name, s in sorted(per_symbol.items()):
        nc = s["n_closed"]
        per_symbol_list.append(
            {
                "symbol": sym_name,
                "n_open": s["n_open"],
                "n_closed": nc,
                "realized_pnl_usd": round(s["realized_pnl_usd"], 4),
                "win_rate": round(s["wins"] / nc, 4) if nc else None,
                "open_notional_usd": round(open_notional_by_sym.get(sym_name, 0.0), 2),
            }
        )

    # most-recent-first, capped feed
    closed_view_recent = list(reversed(closed_view))[:CLOSED_FEED_CAP]
    # cap curve length (keep head anchor + tail)
    if len(equity_curve) > EQUITY_CURVE_CAP:
        head = equity_curve[:1]
        tail = equity_curve[-(EQUITY_CURVE_CAP - 1):]
        equity_curve = head + tail

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _iso(now),
        "window_days": days,
        "bot": bot_meta,
        "summary": {
            "bankroll_usd": round(float(bankroll_usd), 2),
            "equity_usd": round(equity_now, 2),
            "realized_pnl_usd": round(realized_total, 4),
            "unrealized_pnl_usd": round(unrealized_total, 4),
            "total_pnl_usd": round(realized_total + unrealized_total, 4),
            "win_rate": round(win_rate, 4),
            "n_open": len(open_positions),
            "n_settled": n_settled,
            "total_fees_usd": round(total_fees, 4),
            "avg_win_usd": round(avg_win, 4),
            "avg_loss_usd": round(avg_loss, 4),
        },
        "open_positions": open_view,
        "closed_positions": closed_view_recent,
        "equity_curve": equity_curve,
        "exit_mix": exit_mix,
        "per_symbol": per_symbol_list,
    }


# --------------------------------------------------------------------------
# bot metadata (mode / halal / liveness) — best-effort from process + env
# --------------------------------------------------------------------------
def _detect_bot_meta() -> Dict[str, Any]:
    """Best-effort description of the running bot for the dashboard header."""
    import subprocess

    running = False
    symbols: List[str] = []
    mode = "paper"
    halal = os.environ.get("HALAL_MODE", "1") not in ("0", "false", "False", "")
    exit_rule: Optional[str] = None
    try:
        # macOS pgrep has no -a (list) flag, so resolve the PID first, then
        # read that PID's full command line via ps -o command= (portable).
        pids = subprocess.run(
            ["pgrep", "-f", "live_supervisor.py --mode"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.split()
        line = ""
        if pids:
            line = subprocess.run(
                ["ps", "-p", pids[0], "-o", "command="],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        running = bool(line)
        if "--mode live" in line:
            mode = "live"
        if "--no-halal-mode" in line:
            halal = False
        # parse --symbols X,Y,Z
        if "--symbols" in line:
            parts = line.split("--symbols", 1)[1].strip().split()
            if parts:
                symbols = [s for s in parts[0].split(",") if s]
        if "--exit-rule" in line:
            parts = line.split("--exit-rule", 1)[1].strip().split()
            if parts:
                exit_rule = parts[0].strip("'\"")
    except Exception:  # noqa: BLE001
        pass
    return {
        "mode": mode,
        "halal": halal,
        "running": running,
        "symbols": symbols,
        "exit_rule": exit_rule,
    }


# --------------------------------------------------------------------------
# publish
# --------------------------------------------------------------------------
def publish(snapshot: Dict[str, Any], *, ingest_url: str, secret: str, timeout: float = 10.0) -> bool:
    import requests

    try:
        resp = requests.post(
            ingest_url,
            json=snapshot,
            headers={"x-exporter-secret": secret, "content-type": "application/json"},
            timeout=timeout,
        )
        if resp.status_code >= 300:
            print(
                f"[dashboard_exporter] ingest returned HTTP {resp.status_code}: {resp.text[:200]}",
                file=sys.stderr,
            )
            return False
        return True
    except Exception as exc:  # noqa: BLE001 - publish is best-effort, keep looping
        print(f"[dashboard_exporter] publish failed: {exc}", file=sys.stderr)
        return False


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Publish crypto-bot state snapshots to the Vercel dashboard.")
    p.add_argument("--ingest-url", default=os.environ.get("DASHBOARD_INGEST_URL", ""))
    p.add_argument("--secret", default=os.environ.get("DASHBOARD_EXPORTER_SECRET", ""))
    p.add_argument(
        "--bankroll",
        type=float,
        default=float(os.environ.get("DASHBOARD_BANKROLL_USD", "10000") or "10000"),
    )
    p.add_argument("--interval", type=float, default=30.0, help="seconds between publishes")
    p.add_argument("--days", type=int, default=14, help="closed-history window in days")
    p.add_argument("--once", action="store_true", help="publish/print once and exit")
    p.add_argument("--dry-run", action="store_true", help="print JSON to stdout, never POST")
    p.add_argument("--no-marks", action="store_true", help="skip live mark fetch (unrealized=0)")
    p.add_argument("--redis-url", default=None, help="override REDIS_URL for the read")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.dry_run and (not args.ingest_url or not args.secret):
        print(
            "error: publishing requires --ingest-url/--secret (or "
            "DASHBOARD_INGEST_URL/DASHBOARD_EXPORTER_SECRET). Use --dry-run to "
            "print the snapshot without publishing.",
            file=sys.stderr,
        )
        return 2

    store = PositionStore(redis_url=args.redis_url) if args.redis_url else PositionStore()

    def _emit_once() -> None:
        snap = build_snapshot(
            store,
            bankroll_usd=args.bankroll,
            days=args.days,
            fetch_marks=not args.no_marks,
            bot_meta=_detect_bot_meta(),
        )
        if args.dry_run:
            print(json.dumps(snap, indent=2))
        else:
            ok = publish(snap, ingest_url=args.ingest_url, secret=args.secret)
            s = snap["summary"]
            print(
                f"[dashboard_exporter] {'published' if ok else 'FAILED'} "
                f"equity=${s['equity_usd']} open={s['n_open']} settled={s['n_settled']} "
                f"wr={s['win_rate']:.2f} @ {snap['generated_at_utc']}",
                file=sys.stderr,
            )

    if args.once:
        _emit_once()
        return 0

    print(
        f"[dashboard_exporter] publishing every {args.interval:.0f}s "
        f"(window={args.days}d, marks={'off' if args.no_marks else 'on'}). Ctrl-C to stop.",
        file=sys.stderr,
    )
    try:
        while True:
            _emit_once()
            time.sleep(max(1.0, args.interval))
    except KeyboardInterrupt:
        print("[dashboard_exporter] stopped.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
