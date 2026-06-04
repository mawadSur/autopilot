"""Stablecoin yield SHADOW accrual-runner + peg risk monitor (SHADOW / NO ORDERS).

The "automate a known yield" pivot, made into a forward-validating runner: it
does NOT try to beat the market, it COLLECTS a structural stablecoin earn rate
(supply a stablecoin on a CeFi venue such as Kraken Earn) and watches the risk
that actually matters for that yield — the stablecoin PEG (does it still trade at
$1?) and, by extension, the platform holding it. There is no fee edge to chase
here, so this module does NOT fake a "risk-adjusted APY"; it accrues the
operator-supplied earn rate and surfaces the honest depeg signal alongside it.

SHADOW / NO ORDERS — this is observability + bookkeeping ONLY. Per scan it:
    * READS the public Kraken ticker for ``<SYM>/USD`` (the live peg signal),
    * ACCRUES the configured earn rate over the elapsed wall-clock span,
    * writes an append-only JSONL event, and
    * reports cumulative realized yield + the current peg to Discord.

It places NO orders, signs nothing, holds NO key, touches NO wallet/custody and
uses NO Kraken PRIVATE/auth API (no earn-subscribe/deposit/withdraw/balance) —
the ONLY Kraken call is the public ``fetch_ticker``. The earn APY itself is
OPERATOR-SUPPLIED via ``--apy`` because Kraken Earn rates are not exposed through
any clean public API; verify it on Kraken Earn before trusting an accrual.

Restart-safe: state is folded from the ledger each scan, so a crash/restart
resumes the same shadow positions and cumulative yield. Accrual is pure
wall-clock interest: the earn over a span of ``seconds`` at annual rate ``apy``
on ``notional`` is ``apy * notional * seconds / 31_557_600`` (365.25 d/yr). This
samples the configured rate at scan cadence with no look-ahead — only the elapsed
time between the last and current scan is ever credited.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "yield_accrued",
    "peg_deviation",
    "is_depeg",
    "fold_ledger",
    "run_scan",
    "DEFAULT_LEDGER",
    "DEFAULT_APY",
    "DEFAULT_NOTIONAL",
    "DEFAULT_DEPEG_THRESHOLD",
]

# Seconds in a Julian year (365.25 days) — the annualization base for accrual.
SECONDS_PER_YEAR = 31_557_600.0

DEFAULT_LEDGER = "runs/yield_ledger.jsonl"
DEFAULT_APY = 0.045              # OPERATOR-SUPPLIED Kraken Earn APY — verify on Kraken Earn.
DEFAULT_NOTIONAL = 10_000.0      # shadow USD supplied per stablecoin
DEFAULT_DEPEG_THRESHOLD = 0.005  # 0.5% off $1 flags a depeg risk alert

_EVENT_OPEN = "open"
_EVENT_ACCRUE = "accrue"


# ----------------------------------------------------------------------------
# Pure helpers (tested directly, no I/O)
# ----------------------------------------------------------------------------
def yield_accrued(apy: float, notional: float, seconds: float) -> float:
    """Earn (USD) accrued on ``notional`` at annual ``apy`` over ``seconds``.

    Pure wall-clock interest annualized on a 365.25-day year:
    ``apy * notional * (seconds / 31_557_600)``. One full year at apy=0.045 on
    $10,000 returns $450.0; half a year returns $225.0.
    """
    return float(apy) * float(notional) * (float(seconds) / SECONDS_PER_YEAR)


def peg_deviation(price: float) -> float:
    """Signed deviation of a stablecoin price from its $1 peg: ``price - 1.0``.

    Negative when trading below peg (the dangerous direction for a depositor).
    """
    return float(price) - 1.0


def is_depeg(deviation: float, threshold: float) -> bool:
    """True when the absolute peg deviation exceeds ``threshold`` (e.g. 0.005)."""
    return abs(float(deviation)) > float(threshold)


# ----------------------------------------------------------------------------
# Ledger I/O (append-only JSONL, restart-safe fold)
# ----------------------------------------------------------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def _write(path: str, payload: Dict[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def fold_ledger(path: str) -> Dict[str, Dict[str, Any]]:
    """Replay the JSONL into per-stablecoin shadow state.

    Returns ``{symbol: {apy, notional, entry_ts, last_ts_ms, realized_usd,
    n_accruals, last_price, last_deviation}}``. ``open`` sets the position;
    ``accrue`` adds realized yield, advances ``last_ts_ms`` and records the
    latest observed price + deviation.
    """
    state: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return state
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            sym = e.get("symbol")
            if not sym:
                continue
            if e.get("_event") == _EVENT_OPEN:
                state[sym] = {
                    "apy": float(e.get("apy", 0.0)),
                    "notional": float(e.get("notional", 0.0)),
                    "entry_ts": e.get("ts"),
                    "last_ts_ms": float(e.get("ts_ms", 0.0)),
                    "realized_usd": 0.0,
                    "n_accruals": 0,
                    "last_price": None,
                    "last_deviation": None,
                }
            elif e.get("_event") == _EVENT_ACCRUE:
                pos = state.get(sym)
                if pos is None:
                    continue
                pos["realized_usd"] += float(e.get("realized_delta", 0.0))
                pos["last_ts_ms"] = float(e.get("ts_ms", pos["last_ts_ms"]))
                pos["n_accruals"] += 1
                pos["last_price"] = float(e.get("price")) if e.get("price") is not None else pos["last_price"]
                pos["last_deviation"] = (
                    float(e.get("deviation")) if e.get("deviation") is not None else pos["last_deviation"]
                )
    return state


# ----------------------------------------------------------------------------
# Scan
# ----------------------------------------------------------------------------
def run_scan(
    path: str,
    prices: Dict[str, float],
    *,
    now_ms: float,
    apy: float,
    notional: float,
    stablecoins: Sequence[str],
    depeg_threshold: float,
) -> Dict[str, Any]:
    """One scan: open the basket on first run, else accrue + check peg per position.

    ``prices`` maps ``<SYM>/USD`` -> last price (injected so tests need no
    network). Pure w.r.t. I/O except the append-only ledger writes. On the first
    run (empty ledger) it writes an ``open`` event per stablecoin with
    ``last_ts_ms = now`` (no yield is credited at open — no look-ahead). On every
    later scan, for each open position with a current quote it accrues yield over
    ``(now - last_ts_ms)`` seconds, records the peg deviation, writes an
    ``accrue`` event, and collects a DEPEG alert when the deviation breaches the
    threshold. A stablecoin with no current quote this scan is skipped (no
    fabricated accrual). Returns ``{action, n, alerts:[...]}``.
    """
    state = fold_ledger(path)

    if not state:
        for sym in stablecoins:
            _write(path, {
                "_event": _EVENT_OPEN, "ts": _iso(now_ms), "ts_ms": now_ms,
                "symbol": sym, "apy": float(apy), "notional": float(notional),
            })
        return {"action": "opened", "n": len(list(stablecoins)), "alerts": []}

    accrued = 0
    alerts: List[Dict[str, Any]] = []
    for sym, pos in state.items():
        price = prices.get(_ticker(sym))
        if price is None:
            continue  # no current quote this scan; skip (no fabricated accrual)
        seconds = (now_ms - pos["last_ts_ms"]) / 1000.0
        if seconds < 0:
            continue
        delta = yield_accrued(pos["apy"], pos["notional"], seconds)
        dev = peg_deviation(price)
        _write(path, {
            "_event": _EVENT_ACCRUE, "ts": _iso(now_ms), "ts_ms": now_ms,
            "symbol": sym, "price": float(price), "deviation": dev,
            "realized_delta": delta,
        })
        accrued += 1
        if is_depeg(dev, depeg_threshold):
            alerts.append({
                "symbol": sym, "price": float(price), "deviation": dev,
                "threshold": float(depeg_threshold),
            })
    return {"action": "accrued", "n": accrued, "alerts": alerts}


def _ticker(symbol: str) -> str:
    """Map a tracked stablecoin (``USDC``) to its Kraken ticker (``USDC/USD``)."""
    return f"{symbol}/USD"


# ----------------------------------------------------------------------------
# Reporting (console + Discord)
# ----------------------------------------------------------------------------
def _annualized_pct(pos: Dict[str, Any], now_ms: float) -> Optional[float]:
    """Realized-to-date annualized as a % of notional, or None if not enough time
    has elapsed. Honest: derived from realized $ and actual held seconds, no
    forward projection of the configured APY."""
    notional = pos.get("notional") or 0.0
    if notional <= 0:
        return None
    # Held span from entry to the last accrual timestamp (what realized covers).
    last_ms = pos.get("last_ts_ms") or 0.0
    entry_ts = pos.get("entry_ts")
    if not entry_ts:
        return None
    try:
        entry_ms = datetime.fromisoformat(entry_ts).timestamp() * 1000.0
    except (TypeError, ValueError):
        return None
    held_s = (last_ms - entry_ms) / 1000.0
    if held_s <= 0:
        return None
    return (pos["realized_usd"] / notional) * (SECONDS_PER_YEAR / held_s) * 100.0


def _report_console(path: str) -> None:
    state = fold_ledger(path)
    if not state:
        print("  (no positions yet)")
        return
    total_realized = 0.0
    print(f"  {'symbol':<10}{'realized$':>12}{'peg':>10}{'dev(bps)':>10}{'accr':>6}")
    for sym, pos in sorted(state.items()):
        total_realized += pos["realized_usd"]
        price = pos.get("last_price")
        dev = pos.get("last_deviation")
        price_s = "—" if price is None else f"{price:.4f}"
        dev_s = "—" if dev is None else f"{dev * 10_000:+.1f}"
        depeg_flag = ""
        if dev is not None and price is not None:
            depeg_flag = "  DEPEG" if is_depeg(dev, DEFAULT_DEPEG_THRESHOLD) else ""
        print(f"  {sym:<10}{pos['realized_usd']:>12.4f}{price_s:>10}{dev_s:>10}"
              f"{pos['n_accruals']:>6}{depeg_flag}")
    deployed = sum(p["notional"] for p in state.values()) or 1.0
    print(f"  TOTAL realized yield: ${total_realized:.4f} on ${deployed:.0f} supplied (SHADOW)")


def build_discord_message(path: str, *, now_ms: float, depeg_threshold: float) -> str:
    """Build a short Discord status line per stablecoin: cumulative realized
    yield $ + annualized %, current peg + deviation in bps, and a prominent
    depeg warning line for any position currently off its peg."""
    state = fold_ledger(path)
    if not state:
        return "Yield SHADOW — no positions yet (SHADOW, NO ORDERS)."
    lines: List[str] = ["Yield SHADOW (NO ORDERS) — earn accrual + peg monitor"]
    depeg_lines: List[str] = []
    for sym, pos in sorted(state.items()):
        ann = _annualized_pct(pos, now_ms)
        ann_s = "—" if ann is None else f"{ann:+.2f}%/yr"
        price = pos.get("last_price")
        dev = pos.get("last_deviation")
        if price is None or dev is None:
            peg_s = "peg —"
        else:
            peg_s = f"peg ${price:.4f} ({dev * 10_000:+.1f} bps)"
        lines.append(f"{sym}: realized ${pos['realized_usd']:.4f} ({ann_s}) | {peg_s}")
        if price is not None and dev is not None and is_depeg(dev, depeg_threshold):
            depeg_lines.append(
                f"⚠ DEPEG {sym}: ${price:.4f} is {dev * 10_000:+.1f} bps off $1 "
                f"(> {depeg_threshold * 10_000:.0f} bps)"
            )
    return "\n".join(depeg_lines + lines)


def report_to_discord(path: str, notifier: Any, *, now_ms: float, depeg_threshold: float) -> bool:
    """Post the yield + peg status to Discord via :class:`alerts.notifier.Notifier`.

    Routes a depeg through ``notifier.alert`` (action-required) and the routine
    status through ``notifier.info``. Best-effort: a notifier failure is logged
    and swallowed so the loop never crashes. Returns True if a send succeeded.
    """
    message = build_discord_message(path, now_ms=now_ms, depeg_threshold=depeg_threshold)
    has_depeg = "⚠ DEPEG" in message
    try:
        if has_depeg:
            return bool(notifier.alert(message, severity="critical"))
        return bool(notifier.info(message))
    except Exception as exc:  # noqa: BLE001 - a Discord failure must not kill the run.
        print(f"  discord report failed ({type(exc).__name__}: {str(exc)[:80]}); continuing")
        return False


# ----------------------------------------------------------------------------
# Live price fetch (public Kraken ticker ONLY — no auth, no orders)
# ----------------------------------------------------------------------------
def fetch_prices(stablecoins: Sequence[str]) -> Dict[str, float]:
    """Read the live last price of each ``<SYM>/USD`` from the PUBLIC Kraken
    ticker via ccxt. Public market data only — no key, no auth, no order path.
    Factored out so tests inject a fake instead of hitting the network.
    """
    import ccxt
    ex = ccxt.kraken({"timeout": 20000, "enableRateLimit": True})
    prices: Dict[str, float] = {}
    for sym in stablecoins:
        ticker = ex.fetch_ticker(_ticker(sym))
        last = ticker.get("last")
        if last is not None:
            prices[_ticker(sym)] = float(last)
    return prices


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Stablecoin yield SHADOW runner + peg monitor (NO ORDERS).")
    p.add_argument("--stablecoin", action="append", dest="stablecoins",
                   help="stablecoin to track (repeatable); Kraken ticker is <SYM>/USD. Default: USDC")
    p.add_argument("--apy", type=float, default=DEFAULT_APY,
                   help="OPERATOR-SUPPLIED Kraken Earn APY as a fraction (verify on Kraken Earn; "
                        "not available from any clean API). Default 0.045 (4.5%%)")
    p.add_argument("--notional", type=float, default=DEFAULT_NOTIONAL,
                   help="shadow USD supplied per stablecoin (default 10000)")
    p.add_argument("--interval", type=float, default=3600.0,
                   help="seconds between scans (default 1h)")
    p.add_argument("--once", action="store_true")
    p.add_argument("--depeg-threshold", type=float, default=DEFAULT_DEPEG_THRESHOLD,
                   help="abs peg deviation that flags a depeg alert (default 0.005 = 0.5%%)")
    p.add_argument("--discord", action="store_true", help="post status to Discord each scan")
    p.add_argument("--ledger-path", default=DEFAULT_LEDGER)
    args = p.parse_args(argv)

    stablecoins = args.stablecoins or ["USDC"]

    notifier = None
    if args.discord:
        # Lazy + tolerant: load env + build Notifier here so the pure path never
        # depends on alerts/requests being importable.
        from portfolio_reporter import load_env_files
        from alerts.notifier import Notifier
        load_env_files()
        notifier = Notifier()

    def scan() -> None:
        now_ms = time.time() * 1000.0
        try:
            prices = fetch_prices(stablecoins)
        except Exception as exc:  # noqa: BLE001 - skip this scan, do not fabricate.
            print(f"  ticker fetch failed ({type(exc).__name__}: {str(exc)[:80]}); skipping scan")
            return
        res = run_scan(
            args.ledger_path, prices, now_ms=now_ms, apy=args.apy,
            notional=args.notional, stablecoins=stablecoins,
            depeg_threshold=args.depeg_threshold,
        )
        print(f"SHADOW yield scan @ {_utc_now_iso()}: {res['action']} "
              f"{res['n']} position(s), {len(res['alerts'])} depeg alert(s)")
        for a in res["alerts"]:
            print(f"  ⚠ DEPEG {a['symbol']}: ${a['price']:.4f} "
                  f"({a['deviation'] * 10_000:+.1f} bps off $1)")
        _report_console(args.ledger_path)
        if notifier is not None:
            report_to_discord(args.ledger_path, notifier, now_ms=now_ms,
                              depeg_threshold=args.depeg_threshold)

    print("=== stablecoin yield SHADOW runner — NO ORDERS PLACED, public ticker only ===")
    print(f"    apy={args.apy:.4f} (operator-supplied; verify on Kraken Earn) "
          f"notional=${args.notional:.0f}/coin tracking={stablecoins}")
    if args.once:
        scan()
        return 0
    try:
        while True:
            try:
                scan()
            except Exception as exc:  # noqa: BLE001 - a transient error must not kill the run.
                print(f"  scan failed ({type(exc).__name__}: {str(exc)[:80]}); retrying next tick")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped by user. No orders were ever placed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
