"""Calibration drift monitor — daily reliability check for closed trades.

Sprint 2 item #8. Sibling to ``scripts/diagnose_live_features.py``: same
``PositionStore`` + ``cfg`` shape, same CLI style. This script is
intended to run on a daily cron right after ``diagnose_live_features.py``.

The check
---------
Given a window of closed positions, bucket them by entry-time model
confidence and compare the realized winrate per bucket against the
bucket midpoint. A perfectly-calibrated model produces realized winrate
≈ midpoint for every bucket — equivalently, a reliability fit of
``realized_wr = slope * midpoint + intercept`` has slope ≈ 1 and
intercept ≈ 0. Slope drift > ``--slope-tolerance`` (default 0.10) is
the alert signal.

Slope formula
-------------
Weighted least-squares fit with weights ``w_i = sqrt(n_i)`` (standard
heuristic since bucket SE scales as ``1/sqrt(n)``). Closed-form::

    W   = Σ w_i
    Sx  = Σ w_i · x_i
    Sy  = Σ w_i · y_i
    Sxx = Σ w_i · x_i²
    Sxy = Σ w_i · x_i · y_i
    D   = W · Sxx − Sx²

    slope     = (W · Sxy − Sx · Sy) / D
    intercept = (Sxx · Sy − Sx · Sxy) / D

Only buckets meeting ``--min-n-per-bucket`` are admitted to the fit;
the rest are listed as ``(insufficient)`` in the table. Two-or-fewer
admitted buckets collapse the WLS fit (D ≤ 0) and we report
``NO_DATA`` for the verdict — calibration cannot be assessed from a
single point.

Confidence sources
------------------
The live supervisor today does NOT round-trip the signal-time
confidence onto the ``Position`` blob (``model_meta`` is empty on
disk; verified against the 2026-05-18 smoke fixture). The script
therefore looks up confidence in this priority order:

    1. ``position.model_meta["entry_confidence"]`` (forward-compat:
       future supervisor commits may populate this field directly).
    2. ``TradeContextStore.get_signal_snapshot(trade_id=position_id)``
       → ``snap.model_confidence``. Paper-deferred fills bind
       ``position_id == trade_id`` (see ``_drain_pending_paper_fill``).

Positions without either are surfaced as ``unconfidenced`` in the
report and excluded from the bucket calc. When every closed position
in the window is unconfidenced the verdict collapses to ``NO_DATA``.

CLI exit codes
--------------
* ``0`` — verdict ``OK`` or ``NO_DATA`` or Redis unreachable (the
  monitor is best-effort; a missing Redis shouldn't fail the cron).
* ``1`` — verdict ``ALERT`` (only when ``--alert`` is set; without
  ``--alert`` the table prints and exit is still 0 — the script is a
  reporter in that mode).
* ``2`` — a non-connection :class:`redis.exceptions.RedisError` was
  raised, or an internal invariant failed. Logged with a stack trace.

Usage::

    ./.venv/bin/python scripts/diagnose_calibration_drift.py \\
        --date 2026-05-18 \\
        --alert
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import traceback
from datetime import date as date_cls, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
for _p in (SRC, REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import redis.exceptions  # noqa: E402

from state.position_store import Position, PositionStore  # noqa: E402
from state.trade_context_store import TradeContextStore  # noqa: E402


LOGGER = logging.getLogger("diagnose_calibration_drift")

DEFAULT_BINS_CSV = "0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00"
DEFAULT_WINDOW_DAYS = 7
DEFAULT_SLOPE_TOLERANCE = 0.10
DEFAULT_MIN_N_PER_BUCKET = 5
DEFAULT_NAMESPACE = "autopilot"
DEFAULT_REDIS_URL = "redis://localhost:6379/0"

# Verdict labels.
VERDICT_OK = "OK"
VERDICT_ALERT = "ALERT"
VERDICT_NO_DATA = "NO_DATA"


# ---------------------------------------------------------------------------
# Confidence resolution
# ---------------------------------------------------------------------------
def resolve_confidence(
    position: Position,
    *,
    trade_ctx_store: Optional[TradeContextStore],
) -> Optional[float]:
    """Best-effort lookup of entry-time confidence for ``position``.

    Returns None when neither source has a usable float. Non-finite
    confidences (NaN/inf) are coerced to None — a calibration check on
    a NaN confidence is meaningless and would silently poison the bucket
    summary.
    """
    meta = position.model_meta or {}
    raw = meta.get("entry_confidence")
    if raw is None:
        raw = meta.get("model_confidence")
    if raw is not None:
        try:
            v = float(raw)
        except (TypeError, ValueError):
            v = None
        else:
            if math.isfinite(v):
                return v

    if trade_ctx_store is None:
        return None
    try:
        snap = trade_ctx_store.get_signal_snapshot(position.position_id)
    except redis.exceptions.RedisError:
        # Propagate to the caller — Redis errors are surfaced uniformly.
        raise
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        LOGGER.debug(
            "signal snapshot lookup failed for %s: %r", position.position_id, exc
        )
        return None
    if snap is None:
        return None
    try:
        v = float(snap.model_confidence)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------
def _parse_iso_utc(value: str) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utc_date_range(
    *, date: Optional[date_cls], window_days: int, now_utc: datetime
) -> Tuple[datetime, datetime, str]:
    """Return ``(window_start, window_end, label)``.

    ``window_start`` and ``window_end`` are inclusive lower / exclusive
    upper bounds on the ``opened_at_utc`` filter. ``label`` is a short
    human string used in the report header.
    """
    if date is not None:
        start = datetime(date.year, date.month, date.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        return start, end, f"{date.isoformat()}"
    end = datetime(
        now_utc.year, now_utc.month, now_utc.day, tzinfo=timezone.utc
    ) + timedelta(days=1)
    start = end - timedelta(days=window_days)
    return start, end, f"last {window_days} days (≤ {(end - timedelta(days=1)).date().isoformat()})"


def list_window_closed(
    store: PositionStore,
    *,
    window_start: datetime,
    window_end: datetime,
    symbol: Optional[str],
) -> List[Position]:
    """Pull every closed position whose ``opened_at_utc`` falls in the window.

    Walks the ``{ns}:closed:{YYYY-MM-DD}`` sets across the date range and
    pads by one day on each side (positions opened late on day N can land
    in the closed-set for day N+1, and vice versa under clock skew). Each
    position is then filtered by its actual ``opened_at_utc``.
    """
    walk_days: List[date_cls] = []
    cursor = (window_start - timedelta(days=1)).date()
    final = (window_end + timedelta(days=1)).date()
    while cursor <= final:
        walk_days.append(cursor)
        cursor = cursor + timedelta(days=1)

    seen_ids: set[str] = set()
    out: List[Position] = []
    for d in walk_days:
        when = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        closed_key = store._closed_set_key(when)  # noqa: SLF001 - intentional
        try:
            ids = store._redis.smembers(closed_key) or set()  # noqa: SLF001
        except redis.exceptions.RedisError:
            raise
        for pid in ids:
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            try:
                position = store.get(pid)
            except redis.exceptions.RedisError:
                raise
            if position is None:
                continue
            if position.status != "closed":
                continue
            if position.realized_pnl_usd is None:
                continue
            opened_at = _parse_iso_utc(position.opened_at_utc)
            if opened_at is None:
                continue
            if not (window_start <= opened_at < window_end):
                continue
            if symbol is not None and position.symbol != symbol:
                continue
            out.append(position)
    # Stable order so the table is reproducible.
    out.sort(key=lambda p: p.opened_at_utc)
    return out


# ---------------------------------------------------------------------------
# Bucketing + WLS fit
# ---------------------------------------------------------------------------
def parse_bins(bins_csv: str) -> List[float]:
    """Parse the ``--bins`` CSV into a sorted list of unique edges.

    Edges must be strictly increasing and lie within ``[0.0, 1.0]``.
    """
    parts = [s.strip() for s in bins_csv.split(",") if s.strip()]
    edges: List[float] = []
    for s in parts:
        v = float(s)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"bin edge {v!r} not in [0.0, 1.0]")
        edges.append(v)
    if len(edges) < 2:
        raise ValueError("need at least two bin edges to form a bucket")
    for a, b in zip(edges, edges[1:]):
        if not (a < b):
            raise ValueError(f"bin edges must be strictly increasing: {a} !< {b}")
    return edges


def assign_bucket(confidence: float, edges: List[float]) -> Optional[int]:
    """Return the bucket index whose ``[edges[i], edges[i+1])`` contains
    ``confidence`` (final bucket is closed on the right). Returns None
    when ``confidence`` falls outside the full ``[edges[0], edges[-1]]``
    range.
    """
    if not math.isfinite(confidence):
        return None
    if confidence < edges[0] or confidence > edges[-1]:
        return None
    last = len(edges) - 2
    for i in range(last + 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == last:
            if lo <= confidence <= hi:
                return i
        else:
            if lo <= confidence < hi:
                return i
    return None


def is_win(position: Position) -> bool:
    """Realized winner if ``realized_pnl_usd > 0``. Zero is not a win."""
    pnl = position.realized_pnl_usd
    if pnl is None:
        return False
    return float(pnl) > 0.0


def bucket_table(
    positions: List[Position],
    *,
    edges: List[float],
    trade_ctx_store: Optional[TradeContextStore],
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Build the per-bucket rows.

    Returns ``(rows, confidenced_count, unconfidenced_count)``. Each row
    has ``bucket``, ``midpoint``, ``n``, ``wins``, ``realized_wr`` and
    ``residual`` (None when ``n == 0``).
    """
    n_buckets = len(edges) - 1
    counts = [0] * n_buckets
    wins = [0] * n_buckets
    confidenced = 0
    unconfidenced = 0

    for position in positions:
        conf = resolve_confidence(position, trade_ctx_store=trade_ctx_store)
        if conf is None:
            unconfidenced += 1
            continue
        idx = assign_bucket(conf, edges)
        if idx is None:
            unconfidenced += 1
            continue
        confidenced += 1
        counts[idx] += 1
        if is_win(position):
            wins[idx] += 1

    rows: List[Dict[str, Any]] = []
    for i in range(n_buckets):
        lo = edges[i]
        hi = edges[i + 1]
        mid = 0.5 * (lo + hi)
        n = counts[i]
        w = wins[i]
        wr = (w / n) if n > 0 else None
        residual = (wr - mid) if wr is not None else None
        rows.append(
            {
                "bucket": f"{lo:.2f}-{hi:.2f}",
                "lo": lo,
                "hi": hi,
                "midpoint": mid,
                "n": n,
                "wins": w,
                "realized_wr": wr,
                "residual": residual,
            }
        )
    return rows, confidenced, unconfidenced


def wls_slope_intercept(
    rows: List[Dict[str, Any]], *, min_n: int
) -> Tuple[Optional[float], Optional[float], int]:
    """Weighted least-squares fit of realized_wr vs midpoint.

    Weights ``w_i = sqrt(n_i)`` for buckets with ``n_i >= min_n``;
    buckets below the floor are excluded entirely. Returns
    ``(slope, intercept, n_buckets_used)``. When fewer than two buckets
    qualify, or the design matrix is degenerate (D <= 0), returns
    ``(None, None, n_buckets_used)``.
    """
    xs: List[float] = []
    ys: List[float] = []
    ws: List[float] = []
    for r in rows:
        n = int(r["n"])
        wr = r["realized_wr"]
        if n < min_n or wr is None:
            continue
        xs.append(float(r["midpoint"]))
        ys.append(float(wr))
        ws.append(math.sqrt(float(n)))

    used = len(xs)
    if used < 2:
        return None, None, used

    W = sum(ws)
    Sx = sum(w * x for w, x in zip(ws, xs))
    Sy = sum(w * y for w, y in zip(ws, ys))
    Sxx = sum(w * x * x for w, x in zip(ws, xs))
    Sxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))
    D = W * Sxx - Sx * Sx
    if D <= 0:
        return None, None, used
    slope = (W * Sxy - Sx * Sy) / D
    intercept = (Sxx * Sy - Sx * Sxy) / D
    return slope, intercept, used


_TOLERANCE_EPSILON = 1e-9


def classify_verdict(
    slope: Optional[float], *, tolerance: float
) -> str:
    """Map slope/tolerance to ``OK`` / ``ALERT`` / ``NO_DATA``.

    A tiny epsilon is added to the tolerance comparison so that exact-
    boundary inputs like ``slope=1.10, tolerance=0.10`` still classify
    as OK despite floating-point imprecision in ``abs(1.10 - 1.0)``.
    """
    if slope is None:
        return VERDICT_NO_DATA
    if abs(slope - 1.0) <= tolerance + _TOLERANCE_EPSILON:
        return VERDICT_OK
    return VERDICT_ALERT


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _fmt_float(v: Optional[float], width: int = 6) -> str:
    if v is None:
        return f"{'-':>{width}}"
    return f"{v:>{width}.3f}"


def _fmt_signed(v: Optional[float], width: int = 7) -> str:
    if v is None:
        return f"{'-':>{width}}"
    return f"{v:>+{width}.3f}"


def render_report(
    *,
    label: str,
    symbol: Optional[str],
    n_total: int,
    confidenced: int,
    unconfidenced: int,
    rows: List[Dict[str, Any]],
    min_n: int,
    slope: Optional[float],
    intercept: Optional[float],
    tolerance: float,
    verdict: str,
) -> str:
    """Render the human-readable report. Returns the full text block."""
    lines: List[str] = []
    lines.append(f"Calibration drift report — {label}")
    sym = symbol if symbol is not None else "ALL"
    n_eligible = sum(1 for r in rows if int(r["n"]) >= min_n)
    n_buckets = len(rows)
    lines.append(
        f"Symbol: {sym}  |  closed trades: {n_total}  |  "
        f"confidenced: {confidenced}  |  unconfidenced: {unconfidenced}  |  "
        f"bins evaluated: {n_eligible}/{n_buckets}"
    )
    lines.append("")
    lines.append(
        "| bin           | midpoint |  n | wins | realized_wr | residual | note         |"
    )
    lines.append(
        "| ------------- | -------: | -: | ---: | ----------: | -------: | ------------ |"
    )
    for r in rows:
        n = int(r["n"])
        wr = r["realized_wr"]
        resid = r["residual"]
        note = "" if n >= min_n and wr is not None else "(insufficient)"
        lines.append(
            f"| {r['bucket']:<13} | "
            f"{r['midpoint']:>8.3f} | "
            f"{n:>2d} | "
            f"{int(r['wins']):>4d} | "
            f"{_fmt_float(wr, width=11)} | "
            f"{_fmt_signed(resid, width=8)} | "
            f"{note:<12} |"
        )
    lines.append("")

    if slope is None:
        lines.append(
            f"reliability slope: -- (target 1.0, tolerance ±{tolerance:.2f}) — INSUFFICIENT_DATA"
        )
        lines.append("reliability intercept: --")
    else:
        status = (
            "IN_TOLERANCE"
            if abs(slope - 1.0) <= tolerance
            else "OUT_OF_TOLERANCE"
        )
        lines.append(
            f"reliability slope: {slope:.3f} (target 1.0, tolerance ±{tolerance:.2f}) — {status}"
        )
        lines.append(
            f"reliability intercept: {intercept:.3f}"
            if intercept is not None
            else "reliability intercept: --"
        )
    lines.append(f"verdict: {verdict}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------
def build_json_blob(
    *,
    label: str,
    symbol: Optional[str],
    window_start: datetime,
    window_end: datetime,
    n_total: int,
    confidenced: int,
    unconfidenced: int,
    rows: List[Dict[str, Any]],
    min_n: int,
    slope: Optional[float],
    intercept: Optional[float],
    tolerance: float,
    verdict: str,
) -> Dict[str, Any]:
    """Build the downstream-consumable JSON blob written by ``--out``."""
    return {
        "label": label,
        "symbol": symbol,
        "window_start_utc": window_start.isoformat(),
        "window_end_utc": window_end.isoformat(),
        "n_closed_trades": int(n_total),
        "n_confidenced": int(confidenced),
        "n_unconfidenced": int(unconfidenced),
        "min_n_per_bucket": int(min_n),
        "slope_tolerance": float(tolerance),
        "reliability_slope": (None if slope is None else float(slope)),
        "reliability_intercept": (
            None if intercept is None else float(intercept)
        ),
        "verdict": verdict,
        "buckets": [
            {
                "bucket": r["bucket"],
                "lo": float(r["lo"]),
                "hi": float(r["hi"]),
                "midpoint": float(r["midpoint"]),
                "n": int(r["n"]),
                "wins": int(r["wins"]),
                "realized_wr": (
                    None if r["realized_wr"] is None else float(r["realized_wr"])
                ),
                "residual": (
                    None if r["residual"] is None else float(r["residual"])
                ),
                "included_in_fit": int(r["n"]) >= min_n and r["realized_wr"] is not None,
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Per-bucket reliability check for closed positions. Compares "
            "realized winrate against confidence-bucket midpoints and "
            "alerts when the WLS slope deviates from 1.0."
        )
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help=(
            "YYYY-MM-DD UTC. Filter to positions opened on that date. "
            "When omitted, falls back to a rolling --window-days window "
            "ending today (UTC)."
        ),
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help=(
            "Rolling window in days when --date is omitted "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--bins",
        type=str,
        default=DEFAULT_BINS_CSV,
        help=(
            "Comma-separated confidence bin edges (default: %(default)s). "
            "Left-inclusive, right-exclusive; the final edge is inclusive."
        ),
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Filter to one symbol (default: all symbols).",
    )
    parser.add_argument(
        "--slope-tolerance",
        type=float,
        default=DEFAULT_SLOPE_TOLERANCE,
        help=(
            "ALERT when |slope - 1.0| exceeds this tolerance "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--min-n-per-bucket",
        type=int,
        default=DEFAULT_MIN_N_PER_BUCKET,
        help=(
            "Minimum trades per bucket required for the bucket to be "
            "admitted into the slope regression (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--alert",
        action="store_true",
        help=(
            "Post a Telegram + Discord alert when verdict=ALERT. Without "
            "this flag the script is stdout-only."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Optional path to write the report as a JSON blob in addition "
            "to printing the human-readable table."
        ),
    )
    parser.add_argument(
        "--redis-url",
        type=str,
        default=None,
        help=(
            "Redis URL (default: $REDIS_URL or redis://localhost:6379/0). "
            "Matches PositionStore's default chain."
        ),
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=DEFAULT_NAMESPACE,
        help="PositionStore namespace (default: %(default)s).",
    )
    return parser


def _post_alert_via_notifier(
    *,
    notifier: Any,
    label: str,
    symbol: Optional[str],
    slope: float,
    intercept: Optional[float],
    tolerance: float,
    n_total: int,
    confidenced: int,
) -> None:
    """Post a multi-channel alert via the project Notifier. Best-effort."""
    fields: Dict[str, str] = {
        "Window": label,
        "Symbol": symbol if symbol is not None else "ALL",
        "Slope": f"{slope:.3f}",
        "Tolerance": f"±{tolerance:.2f}",
        "Closed trades": str(n_total),
        "Confidenced": str(confidenced),
    }
    if intercept is not None:
        fields["Intercept"] = f"{intercept:.3f}"
    notifier.alert(
        f"Calibration drift: slope {slope:.3f} (|Δ|={abs(slope - 1.0):.3f} > {tolerance:.2f})",
        severity="alert",
        fields=fields,
    )


def run(
    *,
    args: argparse.Namespace,
    store: PositionStore,
    trade_ctx_store: Optional[TradeContextStore],
    notifier: Any = None,
    now_utc: Optional[datetime] = None,
) -> Tuple[int, Dict[str, Any], str]:
    """Core logic — exposed for tests.

    Returns ``(exit_code, json_blob, report_text)``.
    """
    now_utc = now_utc or datetime.now(timezone.utc)

    parsed_date: Optional[date_cls] = None
    if args.date is not None:
        try:
            parsed_date = date_cls.fromisoformat(args.date)
        except ValueError as exc:
            raise ValueError(f"--date must be YYYY-MM-DD, got {args.date!r}") from exc

    edges = parse_bins(args.bins)
    window_start, window_end, label = _utc_date_range(
        date=parsed_date, window_days=int(args.window_days), now_utc=now_utc
    )

    positions = list_window_closed(
        store,
        window_start=window_start,
        window_end=window_end,
        symbol=args.symbol,
    )
    rows, confidenced, unconfidenced = bucket_table(
        positions, edges=edges, trade_ctx_store=trade_ctx_store
    )

    if not positions:
        verdict = VERDICT_NO_DATA
        slope: Optional[float] = None
        intercept: Optional[float] = None
    else:
        slope, intercept, _used = wls_slope_intercept(
            rows, min_n=int(args.min_n_per_bucket)
        )
        verdict = classify_verdict(slope, tolerance=float(args.slope_tolerance))

    report = render_report(
        label=label,
        symbol=args.symbol,
        n_total=len(positions),
        confidenced=confidenced,
        unconfidenced=unconfidenced,
        rows=rows,
        min_n=int(args.min_n_per_bucket),
        slope=slope,
        intercept=intercept,
        tolerance=float(args.slope_tolerance),
        verdict=verdict,
    )
    blob = build_json_blob(
        label=label,
        symbol=args.symbol,
        window_start=window_start,
        window_end=window_end,
        n_total=len(positions),
        confidenced=confidenced,
        unconfidenced=unconfidenced,
        rows=rows,
        min_n=int(args.min_n_per_bucket),
        slope=slope,
        intercept=intercept,
        tolerance=float(args.slope_tolerance),
        verdict=verdict,
    )

    if args.alert and verdict == VERDICT_ALERT and notifier is not None and slope is not None:
        _post_alert_via_notifier(
            notifier=notifier,
            label=label,
            symbol=args.symbol,
            slope=slope,
            intercept=intercept,
            tolerance=float(args.slope_tolerance),
            n_total=len(positions),
            confidenced=confidenced,
        )

    # Exit code policy: ALERT promotes to 1 only when --alert was set
    # (the operator opted into noticing). NO_DATA and OK always exit 0.
    if verdict == VERDICT_ALERT and args.alert:
        exit_code = 1
    else:
        exit_code = 0
    return exit_code, blob, report


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("CALIBRATION_DRIFT_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    redis_url = (
        args.redis_url
        or os.environ.get("REDIS_URL")
        or DEFAULT_REDIS_URL
    )
    try:
        store = PositionStore(redis_url=redis_url, namespace=args.namespace)
        trade_ctx_store = TradeContextStore(
            redis_url=redis_url, namespace=args.namespace
        )
    except redis.exceptions.ConnectionError as exc:
        print(
            f"Redis unreachable at {redis_url}: {exc}. "
            "Skipping today's calibration check."
        )
        return 0
    except redis.exceptions.RedisError:
        LOGGER.error("Redis init failed:\n%s", traceback.format_exc())
        return 2

    # Lazy-construct the notifier only when --alert is on so an unconfigured
    # alerts pipeline doesn't cause an import-time surprise.
    notifier = None
    if args.alert:
        try:
            from alerts.notifier import Notifier

            notifier = Notifier()
        except ImportError as exc:
            LOGGER.warning(
                "alerts.notifier import failed; skipping alert post: %r", exc
            )
            notifier = None

    try:
        exit_code, blob, report = run(
            args=args,
            store=store,
            trade_ctx_store=trade_ctx_store,
            notifier=notifier,
        )
    except redis.exceptions.ConnectionError as exc:
        print(
            f"Redis unreachable mid-scan: {exc}. "
            "Skipping today's calibration check."
        )
        return 0
    except redis.exceptions.RedisError:
        LOGGER.error(
            "Redis error during calibration scan:\n%s", traceback.format_exc()
        )
        return 2
    except ValueError as exc:
        LOGGER.error("input error: %s", exc)
        return 2

    print(report)
    if args.out:
        try:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(blob, indent=2, sort_keys=True))
        except OSError as exc:
            LOGGER.warning("failed to write --out=%s: %r", args.out, exc)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
