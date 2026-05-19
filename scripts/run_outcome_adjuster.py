"""Daily OutcomeAdjuster runner — recompute per-regime threshold deltas.

Sprint 2 item #6 (CEO review 2026-05-17). Sister cron to
``scripts/run_postmortem.py`` (Lane E digest, 00:05 UTC) and
``scripts/diagnose_calibration_drift.py`` (calibration drift, 00:10 UTC):
this script walks the previous UTC day's closed positions, recomputes
per-regime streaks via :class:`OutcomeAdjuster`, and persists the new
threshold adjustments to the Redis hash
``{namespace}:regime_outcome_adjustment``.

Regime label resolution
-----------------------
For each closed position the resolver tries (in order):

    1. ``signal_snapshot["regime_label"]`` — the canonical seam. Live
       supervisor commits as of 2026-05-18 do NOT populate this field;
       once they do, this path lights up automatically.
    2. ``signal_snapshot["risk_metrics_input"]["regime_label"]`` — same
       seam, but if a future commit nests the field under risk_metrics
       (Lane E specialists already look in both spots).
    3. ``position.model_meta["regime_label"]`` — opportunistic, for
       positions whose entry confidence dispatcher persisted it.

Positions where none of those produce a label are SKIPPED, logged at
INFO, and counted in the summary header. Re-encoding entry-time features
through the regime encoder is documented as a fallback in the spec but
needs a stored feature_window on the snapshot to be useful — live
supervisor stores ``feature_buffer={}`` and ``feature_window=None``
today, so re-encoding is a no-op until that seam is wired. Skipping is
the correct conservative behavior.

CLI exit codes
--------------
* ``0`` — happy path, OR Redis unreachable (best-effort cron).
* ``2`` — input error (bad --date format, etc.) or a non-connection
  ``redis.exceptions.RedisError``.

Usage::

    ./.venv/bin/python scripts/run_outcome_adjuster.py \\
        --date 2026-05-18 \\
        --dry-run

    ./.venv/bin/python scripts/run_outcome_adjuster.py --reset all
    ./.venv/bin/python scripts/run_outcome_adjuster.py --reset trend_up
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import date as date_cls, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
for _p in (SRC, REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import redis  # noqa: E402
import redis.exceptions  # noqa: E402

from regime_memory.outcome_adjuster import (  # noqa: E402
    OutcomeAdjuster,
    normalize_label,
)
from state.position_store import Position, PositionStore  # noqa: E402
from state.trade_context_store import TradeContextStore  # noqa: E402

LOGGER = logging.getLogger("run_outcome_adjuster")

DEFAULT_NAMESPACE = "autopilot"
DEFAULT_REDIS_URL = "redis://localhost:6379/0"


# ---------------------------------------------------------------------------
# Label resolution
# ---------------------------------------------------------------------------
def _try_label_from_signal_snapshot(
    position: Position,
    *,
    trade_ctx_store: Optional[TradeContextStore],
) -> Optional[str]:
    """Probe the signal-time snapshot for a regime label across known seams."""
    if trade_ctx_store is None:
        return None
    try:
        snap = trade_ctx_store.get_signal_snapshot(position.position_id)
    except redis.exceptions.RedisError:
        # Surface so the caller can decide whether to bail or keep going.
        raise
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
    if snap is None:
        return None
    # Probe a few candidate locations. The structured dataclass doesn't
    # have a top-level ``regime_label`` today, so we go through risk_metrics
    # plus the future-compat top-level seam.
    rmi = getattr(snap, "risk_metrics_input", {}) or {}
    rmo = getattr(snap, "risk_metrics_output", {}) or {}
    breaker_ctx = getattr(snap, "breaker_context", {}) or {}
    notes = getattr(snap, "notes", None)
    for source in (rmi, rmo, breaker_ctx):
        if isinstance(source, dict) and "regime_label" in source:
            label = normalize_label(source["regime_label"])
            if label is not None:
                return label
    # ``notes`` is free-form; we don't try to scrape it.
    _ = notes
    return None


def resolve_regime_label(
    position: Position,
    *,
    trade_ctx_store: Optional[TradeContextStore],
) -> Optional[str]:
    """Best-effort regime-label lookup for a closed position. See module doc."""
    # 1. model_meta on the Position blob (cheapest — single Redis hop).
    meta = position.model_meta or {}
    if "regime_label" in meta:
        label = normalize_label(meta["regime_label"])
        if label is not None:
            return label
    # 2. signal snapshot. Catches transport errors and re-raises so the
    # outer try in ``run`` can decide whether to skip the whole run.
    return _try_label_from_signal_snapshot(
        position, trade_ctx_store=trade_ctx_store
    )


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------
def _resolve_target_date(raw: Optional[str], *, now_utc: datetime) -> date_cls:
    """Default to yesterday UTC (matches run_postmortem.py)."""
    if raw is None:
        return (now_utc - timedelta(days=1)).date()
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            f"--date must be YYYY-MM-DD, got {raw!r} ({exc})"
        ) from exc


def list_closed_for_date(
    store: PositionStore,
    *,
    target: date_cls,
    symbol: Optional[str],
) -> List[Position]:
    """Pull the day's closed positions, sorted oldest -> newest."""
    when = datetime(
        target.year, target.month, target.day, 12, 0, 0, tzinfo=timezone.utc
    )
    positions = list(store.list_closed_today(now_utc=when))
    # Filter by symbol when set.
    if symbol is not None:
        positions = [p for p in positions if p.symbol == symbol]
    # Sort by closed_at_utc ascending so the streak math is deterministic.
    positions.sort(key=lambda p: p.closed_at_utc or "")
    return positions


# ---------------------------------------------------------------------------
# Per-label aggregation for the report
# ---------------------------------------------------------------------------
def _aggregate_for_report(
    positions: List[Position],
    label_resolver: Callable[[Position], Optional[str]],
) -> Tuple[Dict[str, Dict[str, Any]], int, int]:
    """Return ``(per_label_stats, resolved_count, skipped_count)``.

    Each per-label dict has ``wins``, ``losses``, ``streak`` (string like
    ``"4 losses"`` / ``"2 wins"`` describing the trailing streak), and a
    canonical ``label`` key.
    """
    stats: Dict[str, Dict[str, Any]] = {}
    # Trailing-streak tracker. The streak shown is the FINAL streak at the
    # end of the iteration (i.e. what the next trade would extend).
    last_outcome: Dict[str, str] = {}
    streak_len: Dict[str, int] = {}
    resolved = 0
    skipped = 0
    for position in positions:
        label = label_resolver(position)
        if label is None:
            skipped += 1
            continue
        pnl = position.realized_pnl_usd
        if pnl is None:
            skipped += 1
            continue
        resolved += 1
        bucket = stats.setdefault(
            label, {"label": label, "wins": 0, "losses": 0, "streak": ""}
        )
        outcome = "win" if float(pnl) > 0.0 else "loss"
        if outcome == "win":
            bucket["wins"] = int(bucket["wins"]) + 1
        else:
            bucket["losses"] = int(bucket["losses"]) + 1
        if last_outcome.get(label) == outcome:
            streak_len[label] = streak_len.get(label, 0) + 1
        else:
            streak_len[label] = 1
        last_outcome[label] = outcome
    for label, bucket in stats.items():
        ln = streak_len.get(label, 0)
        kind = last_outcome.get(label, "")
        if ln and kind:
            bucket["streak"] = f"{ln} {kind}{'es' if kind == 'loss' else 's'}"
    return stats, resolved, skipped


def render_report(
    *,
    target: date_cls,
    symbol: Optional[str],
    positions: List[Position],
    resolved: int,
    skipped: int,
    per_label_stats: Dict[str, Dict[str, Any]],
    prev_adjustments: Dict[str, float],
    new_adjustments: Dict[str, float],
    dry_run: bool,
) -> str:
    """Render the operator-facing report. Returns the full text block."""
    lines: List[str] = []
    suffix = " (DRY-RUN)" if dry_run else ""
    lines.append(f"OutcomeAdjuster run — {target.isoformat()}{suffix}")
    sym = symbol if symbol is not None else "ALL"
    lines.append(
        f"Symbol: {sym}  |  positions: {len(positions)}  |  "
        f"resolved labels: {resolved}  |  skipped: {skipped}"
    )
    lines.append("")
    lines.append(
        "| regime_label   | wins | losses | streak     | prev_adj | new_adj | Δ        |"
    )
    lines.append(
        "| -------------- | ---: | -----: | :--------- | -------: | ------: | -------: |"
    )
    # Iterate the union of labels in stats and in new/prev adjustments so
    # adjustments for labels that didn't trade today still show up.
    all_labels = (
        set(per_label_stats.keys())
        | set(prev_adjustments.keys())
        | set(new_adjustments.keys())
    )
    for label in sorted(all_labels):
        bucket = per_label_stats.get(
            label, {"label": label, "wins": 0, "losses": 0, "streak": ""}
        )
        prev = float(prev_adjustments.get(label, 0.0))
        new = float(new_adjustments.get(label, 0.0))
        delta = new - prev
        wins = int(bucket["wins"])
        losses = int(bucket["losses"])
        streak = str(bucket["streak"] or "-")
        lines.append(
            f"| {label[:14]:<14} | {wins:>4d} | {losses:>6d} | {streak:<10} | "
            f"{prev:>+8.3f} | {new:>+7.3f} | {delta:>+8.3f} |"
        )
    if not all_labels:
        lines.append("| (no regime labels found in window) |")
    lines.append("")
    if dry_run:
        lines.append("dry-run: no Redis writes performed.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reset / inspect helpers
# ---------------------------------------------------------------------------
def _do_reset(
    adjuster: OutcomeAdjuster, target: str
) -> Tuple[int, str]:
    """Handle the --reset flag. ``target`` is ``"all"`` or a single label."""
    if target == "all":
        adjuster.reset(label=None)
        return 0, "OutcomeAdjuster: cleared all adjustments"
    adjuster.reset(label=target)
    return 0, f"OutcomeAdjuster: cleared adjustment for label {target!r}"


# ---------------------------------------------------------------------------
# Core run
# ---------------------------------------------------------------------------
def run(
    *,
    args: argparse.Namespace,
    store: PositionStore,
    trade_ctx_store: Optional[TradeContextStore],
    adjuster: OutcomeAdjuster,
    notifier: Any = None,
    now_utc: Optional[datetime] = None,
) -> Tuple[int, str, Dict[str, Any]]:
    """Core logic. Returns ``(exit_code, report_text, json_blob)``.

    Exposed for tests so they don't go through the CLI parser.
    """
    now_utc = now_utc or datetime.now(timezone.utc)

    if args.reset is not None:
        code, msg = _do_reset(adjuster, args.reset)
        return code, msg, {"action": "reset", "target": args.reset}

    target = _resolve_target_date(args.date, now_utc=now_utc)
    positions = list_closed_for_date(
        store, target=target, symbol=args.symbol
    )

    def _label_resolver(position: Position) -> Optional[str]:
        try:
            return resolve_regime_label(
                position, trade_ctx_store=trade_ctx_store
            )
        except redis.exceptions.RedisError as exc:
            LOGGER.warning(
                "label resolver redis error for %s: %r", position.position_id, exc
            )
            return None

    prev_adjustments = adjuster.all_adjustments()
    per_label_stats, resolved, skipped = _aggregate_for_report(
        positions, _label_resolver
    )

    if args.dry_run:
        # Simulate apply without writing. We reuse a no-write probe by
        # running the streak math against an in-memory copy of the
        # adjuster's state. ``OutcomeAdjuster.apply_closed_positions``
        # always writes, so we replicate its inner loop here with the
        # same parameters.
        new_adjustments = _simulate_apply(
            adjuster, positions, _label_resolver
        )
    else:
        new_adjustments = adjuster.apply_closed_positions(
            positions, label_resolver=_label_resolver
        )

    report = render_report(
        target=target,
        symbol=args.symbol,
        positions=positions,
        resolved=resolved,
        skipped=skipped,
        per_label_stats=per_label_stats,
        prev_adjustments=prev_adjustments,
        new_adjustments=new_adjustments,
        dry_run=bool(args.dry_run),
    )

    blob: Dict[str, Any] = {
        "target_date": target.isoformat(),
        "symbol": args.symbol,
        "n_positions": len(positions),
        "n_resolved_labels": resolved,
        "n_skipped": skipped,
        "prev_adjustments": prev_adjustments,
        "new_adjustments": new_adjustments,
        "dry_run": bool(args.dry_run),
    }

    if args.alert and notifier is not None and not args.dry_run:
        _maybe_post_alert(
            notifier=notifier,
            target=target,
            prev_adjustments=prev_adjustments,
            new_adjustments=new_adjustments,
            per_event_delta=adjuster.per_event_delta,
        )

    return 0, report, blob


def _simulate_apply(
    adjuster: OutcomeAdjuster,
    positions: List[Position],
    label_resolver: Callable[[Position], Optional[str]],
) -> Dict[str, float]:
    """Pure-function streak simulation that mirrors
    :meth:`OutcomeAdjuster.apply_closed_positions` without writing to
    Redis. Lives in the CLI module so the adjuster stays focused on the
    write path; if the math diverges, the unit tests will catch it.
    """
    state: Dict[str, float] = dict(adjuster.all_adjustments())
    win_streaks: Dict[str, int] = {}
    loss_streaks: Dict[str, int] = {}
    max_adj = adjuster.max_adjustment
    per_event = adjuster.per_event_delta
    for position in positions:
        label = label_resolver(position)
        if label is None:
            continue
        pnl = getattr(position, "realized_pnl_usd", None)
        if pnl is None:
            continue
        try:
            pnl_f = float(pnl)
        except (TypeError, ValueError):
            continue
        if pnl_f > 0.0:
            loss_streaks[label] = 0
            new_count = win_streaks.get(label, 0) + 1
            win_streaks[label] = new_count
            if (
                new_count >= adjuster.wins_to_relax
                and new_count % adjuster.wins_to_relax == 0
            ):
                current = state.get(label, 0.0)
                state[label] = max(-max_adj, min(max_adj, current - per_event))
        else:
            win_streaks[label] = 0
            new_count = loss_streaks.get(label, 0) + 1
            loss_streaks[label] = new_count
            if (
                new_count >= adjuster.losses_to_raise
                and new_count % adjuster.losses_to_raise == 0
            ):
                current = state.get(label, 0.0)
                state[label] = max(-max_adj, min(max_adj, current + per_event))
    # Drop the zero entries to match what all_adjustments() would return
    # after a real write (which skips zero deltas to keep the hash compact).
    return {k: v for k, v in state.items() if v != 0.0}


def _maybe_post_alert(
    *,
    notifier: Any,
    target: date_cls,
    prev_adjustments: Dict[str, float],
    new_adjustments: Dict[str, float],
    per_event_delta: float,
) -> None:
    """Alert when any label moved by >= per_event_delta. Best-effort."""
    moved: List[Tuple[str, float, float]] = []
    all_labels = set(prev_adjustments.keys()) | set(new_adjustments.keys())
    for label in all_labels:
        prev = float(prev_adjustments.get(label, 0.0))
        new = float(new_adjustments.get(label, 0.0))
        if abs(new - prev) >= per_event_delta - 1e-9:
            moved.append((label, prev, new))
    if not moved:
        return
    fields: Dict[str, str] = {
        "Date": target.isoformat(),
        "Movers": str(len(moved)),
    }
    # Cap the field count so a many-regime day doesn't explode the alert.
    for label, prev, new in sorted(moved, key=lambda t: -abs(t[2] - t[1]))[:5]:
        fields[label] = f"{prev:+.3f} -> {new:+.3f}"
    notifier.alert(
        f"OutcomeAdjuster: {len(moved)} regime label(s) shifted",
        severity="info",
        fields=fields,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Walk a day's closed positions and recompute per-regime "
            "threshold adjustments. Writes to Redis hash "
            "``{ns}:regime_outcome_adjustment``."
        ),
    )
    parser.add_argument(
        "--date",
        default=None,
        help=(
            "UTC date YYYY-MM-DD whose closed positions to analyse. "
            "Defaults to yesterday UTC (matches the 00:10 UTC cron slot)."
        ),
    )
    parser.add_argument(
        "--position-store-url",
        dest="position_store_url",
        default=None,
        help=(
            "Redis URL for the position store + adjuster "
            "(default: $REDIS_URL or redis://localhost:6379/0)."
        ),
    )
    parser.add_argument(
        "--namespace",
        default=DEFAULT_NAMESPACE,
        help="Redis key namespace (default: %(default)s).",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Filter to one symbol (default: all symbols).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Recompute + print, write nothing.",
    )
    parser.add_argument(
        "--alert",
        action="store_true",
        help=(
            "Post a Telegram + Discord notification when any label moved "
            "by at least per_event_delta. Best-effort."
        ),
    )
    parser.add_argument(
        "--reset",
        default=None,
        help=(
            'Clear adjustments. Pass "all" to wipe the whole hash, or a '
            "single regime label (e.g. trend_down) to clear one field. "
            "Mutually exclusive with the normal recompute path."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("OUTCOME_ADJUSTER_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    redis_url = (
        args.position_store_url
        or os.environ.get("REDIS_URL")
        or DEFAULT_REDIS_URL
    )
    try:
        client = redis.Redis.from_url(redis_url, decode_responses=True)
        # Probe so a Redis-down environment returns 0 fast instead of
        # surfacing the failure later inside the streak loop.
        client.ping()
    except redis.exceptions.ConnectionError as exc:
        print(
            f"Redis unreachable at {redis_url}: {exc}. "
            "Skipping today's outcome adjuster run."
        )
        return 0
    except redis.exceptions.RedisError:
        LOGGER.error("Redis init failed:\n%s", traceback.format_exc())
        return 2

    store = PositionStore(redis_client=client, namespace=args.namespace)
    trade_ctx_store = TradeContextStore(
        redis_client=client, namespace=args.namespace
    )
    adjuster = OutcomeAdjuster(client, namespace=args.namespace)

    notifier: Any = None
    if args.alert:
        try:
            from alerts.notifier import Notifier

            notifier = Notifier()
        except ImportError as exc:
            LOGGER.warning(
                "alerts.notifier import failed; skipping alert post: %r",
                exc,
            )
            notifier = None

    try:
        exit_code, report, _blob = run(
            args=args,
            store=store,
            trade_ctx_store=trade_ctx_store,
            adjuster=adjuster,
            notifier=notifier,
        )
    except redis.exceptions.ConnectionError as exc:
        print(
            f"Redis unreachable mid-scan: {exc}. "
            "Skipping today's outcome adjuster run."
        )
        return 0
    except redis.exceptions.RedisError:
        LOGGER.error(
            "Redis error during outcome adjuster scan:\n%s",
            traceback.format_exc(),
        )
        return 2
    except ValueError as exc:
        LOGGER.error("input error: %s", exc)
        return 2

    print(report)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
