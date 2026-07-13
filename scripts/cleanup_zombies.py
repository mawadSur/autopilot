"""Scan Redis for paper-tagged open positions older than ``--hours`` and purge them.

Companion to ``scripts/force_flat_paper.py``. force_flat_paper closes positions
through the exchange-aware close path (slippage + realized PnL + closed-set
membership). cleanup_zombies is the BLUNT-FORCE tool for the case where a paper
position is stuck open with no realistic chance of closing cleanly — typically
because the entry was synthesised against stale ticker data, the supervisor
crashed before fill confirmation, or the position simply outlived a supervisor
restart and now no exit policy applies. It removes the open-set membership and
deletes the position blob outright. NO realized PnL is recorded; NO closed-set
entry is created. The position vanishes from the store.

Safety
------
* Dry-run is the DEFAULT. ``--write`` must be passed explicitly to delete.
* ONLY paper-tagged positions (``position.exchange`` ending in ``-paper``) are
  touched. Live positions are NEVER deleted by this tool — they appear in the
  table marked ``SKIP-LIVE`` so the operator can confirm coverage.
* Positions younger than ``--hours`` (default 24h) are preserved.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
for _p in (SRC, REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from state.position_store import Position, PositionStore  # noqa: E402


DEFAULT_HOURS = 24
PAPER_TAG_SUFFIX = "-paper"


def _parse_opened_at(value: str) -> Optional[datetime]:
    """Parse the ISO ``opened_at_utc`` string. Returns None on malformed input."""
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _age_hours(position: Position, now: datetime) -> Optional[float]:
    opened = _parse_opened_at(position.opened_at_utc)
    if opened is None:
        return None
    delta = now - opened
    return delta.total_seconds() / 3600.0


def _is_paper(position: Position) -> bool:
    return (position.exchange or "").endswith(PAPER_TAG_SUFFIX)


def _classify(position: Position, now: datetime, hours: float) -> str:
    """Return one of ``purge`` / ``too_young`` / ``not_paper`` / ``bad_opened_at``."""
    if not _is_paper(position):
        return "not_paper"
    age = _age_hours(position, now)
    if age is None:
        return "bad_opened_at"
    # Strict "older than" semantics: a position at exactly the threshold
    # is preserved. Cleanup is destructive, so we err on the side of
    # keeping borderline positions.
    if age <= hours:
        return "too_young"
    return "purge"


def _purge_position(store: PositionStore, position_id: str) -> None:
    """Atomically drop a position from the open_set and delete its blob.

    Mirrors the open_set / position-key layout in PositionStore. We deliberately
    do NOT call record_close — this is a zombie reaper, not a fill.
    """
    namespace = store.namespace
    open_set_key = f"{namespace}:open_set"
    position_key = f"{namespace}:positions:{position_id}"
    pipe = store._redis.pipeline()  # noqa: SLF001 - intentional low-level access
    pipe.multi()
    pipe.srem(open_set_key, position_id)
    pipe.delete(position_key)
    pipe.execute()


def _print_table(rows: List[dict]) -> None:
    """Pretty-print a markdown-ish table of candidate positions."""
    if not rows:
        print("(no open positions in store)")
        return
    header = f"| {'id':<8} | {'symbol':<10} | {'age_hours':>9} | {'tag':<20} | {'action':<14} |"
    sep = f"|{'-' * 10}|{'-' * 12}|{'-' * 11}|{'-' * 22}|{'-' * 16}|"
    print(header)
    print(sep)
    for r in rows:
        print(
            f"| {r['id']:<8} | {r['symbol']:<10} | {r['age_hours']:>9} "
            f"| {r['tag']:<20} | {r['action']:<14} |"
        )


def _action_label(verdict: str, write: bool) -> str:
    if verdict == "purge":
        return "PURGE" if write else "WOULD-PURGE"
    if verdict == "too_young":
        return "SKIP-YOUNG"
    if verdict == "not_paper":
        return "SKIP-LIVE"
    return "SKIP-BAD-TS"


def cleanup_zombies(
    *,
    store: PositionStore,
    hours: float = DEFAULT_HOURS,
    write: bool = False,
    now: Optional[datetime] = None,
) -> dict:
    """Core logic — exposed for testing without re-parsing argv.

    Returns a dict ``{"purged": [...], "preserved": [...], "rows": [...]}`` where
    each ``rows`` entry is the table-row dict printed by the CLI.
    """
    now = now or datetime.now(timezone.utc)
    opens = store.list_open()

    rows: List[dict] = []
    purged: List[str] = []
    preserved: List[str] = []

    for position in opens:
        verdict = _classify(position, now, hours)
        age = _age_hours(position, now)
        rows.append(
            {
                "id": position.position_id[:8],
                "full_id": position.position_id,
                "symbol": position.symbol,
                "age_hours": f"{age:.2f}" if age is not None else "??",
                "tag": position.exchange or "?",
                "action": _action_label(verdict, write),
                "verdict": verdict,
            }
        )
        if verdict == "purge":
            if write:
                _purge_position(store, position.position_id)
                purged.append(position.position_id)
            else:
                preserved.append(position.position_id)
        else:
            preserved.append(position.position_id)

    return {"purged": purged, "preserved": preserved, "rows": rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Purge paper-tagged open positions older than --hours from Redis. "
            "Dry-run by default; pass --write to actually delete."
        )
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=DEFAULT_HOURS,
        help="Age threshold in hours (default: %(default)s).",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually delete positions. Without this flag the script is a dry-run.",
    )
    parser.add_argument(
        "--redis-url",
        default=None,
        help=(
            "Redis URL (default: $REDIS_URL or redis://localhost:6379/0). "
            "Same default chain as PositionStore."
        ),
    )
    parser.add_argument(
        "--namespace",
        default="autopilot",
        help="PositionStore namespace (default: %(default)s).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    redis_url = args.redis_url or os.environ.get("REDIS_URL") or "redis://localhost:6379/0"
    store = PositionStore(redis_url=redis_url, namespace=args.namespace)

    mode = "WRITE" if args.write else "DRY-RUN"
    print(
        f"cleanup_zombies: mode={mode} hours={args.hours} "
        f"redis_url={redis_url} ns={args.namespace}"
    )

    result = cleanup_zombies(store=store, hours=args.hours, write=args.write)
    _print_table(result["rows"])

    purge_count = len(result["purged"])
    candidate_count = sum(1 for r in result["rows"] if r["verdict"] == "purge")
    if args.write:
        print(f"\npurged={purge_count} preserved={len(result['preserved'])}")
        if purge_count:
            print("purged ids:")
            for pid in result["purged"]:
                print(f"  {pid}")
    else:
        print(
            f"\nwould-purge={candidate_count} preserved={len(result['preserved'])}"
            " (re-run with --write to actually delete)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
