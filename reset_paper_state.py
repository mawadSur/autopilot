"""Clear paper-mode Redis state before starting a new paper-mode run.

Why this exists: paper-mode supervisor runs accumulate ``Position`` blobs in
Redis under ``autopilot:positions:*``. Mixing artifacts across runs makes any
Redis-based PnL audit meaningless. Run this before launching a fresh paper
session to start from a clean slate.

**Safety guard:** refuses to clear if ANY open position has
``exchange != "coinbase-paper"``. That covers both ``coinbase`` (live fills)
and any future exchange ids you don't want to nuke. Override with
``--force`` only after manually confirming.

Usage::

    ./.venv/bin/python reset_paper_state.py            # safe clear
    ./.venv/bin/python reset_paper_state.py --dry-run  # report only
    ./.venv/bin/python reset_paper_state.py --force    # skip live-position check

Exit codes:
    0  -- cleared (or dry-run report only)
    1  -- aborted because live positions detected
    2  -- aborted because Redis unreachable / other error
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from state.position_store import PositionStore, get_default_store


PAPER_EXCHANGES = {"coinbase-paper"}


def _classify(store: PositionStore) -> tuple[List[str], List[str]]:
    """Return (paper_ids, live_ids) currently in the open_set."""
    paper, live = [], []
    for pos in store.list_open():
        if pos.exchange in PAPER_EXCHANGES:
            paper.append(pos.position_id)
        else:
            live.append(pos.position_id)
    return paper, live


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be cleared, don't actually delete")
    p.add_argument("--force", action="store_true",
                   help="Clear even if live positions are detected (dangerous)")
    args = p.parse_args(argv)

    try:
        store = get_default_store()
    except Exception as exc:
        print(f"error: cannot reach Redis: {exc}", file=sys.stderr)
        return 2

    paper, live = _classify(store)
    print(f"open positions: {len(paper)} paper, {len(live)} non-paper")

    if live and not args.force:
        print("aborting: refuse to clear with live positions in open_set", file=sys.stderr)
        print("review them with `redis-cli --scan --pattern autopilot:positions:*` and", file=sys.stderr)
        print("rerun with --force if you really mean it.", file=sys.stderr)
        return 1

    # Count all keys before clear for the report.
    pattern = f"{store.namespace}:*"
    all_keys = list(store._redis.scan_iter(match=pattern))
    print(f"keys under namespace {store.namespace!r}: {len(all_keys)}")

    if args.dry_run:
        print("(dry-run) would delete all of the above.")
        return 0

    deleted = store.clear_namespace()
    print(f"deleted {deleted} keys.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
