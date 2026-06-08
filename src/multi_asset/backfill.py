"""Backfill OHLCV for an entire universe into the on-disk data tree.

CLI::

    ./.venv/bin/python src/multi_asset/backfill.py \\
        --granularity 1d --days 730 --data-root data/

Writes ``data/<tree>/<SAFE_SYMBOL>/<granularity>/ohlcv.csv`` for every
instrument in the universe (default :data:`universe.DEFAULT_UNIVERSE`, or load
one with ``--universe path.json``). One bad symbol logs and is skipped — it does
not abort the run. Re-running merges with any existing CSV (de-dupe by
timestamp, last-wins) so backfills are resumable/extendable.

No orders, no live trading — pure data acquisition.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd

from multi_asset.sources import OHLCV_COLUMNS, get_source
from multi_asset.universe import DEFAULT_UNIVERSE, Instrument, Universe


def _merge_existing(path: Path, fresh: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        try:
            prior = pd.read_csv(path)
            combined = pd.concat([prior[OHLCV_COLUMNS], fresh[OHLCV_COLUMNS]], ignore_index=True)
        except Exception:
            combined = fresh
    else:
        combined = fresh
    combined = combined.drop_duplicates(subset="timestamp", keep="last")
    combined = combined.assign(_k=pd.to_datetime(combined["timestamp"], utc=True))
    combined = combined.sort_values("_k").drop(columns="_k").reset_index(drop=True)
    return combined


def backfill_instrument(
    inst: Instrument, *, data_root: Path, granularity: str, days: float
) -> int:
    """Fetch + persist one instrument. Returns the row count written."""
    kwargs = {"exchange_id": inst.exchange} if (inst.asset_class == "crypto" and inst.exchange) else {}
    src = get_source(inst.asset_class, **kwargs)
    fresh = src.fetch_ohlcv(inst.symbol, granularity=granularity, days=days)
    if fresh.empty:
        print(f"  {inst.asset_id}: no data returned (skipped)")
        return 0
    out_dir = inst.data_dir(data_root, granularity)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ohlcv.csv"
    merged = _merge_existing(out_path, fresh)
    merged.to_csv(out_path, index=False)
    print(f"  {inst.asset_id}: {len(merged)} rows -> {out_path}")
    return len(merged)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Backfill OHLCV for the multi-asset universe.")
    p.add_argument("--universe", default=None, help="path to a universe JSON (default: built-in)")
    p.add_argument("--granularity", default=None, help="override the universe granularity")
    p.add_argument("--days", type=float, default=730.0, help="trailing days to fetch")
    p.add_argument("--data-root", default="data", help="root of the data tree")
    args = p.parse_args(argv)

    universe = Universe.from_json(Path(args.universe)) if args.universe else DEFAULT_UNIVERSE
    granularity = args.granularity or universe.granularity
    data_root = Path(args.data_root)

    print(f"=== multi-asset backfill @ {granularity}, {args.days:g}d, {len(universe.instruments)} instruments ===")
    total = 0
    for inst in universe.instruments:
        try:
            total += backfill_instrument(inst, data_root=data_root, granularity=granularity, days=args.days)
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not kill the run
            print(f"  {inst.asset_id}: FETCH FAILED ({type(exc).__name__}: {str(exc)[:80]})")
    print(f"done: {total} total rows across the universe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
