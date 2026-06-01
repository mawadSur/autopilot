"""Dataset assembly utility for the calibration baseline.

Walks ``trade_execution_<id>.json`` logs, extracts ``features_window`` +
``market_outcome`` pairs from settled trades, and emits a tabular dataset for
training the XGBoost calibration baseline. Output column order mirrors
:data:`ALL_FEATURE_COLUMNS` so the trained model lines up with the runtime
extractor in :mod:`calibration_agent.ml_service`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# Mirror the sys.path shim used by main.py / orchestrator.py so this CLI runs
# without the caller setting PYTHONPATH. ``calibration_agent.__init__`` imports
# ``analyzer`` which uses flat ``from models import Market`` against ``src/``.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd

from calibration_agent.ml_service import ALL_FEATURE_COLUMNS
from storage import SQLITE_PATH_ENV_VAR, SQLiteStore, is_sqlite_enabled

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

SKIP_NOT_SETTLED = "not_settled"
SKIP_NO_FEATURES_WINDOW = "no_features_window"
SKIP_MISSING_FEATURE_COLUMNS = "missing_feature_columns"
SKIP_NO_MARKET_OUTCOME = "no_market_outcome"
SKIP_INVALID_FEATURES = "invalid_features"
SKIP_BACKFILL_EXCLUDED = "backfill_excluded"

# Sources that capture features at decision time (orchestrator) or close to it
# (shadow_capture polling active markets). Backfilled rows are excluded by
# default because their ``features_window`` reflects post-resolution market
# state — see ``backfill_from_polymarket.BACKFILL_NOTES`` for the full caveat.
FULL_FIDELITY_SOURCES: Tuple[str, ...] = ("orchestrator", "shadow")
DEFAULT_SOURCE = "orchestrator"

OUTPUT_COLUMNS: Tuple[str, ...] = (
    "trade_id",
    "captured_at_utc",
    "settled_at",
    "source",
    *ALL_FEATURE_COLUMNS,
    "market_outcome",
    "final_outcome",
)


def _load_trade_log(path: Path) -> Optional[Dict[str, Any]]:
    """Tolerantly load a trade log; returns None (and warns) on parse errors."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping unreadable trade log %s: %s", path, exc)
        return None

    if not isinstance(payload, dict):
        logger.warning(
            "Skipping non-object trade log %s (got %s)", path, type(payload).__name__
        )
        return None
    return payload


def _qualifies_for_training(
    payload: Dict[str, Any], include_unlabeled: bool
) -> Tuple[bool, Optional[str]]:
    """Return ``(qualifies, skip_reason)`` for a single trade payload."""

    status = str(payload.get("status") or "").strip().lower()
    if status != "settled" and not include_unlabeled:
        return False, SKIP_NOT_SETTLED

    features_window = payload.get("features_window")
    if features_window is None:
        return False, SKIP_NO_FEATURES_WINDOW
    if not isinstance(features_window, dict):
        return False, SKIP_INVALID_FEATURES

    missing = [col for col in ALL_FEATURE_COLUMNS if col not in features_window]
    if missing:
        return False, SKIP_MISSING_FEATURE_COLUMNS

    try:
        for col in ALL_FEATURE_COLUMNS:
            float(features_window[col])
    except (TypeError, ValueError):
        return False, SKIP_INVALID_FEATURES

    if payload.get("market_outcome") is None:
        return False, SKIP_NO_MARKET_OUTCOME

    return True, None


def _coerce_optional_bool_int(value: Any) -> Optional[int]:
    """Convert a tri-state bool/None into 0/1/None for tabular storage."""
    return None if value is None else int(bool(value))


def _resolve_source(payload: Dict[str, Any]) -> Tuple[str, bool]:
    """Return ``(source, was_missing)`` for a trade payload.

    Pre-schema artifacts (no ``source`` field) are treated as
    ``"orchestrator"`` for back-compat; the second tuple element flags those
    so the caller can emit a single aggregated WARNING per assemble call.
    """

    raw = payload.get("source")
    if raw is None:
        return DEFAULT_SOURCE, True
    return str(raw), False


def _build_row(payload: Dict[str, Any], *, source_file: Path, source: str) -> Dict[str, Any]:
    """Materialize a single row from a qualifying trade payload."""

    features_window: Dict[str, Any] = payload["features_window"]
    trade_id = (
        payload.get("trade_id")
        or payload.get("event_id")
        or source_file.stem.replace("trade_execution_", "", 1)
    )
    row: Dict[str, Any] = {
        "trade_id": str(trade_id),
        "captured_at_utc": features_window.get("captured_at_utc"),
        "settled_at": payload.get("settled_at"),
        "source": source,
    }
    for col in ALL_FEATURE_COLUMNS:
        row[col] = float(features_window[col])
    row["market_outcome"] = _coerce_optional_bool_int(payload.get("market_outcome"))
    row["final_outcome"] = _coerce_optional_bool_int(payload.get("final_outcome"))
    return row


def _empty_dataframe() -> pd.DataFrame:
    """Construct an empty DataFrame with the canonical column dtypes."""

    return pd.DataFrame({col: pd.Series(dtype="object") for col in OUTPUT_COLUMNS})


def _write_output(df: pd.DataFrame, output_path: Path) -> None:
    """Write the assembled DataFrame to Parquet or CSV based on extension."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(output_path, index=False)
        return
    # Default to Parquet for unknown / missing extension.
    if suffix not in {".parquet", ""}:
        logger.warning(
            "Unknown output extension %r; writing Parquet instead.", suffix
        )
    df.to_parquet(output_path, index=False)


def _iter_payloads_from_sqlite(store: SQLiteStore) -> Iterable[Tuple[Dict[str, Any], Path]]:
    """Yield ``(payload, synthetic_source_path)`` tuples mirroring the JSON walk.

    SQLite rows are reassembled into the same ``event_payload`` shape that
    ``_qualifies_for_training`` / ``_build_row`` already understand so the
    rest of the pipeline stays unchanged. Rows are emitted ordered by
    ``trade_id`` to keep dataset assembly deterministic.
    """

    # ``limit`` is required by the public API; pass a generous ceiling so a
    # full mirror walk doesn't truncate. Tests / CI are well below this cap.
    rows = store.list_trades(limit=10_000_000)
    rows.sort(key=lambda r: str(r.get("trade_id") or ""))
    for row in rows:
        payload: Dict[str, Any] = {
            "trade_id": row.get("trade_id"),
            "event_id": row.get("event_id"),
            "status": row.get("status"),
            "source": row.get("source"),
            "created_at_utc": row.get("created_at_utc"),
            "settled_at": row.get("settled_at"),
            "final_outcome": row.get("final_outcome"),
            "market_outcome": row.get("market_outcome"),
            "entry_price": row.get("entry_price"),
            "exit_price": row.get("exit_price"),
            "position_size_usd": row.get("position_size_usd"),
            "realized_pnl_usd": row.get("realized_pnl_usd"),
            "max_loss_usd": row.get("max_loss_usd"),
            "features_window": row.get("features_window"),
            "research": row.get("research"),
            "calibration": row.get("calibration"),
            "risk": row.get("risk"),
            "notes": row.get("notes"),
        }
        source_file = Path(row.get("source_file") or f"sqlite://{row.get('trade_id') or 'unknown'}")
        yield payload, source_file


def assemble_dataset(
    trade_store_dir: PathLike,
    *,
    output_path: Optional[PathLike] = None,
    include_unlabeled: bool = False,
    include_backfill: bool = False,
    source: str = "auto",
    sqlite_store: Optional[SQLiteStore] = None,
) -> pd.DataFrame:
    """Walk ``trade_store_dir`` (or SQLite) and emit a (features, label) DataFrame.

    Source resolution (``source`` arg):
        - ``"files"`` (legacy) — walk ``trade_store_dir`` for JSON files.
        - ``"sqlite"`` — pull rows from the SQLite mirror; trade_store_dir is
          ignored. Requires an explicit ``sqlite_store`` or
          ``AUTOPILOT_SQLITE_PATH`` to be set.
        - ``"auto"`` (default) — use SQLite when ``AUTOPILOT_SQLITE_PATH`` is
          set, otherwise fall back to the filesystem walk.

    Files are processed in sorted-filename order (or sorted-trade_id order
    for SQLite). Skip-reason counts are printed to stderr. When
    ``output_path`` is provided the DataFrame is also written to disk
    (Parquet for ``.parquet``/no extension, CSV for ``.csv``).

    Source filtering: by default only rows with ``source`` in
    :data:`FULL_FIDELITY_SOURCES` (``orchestrator`` / ``shadow``) are kept;
    rows tagged ``source="backfill"`` are skipped and bucketed under the
    :data:`SKIP_BACKFILL_EXCLUDED` reason. Pass ``include_backfill=True`` to
    keep them (smoke-testing the training pipeline only — see
    ``backfill_from_polymarket.BACKFILL_NOTES``). Trade logs without an
    explicit ``source`` field are treated as ``"orchestrator"`` for
    back-compat with pre-schema artifacts.
    """

    rows: List[Dict[str, Any]] = []
    skip_reasons: Counter[str] = Counter()
    total_files = 0
    missing_source_count = 0

    resolved_source = (source or "auto").strip().lower()
    if resolved_source not in {"auto", "files", "sqlite"}:
        raise ValueError(f"Unknown source mode {source!r}; expected one of auto|files|sqlite.")
    if resolved_source == "auto":
        resolved_source = "sqlite" if (sqlite_store is not None or is_sqlite_enabled()) else "files"

    use_sqlite = resolved_source == "sqlite"
    owns_store = False

    if use_sqlite:
        store = sqlite_store
        if store is None:
            db_path_raw = (os.environ.get(SQLITE_PATH_ENV_VAR) or "").strip()
            if not db_path_raw:
                raise RuntimeError(
                    f"source='sqlite' requested but {SQLITE_PATH_ENV_VAR} is unset and no store passed."
                )
            store = SQLiteStore(db_path_raw)
            owns_store = True
        try:
            payload_iter = list(_iter_payloads_from_sqlite(store))
        finally:
            if owns_store:
                store.close()
        for payload, source_file in payload_iter:
            total_files += 1
            payload_source, was_missing = _resolve_source(payload)
            if was_missing:
                missing_source_count += 1
            if not include_backfill and payload_source == "backfill":
                skip_reasons[SKIP_BACKFILL_EXCLUDED] += 1
                continue
            qualifies, reason = _qualifies_for_training(payload, include_unlabeled)
            if not qualifies:
                assert reason is not None  # for type-checkers
                skip_reasons[reason] += 1
                continue
            rows.append(_build_row(payload, source_file=source_file, source=payload_source))
    else:
        store_dir = Path(trade_store_dir)
        if store_dir.is_dir():
            candidate_files = sorted(store_dir.glob("trade_execution_*.json"))
        else:
            logger.warning(
                "Trade store dir %s does not exist or is not a directory; "
                "returning empty dataset.",
                store_dir,
            )
            candidate_files = []

        for file_path in candidate_files:
            total_files += 1
            payload = _load_trade_log(file_path)
            if payload is None:
                skip_reasons[SKIP_INVALID_FEATURES] += 1
                continue
            payload_source, was_missing = _resolve_source(payload)
            if was_missing:
                missing_source_count += 1
            if not include_backfill and payload_source == "backfill":
                skip_reasons[SKIP_BACKFILL_EXCLUDED] += 1
                continue
            qualifies, reason = _qualifies_for_training(payload, include_unlabeled)
            if not qualifies:
                assert reason is not None  # for type-checkers
                skip_reasons[reason] += 1
                continue
            rows.append(_build_row(payload, source_file=file_path, source=payload_source))

    if missing_source_count > 0:
        logger.warning(
            "%d trade log(s) without explicit 'source' — treating as 'orchestrator'",
            missing_source_count,
        )

    df = pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS)) if rows else _empty_dataframe()

    if output_path is not None:
        _write_output(df, Path(output_path))

    skipped_total = sum(skip_reasons.values())
    if skip_reasons:
        skip_breakdown = ", ".join(
            f"{reason}={count}" for reason, count in sorted(skip_reasons.items())
        )
    else:
        skip_breakdown = "none"
    source_label = "sqlite" if use_sqlite else "files"
    print(
        f"Assembled {len(df)} rows from {total_files} {source_label} "
        f"(skipped {skipped_total}: {skip_breakdown})",
        file=sys.stderr,
    )
    return df


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble a (features, label) dataset from settled trade execution "
            "logs for the XGBoost calibration baseline."
        )
    )
    parser.add_argument(
        "trade_store_dir",
        type=Path,
        help="Directory containing trade_execution_<id>.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path. Use .parquet (default) or .csv. Omit to skip writing.",
    )
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Include open / unsettled trades for diagnostic dumps.",
    )
    parser.add_argument(
        "--include-backfill",
        action="store_true",
        help=(
            "Include rows with source='backfill' (degraded-fidelity, "
            "post-resolution feature snapshots). Excluded by default — useful "
            "only for smoke-testing the training pipeline."
        ),
    )
    parser.add_argument(
        "--source",
        choices=("auto", "files", "sqlite"),
        default="auto",
        help=(
            "Where to read trade payloads from. 'auto' (default) uses SQLite when "
            f"${SQLITE_PATH_ENV_VAR} is set, else falls back to JSON files."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    df = assemble_dataset(
        args.trade_store_dir,
        output_path=args.output,
        include_unlabeled=args.include_unlabeled,
        include_backfill=args.include_backfill,
        source=args.source,
    )
    if not df.empty:
        print(df.describe(include="all"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
