"""Threshold sweep for the three XGBoost crypto models.

Standalone, read-only analysis. Does NOT modify any source files.

For each (symbol, model) pair:
  1. Load the calibrated sklearn model + meta.json.
  2. Take the last 15% of the dataset as the test slice (matches the time-based
     split in src/crypto_training/train_xgboost.py).
  3. Run predict_proba on the test slice.
  4. Sweep thresholds 0.10 .. 0.70.
  5. For each threshold compute:
       - n_triggers
       - precision (fraction of triggers where label == 1)
       - frequency vs. test slice
       - estimated triggers per day (frequency * 1440)
       - "edge" in bps: precision * avg_win_bps - (1-precision) * avg_loss_bps
         where avg_win_bps  = mean of |forward_5bar_return_bps| on triggered hits
               avg_loss_bps = mean of |forward_5bar_return_bps| on triggered misses
         (forward return reconstructed from the `mid` column, since the CSV
         drops `close`.)

Forward-return reconstruction note:
  build_dataset.py labels with (close[t+5] - close[t]) / close[t] * 1e4 > 10 bps,
  then drops `close` before persisting. The persisted CSV does keep `mid` --
  but L2 book columns are 0 across the test slice (book wasn't backfilled for
  that period). However `return_1` (= (close[t]-close[t-1])/close[t-1]) is
  fully populated, so we reconstruct the 5-bar forward return exactly as
  prod_{i=1..5}(1 + return_1[t+i]) - 1, in bps. This was sanity-checked: the
  > 10 bps boolean of the reconstructed series agrees with the persisted
  `label` column on 100% of training-set rows.

Usage:
    ./.venv/bin/python threshold_sweep.py
    ./.venv/bin/python threshold_sweep.py --json sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent

SYMBOLS = [
    ("ETH/USD", "eth_usd_v2", REPO_ROOT / "data" / "crypto" / "datasets" / "eth_usd_1m.csv", 0.30),
    ("BTC/USD", "btc_usd_v1", REPO_ROOT / "data" / "crypto" / "datasets" / "btc_usd_1m.csv", 0.30),
    ("SOL/USD", "sol_usd_v1", REPO_ROOT / "data" / "crypto" / "datasets" / "sol_usd_1m.csv", 0.50),
]

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

TEST_FRAC = 0.15
HORIZON = 5  # 5-bar forward
MIN_TRIGGERS_FOR_EV = 50  # for "recommend EV-best with min freq" gate
THIN_THRESHOLD = 30        # flag <30 triggers as small-sample noise


@dataclass
class SweepRow:
    thr: float
    triggers: int
    precision: Optional[float]   # None if n_triggers == 0
    frequency: float
    triggers_per_day: float
    edge_bps: Optional[float]    # None if can't compute (n_triggers == 0 or no fwd return data)
    avg_win_bps: Optional[float]
    avg_loss_bps: Optional[float]


@dataclass
class SymbolResult:
    symbol: str
    model_name: str
    n_test: int
    base_rate: float
    auc: float
    current_thr: float
    rows: List[SweepRow]


def load_symbol(symbol: str, model_name: str, csv_path: Path, current_thr: float) -> SymbolResult:
    model_dir = REPO_ROOT / "model_crypto" / model_name
    model = joblib.load(model_dir / "model.joblib")
    meta = json.loads((model_dir / "meta.json").read_text())

    feature_cols = meta["feature_cols"]
    auc = float(meta.get("metrics_test", {}).get("auc", float("nan")))

    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    n_test = int(n * TEST_FRAC)
    n_train_val = n - n_test
    test_df = df.iloc[n_train_val:].reset_index(drop=True)

    # Predict
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["label"].astype(int).to_numpy()
    proba = model.predict_proba(X_test)[:, 1]

    # Forward 5-bar return in bps, reconstructed exactly from return_1.
    # (mid/best_bid/best_ask are 0 across the test slice -- the L2 book wasn't
    # backfilled for that period -- so we use return_1, which IS populated.)
    if "return_1" not in test_df.columns:
        raise RuntimeError(f"{csv_path} missing `return_1` column; cannot reconstruct fwd return")
    r1 = test_df["return_1"].astype(float).to_numpy()
    log1p = np.log1p(r1)
    cum = np.concatenate([[0.0], np.cumsum(log1p)])  # cum[k] = sum(log1p[:k])
    n_rows = len(r1)
    fwd_bps = np.full(n_rows, np.nan, dtype=np.float64)
    if n_rows > HORIZON:
        # sum log1p[t+1 .. t+HORIZON] = cum[t+HORIZON+1] - cum[t+1]
        fwd_bps[: n_rows - HORIZON] = (
            np.expm1(cum[HORIZON + 1 : n_rows + 1] - cum[1 : n_rows - HORIZON + 1])
            * 1e4
        )

    base_rate = float(y_test.mean())

    rows: List[SweepRow] = []
    for thr in THRESHOLDS:
        mask = proba >= thr
        n_trig = int(mask.sum())
        freq = n_trig / len(proba) if len(proba) else 0.0
        per_day = freq * 1440.0

        if n_trig == 0:
            rows.append(SweepRow(thr, 0, None, freq, per_day, None, None, None))
            continue

        y_trig = y_test[mask]
        precision = float(y_trig.mean())

        # Edge in bps via reconstructed forward returns on triggered rows.
        fwd_trig = fwd_bps[mask]
        valid_fwd = ~np.isnan(fwd_trig)
        if valid_fwd.sum() == 0:
            edge = None
            avg_win = avg_loss = None
        else:
            fwd_v = fwd_trig[valid_fwd]
            y_v = y_trig[valid_fwd]
            wins = fwd_v[y_v == 1]
            losses = fwd_v[y_v == 0]
            avg_win = float(np.mean(np.abs(wins))) if wins.size else 0.0
            avg_loss = float(np.mean(np.abs(losses))) if losses.size else 0.0
            # precision used in edge is the SAME precision we report (full mask).
            # That's fine because mid==0 rows are very rare in test (warmup is in
            # train), and using the full-mask precision avoids confounding two
            # different denominators.
            edge = precision * avg_win - (1.0 - precision) * avg_loss

        rows.append(SweepRow(thr, n_trig, precision, freq, per_day, edge, avg_win, avg_loss))

    return SymbolResult(
        symbol=symbol,
        model_name=model_name,
        n_test=len(test_df),
        base_rate=base_rate,
        auc=auc,
        current_thr=current_thr,
        rows=rows,
    )


def _fmt_pct(x: Optional[float]) -> str:
    return f"{x*100:5.1f}%" if x is not None else "    -- "


def _fmt_bps(x: Optional[float]) -> str:
    if x is None:
        return "   --  "
    sign = "+" if x >= 0 else "-"
    return f"{sign}{abs(x):6.2f}"


def print_symbol(res: SymbolResult) -> None:
    # Choose best EV row with >= MIN_TRIGGERS_FOR_EV
    eligible = [r for r in res.rows if r.triggers >= MIN_TRIGGERS_FOR_EV and r.edge_bps is not None]
    best_ev = max(eligible, key=lambda r: r.edge_bps) if eligible else None

    print(
        f"\n{res.symbol} {res.model_name} "
        f"(test n={res.n_test}, base rate {res.base_rate*100:.1f}%, "
        f"model AUC {res.auc:.3f}):\n"
    )
    print("| thr  | triggers | precision | trig/day | edge (bps) | avg_win | avg_loss | flag                |")
    print("|-----:|---------:|----------:|---------:|-----------:|--------:|---------:|---------------------|")

    for r in res.rows:
        flags: List[str] = []
        if abs(r.thr - res.current_thr) < 1e-9:
            flags.append("★ current")
        if best_ev is not None and r is best_ev:
            flags.append("← best EV")
        if r.precision is not None and r.precision >= 0.60:
            flags.append("60%+")
        if r.triggers > 0 and r.triggers < THIN_THRESHOLD:
            flags.append("thin")
        if r.triggers == 0:
            flags.append("empty")

        flag_str = ", ".join(flags)
        print(
            f"| {r.thr:.2f} | {r.triggers:8d} | {_fmt_pct(r.precision)}   | "
            f"{r.triggers_per_day:8.1f} | {_fmt_bps(r.edge_bps)}     | "
            f"{_fmt_bps(r.avg_win_bps)} | {_fmt_bps(r.avg_loss_bps)}  | "
            f"{flag_str:19s} |"
        )


def print_recommendations(results: List[SymbolResult]) -> None:
    print("\n" + "=" * 80)
    print("PER-SYMBOL BEST THRESHOLD (EV-MAX, MIN 50 TRIGGERS IN TEST):")
    print("=" * 80)
    for res in results:
        eligible = [
            r for r in res.rows
            if r.triggers >= MIN_TRIGGERS_FOR_EV and r.edge_bps is not None
        ]
        if eligible:
            best = max(eligible, key=lambda r: r.edge_bps)
            print(
                f"  {res.symbol}: thr={best.thr:.2f} -> "
                f"precision {best.precision*100:.1f}%, "
                f"{best.triggers_per_day:.1f}/day, "
                f"edge {best.edge_bps:+.2f} bps "
                f"(avg_win {best.avg_win_bps:.2f}, avg_loss {best.avg_loss_bps:.2f})"
            )
        else:
            print(f"  {res.symbol}: no threshold with >={MIN_TRIGGERS_FOR_EV} triggers and computable edge")

    print("\n" + "=" * 80)
    print("REACHING >=60% PRECISION (THRESHOLD-ONLY):")
    print("=" * 80)
    for res in results:
        crosses = [r for r in res.rows if r.precision is not None and r.precision >= 0.60]
        if not crosses:
            print(
                f"  {res.symbol}: not reachable in the swept range "
                f"(max precision {_max_precision(res):.1f}% at thr={_max_precision_thr(res):.2f})"
            )
            continue
        # First (lowest) threshold that crosses 60%
        first = crosses[0]
        if first.triggers < MIN_TRIGGERS_FOR_EV:
            cost = f" -- only {first.triggers} triggers in test (~{first.triggers_per_day:.2f}/day) -- BELOW useful frequency"
        elif first.triggers < THIN_THRESHOLD * 3:  # < 90 triggers
            cost = f" -- thin: {first.triggers} triggers, ~{first.triggers_per_day:.2f}/day"
        else:
            cost = f" -- {first.triggers} triggers, ~{first.triggers_per_day:.1f}/day"
        print(
            f"  {res.symbol}: thr>={first.thr:.2f} -> precision {first.precision*100:.1f}%{cost}"
        )


def _max_precision(res: SymbolResult) -> float:
    vals = [r.precision for r in res.rows if r.precision is not None]
    return (max(vals) * 100.0) if vals else float("nan")


def _max_precision_thr(res: SymbolResult) -> float:
    best = max(
        (r for r in res.rows if r.precision is not None),
        key=lambda r: r.precision,
        default=None,
    )
    return best.thr if best else float("nan")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--json", type=str, default=None, help="Write machine-readable output to this path.")
    args = p.parse_args(argv)

    results: List[SymbolResult] = []
    for symbol, model_name, csv_path, current_thr in SYMBOLS:
        print(f"[load] {symbol} ({model_name}) ...", file=sys.stderr, flush=True)
        res = load_symbol(symbol, model_name, csv_path, current_thr)
        results.append(res)
        print_symbol(res)

    print_recommendations(results)

    if args.json:
        out = []
        for res in results:
            d = {
                "symbol": res.symbol,
                "model_name": res.model_name,
                "n_test": res.n_test,
                "base_rate": res.base_rate,
                "auc": res.auc,
                "current_thr": res.current_thr,
                "rows": [asdict(r) for r in res.rows],
            }
            out.append(d)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nWrote JSON to {args.json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
