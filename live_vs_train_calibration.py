"""Compare each symbol's *live* XGBoost precision to its 90-day test-slice projection.

Background
----------
``crypto_training.train_xgboost`` trains a CalibratedClassifierCV(XGBClassifier)
per symbol, splits the dataset chronologically into 70/15/15 train/val/test, and
saves a ``meta.json`` summarizing AUC / Brier / log-loss / accuracy on the held-
out test slice. What it does NOT save (today) is precision at the live trigger
threshold -- so when the live predictor starts firing, there is no automated way
to tell whether realized hit-rate matches the projection.

This script closes that gap. For a given completed (or in-flight) supervisor
run, it:

  1. Parses ``supervisor.log`` to extract:
       * every ``xgb predictor: ... P(long)=X`` line (raw probability per minute
         per symbol), with the threshold the predictor was using.
       * every ``tick #N | SYMBOL | action=allowed`` line (a paper fill).
     Log timestamps are local-time; the local tz is read from the system and
     converted to UTC for matching against Coinbase candles.

  2. For each fill, fetches Coinbase 1m candles via
     ``CoinbaseExchange().fetch_candles_window`` covering ``[fill_min,
     fill_min + 6]`` and computes the forward 5-bar return matching the
     training label exactly (``build_dataset.label_forward_return_binary``:
     ``(close[t+5] - close[t]) / close[t] * 10_000 > 10`` bps). Fills whose
     forward window isn't yet closed (or whose candle fetch returns short) are
     marked unresolved and excluded from the precision count.

  3. For each model, re-loads ``model.joblib`` and replays it on the held-out
     last-15% slice of ``data/crypto/datasets/{symbol}_1m.csv`` to derive the
     test-set precision at the live threshold (since ``meta.json`` only stores
     AUC / Brier / log-loss / accuracy -- no precision at chosen threshold).

  4. Prints a per-symbol calibration report + GREEN / YELLOW / RED verdict:
       * GREEN: live precision within +/- 5 points of projection.
       * YELLOW: within +/- 15 points.
       * RED: outside +/- 15 points, OR live max raw prob never crossed the
         threshold (silent symbol -- model can't fire).

Usage
-----
::

  ./.venv/bin/python live_vs_train_calibration.py \
      --run-dir runs/2026-05-23T17-51-57Z_ETH-USD,BTC-USD,SOL-USD

  ./.venv/bin/python live_vs_train_calibration.py --run-dir ... --json out.json

The ``--model-dir`` flag (default ``model_crypto/``) picks the per-symbol
sub-directory for each ETH/BTC/SOL log line. Datasets live under
``data/crypto/datasets/`` and are discovered by symbol.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

LOGGER = logging.getLogger("calibration")

# ---------------------------------------------------------------------------
# Constants tied to the trainer defaults
# (src/crypto_training/build_dataset.py).
# ---------------------------------------------------------------------------
HORIZON_BARS = 5
THRESHOLD_BPS = 10.0

# How wide a slop window to add around the fill minute when asking Coinbase
# for the forward-return candle. 1m granularity is fine but the public REST
# endpoint occasionally returns short ranges, so we ask for [t-1, t+10].
FORWARD_FETCH_PADDING_BEFORE = 1
FORWARD_FETCH_PADDING_AFTER = 10

# Symbol -> dataset filename + meta sub-dir name. Easy to extend.
SYMBOL_TO_DATASET = {
    "ETH/USD": "eth_usd_1m.csv",
    "BTC/USD": "btc_usd_1m.csv",
    "SOL/USD": "sol_usd_1m.csv",
}
# Symbol -> default model sub-directory under --model-dir. Matches the
# CRYPTO_MODEL_MAP that supervisor runs use in practice.
SYMBOL_TO_MODEL_SUBDIR = {
    "ETH/USD": "eth_usd_v2",
    "BTC/USD": "btc_usd_v1",
    "SOL/USD": "sol_usd_v1",
}

# ---------------------------------------------------------------------------
# Log line regexes
# ---------------------------------------------------------------------------
# Example: "2026-05-23 13:52:21,425 INFO predictor: xgb predictor: ETH/USD P(long)=0.105 (thr=0.30 -> neutral)"
RE_PRED = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,.]?\d*\s+"
    r"\S+\s+\S+:\s+xgb predictor:\s+(?P<symbol>\S+)\s+"
    r"P\(long\)=(?P<p>[\d.]+)\s+\(thr=(?P<thr>[\d.]+)\s+->\s+(?P<verdict>\w+)\)"
)
# Example: "2026-05-23 16:41:31,367 INFO __main__: tick #167 | ETH/USD | action=allowed | confidence=0.351 -- paper"
RE_FILL = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,.]?\d*\s+"
    r"\S+\s+\S+:\s+tick #(?P<tick>\d+)\s+\|\s+(?P<symbol>\S+)\s+\|\s+"
    r"action=allowed\s+\|\s+confidence=(?P<conf>[\d.]+)"
)


@dataclasses.dataclass
class Prediction:
    ts_utc: datetime
    symbol: str
    proba: float
    threshold: float
    fired: bool


@dataclasses.dataclass
class Fill:
    ts_utc: datetime
    symbol: str
    confidence: float
    tick: int
    forward_ret_bps: Optional[float] = None  # None until resolved
    is_win: Optional[bool] = None  # None = unresolved
    resolution_note: str = ""


# ---------------------------------------------------------------------------
# Local-time -> UTC conversion
# ---------------------------------------------------------------------------
def _local_to_utc(ts_local_str: str) -> datetime:
    """Parse a 'YYYY-MM-DD HH:MM:SS' local-time string into a UTC datetime.

    Reads the system's local tz via ``datetime.astimezone()`` after marking the
    naive datetime as local. This sidesteps having to depend on tzdata / pytz
    being present.
    """
    naive = datetime.strptime(ts_local_str, "%Y-%m-%d %H:%M:%S")
    # On POSIX the "no tzinfo + astimezone()" idiom assumes local time, which is
    # exactly what we want (log timestamps are written in local).
    local = naive.astimezone()
    return local.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Supervisor log parsing
# ---------------------------------------------------------------------------
def parse_supervisor_log(
    log_path: Path,
) -> Tuple[Dict[str, List[Prediction]], Dict[str, List[Fill]]]:
    preds: Dict[str, List[Prediction]] = {}
    fills: Dict[str, List[Fill]] = {}

    if not log_path.exists():
        raise FileNotFoundError(f"supervisor.log not found at {log_path}")

    with log_path.open() as fh:
        for line in fh:
            m = RE_PRED.match(line)
            if m:
                sym = m.group("symbol")
                preds.setdefault(sym, []).append(
                    Prediction(
                        ts_utc=_local_to_utc(m.group("ts")),
                        symbol=sym,
                        proba=float(m.group("p")),
                        threshold=float(m.group("thr")),
                        fired=m.group("verdict").lower() == "trigger",
                    )
                )
                continue
            m = RE_FILL.match(line)
            if m:
                sym = m.group("symbol")
                fills.setdefault(sym, []).append(
                    Fill(
                        ts_utc=_local_to_utc(m.group("ts")),
                        symbol=sym,
                        confidence=float(m.group("conf")),
                        tick=int(m.group("tick")),
                    )
                )
    return preds, fills


# ---------------------------------------------------------------------------
# Coinbase candle fetch + forward-return resolution
# ---------------------------------------------------------------------------
def _build_exchange():
    """Construct a sandbox-mode CoinbaseExchange. Public REST candle reads need
    no credentials, so it's safe to default sandbox=True (avoids any chance of
    accidentally placing an order)."""
    # Import inside the function so a missing ccxt only blows up at fetch time,
    # not at module import.
    from exchanges.coinbase import CoinbaseExchange

    return CoinbaseExchange(sandbox=True)


def _fetch_close_at(exchange, symbol: str, ts_utc: datetime) -> Optional[float]:
    """Fetch the 1m candle whose start is ``ts_utc`` (truncated to the minute)."""
    minute_start = int(ts_utc.timestamp() // 60 * 60)
    candles = exchange.fetch_candles_window(
        symbol,
        granularity="ONE_MINUTE",
        start_unix=minute_start,
        end_unix=minute_start + 60,
    )
    # Coinbase returns oldest-first; we want the candle whose epoch start
    # matches ``minute_start``.
    for c in candles:
        if c["timestamp"].startswith(
            datetime.fromtimestamp(minute_start, tz=timezone.utc)
            .isoformat()
            .split("+")[0]
        ):
            return float(c["close"])
    # Fallback: take the first row if the timestamp prefix didn't line up
    # (occasionally Coinbase returns 'Z' vs '+00:00' differences).
    return float(candles[0]["close"]) if candles else None


def _fetch_close_pair(
    exchange, symbol: str, fill_ts_utc: datetime
) -> Tuple[Optional[float], Optional[float], str]:
    """Fetch the close at the fill minute and at ``fill_min + HORIZON_BARS``.

    Returns ``(close_t, close_t_plus_5, note)``. ``note`` is non-empty when the
    pair could not be resolved (e.g. forward candle not yet available).
    """
    fill_minute = int(fill_ts_utc.timestamp() // 60 * 60)
    start_unix = fill_minute - 60 * FORWARD_FETCH_PADDING_BEFORE
    end_unix = fill_minute + 60 * (HORIZON_BARS + FORWARD_FETCH_PADDING_AFTER)
    try:
        candles = exchange.fetch_candles_window(
            symbol,
            granularity="ONE_MINUTE",
            start_unix=start_unix,
            end_unix=end_unix,
        )
    except Exception as exc:  # noqa: BLE001 -- never crash the whole report
        return None, None, f"fetch error: {exc}"

    by_unix: Dict[int, float] = {}
    for c in candles:
        # Parse 'YYYY-MM-DDTHH:MM:SS+00:00' to a unix int once.
        ts_iso = c["timestamp"].replace("Z", "+00:00")
        try:
            ts_unix = int(datetime.fromisoformat(ts_iso).timestamp())
        except ValueError:
            continue
        by_unix[ts_unix] = float(c["close"])

    close_t = by_unix.get(fill_minute)
    close_t_plus = by_unix.get(fill_minute + 60 * HORIZON_BARS)
    if close_t is None:
        return None, None, "fill-minute candle missing"
    if close_t_plus is None:
        return close_t, None, "forward candle not yet available"
    return close_t, close_t_plus, ""


def resolve_fills(
    fills: Dict[str, List[Fill]], *, sleep_between_calls: float = 0.15
) -> None:
    """Mutate ``fills`` in place: populate ``forward_ret_bps`` + ``is_win``.

    A "win" matches the trainer's definition exactly:
    ``(close[t+5] - close[t]) / close[t] * 10_000 > 10`` bps.
    """
    if not any(fills.values()):
        return
    exchange = _build_exchange()
    for sym, sym_fills in fills.items():
        for fill in sym_fills:
            close_t, close_t_plus, note = _fetch_close_pair(
                exchange, sym, fill.ts_utc
            )
            if close_t is None or close_t_plus is None:
                fill.is_win = None
                fill.resolution_note = note or "incomplete"
                time.sleep(sleep_between_calls)
                continue
            ret_bps = (close_t_plus - close_t) / close_t * 10_000.0
            fill.forward_ret_bps = ret_bps
            fill.is_win = ret_bps > THRESHOLD_BPS
            time.sleep(sleep_between_calls)


# ---------------------------------------------------------------------------
# Test-set precision (replay the saved model on the last 15% of the dataset)
# ---------------------------------------------------------------------------
def test_set_precision(
    *, model_path: Path, meta_path: Path, dataset_path: Path, threshold: float
) -> Tuple[Optional[float], int, int, str]:
    """Replay the saved CalibratedClassifierCV on the held-out test slice and
    return (precision, triggers, wins, note). ``note`` non-empty on error."""
    import joblib
    import numpy as np
    import pandas as pd

    if not model_path.exists():
        return None, 0, 0, f"model missing: {model_path}"
    if not meta_path.exists():
        return None, 0, 0, f"meta missing: {meta_path}"
    if not dataset_path.exists():
        return None, 0, 0, f"dataset missing: {dataset_path}"

    with meta_path.open() as fh:
        meta = json.load(fh)
    feature_cols = list(meta.get("feature_cols") or [])
    if not feature_cols:
        return None, 0, 0, "meta.json missing feature_cols"

    df = pd.read_csv(dataset_path)
    # Trainer drops rows missing label, sorts by timestamp, splits 70/15/15.
    df = df.dropna(subset=["label"]).sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    n_test = int(n * 0.15)
    test_df = df.iloc[n - n_test :].copy()

    missing_cols = [c for c in feature_cols if c not in test_df.columns]
    if missing_cols:
        return None, 0, 0, f"dataset missing feature cols: {missing_cols[:3]}..."
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["label"].astype(int).to_numpy()

    model = joblib.load(model_path)
    probs = model.predict_proba(X_test)[:, 1]

    triggers_mask = probs >= threshold
    triggers = int(triggers_mask.sum())
    if triggers == 0:
        return 0.0, 0, 0, "no triggers in test slice"
    wins = int((y_test[triggers_mask] == 1).sum())
    return wins / triggers, triggers, wins, ""


# ---------------------------------------------------------------------------
# Reporting + verdict
# ---------------------------------------------------------------------------
def _verdict_from_drift(
    *, drift_pts: Optional[float], silent: bool
) -> Tuple[str, str]:
    if silent:
        return "RED", "silent symbol -- live prob never crossed threshold"
    if drift_pts is None:
        return "YELLOW", "no resolved fills -- cannot compare"
    mag = abs(drift_pts)
    if mag <= 5.0:
        return "GREEN", "within +/- 5 points of projection"
    if mag <= 15.0:
        return "YELLOW", "drift between 5 and 15 points"
    return "RED", "drift > 15 points -- model not matching live behavior"


def _quantile(vals: List[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def build_report(
    *,
    run_dir: Path,
    model_dir: Path,
    dataset_dir: Path,
) -> Dict[str, Dict]:
    log_path = run_dir / "supervisor.log"
    preds, fills = parse_supervisor_log(log_path)

    # Collect every symbol we saw a *prediction* for (fills are a subset).
    symbols = sorted(set(preds.keys()) | set(fills.keys()))
    if not symbols:
        raise RuntimeError(f"No predictor lines found in {log_path}")

    LOGGER.info(
        "parsed %d prediction lines across %d symbols, %d fills",
        sum(len(v) for v in preds.values()),
        len(symbols),
        sum(len(v) for v in fills.values()),
    )

    resolve_fills(fills)

    report: Dict[str, Dict] = {}
    for sym in symbols:
        sym_preds = preds.get(sym, [])
        sym_fills = fills.get(sym, [])
        proba_vals = [p.proba for p in sym_preds]
        thresholds = sorted({p.threshold for p in sym_preds})
        threshold = thresholds[-1] if thresholds else float("nan")

        # Live raw-prob distribution
        dist = {
            "n": len(proba_vals),
            "mean": sum(proba_vals) / len(proba_vals) if proba_vals else float("nan"),
            "p50": _quantile(proba_vals, 0.5),
            "p90": _quantile(proba_vals, 0.9),
            "p95": _quantile(proba_vals, 0.95),
            "p99": _quantile(proba_vals, 0.99),
            "max": max(proba_vals) if proba_vals else float("nan"),
        }

        # Live triggers + resolved precision
        live_triggers = sum(1 for p in sym_preds if p.fired)
        resolved = [f for f in sym_fills if f.is_win is not None]
        unresolved = [f for f in sym_fills if f.is_win is None]
        wins = sum(1 for f in resolved if f.is_win)
        live_precision = (wins / len(resolved)) if resolved else None

        # Test-set precision via re-load + replay.
        model_subdir = SYMBOL_TO_MODEL_SUBDIR.get(sym)
        if model_subdir is None:
            test_precision, test_trig, test_wins, test_note = (
                None,
                0,
                0,
                f"no model mapping for {sym}",
            )
        else:
            test_precision, test_trig, test_wins, test_note = test_set_precision(
                model_path=model_dir / model_subdir / "model.joblib",
                meta_path=model_dir / model_subdir / "meta.json",
                dataset_path=dataset_dir / SYMBOL_TO_DATASET.get(sym, ""),
                threshold=threshold,
            )

        # Silent symbol: predictor never crossed threshold.
        silent = (
            proba_vals and max(proba_vals) < threshold and not sym_fills
        )

        if live_precision is not None and test_precision is not None:
            drift_pts = (live_precision - test_precision) * 100.0
        else:
            drift_pts = None
        verdict, verdict_reason = _verdict_from_drift(
            drift_pts=drift_pts, silent=bool(silent)
        )

        report[sym] = {
            "model_subdir": model_subdir,
            "threshold": threshold,
            "test_precision": test_precision,
            "test_triggers": test_trig,
            "test_wins": test_wins,
            "test_note": test_note,
            "live_prob_distribution": dist,
            "live_triggers": live_triggers,
            "live_fills": len(sym_fills),
            "live_resolved": len(resolved),
            "live_unresolved": len(unresolved),
            "live_wins": wins,
            "live_precision": live_precision,
            "drift_points": drift_pts,
            "silent": bool(silent),
            "verdict": verdict,
            "verdict_reason": verdict_reason,
        }
    return report


def format_report(report: Dict[str, Dict]) -> str:
    lines: List[str] = []
    for sym in sorted(report.keys()):
        r = report[sym]
        sub = r["model_subdir"] or "(no model mapped)"
        thr = r["threshold"]
        lines.append(f"{sym} {sub}:")

        # Test-set precision
        if r["test_precision"] is None:
            lines.append(
                f"  Test-set precision @ thr={thr:.2f}:       N/A ({r['test_note']})"
            )
        else:
            lines.append(
                f"  Test-set precision @ thr={thr:.2f}:       "
                f"{r['test_precision'] * 100:.1f}%  "
                f"({r['test_wins']}/{r['test_triggers']} triggers)"
            )

        # Live raw-prob distribution
        d = r["live_prob_distribution"]
        if d["n"] == 0:
            lines.append("  Live raw-prob distribution: no predictor lines parsed")
        else:
            lines.append(f"  Live raw-prob distribution (n={d['n']}):")
            lines.append(
                f"    mean={d['mean']:.3f}  p50={d['p50']:.3f}  p90={d['p90']:.3f}  "
                f"p95={d['p95']:.3f}  p99={d['p99']:.3f}  max={d['max']:.3f}"
            )

        # Live triggers
        if d["n"]:
            pct = r["live_triggers"] / d["n"] * 100
            lines.append(
                f"  Live triggers fired: {r['live_triggers']} ({pct:.1f}%)"
            )
        else:
            lines.append(f"  Live triggers fired: {r['live_triggers']}")

        # Live realized precision
        if r["live_resolved"] == 0:
            note = (
                "no fills"
                if r["live_fills"] == 0
                else f"{r['live_unresolved']} fills unresolved (forward window not closed)"
            )
            lines.append(f"  Live realized precision: N/A ({note})")
        else:
            unres_note = (
                f", {r['live_unresolved']} unresolved" if r["live_unresolved"] else ""
            )
            lines.append(
                f"  Live realized precision: {r['live_precision'] * 100:.1f}%  "
                f"({r['live_wins']} wins / {r['live_resolved']} trades, "
                f"5-bar fwd return > {THRESHOLD_BPS:.0f} bps{unres_note})"
            )

        # Drift
        if r["drift_points"] is None:
            lines.append(f"  Calibration drift: N/A ({r['verdict_reason']})")
        else:
            sign = "+" if r["drift_points"] >= 0 else ""
            direction = "live > projected" if r["drift_points"] >= 0 else "live < projected"
            lines.append(
                f"  Calibration drift: {sign}{r['drift_points']:.1f} points "
                f"({direction}). {r['verdict_reason']}"
            )
        lines.append("")

    # Verdict block
    lines.append("Verdict")
    lines.append("-------")
    for sym in sorted(report.keys()):
        r = report[sym]
        lines.append(f"  {sym}: {r['verdict']}  -- {r['verdict_reason']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to a supervisor run dir (must contain supervisor.log)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_ROOT / "model_crypto",
        help="Root of per-symbol model bundles (default model_crypto/)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "data" / "crypto" / "datasets",
        help="Root of per-symbol dataset CSVs (default data/crypto/datasets/)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to write the report as JSON for machine-readable use",
    )
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"ERROR: --run-dir {run_dir} is not a directory", file=sys.stderr)
        return 2

    report = build_report(
        run_dir=run_dir,
        model_dir=args.model_dir.resolve(),
        dataset_dir=args.dataset_dir.resolve(),
    )
    print(format_report(report))

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\nJSON written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
