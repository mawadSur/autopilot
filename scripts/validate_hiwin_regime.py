"""Validate the hi-win regime signature for ETH v2 blend09_alt.

Reproduces the 2026-05-13 sweep's test-split predictions, segments fires at
thr=0.57 by calendar day, and compares feature distributions on hi-win days
(>= 60% winrate) vs lo-win days. Output answers: is the 2-day hi-win cluster
real and sharp enough to gate on?
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
for p in (SRC, REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import joblib
import numpy as np
import pandas as pd

from crypto_training.train_xgboost import _time_based_split  # noqa: E402

DATASET = REPO / "data/crypto/datasets/eth_usd_voln.parquet"
MODEL_DIR = REPO / "model_crypto/eth_usd_voln_v2_blend09_alt"
THR = 0.57
HIWIN_FLOOR = 0.60

# Hypothesized hi-win signature from autopilot_v2_edge_diagnosed_2026_05_15.md
SIGNATURE_FEATURES = (
    "vol_ma_20",
    "bb_width_20",
    "zret_20",
    "zret_60",
    "range_pct",
    "log_ret",
    "macd_hist",
)


def main() -> int:
    df = pd.read_parquet(DATASET)
    print(f"loaded dataset: rows={len(df)} cols={len(df.columns)}")
    # Same split as the sweep
    train_df, val_df, test_df = _time_based_split(df, val_frac=0.15, test_frac=0.15)
    print(
        f"test split: rows={len(test_df)} "
        f"{test_df['timestamp'].iloc[0]} -> {test_df['timestamp'].iloc[-1]}"
    )

    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    feature_cols = meta["feature_cols"]
    print(f"feature count: {len(feature_cols)}")

    model = joblib.load(MODEL_DIR / "model.joblib")
    X_test = test_df[feature_cols].to_numpy(np.float32)
    y_test = test_df["label"].astype(int).to_numpy()
    proba = model.predict_proba(X_test)[:, 1]

    fires_mask = proba >= THR
    n_fires = int(fires_mask.sum())
    if n_fires == 0:
        print("no fires at thr={:.2f} -- abort".format(THR))
        return 1

    fired = test_df.loc[fires_mask].copy()
    fired["proba"] = proba[fires_mask]
    fired["label"] = y_test[fires_mask]
    fired["timestamp"] = pd.to_datetime(fired["timestamp"], utc=True)
    fired["date"] = fired["timestamp"].dt.date

    overall_wr = float(fired["label"].mean())
    print(
        f"fires: n={n_fires} overall_wr={overall_wr:.4f} "
        f"(memory claimed 62.5% @ n=192)"
    )

    by_day = (
        fired.groupby("date")
        .agg(n=("label", "size"), wr=("label", "mean"))
        .sort_values("date")
    )
    print("\nper-day fires:")
    print(by_day.to_string())

    hiwin_days = set(by_day.index[by_day["wr"] >= HIWIN_FLOOR])
    lowin_days = set(by_day.index[by_day["wr"] < HIWIN_FLOOR])
    print(
        f"\nhi-win days (wr >= {HIWIN_FLOOR}): {len(hiwin_days)} "
        f"({sorted(map(str, hiwin_days))})"
    )
    print(f"lo-win days (wr <  {HIWIN_FLOOR}): {len(lowin_days)}")

    hi_mask = fired["date"].isin(hiwin_days)
    lo_mask = ~hi_mask
    print(
        f"hi-win fires: n={int(hi_mask.sum())} wr={float(fired.loc[hi_mask, 'label'].mean()):.4f}"
    )
    print(
        f"lo-win fires: n={int(lo_mask.sum())} wr={float(fired.loc[lo_mask, 'label'].mean()):.4f}"
    )

    # Feature distribution comparison
    print("\nfeature distribution (hi vs lo on fired rows):")
    print(
        f"  {'feature':<22} {'hi_mean':>10} {'lo_mean':>10} "
        f"{'hi_med':>10} {'lo_med':>10} {'cohen_d':>9} {'separation':>10}"
    )
    print("  " + "-" * 86)
    rows = []
    for feat in SIGNATURE_FEATURES + tuple(
        f
        for f in feature_cols
        if f
        in (
            "bb_pctb_20",
            "ema_spread_9_21",
            "ema_spread_21_50",
            "vol_z_20",
            "price_pos_donchian20",
            "in_golden_pocket",
        )
    ):
        if feat not in fired.columns:
            continue
        a = fired.loc[hi_mask, feat].astype(float).to_numpy()
        b = fired.loc[lo_mask, feat].astype(float).to_numpy()
        if len(a) == 0 or len(b) == 0:
            continue
        mu_a, mu_b = float(np.mean(a)), float(np.mean(b))
        med_a, med_b = float(np.median(a)), float(np.median(b))
        sd_pooled = float(np.sqrt((np.var(a) + np.var(b)) / 2.0)) or 1e-12
        cohen_d = (mu_a - mu_b) / sd_pooled
        # Separation: fraction of lo-win mass outside the hi-win 10-90 IQ band
        q10, q90 = np.quantile(a, [0.10, 0.90])
        sep = float(((b < q10) | (b > q90)).mean())
        print(
            f"  {feat:<22} {mu_a:>10.4f} {mu_b:>10.4f} "
            f"{med_a:>10.4f} {med_b:>10.4f} {cohen_d:>9.3f} {sep:>10.3f}"
        )
        rows.append(
            {
                "feature": feat,
                "hi_mean": mu_a,
                "lo_mean": mu_b,
                "hi_med": med_a,
                "lo_med": med_b,
                "cohen_d": cohen_d,
                "separation": sep,
                "hi_q10": float(q10),
                "hi_q90": float(q90),
            }
        )

    out = REPO / "model_sanity/hiwin_regime_validation.json"
    out.write_text(
        json.dumps(
            {
                "thr": THR,
                "hiwin_floor": HIWIN_FLOOR,
                "n_fires": n_fires,
                "overall_winrate": overall_wr,
                "by_day": [
                    {"date": str(d), "n": int(by_day.loc[d, "n"]), "wr": float(by_day.loc[d, "wr"])}
                    for d in by_day.index
                ],
                "hiwin_days": sorted(map(str, hiwin_days)),
                "lowin_days": sorted(map(str, lowin_days)),
                "feature_stats": rows,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
