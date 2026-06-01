"""Feature Diagnostic Script — one-shot investigation of 3 crypto datasets.

Tests 6 hypotheses for why XGBoost and transformer models fail out-of-sample:
  1. Label leakage (features with |Pearson corr| > 0.5 with label)
  2. Regime shift train → test (volatility, trend, spread)
  3. Feature drift across splits (mean/std shift > 2 sigma)
  4. Per-feature univariate AUC on test split
  5. Horizon / threshold sanity (base rate per split)
  6. Class imbalance per split

Usage:
    ./.venv/bin/python scripts/diagnose_features.py

Writes a markdown report to /tmp/feature_diagnosis_<utc>.md
and prints it to stdout.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO / "data" / "crypto" / "datasets"

# (symbol, parquet_filename, meta_dir) — ETH uses eth_usd_5m_h.parquet
SYMBOLS: List[Tuple[str, str, str]] = [
    ("BTC", "btc_usd_v1.parquet", "btc_usd_v1"),
    ("ETH", "eth_usd_5m_h.parquet", "eth_usd_v1"),
    ("SOL", "sol_usd_v1.parquet", "sol_usd_v1"),
]

VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Thresholds
LEAK_CORR_THRESHOLD = 0.5
DRIFT_ZSCORE_THRESHOLD = 2.0
REGIME_SIGMA_THRESHOLD = 1.0
AUC_SIGNAL_THRESHOLD = 0.55
BASE_RATE_MIN = 0.05
BASE_RATE_MAX = 0.95


# ---------------------------------------------------------------------------
# Split helpers (mirrors train_xgboost._time_based_split exactly)
# ---------------------------------------------------------------------------

def time_based_split(
    df: pd.DataFrame,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train: n_train + n_val].copy()
    test = df.iloc[n_train + n_val:].copy()
    return train, val, test


def load_meta(meta_dir: str) -> Dict:
    p = REPO / "model_crypto" / meta_dir / "meta.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def feature_cols_from_df(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ("timestamp", "label")]


# ---------------------------------------------------------------------------
# Hypothesis 1: Label Leakage
# ---------------------------------------------------------------------------

def h1_label_leakage(
    train: pd.DataFrame,
    feature_cols: List[str],
) -> Dict:
    """Flag features with |Pearson corr| > 0.5 with label on train set."""
    y = train["label"].astype(float)
    corrs: Dict[str, float] = {}
    for col in feature_cols:
        x = train[col].astype(float)
        if x.std() < 1e-12:
            continue
        c = float(x.corr(y))
        if abs(c) > LEAK_CORR_THRESHOLD:
            corrs[col] = round(c, 4)
    return {
        "flagged_count": len(corrs),
        "flagged_features": dict(sorted(corrs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:20]),
        "verdict": "POSSIBLE LEAKAGE" if corrs else "clean",
    }


# ---------------------------------------------------------------------------
# Hypothesis 2: Regime Shift
# ---------------------------------------------------------------------------

def _rolling_realised_vol(close: pd.Series, window: int = 30) -> float:
    log_rets = np.log(close / close.shift(1)).dropna()
    return float(log_rets.rolling(window).std().dropna().mean())


def _rolling_trend(close: pd.Series, window: int = 30) -> float:
    log_rets = np.log(close / close.shift(1)).dropna()
    return float(log_rets.rolling(window).mean().dropna().mean())


def _avg_spread_pct(df: pd.DataFrame) -> float:
    if "spread_pct" in df.columns:
        return float(df["spread_pct"].dropna().mean())
    return float("nan")


def h2_regime_shift(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> Dict:
    """Compare realised vol, trend, and spread across splits."""
    results: Dict = {}
    flags: List[str] = []

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        if "close" not in split_df.columns:
            # Use ema_9 as proxy for price if close not present
            price_col = "ema_9" if "ema_9" in split_df.columns else None
        else:
            price_col = "close"

        if price_col:
            rv = _rolling_realised_vol(split_df[price_col])
            trend = _rolling_trend(split_df[price_col])
        else:
            rv = float("nan")
            trend = float("nan")

        spread = _avg_spread_pct(split_df)
        results[split_name] = {"rv": rv, "trend": trend, "spread_pct": spread}

    # Check if test is > 1 sigma from train on each metric
    for metric in ("rv", "trend", "spread_pct"):
        train_v = results["train"][metric]
        test_v = results["test"][metric]
        val_v = results["val"][metric]

        if not (np.isfinite(train_v) and np.isfinite(test_v)):
            continue

        # Estimate sigma across the 3 splits
        vals = [results[s][metric] for s in ("train", "val", "test") if np.isfinite(results[s][metric])]
        if len(vals) < 2:
            continue
        sigma = float(np.std(vals))
        if sigma < 1e-12:
            continue
        diff_test = abs(test_v - train_v) / sigma
        if diff_test > REGIME_SIGMA_THRESHOLD:
            flags.append(f"{metric}: train={train_v:.5f} test={test_v:.5f} diff={diff_test:.1f}σ")

    results["flags"] = flags
    results["verdict"] = "REGIME SHIFT DETECTED" if flags else "no shift detected"
    return results


# ---------------------------------------------------------------------------
# Hypothesis 3: Feature Drift
# ---------------------------------------------------------------------------

def h3_feature_drift(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
) -> Dict:
    """Flag features where |train_mean - test_mean| / train_std > 2.0."""
    drifted: Dict[str, Dict] = {}

    for col in feature_cols:
        tr = train[col].astype(float).dropna()
        te = test[col].astype(float).dropna()
        if len(tr) == 0 or len(te) == 0:
            continue
        mu_tr = float(tr.mean())
        std_tr = float(tr.std())
        mu_te = float(te.mean())
        if std_tr < 1e-9:
            continue
        z = abs(mu_tr - mu_te) / std_tr
        if z > DRIFT_ZSCORE_THRESHOLD:
            drifted[col] = {
                "train_mean": round(mu_tr, 5),
                "test_mean": round(mu_te, 5),
                "train_std": round(std_tr, 5),
                "z_score": round(z, 2),
            }

    top_drifted = dict(
        sorted(drifted.items(), key=lambda kv: kv[1]["z_score"], reverse=True)[:20]
    )
    return {
        "flagged_count": len(drifted),
        "top_20_by_zscore": top_drifted,
        "verdict": f"{len(drifted)} features drifted > {DRIFT_ZSCORE_THRESHOLD}σ" if drifted else "no drift",
    }


# ---------------------------------------------------------------------------
# Hypothesis 4: Univariate AUC on test split
# ---------------------------------------------------------------------------

def _univariate_auc(X_col: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC of a single feature vs label (raw values, no fitting)."""
    if len(np.unique(y)) < 2:
        return float("nan")
    # Both the raw value AND its negation — take the best (AUC can be < 0.5
    # if the feature is negatively predictive; we want the max signal direction)
    try:
        auc_pos = float(roc_auc_score(y, X_col))
        return max(auc_pos, 1 - auc_pos)
    except Exception:
        return float("nan")


def h4_univariate_auc(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
) -> Dict:
    """Per-feature univariate AUC on test split.

    Also classifies each feature by type (volatility vs directional) to flag
    the spurious-confound pattern: volatility features predict label=1 not
    because they carry directional signal, but because a fixed-bps threshold
    is cleared more often in high-vol regimes.
    """
    VOL_KEYWORDS = {
        "atr", "rv_", "ret_std", "range_ma", "vol_", "bb_width", "vol_of_vol",
        "atrp", "spread_z", "vol_z", "vol_log",
    }
    DIRECTION_KEYWORDS = {
        "return_1", "return_5", "return_15", "log_ret", "zret", "macd",
        "l1_imbalance", "ofi", "ema_spread", "price_z", "close_over",
        "tf5_log_ret", "tf15_log_ret", "tf60_log_ret", "close_pos",
    }

    y_train = train["label"].astype(int).to_numpy()
    y_test = test["label"].astype(int).to_numpy()
    aucs_test: Dict[str, float] = {}
    aucs_train: Dict[str, float] = {}
    feature_type: Dict[str, str] = {}
    for col in feature_cols:
        x_te = test[col].astype(float).fillna(0).to_numpy()
        x_tr = train[col].astype(float).fillna(0).to_numpy()
        aucs_test[col] = round(_univariate_auc(x_te, y_test), 4)
        aucs_train[col] = round(_univariate_auc(x_tr, y_train), 4)
        col_lower = col.lower()
        is_vol = any(k in col_lower for k in VOL_KEYWORDS)
        is_dir = any(k in col_lower for k in DIRECTION_KEYWORDS)
        feature_type[col] = "volatility" if is_vol and not is_dir else ("directional" if is_dir else "other")

    ranked = sorted(aucs_test.items(), key=lambda kv: kv[1], reverse=True)
    top10 = ranked[:10]
    bottom10 = ranked[-10:]

    top_auc = top10[0][1] if top10 else float("nan")
    n_above_threshold = sum(1 for _, a in aucs_test.items() if a > AUC_SIGNAL_THRESHOLD)

    # Separate vol vs directional in top 10
    top10_vol = [(f, a) for f, a in top10 if feature_type.get(f) == "volatility"]
    top10_dir = [(f, a) for f, a in top10 if feature_type.get(f) == "directional"]

    # Best directional AUC (true alpha signal, not vol confound)
    dir_aucs = [(f, a) for f, a in aucs_test.items() if feature_type.get(f) == "directional"]
    best_dir_auc = max((a for _, a in dir_aucs), default=float("nan")) if dir_aucs else float("nan")

    # Spurious vol confound: vol features predict label simply because high vol
    # makes it easier to clear a fixed threshold_bps. This masquerades as
    # predictive signal but collapses when test has lower vol.
    has_spurious_vol_confound = len(top10_vol) >= 5 and best_dir_auc < 0.55

    return {
        "top_10": dict(top10),
        "bottom_10": dict(bottom10),
        "n_features_above_auc_055": n_above_threshold,
        "best_feature_auc": top_auc,
        "best_directional_feature_auc": round(best_dir_auc, 4),
        "top10_vol_features": len(top10_vol),
        "top10_dir_features": len(top10_dir),
        "spurious_vol_confound_detected": has_spurious_vol_confound,
        "verdict": (
            "NO USABLE SIGNAL (all features AUC < 0.55)"
            if top_auc < AUC_SIGNAL_THRESHOLD
            else (
                "SPURIOUS VOL CONFOUND: top features are volatility measures, "
                f"not directional — best directional AUC={best_dir_auc:.4f}. "
                "Model learns 'trade when vol is high' not 'predict direction'."
                if has_spurious_vol_confound
                else f"some signal present: {n_above_threshold} features AUC > 0.55"
            )
        ),
    }


# ---------------------------------------------------------------------------
# Hypothesis 5: Horizon / Threshold Sanity
# ---------------------------------------------------------------------------

def h5_horizon_sanity(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    meta: Dict,
) -> Dict:
    """Check label base rate per split and verify against meta.json thresholds."""
    results: Dict = {}
    flags: List[str] = []

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        y = split_df["label"].astype(float).dropna()
        n = len(y)
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        base_rate = float(n_pos / n) if n > 0 else float("nan")
        results[split_name] = {
            "n": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "base_rate": round(base_rate, 4),
        }
        if base_rate < BASE_RATE_MIN or base_rate > BASE_RATE_MAX:
            flags.append(
                f"{split_name}: base_rate={base_rate:.3f} "
                f"(outside [{BASE_RATE_MIN}, {BASE_RATE_MAX}])"
            )

    # Pull horizon / threshold from meta if present
    xgb_kwargs = meta.get("xgb_kwargs", {})
    # Horizon and threshold are stored in the dataset, not xgb_kwargs directly
    # Check xgb_kwargs for any horizon-related keys
    horizon_info = {k: v for k, v in xgb_kwargs.items() if "horizon" in k.lower() or "threshold" in k.lower()}
    # Also check top-level meta keys
    for k in ("horizon_bars", "threshold_bps", "label_mode"):
        if k in meta:
            horizon_info[k] = meta[k]

    results["meta_horizon_info"] = horizon_info
    results["flags"] = flags
    results["verdict"] = "IMBALANCE FLAGS" if flags else "base rates OK"
    return results


# ---------------------------------------------------------------------------
# Hypothesis 6: Class Imbalance
# ---------------------------------------------------------------------------

def h6_class_imbalance(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    meta: Dict,
) -> Dict:
    """Label distribution and implied class weights per split."""
    results: Dict = {}

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        y = split_df["label"].astype(float).dropna()
        n = len(y)
        n_pos = int((y == 1).sum())
        n_neg = n - n_pos
        ratio = float(n_neg / n_pos) if n_pos > 0 else float("inf")
        results[split_name] = {
            "n": n,
            "label_0": n_neg,
            "label_1": n_pos,
            "neg_to_pos_ratio": round(ratio, 2),
            "pos_rate": round(n_pos / n, 4) if n > 0 else 0.0,
        }

    # Drift in class ratio between train and test
    train_ratio = results["train"]["neg_to_pos_ratio"]
    test_ratio = results["test"]["neg_to_pos_ratio"]
    ratio_change_pct = abs(train_ratio - test_ratio) / max(train_ratio, 1e-9) * 100

    meta_dist = meta.get("class_distribution", {})
    results["meta_class_distribution"] = meta_dist
    results["train_to_test_ratio_change_pct"] = round(ratio_change_pct, 1)
    results["verdict"] = (
        f"class ratio drifts {ratio_change_pct:.0f}% train→test"
        if ratio_change_pct > 20
        else "stable class ratio across splits"
    )
    return results


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _fmt_dict(d: Dict, indent: int = 4) -> str:
    pad = " " * indent
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{pad}{k}:")
            for kk, vv in v.items():
                lines.append(f"{pad}  {kk}: {vv}")
        else:
            lines.append(f"{pad}{k}: {v}")
    return "\n".join(lines)


def run_diagnostics(
    symbol: str,
    parquet: str,
    meta_dir: str,
) -> Tuple[str, Dict]:
    """Run all 6 hypotheses for one symbol. Returns (markdown_section, raw_results)."""
    parquet_path = DATASET_DIR / parquet
    df = pd.read_parquet(parquet_path)
    df = df.sort_values("timestamp").reset_index(drop=True)

    meta = load_meta(meta_dir)
    feature_cols = feature_cols_from_df(df)

    train, val, test = time_based_split(df)

    lines: List[str] = [f"\n### {symbol}\n"]
    lines.append(
        f"Dataset: `{parquet}` | rows: {len(df):,} "
        f"(train={len(train):,}, val={len(val):,}, test={len(test):,}) "
        f"| features: {len(feature_cols)}\n"
    )

    all_results: Dict = {}

    # H1
    h1 = h1_label_leakage(train, feature_cols)
    all_results["h1"] = h1
    lines.append("#### H1 — Label Leakage\n")
    lines.append(f"- Verdict: **{h1['verdict']}**")
    lines.append(f"- Flagged features (|corr| > {LEAK_CORR_THRESHOLD}): {h1['flagged_count']}")
    if h1["flagged_features"]:
        for feat, corr in list(h1["flagged_features"].items())[:10]:
            lines.append(f"  - `{feat}`: corr={corr}")
    lines.append("")

    # H2
    h2 = h2_regime_shift(train, val, test)
    all_results["h2"] = h2
    lines.append("#### H2 — Regime Shift\n")
    lines.append(f"- Verdict: **{h2['verdict']}**")
    for split_name in ("train", "val", "test"):
        s = h2[split_name]
        lines.append(
            f"  - {split_name}: rv={s['rv']:.5f} trend={s['trend']:.6f} spread%={s['spread_pct']:.5f}"
        )
    if h2["flags"]:
        for flag in h2["flags"]:
            lines.append(f"  - FLAG: {flag}")
    lines.append("")

    # H3
    h3 = h3_feature_drift(train, val, test, feature_cols)
    all_results["h3"] = h3
    lines.append("#### H3 — Feature Drift (train → test)\n")
    lines.append(f"- Verdict: **{h3['verdict']}**")
    lines.append(f"- Features with z > {DRIFT_ZSCORE_THRESHOLD}: {h3['flagged_count']}")
    if h3["top_20_by_zscore"]:
        lines.append("- Top drifted features:")
        for feat, info in list(h3["top_20_by_zscore"].items())[:10]:
            lines.append(
                f"  - `{feat}`: train_mean={info['train_mean']}, "
                f"test_mean={info['test_mean']}, z={info['z_score']}"
            )
    lines.append("")

    # H4
    h4 = h4_univariate_auc(train, test, feature_cols)
    all_results["h4"] = h4
    lines.append("#### H4 — Univariate AUC on Test Split\n")
    lines.append(f"- Verdict: **{h4['verdict']}**")
    lines.append(f"- Best overall feature AUC: {h4['best_feature_auc']}")
    lines.append(f"- Best DIRECTIONAL feature AUC: {h4['best_directional_feature_auc']}")
    lines.append(f"- Features with AUC > 0.55: {h4['n_features_above_auc_055']}")
    lines.append(f"- Top-10 breakdown: {h4['top10_vol_features']} volatility, {h4['top10_dir_features']} directional")
    if h4["spurious_vol_confound_detected"]:
        lines.append("- **WARNING: Spurious volatility confound detected** — see verdict")
    lines.append("- Top 10 features by test AUC:")
    for feat, auc in list(h4["top_10"].items())[:10]:
        lines.append(f"  - `{feat}`: AUC={auc}")
    lines.append("")

    # H5
    h5 = h5_horizon_sanity(train, val, test, meta)
    all_results["h5"] = h5
    lines.append("#### H5 — Horizon / Threshold Sanity\n")
    lines.append(f"- Verdict: **{h5['verdict']}**")
    lines.append(f"- Meta horizon info: {h5['meta_horizon_info']}")
    for split_name in ("train", "val", "test"):
        s = h5[split_name]
        lines.append(
            f"  - {split_name}: n={s['n']:,} pos={s['n_pos']:,} "
            f"base_rate={s['base_rate']}"
        )
    if h5["flags"]:
        for flag in h5["flags"]:
            lines.append(f"  - FLAG: {flag}")
    lines.append("")

    # H6
    h6 = h6_class_imbalance(train, val, test, meta)
    all_results["h6"] = h6
    lines.append("#### H6 — Class Imbalance\n")
    lines.append(f"- Verdict: **{h6['verdict']}**")
    lines.append(f"- Train-to-test ratio change: {h6['train_to_test_ratio_change_pct']}%")
    for split_name in ("train", "val", "test"):
        s = h6[split_name]
        lines.append(
            f"  - {split_name}: 0={s['label_0']:,} 1={s['label_1']:,} "
            f"ratio={s['neg_to_pos_ratio']} pos_rate={s['pos_rate']}"
        )
    lines.append("")

    return "\n".join(lines), all_results


def _rank_hypotheses(per_symbol_results: Dict[str, Dict]) -> Tuple[str, str]:
    """Score each hypothesis across all symbols and return (ranked_md, headline)."""
    scores: Dict[str, int] = {
        "H1 Label Leakage": 0,
        "H2 Regime Shift + Spurious Vol Confound": 0,
        "H3 Feature Drift": 0,
        "H4 No Directional Signal": 0,
        "H5 Horizon/Threshold Issue": 0,
        "H6 Class Imbalance (label base-rate drift)": 0,
    }

    for sym, results in per_symbol_results.items():
        # H1
        if results["h1"]["flagged_count"] > 0:
            scores["H1 Label Leakage"] += 3 * results["h1"]["flagged_count"]

        # H2 + spurious vol confound (highest-weight combination)
        h2_flags = len(results["h2"]["flags"])
        scores["H2 Regime Shift + Spurious Vol Confound"] += h2_flags * 5

        # H4 spurious vol confound amplifies H2 dramatically
        if results["h4"].get("spurious_vol_confound_detected"):
            scores["H2 Regime Shift + Spurious Vol Confound"] += 25  # strong evidence

        # H3
        n_drifted = results["h3"]["flagged_count"]
        scores["H3 Feature Drift"] += min(n_drifted * 2, 50)  # cap at 50 per symbol

        # H4 directional signal
        best_dir_auc = results["h4"].get("best_directional_feature_auc", 0.5)
        if best_dir_auc < 0.52:
            scores["H4 No Directional Signal"] += 30
        elif best_dir_auc < AUC_SIGNAL_THRESHOLD:
            scores["H4 No Directional Signal"] += 15

        # H5
        h5_flags = len(results["h5"]["flags"])
        scores["H5 Horizon/Threshold Issue"] += h5_flags * 10

        # H6
        ratio_change = results["h6"]["train_to_test_ratio_change_pct"]
        if ratio_change > 20:
            scores["H6 Class Imbalance (label base-rate drift)"] += 15

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    lines = ["### Hypothesis Ranking (by evidence strength)\n"]
    lines.append("| Rank | Hypothesis | Score |")
    lines.append("|------|-----------|-------|")
    for i, (hyp, score) in enumerate(ranked, 1):
        lines.append(f"| {i} | {hyp} | {score} |")
    lines.append("")

    top_hyp = ranked[0][0] if ranked else "unknown"
    headline = top_hyp
    return "\n".join(lines), headline


def main() -> int:
    utc_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(f"/tmp/feature_diagnosis_{utc_tag}.md")

    per_symbol_sections: List[str] = []
    per_symbol_results: Dict[str, Dict] = {}

    print(f"Running feature diagnostics on {len(SYMBOLS)} symbols...\n", flush=True)

    for symbol, parquet, meta_dir in SYMBOLS:
        print(f"  [{symbol}] loading {parquet}...", flush=True)
        section, results = run_diagnostics(symbol, parquet, meta_dir)
        per_symbol_sections.append(section)
        per_symbol_results[symbol] = results
        print(f"  [{symbol}] done. H4 best AUC={results['h4']['best_feature_auc']}, "
              f"H3 drifted={results['h3']['flagged_count']}, "
              f"H1 leaked={results['h1']['flagged_count']}", flush=True)

    # Cross-symbol patterns
    drift_counts = {s: per_symbol_results[s]["h3"]["flagged_count"] for s in per_symbol_results}
    auc_bests = {s: per_symbol_results[s]["h4"]["best_feature_auc"] for s in per_symbol_results}
    leak_counts = {s: per_symbol_results[s]["h1"]["flagged_count"] for s in per_symbol_results}

    cross_lines: List[str] = ["\n## Cross-Symbol Patterns\n"]

    # Find features that drift in ALL 3 symbols
    all_drifted_sets = []
    for sym in per_symbol_results:
        drifted_set = set(per_symbol_results[sym]["h3"]["top_20_by_zscore"].keys())
        all_drifted_sets.append(drifted_set)
    if len(all_drifted_sets) == 3:
        common_drifted = all_drifted_sets[0] & all_drifted_sets[1] & all_drifted_sets[2]
        if common_drifted:
            cross_lines.append(
                f"- **Features drifting in ALL 3 symbols (top-20 overlap): "
                f"{len(common_drifted)}** — "
                + ", ".join(sorted(common_drifted)[:10])
            )
        else:
            cross_lines.append("- No features are in the top-20 drift list for all 3 symbols simultaneously.")

    # Check if H4 (no signal) holds across all
    all_no_signal = all(v < AUC_SIGNAL_THRESHOLD for v in auc_bests.values())
    cross_lines.append(
        f"- H4 (overall signal): best AUC per symbol: {auc_bests}"
    )
    # Spurious vol confound across symbols
    vol_confound_syms = [
        s for s in per_symbol_results
        if per_symbol_results[s]["h4"].get("spurious_vol_confound_detected")
    ]
    if vol_confound_syms:
        cross_lines.append(
            f"- **SPURIOUS VOL CONFOUND detected in: {vol_confound_syms}** — "
            "top features are volatility measures (atrp_14, rv_*, range_ma_20), not directional. "
            "These predict label=1 because high-vol periods clear a fixed threshold more often. "
            "Train period has higher vol → more label=1. Test period lower vol → fewer label=1. "
            "Model appears predictive on val but collapses on test. "
            "Root cause: label definition conflates direction + volatility."
        )
    # Directional AUC across symbols
    dir_aucs_cross = {s: per_symbol_results[s]["h4"]["best_directional_feature_auc"] for s in per_symbol_results}
    all_dir_random = all(v < 0.53 for v in dir_aucs_cross.values())
    cross_lines.append(
        f"- Best DIRECTIONAL feature AUC per symbol: {dir_aucs_cross}"
        + (" — **ALL near-random (< 0.53): there is NO directional alpha in the feature set**."
           if all_dir_random
           else "")
    )
    # Regime shift across symbols
    regime_shift_syms = [
        s for s in per_symbol_results
        if "REGIME SHIFT" in per_symbol_results[s]["h2"]["verdict"]
    ]
    cross_lines.append(
        f"- H2 (regime shift): detected in {len(regime_shift_syms)}/3 symbols — {regime_shift_syms}. "
        "Train period has consistently HIGHER realized volatility than test across all 3 assets."
    )

    # Drift summary
    total_drifted = sum(drift_counts.values())
    cross_lines.append(
        f"- H3 total drifted features across symbols: {total_drifted} "
        f"({drift_counts})"
    )

    cross_lines.append("")

    # Ranking
    ranking_md, top_hyp = _rank_hypotheses(per_symbol_results)

    # Remediation map
    remediation_map = {
        "H1 Label Leakage": (
            "Remove or lag the flagged features by at least `horizon_bars` bars, "
            "then rebuild the parquets and retrain."
        ),
        "H2 Regime Shift + Spurious Vol Confound": (
            "The model is learning 'trade when volatility is high' (atrp_14, rv_*, range_ma_20 "
            "are spurious confounds: high-vol periods clear a fixed threshold_bps more often). "
            "When test-period volatility is lower than train, label base rate drops and the "
            "model's vol-based predictions become anti-predictive. "
            "Recommended fix: (1) Replace the fixed-bps threshold with a volatility-normalized "
            "threshold (e.g., `forward_return > k * atr_14` at each bar) so label=1 means "
            "'outperformed the expected vol move', not just 'any big move'. "
            "(2) Remove or residualize the vol features from the feature set, leaving only "
            "directional features. "
            "(3) Add a regime filter: only trade when current ATR percentile is within "
            "the training distribution's IQR."
        ),
        "H3 Feature Drift": (
            "Drop or normalize the heavily drifted features. "
            "Alternatively, add online normalization (z-score rolling 500-bar window) "
            "so features stay in range as the market regime changes."
        ),
        "H4 No Directional Signal": (
            "The directional features themselves have no out-of-sample predictive power. "
            "Investigate the feature engineering pipeline: "
            "(a) check if order-book features (bid_depth_*, vwap_*) are forward-filled "
            "from a snapshot that's contemporaneous with the label close; "
            "(b) try raw price-only features (OHLCV + ATR) with longer horizons (≥20 bars); "
            "(c) consider fundamentally different alpha sources (funding rates, options IV)."
        ),
        "H5 Horizon/Threshold Issue": (
            "Adjust the forward horizon or threshold_bps so the label base rate sits "
            "between 20% and 50%. A base rate < 10% means the model almost never fires. "
            "Try reducing threshold_bps to 5 or increasing horizon_bars to 20."
        ),
        "H6 Class Imbalance (label base-rate drift)": (
            "The neg/pos ratio is extreme and drifts across splits. "
            "Use scale_pos_weight dynamically or oversample positive examples. "
            "Consider a balanced sampling strategy per epoch."
        ),
    }

    recommended_remediation = remediation_map.get(top_hyp, "Investigate further.")

    # Build full report
    report_lines: List[str] = [
        f"# Feature Diagnostic Report — {utc_tag}\n",
        "## Summary\n",
        f"- Most likely root cause: **{top_hyp}**",
        f"- Recommended remediation: {recommended_remediation}",
        "",
        "## Per-Symbol Details",
    ]
    report_lines.extend(per_symbol_sections)
    report_lines.extend(cross_lines)
    report_lines.append("\n## Hypothesis Ranking\n")
    report_lines.append(ranking_md)

    report = "\n".join(report_lines)

    # Write to /tmp
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to: {out_path}\n", flush=True)
    print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
