#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# retrain_all_crypto_models.sh
#
# Retrains BTC/ETH/SOL XGBoost calibration models in-place and verifies
# that each emitted meta.json carries the new feature_means + feature_stds
# fields required by A1 SignalForensics' Mahalanobis OOD check.
#
# Why this script exists
#   The trainer change at commit 906f843 ("Trainer: persist feature_means
#   + feature_stds for A1 Mahalanobis check") added new fields to
#   meta.json. Bundles trained before that commit ship without them, so
#   A1 silently skips the Mahalanobis distance check on production
#   bundles. Rerunning the trainer once on each dataset is the cleanest
#   way to populate the fields.
#
# Prerequisites
#   1. Datasets must exist (the real on-disk CSVs, not the old .parquet
#      names this script used to reference):
#        data/crypto/datasets/eth_usd_1m.csv
#        data/crypto/datasets/btc_usd_1m.csv
#        data/crypto/datasets/sol_usd_1m.csv
#   2. .venv exists at repo root (the trainer is heavy: xgboost,
#      sklearn, pandas, joblib).
#   3. ~30-60 minutes of developer-machine wall-clock time. XGBoost +
#      isotonic CalibratedClassifierCV(cv="prefit") is the bottleneck.
#
# IMPORTANT — which model trio?  (operator decision, has money impact)
#   This script RETRAINS IN PLACE and overwrites model.joblib + meta.json
#   in the target dirs. It regenerates the whole model (not just the new
#   feature_means/feature_stds stats), so spot-check AUC/reliability_slope
#   against the prior bundle — the bundles are git-tracked, so `git checkout`
#   restores them if the numbers drift. The SYMBOLS array below defaults to
#   the +10bps sigmoid trio (eth_usd_v3_sigmoid / btc_usd_v2_sigmoid /
#   sol_usd_v1), the set diagnosed as usable. If your production set is the
#   +20bps Tier-2 trio instead, switch to the commented-out block below.
#
# Usage
#   chmod +x scripts/retrain_all_crypto_models.sh
#   bash scripts/retrain_all_crypto_models.sh
#
# Notes
#   * Operator must `chmod +x` after checkout — git tracks the executable
#     bit but a fresh clone without the bit will silently fall back to
#     `bash <path>` invocation.
#   * The script does NOT push to origin. It writes only into
#     model_crypto/<symbol>_v*/. Operator commits those artefacts after
#     spot-checking AUC / reliability_slope.
#   * Failure of one symbol does NOT abort the run — we want a full
#     summary table at the end so the operator can see which symbols
#     succeeded vs failed in one pass.
# ---------------------------------------------------------------------------

set -uo pipefail

# Resilient venv activation. The script is callable from a fresh shell
# (operator may not have `source .venv/bin/activate` in their session);
# `|| true` swallows the failure if the venv layout is non-standard.
if [ -f "./.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source ./.venv/bin/activate || true
fi

PYTHON="${PYTHON:-./.venv/bin/python}"
TRAINER="src/crypto_training/train_xgboost.py"

# (symbol; dataset_path; output_dir) tuples for each crypto symbol.
# Default: +10bps sigmoid trio (the set diagnosed as usable on 2026-05-30).
SYMBOLS=(
    "ETH/USD;data/crypto/datasets/eth_usd_1m.csv;model_crypto/eth_usd_v3_sigmoid/"
    "BTC/USD;data/crypto/datasets/btc_usd_1m.csv;model_crypto/btc_usd_v2_sigmoid/"
    "SOL/USD;data/crypto/datasets/sol_usd_1m.csv;model_crypto/sol_usd_v1/"
)

# Alternative: +20bps Tier-2 trio. Uncomment (and comment the block above)
# if the +20bps models are your production set.
# SYMBOLS=(
#     "ETH/USD;data/crypto/datasets/eth_usd_1m_20bps.csv;model_crypto/eth_usd_v4_20bps_sigmoid/"
#     "BTC/USD;data/crypto/datasets/btc_usd_1m_20bps.csv;model_crypto/btc_usd_v3_20bps_sigmoid/"
#     "SOL/USD;data/crypto/datasets/sol_usd_1m_20bps.csv;model_crypto/sol_usd_v2_20bps_sigmoid/"
# )

# Per-symbol summary rows accumulated during the run; printed at the end.
SUMMARY_ROWS=()

# ---------------------------------------------------------------------------
# Helper: inspect_meta_json
#   Reads meta.json from $1 and prints "y/n" for whether feature_means and
#   feature_stds are populated. Also extracts AUC + reliability_slope so
#   the summary table is informative.
# ---------------------------------------------------------------------------
inspect_meta_json () {
    local meta_path="$1"
    if [ ! -f "$meta_path" ]; then
        echo "missing|n|n|n/a|n/a"
        return
    fi
    "$PYTHON" - "$meta_path" <<'PY'
import json
import sys
from pathlib import Path

meta_path = Path(sys.argv[1])
try:
    meta = json.loads(meta_path.read_text())
except Exception as exc:  # noqa: BLE001 - script-level only
    print(f"unreadable|n|n|n/a|n/a")
    sys.exit(0)

# feature_means / feature_stds may live at the top level or nested under
# fold_<n> entries. Accept either; report y/n based on presence at any
# depth and non-empty.
def _has_stat(meta_obj, key):
    if isinstance(meta_obj, dict):
        if key in meta_obj and meta_obj[key]:
            return True
        for v in meta_obj.values():
            if _has_stat(v, key):
                return True
    elif isinstance(meta_obj, list):
        for v in meta_obj:
            if _has_stat(v, key):
                return True
    return False

means_y = "y" if _has_stat(meta, "feature_means") else "n"
stds_y = "y" if _has_stat(meta, "feature_stds") else "n"

# Pull AUC + reliability_slope where available. Try the most common
# layouts; fall back to "n/a" so the summary doesn't crash.
def _first(d, *keys):
    for key in keys:
        if isinstance(d, dict) and key in d and d[key] is not None:
            return d[key]
    return None

auc = (
    _first(meta, "auc")
    or _first(meta.get("metrics_test", {}) if isinstance(meta, dict) else {}, "auc")
    or _first(meta.get("metrics", {}) if isinstance(meta, dict) else {}, "auc")
)
slope = (
    _first(meta, "reliability_slope")
    or _first(meta.get("metrics_test", {}) if isinstance(meta, dict) else {}, "reliability_slope")
    or _first(meta.get("metrics", {}) if isinstance(meta, dict) else {}, "reliability_slope")
)

def _fmt(v):
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.3f}"
    except Exception:
        return str(v)

print(f"present|{means_y}|{stds_y}|{_fmt(auc)}|{_fmt(slope)}")
PY
}

# ---------------------------------------------------------------------------
# Main loop: train + inspect + record summary row.
# ---------------------------------------------------------------------------
echo "=========================================================="
echo "retrain_all_crypto_models.sh — start  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================================="
echo ""

for entry in "${SYMBOLS[@]}"; do
    SYMBOL="$(echo "$entry" | cut -d';' -f1)"
    DATASET="$(echo "$entry" | cut -d';' -f2)"
    OUTDIR="$(echo "$entry" | cut -d';' -f3)"
    META_PATH="${OUTDIR}meta.json"

    echo "----------------------------------------------------------"
    echo "[${SYMBOL}] dataset=${DATASET}  out=${OUTDIR}"
    echo "----------------------------------------------------------"

    if [ ! -f "$DATASET" ]; then
        echo "[${SYMBOL}] ABORT: dataset $DATASET not found. Skipping."
        SUMMARY_ROWS+=("${SYMBOL}|missing-dataset|n|n|n/a|n/a")
        continue
    fi

    mkdir -p "$OUTDIR"

    # Run the trainer. Don't `set -e` because we want all symbols to
    # attempt training even if one fails — see the SUMMARY_ROWS pattern.
    if "$PYTHON" "$TRAINER" --dataset "$DATASET" --out "$OUTDIR"; then
        STATUS="ok"
    else
        STATUS="train-failed(exit=$?)"
    fi

    INSPECTION="$(inspect_meta_json "$META_PATH")"
    META_PRESENT="$(echo "$INSPECTION" | cut -d'|' -f1)"
    MEANS_Y="$(echo "$INSPECTION" | cut -d'|' -f2)"
    STDS_Y="$(echo "$INSPECTION" | cut -d'|' -f3)"
    AUC="$(echo "$INSPECTION" | cut -d'|' -f4)"
    SLOPE="$(echo "$INSPECTION" | cut -d'|' -f5)"

    echo ""
    echo "[${SYMBOL}] status=${STATUS}  meta=${META_PRESENT}  feature_means=${MEANS_Y}  feature_stds=${STDS_Y}  AUC=${AUC}  slope=${SLOPE}"
    echo ""

    SUMMARY_ROWS+=("${SYMBOL}|${STATUS}|${MEANS_Y}|${STDS_Y}|${AUC}|${SLOPE}")
done

# ---------------------------------------------------------------------------
# Final summary table
# ---------------------------------------------------------------------------
echo ""
echo "=========================================================="
echo "Summary  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================================="
printf "%-10s %-22s %-12s %-12s %-8s %-8s\n" "symbol" "status" "means_set?" "stds_set?" "auc" "slope"
printf "%-10s %-22s %-12s %-12s %-8s %-8s\n" "------" "------" "----------" "---------" "----" "-----"
for row in "${SUMMARY_ROWS[@]}"; do
    SYMBOL="$(echo "$row" | cut -d'|' -f1)"
    STATUS="$(echo "$row" | cut -d'|' -f2)"
    MEANS_Y="$(echo "$row" | cut -d'|' -f3)"
    STDS_Y="$(echo "$row" | cut -d'|' -f4)"
    AUC="$(echo "$row" | cut -d'|' -f5)"
    SLOPE="$(echo "$row" | cut -d'|' -f6)"
    printf "%-10s %-22s %-12s %-12s %-8s %-8s\n" "$SYMBOL" "$STATUS" "$MEANS_Y" "$STDS_Y" "$AUC" "$SLOPE"
done

echo ""
echo "Next steps for the operator:"
echo "  1. Spot-check AUC + reliability_slope per symbol against the previous bundle."
echo "  2. If numbers look sane, commit the regenerated model_crypto/<symbol>/ artefacts."
echo "  3. Re-run a tick of the supervisor against a real ticker; A1 SignalForensics"
echo "     should now compute Mahalanobis distance instead of skipping."
echo ""
echo "retrain_all_crypto_models.sh — done  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
