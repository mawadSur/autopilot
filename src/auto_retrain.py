#!/usr/bin/env python3
"""
Automated retraining script for the autopilot trading system.

This script:
1. Locates the live data store (HDF5).
2. Triggers model retraining using the latest data.
3. Evaluates the new model.
4. If successful, updates the production model used for inference.
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Configuration
HDF5_STORE_PATH = os.getenv("HDF5_STORE_PATH", "market_data_store.h5")
MODEL_DIR = os.getenv("MODEL_DIR", "./model")
RETRAIN_DIR = os.getenv("RETRAIN_DIR", "./model_retrain")
SYMBOLS = os.getenv("SYMBOLS", os.getenv("TRADE_SYMBOL", "ETHUSDT"))

def _get_best_metric(summary: dict) -> tuple[float, str]:
    """Return (value, key_name) for the best available validation metric."""
    if "val_net_profit_best" in summary:
        return float(summary["val_net_profit_best"]), "val_net_profit_best"
    if "val_acc_best" in summary:
        return float(summary["val_acc_best"]), "val_acc_best"
    return 0.0, "none"


def run_retraining():
    """Execute the retraining pipeline."""
    print(f"\n{'='*60}")
    print(f"🚀 STARTING AUTOMATED RETRAINING: {datetime.now().isoformat()}")
    print(f"{'='*60}")

    if not os.path.exists(HDF5_STORE_PATH):
        print(f"❌ HDF5 store not found at {HDF5_STORE_PATH}")
        return False

    # 1. Prepare retraining directory
    if os.path.exists(RETRAIN_DIR):
        shutil.rmtree(RETRAIN_DIR)
    os.makedirs(RETRAIN_DIR)

    # 2. Trigger train_model.py
    print(f"\n📂 Training from {HDF5_STORE_PATH}...")
    
    cmd = [
        sys.executable, "src/train_model.py",
        "--data-path", HDF5_STORE_PATH,
        "--data-format", "hdf5",
        "--symbols", SYMBOLS,
        "--output-dir", RETRAIN_DIR,
        "--epochs", "20",  # Faster retraining for daily updates
        "--batch-size", "512"
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        for line in process.stdout:
            print(line, end="")

        process.wait()

        if process.returncode != 0:
            print(f"❌ Retraining failed with exit code {process.returncode}")
            return False

    except Exception as e:
        print(f"❌ Error during training execution: {e}")
        return False

    # 3. Evaluate and Deploy
    print("\n⚖️ Evaluating new model...")

    new_summary_path = Path(RETRAIN_DIR) / "training_summary.json"
    if not new_summary_path.exists():
        print("❌ Training summary not found. Retraining likely failed.")
        return False

    with open(new_summary_path, "r") as f:
        new_summary = json.load(f)

    new_metric, new_key = _get_best_metric(new_summary)
    if new_key == "none":
        print("❌ New model has no recognised metric key. Training likely failed silently.")
        return False
    print(f"📈 New model best metric ({new_key}): {new_metric:.5f}")

    # Compare against the currently deployed model before replacing it.
    current_summary_path = Path(MODEL_DIR) / "training_summary.json"
    if current_summary_path.exists():
        with open(current_summary_path, "r") as f:
            current_summary = json.load(f)
        current_metric, current_key = _get_best_metric(current_summary)
        print(f"📊 Current deployed model metric ({current_key}): {current_metric:.5f}")
        if new_key != current_key:
            # Different metric types (e.g. acc vs profit) can't be compared — deploy unconditionally.
            print(f"ℹ️  Metric type changed ({current_key} → {new_key}), deploying unconditionally.")
        elif new_metric < current_metric:
            print(f"⚠️  New model ({new_metric:.5f}) is not better than current ({current_metric:.5f}). Skipping deployment.")
            return False
    else:
        print("ℹ️  No current model found — deploying unconditionally.")

    # Verify all required files exist before touching MODEL_DIR.
    required_files = ["model.pt", "scaler.joblib", "model_meta.json", "training_summary.json"]
    missing_files = [f for f in required_files if not (Path(RETRAIN_DIR) / f).exists()]
    if missing_files:
        print(f"❌ Retraining output incomplete. Missing: {missing_files}")
        return False

    print("\n✅ Deploying new model...")
    
    # Backup old model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(MODEL_DIR) / "backups" / timestamp
    os.makedirs(backup_dir, exist_ok=True)
    
    for f in ["model.pt", "scaler.joblib", "model_meta.json", "training_summary.json"]:
        src = Path(MODEL_DIR) / f
        if src.exists():
            shutil.copy(src, backup_dir / f)

    # Copy new model files
    for f in ["model.pt", "scaler.joblib", "model_meta.json", "training_summary.json"]:
        src = Path(RETRAIN_DIR) / f
        dst = Path(MODEL_DIR) / f
        if src.exists():
            shutil.copy(src, dst)

    print(f"🚀 New model deployed to {MODEL_DIR}")
    print(f"📦 Backup created at {backup_dir}")

    return True

if __name__ == "__main__":
    success = run_retraining()
    if not success:
        sys.exit(1)
