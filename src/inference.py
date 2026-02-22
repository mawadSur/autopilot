#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference.py — SageMaker-compatible inference script.

Loads model_meta.json, model.pt, scaler.joblib from model_dir,
accepts JSON or CSV, builds windows, returns probabilities for class=1 (buy).
"""

from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

# Optional deps
try:
    import pandas as pd
except Exception:
    pd = None

try:
    import joblib
except Exception:
    joblib = None

# Project utils/models
from utils import load_meta, build_windows
from models import build_model_from_meta
from config import cfg


# ---------- SageMaker entrypoints ----------

def model_fn(model_dir: str):
    """Load model/scaler/meta using the shared builders (SageMaker entrypoint)."""
    meta = load_meta(model_dir)
    feature_cols: List[str] = list(meta["feature_cols"])  # strict: must exist

    model = build_model_from_meta(meta)
    weights_path = os.path.join(model_dir, meta.get("model_state_path", "model.pt"))
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    scaler = None
    spath = meta.get("scaler_path", "scaler.joblib")
    if spath and joblib is not None:
        sfile = os.path.join(model_dir, spath)
        if os.path.exists(sfile):
            scaler = joblib.load(sfile)

    return (model, scaler, meta)


def _json_to_matrix(payload: Dict[str, Any], feature_cols: List[str]) -> np.ndarray:
    """
    Accept either:
      {"rows":[{feature:value,...}, ...]}
      {"instances":[[f1,f2,...], ...]}
    Returns np.ndarray [N, F] ordered by feature_cols.
    """
    if "rows" in payload and isinstance(payload["rows"], list):
        rows = payload["rows"]
        if not rows:
            return np.zeros((0, len(feature_cols)), dtype=np.float32)
        out = []
        for r in rows:
            out.append([float(r[c]) for c in feature_cols])
        return np.array(out, dtype=np.float32)

    if "instances" in payload and isinstance(payload["instances"], list):
        arr = np.array(payload["instances"], dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("'instances' must be 2D: [N, F]")
        if arr.shape[1] != len(feature_cols):
            raise ValueError(f"instances columns ({arr.shape[1]}) != required features ({len(feature_cols)})")
        return arr

    raise ValueError("JSON must contain 'rows' (list of dicts) or 'instances' (2D list).")


def _csv_to_matrix(body: str, feature_cols: List[str]) -> np.ndarray:
    """
    CSV with header including feature columns; extra columns ignored but must contain at least all needed features.
    """
    if pd is None:
        raise RuntimeError("pandas is required for CSV input. Install pandas or send JSON instead.")
    df = pd.read_csv(io.StringIO(body))
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required features: {missing}")
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    return X


def input_fn(request_body: str, content_type: str):
    """
    Parse incoming request to a flat features matrix [N, F].
    """
    if not request_body:
        raise ValueError("Empty request body.")

    content_type = (content_type or "application/json").lower().strip()
    input_obj: Dict[str, Any] = {"flat": None, "content_type": content_type}

    if content_type == "application/json":
        payload = json.loads(request_body)
        input_obj["payload"] = payload
        return input_obj

    if content_type in ("text/csv", "application/csv"):
        input_obj["csv"] = request_body
        return input_obj

    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_object: Dict[str, Any], model_bundle):
    """
    Build windows and predict class probabilities.
    Output dict includes probs and hard labels using thresholds + margin gating.
    """
    model, scaler, meta = model_bundle
    feature_cols: List[str] = list(meta["feature_cols"])  # strict: must exist
    window_size = int(meta.get("window_size", 150))
    thr_long = float(meta.get("thr_long", meta.get("buy_threshold", 0.60)))
    thr_short = float(meta.get("thr_short", thr_long))
    p_margin = float(meta.get("p_margin", meta.get("margin", 0.15)))

    # Parse input into flat matrix [N,F]
    if "payload" in input_object:
        X_flat = _json_to_matrix(input_object["payload"], feature_cols)
    elif "csv" in input_object:
        X_flat = _csv_to_matrix(input_object["csv"], feature_cols)
    else:
        raise ValueError("Missing payload.")

    if X_flat.shape[0] < window_size:
        # Not enough rows to form a single window
        return {"probs": [], "labels": [], "thresholds": {"long": thr_long, "short": thr_short}, "p_margin": p_margin}

    # Build windows and scale
    X = build_windows(X_flat, window_size)  # [W, T, F]
    if scaler is not None:
        W, T, F = X.shape
        X = scaler.transform(X.reshape(W * T, F)).reshape(W, T, F)

    # Predict
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    with torch.no_grad():
        xb = torch.from_numpy(X).to(device)
        logits = model(xb)
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()

    # Margin-gated signals
    signals: List[int] = []
    for row in probs:
        row = np.asarray(row).flatten()
        if row.size >= 3:
            p_short, p_hold, p_long = row[:3]
            max_prob = max(p_long, p_short)
            top_two = np.sort(row[:3])[-2:]
            gap = float(top_two[-1] - top_two[-2]) if len(top_two) == 2 else 0.0
            if p_long >= p_short and p_long >= thr_long and gap >= p_margin:
                signals.append(1)
            elif p_short > p_long and p_short >= thr_short and gap >= p_margin:
                signals.append(-1)
            else:
                signals.append(0)
        elif row.size == 2:
            p_short, p_long = float(row[0]), float(row[1])
            max_prob = max(p_long, p_short)
            gap = abs(p_long - p_short)
            if p_long >= p_short and p_long >= thr_long and gap >= p_margin:
                signals.append(1)
            elif p_short > p_long and p_short >= thr_short and gap >= p_margin:
                signals.append(-1)
            else:
                signals.append(0)
        else:
            p_long = float(row[0])
            signals.append(1 if p_long >= thr_long else 0)

    # Preserve old behavior: if binary, expose class-1 probs; else full probs
    probs_out = probs[:, 1].tolist() if probs.shape[1] == 2 else probs.tolist()
    return {
        "probs": probs_out,
        "signals": signals,
        "thresholds": {"long": thr_long, "short": thr_short},
        "p_margin": p_margin,
    }


def output_fn(prediction: Dict[str, Any], accept: str = "application/json"):
    """
    Format the response.
    """
    accept = (accept or "application/json").lower().strip()
    body = json.dumps(prediction)
    if accept in ("application/json", "text/json"):
        return body, accept
    # Fallback to JSON
    return body, "application/json"
