"""
Production-safe inference entrypoint that:
  - Reads `model_meta.json`
  - Builds the LSTM model dynamically from metadata
  - Loads the correct weights and scaler artifacts
  - Supports SageMaker (model_fn / input_fn / predict_fn / output_fn)
  - Works locally via `python inference.py --input path/to.json`

Expected input JSON (examples):

1) Batch of sequences (recommended)
{
  "features": [[[... F features ...], ... T timesteps ...], ... B batch ...]
}

2) Single sequence
{
  "features": [[... F features ...], ... T timesteps ...]
}

Output:
{
  "logits": [[... C ...], ... B ...],
  "probs":  [[... C ...], ... B ...],
  "preds":  [class_index, ... B ...]
}

If you use thresholds/decision rules post-probabilities, apply them in your caller
or extend `postprocess()` below to add your strategy.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from typing import Any, Dict, Tuple, Union, List

import numpy as np
import torch
import torch.nn.functional as F

from models import (
    build_model_from_meta,
    load_meta,
    load_model_state,
    load_scaler,
    resolve_path,
)

# ----------------------------
# Core load
# ----------------------------

def _detect_model_dir(env: Dict[str, str]) -> str:
    """
    For SageMaker, SM_MODEL_DIR points to the model artifact directory.
    Locally, default to current working directory.
    """
    return env.get("SM_MODEL_DIR") or os.getcwd()


def _get_paths(model_dir: str) -> Tuple[str, str, Union[str, None]]:
    """
    Returns (meta_path, weights_path, scaler_path)
    """
    meta_path = os.path.join(model_dir, "model_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"model_meta.json not found in: {model_dir}. "
            "Ensure training exported metadata alongside weights."
        )

    meta = load_meta(meta_path)
    weights_path = resolve_path(model_dir, meta.model_state_path)
    scaler_path = resolve_path(model_dir, meta.scaler_path) if meta.feature_scaling else None

    if not os.path.exists(weights_path):
        # Backward-compat: check common alternatives
        alt_candidates = ["model.pt", "best_model.pth", "weights.pt"]
        for cand in alt_candidates:
            alt = os.path.join(model_dir, cand)
            if os.path.exists(alt):
                weights_path = alt
                break

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found. Tried: {weights_path}. "
            "Check model_meta.json->model_state_path or export path."
        )

    return meta_path, weights_path, scaler_path


def model_fn(model_dir: str):
    """
    SageMaker entrypoint: load model + scaler from artifacts dir.
    """
    meta_path, weights_path, scaler_path = _get_paths(model_dir)
    meta = load_meta(meta_path)

    model = build_model_from_meta(meta)
    load_model_state(model, weights_path, strict=False)
    model.eval()

    scaler = load_scaler(scaler_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return {
        "model": model,
        "scaler": scaler,
        "device": device,
        "meta": meta.to_dict(),
    }


# ----------------------------
# Input / preprocessing
# ----------------------------

def _ensure_batch(features: Union[List, np.ndarray]) -> np.ndarray:
    """
    Converts features to shape [B, T, F]. Adds batch dim if needed.
    """
    arr = np.array(features, dtype=np.float32)
    if arr.ndim == 2:
        # [T, F] -> [1, T, F]
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array for features, got shape {arr.shape}")
    return arr


def _apply_scaler_if_any(arr_btf: np.ndarray, scaler) -> np.ndarray:
    """
    If scaler is provided (sklearn StandardScaler/MinMax/etc.), apply it across
    the feature dimension. We reshape to [B*T, F], transform, then reshape back.
    """
    if scaler is None:
        return arr_btf

    b, t, f = arr_btf.shape
    flat = arr_btf.reshape(b * t, f)
    flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(b, t, f)


def input_fn(serialized_input_data: Union[str, bytes], content_type: str = "application/json"):
    """
    SageMaker input deserialization.
    """
    if "json" not in content_type:
        raise ValueError(f"Unsupported content_type: {content_type}")

    if isinstance(serialized_input_data, (bytes, bytearray)):
        serialized_input_data = serialized_input_data.decode("utf-8")

    payload = json.loads(serialized_input_data)
    if "features" not in payload:
        raise KeyError("Input JSON must contain a 'features' field.")

    return payload


def _to_tensor(batch_btf: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(batch_btf).to(device)


# ----------------------------
# Predict / postprocess
# ----------------------------

def predict_fn(payload: Dict[str, Any], model_artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """
    SageMaker predict.
    """
    model = model_artifacts["model"]
    scaler = model_artifacts["scaler"]
    device = model_artifacts["device"]
    meta = model_artifacts["meta"]

    arr_btf = _ensure_batch(payload["features"])
    arr_btf = _apply_scaler_if_any(arr_btf, scaler)

    x = _to_tensor(arr_btf, device)
    with torch.no_grad():
        logits = model(x)  # [B, C]
        probs = F.softmax(logits, dim=-1)  # [B, C]
        preds = probs.argmax(dim=-1)       # [B]

    return {
        "logits": logits.cpu().numpy().tolist(),
        "probs": probs.cpu().numpy().tolist(),
        "preds": preds.cpu().numpy().tolist(),
        "meta": meta,
    }


def output_fn(prediction: Dict[str, Any], accept: str = "application/json"):
    """
    SageMaker output serialization.
    """
    if "json" not in accept:
        raise ValueError(f"Unsupported accept: {accept}")
    return json.dumps(prediction)


# ----------------------------
# Local CLI
# ----------------------------

def _local_load() -> Dict[str, Any]:
    """
    Load model artifacts for local runs (no SageMaker).
    """
    model_dir = _detect_model_dir(os.environ)
    return model_fn(model_dir)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Local inference runner")
    parser.add_argument("--input", type=str, required=True, help="Path to JSON input with 'features'")
    parser.add_argument("--accept", type=str, default="application/json")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        payload = json.load(f)

    artifacts = _local_load()
    result = predict_fn(payload, artifacts)
    out = output_fn(result, accept=args.accept)
    print(out)


if __name__ == "__main__":
    _cli()
