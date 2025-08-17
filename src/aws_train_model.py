#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aws_train_model.py

Chunked training for LSTM classifier on many CSVs under a directory (e.g., eth_1m_data/).
- Streams data in chunks with a (T-1) overlap buffer so sequences don't break at boundaries
- Two-pass StandardScaler fit for best accuracy (fit on train portion only)
- Validation split:
    * default: last --val-frac of sequences
    * legacy flag: --val_months N  -> last N CSV files (by filename) comprise the validation set
- Saves: model.pt, scaler.joblib (if enabled), model_meta.json
- Works locally and on SageMaker (uses SM_MODEL_DIR if available)
- Reads CHUNK_SIZE from .env (default 200000)

Backward-compat flags accepted (mapped internally):
  --batch_size   -> --batch-size
  --hidden_size  -> --hidden-size
  --window_size  -> --seq-len
  --val_months   -> activates month-based validation split
  --accumulate   -> gradient accumulation steps

Usage (defaults assume ./eth_1m_data exists and label column is 'label'):
  python aws_train_model.py --data eth_1m_data --seq-len 60 --epochs 10
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from dataclasses import asdict
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

# env (.env) support
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False

# Prefer joblib for sklearn persistence
try:
    import joblib
except Exception:
    joblib = None

# Optional scaler (enabled via --scale-features)
try:
    from sklearn.preprocessing import StandardScaler
except Exception:
    StandardScaler = None  # guard if sklearn isn't installed

# ---- Unified model and metadata helpers (models.py in project root) ----
from models import (
    LSTMClassifier,
    ModelMeta,
)

# -----------------------------
# Utils & Repro
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def list_csvs_sorted(path: str) -> List[str]:
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        return files
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if os.path.splitext(path)[1].lower() != ".csv":
        raise ValueError(f"Only .csv supported in chunked mode; got {path}")
    return [path]

def infer_feature_cols_from_sample(
    sample_df: pd.DataFrame,
    label_col: Optional[str],
    time_col: Optional[str],
    price_col: Optional[str],
    explicit_feature_cols: Optional[List[str]],
) -> List[str]:
    if explicit_feature_cols:
        return explicit_feature_cols
    drop = {c for c in [label_col, time_col, price_col] if c is not None}
    numeric = sample_df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric if c not in drop]
    if not feats:
        raise ValueError("Could not infer numeric feature columns from the sample.")
    return feats

def build_windows_from_flat(features: np.ndarray, seq_len: int) -> np.ndarray:
    """
    features: [N, F] -> windows: [N - seq_len + 1, seq_len, F]
    """
    N, F = features.shape
    if N < seq_len:
        return np.empty((0, seq_len, F), dtype=np.float32)
    stride0, stride1 = features.strides
    shape = (N - seq_len + 1, seq_len, F)
    strides = (stride0, stride0, stride1)
    win = np.lib.stride_tricks.as_strided(features, shape=shape, strides=strides).copy()
    return win

# -----------------------------
# Chunked iterators with overlap buffer (T-1)
# -----------------------------

def iter_chunked_windows_and_labels(
    files: List[str],
    feature_cols: List[str],
    label_col: str,
    seq_len: int,
    chunksize: int,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Yields (X_chunk, y_chunk) where:
      X_chunk: [B, T, F], y_chunk: [B]
    Maintains a (T-1)-row buffer across chunks & files so windows are continuous.
    """
    buffer_df = None
    for fpath in files:
        for chunk in pd.read_csv(fpath, chunksize=chunksize):
            dfc = chunk if buffer_df is None else pd.concat([buffer_df, chunk], ignore_index=True)
            L = len(dfc)
            if L < seq_len:
                buffer_df = dfc
                continue

            feats = dfc[feature_cols].to_numpy(dtype=np.float32)
            X_all = build_windows_from_flat(feats, seq_len)            # [W, T, F], W = L - seq_len + 1
            y_all = dfc[label_col].to_numpy()                          # [L]
            y_all = y_all[seq_len - 1:]                                # [W], aligned to window end

            # Defer last (seq_len - 1) windows to next loop to preserve continuity
            emit = max(0, X_all.shape[0] - (seq_len - 1))
            if emit > 0:
                yield X_all[:emit], y_all[:emit]

            buffer_df = dfc.iloc[-(seq_len - 1):].copy()

def count_total_sequences(
    files: List[str],
    feature_cols: List[str],
    label_col: str,
    seq_len: int,
    chunksize: int,
) -> int:
    total = 0
    buffer_df = None
    for fpath in files:
        for chunk in pd.read_csv(fpath, chunksize=chunksize):
            dfc = chunk if buffer_df is None else pd.concat([buffer_df, chunk], ignore_index=True)
            L = len(dfc)
            if L < seq_len:
                buffer_df = dfc
                continue
            W = L - seq_len + 1
            emit = max(0, W - (seq_len - 1))
            total += emit
            buffer_df = dfc.iloc[-(seq_len - 1):].copy()
    return total

# -----------------------------
# IterableDatasets for streaming train / val (index-based split)
# -----------------------------

class SeqIterableDataset(IterableDataset):
    """
    Streams sequences directly from disk, yielding mini-batches.
    Subsets:
      - 'train': indices [0, n_train)
      - 'val'  : indices [n_train, n_total)
    """
    def __init__(
        self,
        files: List[str],
        feature_cols: List[str],
        label_col: str,
        seq_len: int,
        chunksize: int,
        subset: str,
        n_train: int,
        n_total: int,
        scaler: Optional[StandardScaler] = None,
        batch_size: int = 128,
    ):
        super().__init__()
        assert subset in ("train", "val")
        self.files = files
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.seq_len = seq_len
        self.chunksize = chunksize
        self.subset = subset
        self.n_train = n_train
        self.n_total = n_total
        self.scaler = scaler
        self.batch_size = int(batch_size)

    def __iter__(self):
        start_idx = 0
        target_lo = 0 if self.subset == "train" else self.n_train
        target_hi = self.n_train if self.subset == "train" else self.n_total

        buf_X, buf_y = [], []

        for X_chunk, y_chunk in iter_chunked_windows_and_labels(
            self.files, self.feature_cols, self.label_col, self.seq_len, self.chunksize
        ):
            B = len(X_chunk)
            if B == 0:
                continue

            end_idx = start_idx + B
            take_lo = max(target_lo, start_idx)
            take_hi = min(target_hi, end_idx)
            if take_hi <= take_lo:
                start_idx = end_idx
                continue

            s = take_lo - start_idx
            e = take_hi - start_idx
            Xb = X_chunk[s:e]
            yb = y_chunk[s:e]

            if self.scaler is not None:
                k, t, f = Xb.shape
                Xb = self.scaler.transform(Xb.reshape(k * t, f)).reshape(k, t, f)

            buf_X.append(Xb)
            buf_y.append(yb)

            # Flush in mini-batches
            while True:
                total_k = sum(x.shape[0] for x in buf_X)
                if total_k < self.batch_size:
                    break
                need = self.batch_size
                out_X, out_y = [], []
                while need > 0 and buf_X:
                    x0, y0 = buf_X[0], buf_y[0]
                    if x0.shape[0] <= need:
                        out_X.append(x0); out_y.append(y0)
                        buf_X.pop(0); buf_y.pop(0)
                        need -= x0.shape[0]
                    else:
                        out_X.append(x0[:need]); out_y.append(y0[:need])
                        buf_X[0] = x0[need:]; buf_y[0] = y0[need:]
                        need = 0
                X_out = np.concatenate(out_X, axis=0)
                y_out = np.concatenate(out_y, axis=0)
                yield torch.from_numpy(X_out.astype(np.float32)), torch.from_numpy(y_out.astype(np.int64))

            start_idx = end_idx

        if buf_X:
            X_out = np.concatenate(buf_X, axis=0)
            y_out = np.concatenate(buf_y, axis=0)
            yield torch.from_numpy(X_out.astype(np.float32)), torch.from_numpy(y_out.astype(np.int64))

# -----------------------------
# Train / Eval Epochs (Iterable) with accumulation
# -----------------------------

def train_epoch_iter(model: nn.Module,
                     loader: DataLoader,
                     device: str,
                     criterion: nn.Module,
                     optimizer: torch.optim.Optimizer,
                     accumulate: int = 1) -> Tuple[float, float, int]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    n = 0
    step = 0
    optimizer.zero_grad(set_to_none=True)
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb) / max(1, accumulate)
        loss.backward()

        step += 1
        if step % max(1, accumulate) == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        bs = xb.shape[0]
        total_loss += float(loss.item()) * bs * max(1, accumulate)  # undo the /accumulate for reporting
        total_correct += int((logits.argmax(dim=-1) == yb).sum().item())
        n += bs

    # Flush leftover grads
    if step % max(1, accumulate) != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    avg_loss = total_loss / max(1, n)
    acc = total_correct / max(1, n)
    return avg_loss, acc, n

def eval_epoch_iter(model: nn.Module,
                    loader: DataLoader,
                    device: str,
                    criterion: nn.Module) -> Tuple[float, float, int]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            bs = xb.shape[0]
            total_loss += float(loss.item()) * bs
            total_correct += int((logits.argmax(dim=-1) == yb).sum().item())
            n += bs
    return total_loss / max(1, n), total_correct / max(1, n), n

# -----------------------------
# Argparse (new + legacy aliases)
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Chunked LSTM training (directory or single CSV).")
    # IO
    p.add_argument("--data", type=str, default="eth_1m_data",
                   help="Path to a CSV file or a directory of CSVs (default: eth_1m_data)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Where to save artifacts. Defaults to SM_MODEL_DIR if set, else ./model")
    # Columns & sequences
    p.add_argument("--label-col", type=str, default="label",
                   help="Ground-truth label column (int classes). Default: 'label'")
    p.add_argument("--price-col", type=str, default="close", help="Optional price column (unused for loss).")
    p.add_argument("--time-col", type=str, default=None, help="Optional timestamp column (not required).")
    p.add_argument("--feature-cols", type=str, nargs="*", default=None,
                   help="Explicit list of feature columns; if omitted, auto-detect numeric minus label/price/time from a sample.")
    p.add_argument("--seq-len", type=int, default=60, help="Sequence length for windowing.")
    # Legacy alias for --seq-len
    p.add_argument("--window_size", dest="seq_len", type=int, help=argparse.SUPPRESS)

    # Scaling
    p.add_argument("--scale-features", type=int, default=1, help="1 to fit/apply StandardScaler; 0 to disable.")
    # Model hyperparams (+ legacy aliases)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--hidden_size", dest="hidden-size", type=int, help=argparse.SUPPRESS)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--bidirectional", type=int, default=0)
    p.add_argument("--num-classes", type=int, default=3)
    # Training hyperparams (+ legacy aliases)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--batch_size", dest="batch-size", type=int, help=argparse.SUPPRESS)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--val-frac", type=float, default=0.2, help="Fraction of sequences at the END used for validation.")
    p.add_argument("--val_months", type=int, default=None,
                   help="Legacy: use last N CSV files as validation (overrides --val-frac when set).")
    p.add_argument("--shuffle-train", type=int, default=0, help="(unused for streaming; kept for API parity)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-class-weights", type=int, default=1)
    p.add_argument("--accumulate", type=int, default=1, help="Gradient accumulation steps (legacy flag supported).")
    return p

# -----------------------------
# Main
# -----------------------------

def main():
    load_dotenv()
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    # Normalize hyphenated dest names that argparse stores as attributes with dashes replaced by underscores
    # e.g. --hidden-size becomes args.hidden_size
    if hasattr(args, "hidden-size"):
        args.hidden_size = getattr(args, "hidden-size")
    if hasattr(args, "batch-size"):
        args.batch_size = getattr(args, "batch-size")

    # Resolve output dir (SageMaker-compatible)
    output_dir = args.output_dir or os.environ.get("SM_MODEL_DIR", "./model")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Files to read
    files = list_csvs_sorted(args.data)

    # Infer feature columns from a small sample (first file head)
    sample = pd.read_csv(files[0], nrows=2000)
    feature_cols = infer_feature_cols_from_sample(
        sample_df=sample,
        label_col=args.label_col,
        time_col=args.time_col,
        price_col=args.price_col,
        explicit_feature_cols=args.feature_cols,
    )
    print(f"[INFO] Using {len(feature_cols)} feature columns")

    if args.label_col not in sample.columns:
        raise ValueError(f"Label column '{args.label_col}' not found in sample. "
                         f"Pass --label-col or create a '{args.label_col}' column.")

    # Chunk size from .env
    chunk_size_env = os.getenv("CHUNK_SIZE", "200000")
    try:
        chunksize = max(10_000, int(chunk_size_env))
    except Exception:
        chunksize = 200_000
    print(f"[INFO] CHUNK_SIZE = {chunksize}")

    # -----------------------------
    # PASS 0: Count sequences & decide split
    # -----------------------------
    # Default: index-based split by --val-frac.
    total_sequences = count_total_sequences(
        files=files,
        feature_cols=feature_cols,
        label_col=args.label_col,
        seq_len=args.seq_len,
        chunksize=chunksize,
    )
    if total_sequences == 0:
        raise RuntimeError("No sequences produced. Check seq-len and your data.")

    if args.val_months and len(files) > 1 and args.val_months > 0:
        # Last N files' sequences are validation; rest are training
        n_val = count_total_sequences(
            files=files[-int(args.val_months):],
            feature_cols=feature_cols,
            label_col=args.label_col,
            seq_len=args.seq_len,
            chunksize=chunksize,
        )
        n_train = max(1, total_sequences - n_val)
        print(f"[INFO] Validation by last {args.val_months} file(s): train={n_train} | val={n_val}")
    else:
        n_val = max(1, int(total_sequences * float(args.val_frac)))
        n_train = max(1, total_sequences - n_val)
        print(f"[INFO] Validation by fraction {args.val_frac:.3f}: train={n_train} | val={n_val}")

    # -----------------------------
    # PASS 1: Fit scaler on TRAIN portion only
    # -----------------------------
    scaler = None
    if int(args.scale_features) == 1:
        if StandardScaler is None:
            raise RuntimeError("scikit-learn not available but --scale-features=1 was set")
        scaler = StandardScaler()
        seen = 0
        for X_chunk, _ in iter_chunked_windows_and_labels(
            files, feature_cols, args.label_col, args.seq_len, chunksize
        ):
            if seen >= n_train:
                break
            B = len(X_chunk)
            if B == 0:
                continue
            take = min(B, n_train - seen)
            Xb = X_chunk[:take]
            b, t, f = Xb.shape
            scaler.partial_fit(Xb.reshape(b * t, f))
            seen += take
        print("[INFO] Scaler fitted on training portion.")
    else:
        print("[INFO] Feature scaling disabled.")

    # -----------------------------
    # Build model, loss, optimizer
    # -----------------------------
    input_size = len(feature_cols)
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=int(args.hidden_size),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        bidirectional=bool(args.bidirectional),
        num_classes=int(args.num_classes),
    ).to(device)

    # Class weights (stream over train portion)
    if int(args.use_class_weights) == 1:
        counts = np.zeros(args.num_classes, dtype=np.int64)
        seen = 0
        for _, y_chunk in iter_chunked_windows_and_labels(
            files, feature_cols, args.label_col, args.seq_len, chunksize
        ):
            if seen >= n_train:
                break
            B = len(y_chunk)
            if B == 0:
                continue
            take = min(B, n_train - seen)
            yb = y_chunk[:take]
            classes, cnt = np.unique(yb, return_counts=True)
            for c, k in zip(classes, cnt):
                if 0 <= int(c) < args.num_classes:
                    counts[int(c)] += int(k)
            seen += take
        weights = np.zeros(args.num_classes, dtype=np.float32)
        for c in range(args.num_classes):
            weights[c] = 0.0 if counts[c] == 0 else 1.0 / counts[c]
        if weights.sum() > 0:
            weights = weights * (len(weights) / (weights.sum()))
        cw = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"[INFO] Class weights: {np.round(weights,4).tolist()}")
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    # -----------------------------
    # Dataloaders (streaming)
    # -----------------------------
    train_ds = SeqIterableDataset(
        files=files,
        feature_cols=feature_cols,
        label_col=args.label_col,
        seq_len=args.seq_len,
        chunksize=chunksize,
        subset="train",
        n_train=n_train,
        n_total=total_sequences,
        scaler=scaler,
        batch_size=int(args.batch_size),
    )
    val_ds = SeqIterableDataset(
        files=files,
        feature_cols=feature_cols,
        label_col=args.label_col,
        seq_len=args.seq_len,
        chunksize=chunksize,
        subset="val",
        n_train=n_train,
        n_total=total_sequences,
        scaler=scaler,
        batch_size=int(args.batch_size),
    )

    train_loader = DataLoader(train_ds, batch_size=None, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=None, num_workers=0, pin_memory=True)

    # -----------------------------
    # Training loop (best val loss)
    # -----------------------------
    best_val_loss = float("inf")
    best_state = None
    for epoch in range(1, int(args.epochs) + 1):
        tr_loss, tr_acc, tr_n = train_epoch_iter(
            model, train_loader, device, criterion, optimizer, accumulate=int(args.accumulate)
        )
        va_loss, va_acc, va_n = eval_epoch_iter(model, val_loader, device, criterion)

        print(f"[EPOCH {epoch:03d}] "
              f"train_n={tr_n} train_loss={tr_loss:.5f} train_acc={tr_acc:.4f} | "
              f"val_n={va_n} val_loss={va_loss:.5f} val_acc={va_acc:.4f}")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # -----------------------------
    # Save artifacts
    # -----------------------------
    weights_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"[SAVE] Weights -> {weights_path}")

    scaler_path = None
    if scaler is not None:
        scaler_path = os.path.join(output_dir, "scaler.joblib")
        if joblib is not None:
            joblib.dump(scaler, scaler_path)
        else:
            import pickle
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
        print(f"[SAVE] Scaler -> {scaler_path}")

    meta = ModelMeta(
        input_size=input_size,
        hidden_size=int(args.hidden_size),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        bidirectional=bool(args.bidirectional),
        num_classes=int(args.num_classes),
        model_state_path="model.pt",
        scaler_path="scaler.joblib" if scaler_path is not None else None,
        feature_scaling=bool(scaler_path is not None),
        framework="pytorch",
        model_type="lstm_classifier",
    )
    meta_path = os.path.join(output_dir, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(asdict(meta), f, indent=2)
    print(f"[SAVE] Meta -> {meta_path}")

    print("[DONE] Training complete.")

if __name__ == "__main__":
    main()
