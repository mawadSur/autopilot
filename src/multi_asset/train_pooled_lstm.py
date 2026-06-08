"""Train ONE pooled LSTM across the whole multi-asset universe.

Mirrors the discipline of ``src/crypto_training/train_xgboost.py`` — global
time-ordered split (no look-ahead), scaler fit on train only, an honest
test-gate that refuses to bless a threshold the test slice doesn't support —
but for sequences + the asset-embedding network in
:class:`multi_asset.model.PooledLSTMClassifier`.

Artifacts written to ``--out``::

    model.pt        # state_dict of the PooledLSTMClassifier
    scaler.joblib   # StandardScaler fit on TRAIN feature rows only
    meta.json       # config, asset vocab, feature_cols, window, threshold, metrics

CLI::

    ./.venv/bin/python src/multi_asset/train_pooled_lstm.py \\
        --dataset data/pooled/pooled_1d.parquet --out model_multi/pooled_1d_v1/

Research / SHADOW only — produces a model, never an order.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None
from sklearn.preprocessing import StandardScaler

from multi_asset.model import PooledLSTMClassifier
from multi_asset.sequences import SequenceBatch, build_asset_vocab, build_sequences

_META_COLS = ("timestamp", "asset_id", "asset_class", "label")


# --------------------------------------------------------------------------- #
# Reproducibility / device
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 1337) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_pooled(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet" and path.exists():
        return pd.read_parquet(path)
    csv = path.with_suffix(".csv")
    if csv.exists():
        return pd.read_csv(csv)
    if path.exists():
        return pd.read_csv(path)
    raise FileNotFoundError(f"No pooled dataset at {path} (or its .csv fallback).")


def feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in _META_COLS]


def time_split(
    df: pd.DataFrame, *, val_frac: float = 0.15, test_frac: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the pooled table on the GLOBAL time axis (no look-ahead).

    Boundaries are chosen at row quantiles of the time-sorted frame, then rows
    are assigned by their timestamp so an entire instant lands in one split.
    """
    d = df.assign(_k=pd.to_datetime(df["timestamp"], utc=True)).sort_values("_k").reset_index(drop=True)
    n = len(d)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError(f"Not enough rows ({n}) for val={val_frac}+test={test_frac}")
    val_start_ts = d["_k"].iloc[n_train]
    test_start_ts = d["_k"].iloc[n_train + n_val]
    train = d[d["_k"] < val_start_ts]
    val = d[(d["_k"] >= val_start_ts) & (d["_k"] < test_start_ts)]
    test = d[d["_k"] >= test_start_ts]
    return (train.drop(columns="_k").reset_index(drop=True),
            val.drop(columns="_k").reset_index(drop=True),
            test.drop(columns="_k").reset_index(drop=True))


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on flattened [N*W, F] train features."""
    n, w, f = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(n * w, f))
    return scaler


def apply_scaler(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    if X.shape[0] == 0:
        return X
    n, w, f = X.shape
    return scaler.transform(X.reshape(n * w, f)).reshape(n, w, f).astype(np.float32)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def _loader(batch: SequenceBatch, scaler: StandardScaler, *, batch_size: int, shuffle: bool) -> DataLoader:
    X = apply_scaler(scaler, batch.X)
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(batch.y.astype(np.int64)),
        torch.from_numpy(batch.asset_idx.astype(np.int64)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _run_epoch(model, loader, device, criterion, optimizer=None) -> float:
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    for xb, yb, ab in loader:
        xb, yb, ab = xb.to(device), yb.to(device), ab.to(device)
        with torch.set_grad_enabled(train):
            logits = model(xb, ab)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


@torch.no_grad()
def _predict_proba(model, batch: SequenceBatch, scaler, device, *, batch_size: int = 512) -> np.ndarray:
    """P(class==1) for every sequence in ``batch``."""
    if len(batch) == 0:
        return np.empty((0,), dtype=np.float64)
    loader = _loader(batch, scaler, batch_size=batch_size, shuffle=False)
    model.eval()
    out: List[np.ndarray] = []
    for xb, _, ab in loader:
        xb, ab = xb.to(device), ab.to(device)
        probs = torch.softmax(model(xb, ab), dim=-1)[:, 1]
        out.append(probs.cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------- #
# Evaluation / threshold gate
# --------------------------------------------------------------------------- #
def sweep_threshold(
    y: np.ndarray, proba: np.ndarray, *, min_trades: int, lo: float = 0.50, hi: float = 0.95, step: float = 0.05
) -> Tuple[Optional[float], float, int]:
    """Pick the threshold with the best precision (win-rate) clearing min_trades."""
    best_thr: Optional[float] = None
    best_wr, best_n = 0.0, 0
    for thr in np.arange(lo, hi + 1e-9, step):
        mask = proba >= thr
        n = int(mask.sum())
        if n < min_trades:
            continue
        wr = float(y[mask].mean())
        if wr > best_wr:
            best_wr, best_n, best_thr = wr, n, float(thr)
    return best_thr, best_wr, best_n


def per_asset_winrate(
    asset_idx: np.ndarray, y: np.ndarray, proba: np.ndarray, threshold: float, idx_to_asset: Dict[int, str]
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    mask = proba >= threshold
    for a in np.unique(asset_idx):
        am = mask & (asset_idx == a)
        n = int(am.sum())
        wr = float(y[am].mean()) if n else 0.0
        out[idx_to_asset[int(a)]] = {"win_rate": wr, "n_trades": n}
    return out


@dataclass
class TrainSummary:
    dataset_path: str
    output_dir: str
    n_assets: int
    asset_ids: List[str]
    feature_count: int
    window: int
    n_train_seq: int
    n_val_seq: int
    n_test_seq: int
    epochs_ran: int
    best_val_loss: float
    optimal_threshold: Optional[float]
    threshold_status: str
    test_winrate_at_optimal: float
    test_ntrades_at_optimal: float
    per_asset_test: Dict[str, Dict[str, float]]


# --------------------------------------------------------------------------- #
# Orchestration (importable for tests)
# --------------------------------------------------------------------------- #
def train(
    *,
    dataset_path: Path,
    output_dir: Path,
    window: int = 32,
    embed_dim: int = 8,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    min_test_winrate: float = 0.55,
    min_test_ntrades: int = 20,
    gap_multiplier: float = 5.0,
    seed: int = 1337,
    force_cpu: bool = False,
    verbose: bool = True,
) -> TrainSummary:
    set_seed(seed)
    device = get_device(force_cpu=force_cpu)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_pooled(Path(dataset_path))
    feat_cols = feature_columns(df)
    vocab = build_asset_vocab(df)
    idx_to_asset = {v: k for k, v in vocab.items()}
    num_classes = int(df["label"].nunique())
    num_classes = max(num_classes, 2)

    train_df, val_df, test_df = time_split(df, val_frac=val_frac, test_frac=test_frac)
    seq_kw = dict(feature_cols=feat_cols, asset_to_idx=vocab, window=window, gap_multiplier=gap_multiplier)
    train_seq = build_sequences(train_df, **seq_kw)
    val_seq = build_sequences(val_df, **seq_kw)
    test_seq = build_sequences(test_df, **seq_kw)
    if len(train_seq) == 0 or len(val_seq) == 0:
        raise ValueError(
            f"Too few sequences (train={len(train_seq)}, val={len(val_seq)}). "
            f"Reduce --window (now {window}) or backfill more history."
        )

    scaler = fit_scaler(train_seq.X)
    train_loader = _loader(train_seq, scaler, batch_size=batch_size, shuffle=True)
    val_loader = _loader(val_seq, scaler, batch_size=batch_size, shuffle=False)

    model = PooledLSTMClassifier(
        n_features=len(feat_cols), n_assets=len(vocab), embed_dim=embed_dim,
        hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, num_classes=num_classes,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(train_seq.y, num_classes).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    epochs_ran = 0
    since_improved = 0
    for ep in range(1, epochs + 1):
        epochs_ran = ep
        tr_loss = _run_epoch(model, train_loader, device, criterion, optimizer)
        va_loss = _run_epoch(model, val_loader, device, criterion, optimizer=None)
        if verbose:
            print(f"epoch {ep:3d}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")
        if va_loss < best_val - 1e-5:
            best_val = va_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= patience:
                if verbose:
                    print(f"early stop @ epoch {ep} (no val improvement in {patience})")
                break

    model.load_state_dict(best_state)

    # Honest test-gate on the held-out tail.
    opt_thr = None
    status = "no_test_data"
    test_wr, test_n = 0.0, 0
    per_asset: Dict[str, Dict[str, float]] = {}
    if len(test_seq) > 0:
        proba = _predict_proba(model, test_seq, scaler, device)
        cand_thr, cand_wr, cand_n = sweep_threshold(test_seq.y, proba, min_trades=min_test_ntrades)
        if cand_thr is not None and cand_wr >= min_test_winrate and cand_n >= min_test_ntrades:
            opt_thr, test_wr, test_n, status = cand_thr, cand_wr, cand_n, "passed"
        else:
            status = "test_gate_failed"
            opt_thr_fallback = cand_thr if cand_thr is not None else 0.5
            test_wr, test_n = cand_wr, cand_n
            per_asset = per_asset_winrate(test_seq.asset_idx, test_seq.y, proba, opt_thr_fallback, idx_to_asset)
        if status == "passed":
            per_asset = per_asset_winrate(test_seq.asset_idx, test_seq.y, proba, opt_thr, idx_to_asset)

    # Persist artifacts.
    torch.save(model.state_dict(), output_dir / "model.pt")
    if joblib is not None:
        joblib.dump(scaler, output_dir / "scaler.joblib")
    meta = {
        "model_type": "pooled_lstm",
        "framework": "pytorch",
        "config": model.config,
        "window": window,
        "gap_multiplier": gap_multiplier,
        "feature_cols": feat_cols,
        "asset_vocab": vocab,
        "num_classes": num_classes,
        "optimal_threshold": opt_thr,
        "threshold_status": status,
        "test_winrate_at_optimal": test_wr,
        "test_ntrades_at_optimal": test_n,
        "min_test_winrate": min_test_winrate,
        "min_test_ntrades": min_test_ntrades,
        "best_val_loss": best_val,
        "epochs_ran": epochs_ran,
        "scaler_path": "scaler.joblib",
        "model_state_path": "model.pt",
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return TrainSummary(
        dataset_path=str(dataset_path),
        output_dir=str(output_dir),
        n_assets=len(vocab),
        asset_ids=sorted(vocab),
        feature_count=len(feat_cols),
        window=window,
        n_train_seq=len(train_seq),
        n_val_seq=len(val_seq),
        n_test_seq=len(test_seq),
        epochs_ran=epochs_ran,
        best_val_loss=best_val,
        optimal_threshold=opt_thr,
        threshold_status=status,
        test_winrate_at_optimal=test_wr,
        test_ntrades_at_optimal=test_n,
        per_asset_test=per_asset,
    )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the pooled multi-asset LSTM.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--embed-dim", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--min-test-winrate", type=float, default=0.55)
    p.add_argument("--min-test-ntrades", type=int, default=20)
    p.add_argument("--gap-multiplier", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--force-cpu", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    summary = train(
        dataset_path=Path(args.dataset), output_dir=Path(args.out),
        window=args.window, embed_dim=args.embed_dim, hidden_size=args.hidden_size,
        num_layers=args.num_layers, dropout=args.dropout, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, patience=args.patience,
        val_frac=args.val_frac, test_frac=args.test_frac,
        min_test_winrate=args.min_test_winrate, min_test_ntrades=args.min_test_ntrades,
        gap_multiplier=args.gap_multiplier, seed=args.seed, force_cpu=args.force_cpu,
    )
    print(json.dumps(asdict(summary), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
