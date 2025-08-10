import os, argparse, json, random, glob
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump
from utils import build_features, FeatureSpec, load_meta, DEFAULT_FEATURE_COLS
from typing import Tuple 
SM_TRAIN_DIR = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

# ------------------------- Model -------------------------
class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_size = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_size, out_size),
            nn.ReLU(),
            nn.Linear(out_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        _, (hn, _) = self.lstm(x)
        last = hn[-1]  # (B, H) from last layer
        return self.head(last).squeeze(-1)  # logits


# ------------------------- Utilities -------------------------
def set_seeds(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_sequences(
    feat_df: pd.DataFrame, feature_cols, window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) where y is next-bar up (close[t+1] > close[t]).
    """
    Xs, ys = [], []
    close = feat_df["close"].values
    mat = feat_df[feature_cols].to_numpy(dtype=np.float32)

    for i in range(window_size, len(mat) - 1):
        Xs.append(mat[i - window_size : i, :])
        ys.append(1.0 if close[i + 1] > close[i] else 0.0)

    X = np.stack(Xs, axis=0) if Xs else np.zeros((0, window_size, len(feature_cols)), dtype=np.float32)
    y = np.array(ys, dtype=np.float32)
    return X, y


# ------------------------- Training entrypoint -------------------------
def train(
    data_path: str,
    model_out: str = "model.pt",
    scaler_out: str = "scaler.joblib",
    meta_out: str = "model_meta.json",
) -> None:
    set_seeds(42)

    # 1) Load raw OHLCV (expects columns: timestamp/index, open, high, low, close, volume)
    raw = pd.read_csv(data_path)
    if "timestamp" in raw.columns:
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
        raw.set_index("timestamp", inplace=True)

    # 2) Build features (single source of truth)
    feat = build_features(raw)

    # 3) Meta (feature schema & window)
    prev_meta = load_meta(meta_out) if os.path.exists(meta_out) else load_meta()
    feature_cols = prev_meta.get("feature_cols", DEFAULT_FEATURE_COLS)
    window_size = int(prev_meta.get("window_size", 150))

    # 4) Split, scale (fit ONLY on train!)
    X, y = build_sequences(feat, feature_cols, window_size)
    assert len(X) == len(y), "X and y must align"

    if len(X) < 100:
        raise RuntimeError("Not enough sequences to train. Provide more data.")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    B, T, F = X_train.shape
    X_train_2d = X_train.reshape(B * T, F)
    scaler.fit(X_train_2d)
    X_train = scaler.transform(X_train_2d).reshape(B, T, F)

    Bv, Tv, Fv = X_val.shape
    X_val = scaler.transform(X_val.reshape(Bv * Tv, Fv)).reshape(Bv, Tv, Fv)

    # 5) Model
    hidden_size = int(prev_meta.get("hidden_size", 64))
    num_layers  = int(prev_meta.get("num_layers", 2))
    dropout     = float(prev_meta.get("dropout", 0.1))
    bidir       = bool(prev_meta.get("bidirectional", False))

    model = LSTMModel(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidir,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def to_torch(x, y):
        return torch.tensor(x, dtype=torch.float32, device=device), torch.tensor(y, dtype=torch.float32, device=device)

    best_val = float("inf")
    for epoch in range(20):
        model.train()
        xb, yb = to_torch(X_train, y_train)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            xv, yv = to_torch(X_val, y_val)
            v_logits = model(xv)
            v_loss = criterion(v_logits, yv).item()

        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), model_out)

        print(f"epoch {epoch+1:02d} | train {loss.item():.4f} | val {v_loss:.4f}")

    # 6) Persist artifacts
    dump(scaler, scaler_out)

    # 7) Write/merge meta
    meta: Dict[str, Any] = {
        **prev_meta,
        "feature_cols": feature_cols,
        "window_size": window_size,
        "input_size": len(feature_cols),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "bidirectional": bidir,
        "scaler_type": "standard",
        # Set/keep thresholds (tune later)
        "buy_threshold": float(prev_meta.get("buy_threshold", 0.5)),
        "sell_threshold": float(prev_meta.get("sell_threshold", 0.5)),
        "label_def": "next_bar_up",
    }
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {model_out}, {scaler_out}, {meta_out}")
    

if __name__ == "__main__":
    # Example:
    # python aws_train_model.py -- just wire your own CLI as needed
    # or call train("your_data.csv")
    pass
