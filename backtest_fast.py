# backtest_fast.py
from __future__ import annotations
import argparse, os, sys
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
from joblib import load

from utils import (
    load_meta, FeatureSpec, load_ohlc_chunks, build_features,
    make_model_window, proba_to_signal,
)

# Reuse the same model arch you trained with (bidirectional configurable via meta)
import torch.nn as nn
class LSTMModel(nn.Module):
    def __init__(self, input_size:int, hidden_size:int=64, num_layers:int=2, dropout:float=0.1, bidirectional:bool=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers>1 else 0.0, bidirectional=bidirectional
        )
        out = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Linear(out, out), nn.ReLU(), nn.Linear(out, 1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hn, _) = self.lstm(x)
        last = hn[-1]
        return self.head(last).squeeze(-1)  # logits


def load_artifacts(model_path="model.pt", scaler_path="scaler.joblib", meta_path="model_meta.json"):
    meta = load_meta(meta_path)
    spec = FeatureSpec(meta["feature_cols"], int(meta["window_size"]))
    model = LSTMModel(
        input_size=len(spec.feature_cols),
        hidden_size=int(meta.get("hidden_size",64)),
        num_layers=int(meta.get("num_layers",2)),
        dropout=float(meta.get("dropout",0.1)),
        bidirectional=bool(meta.get("bidirectional",False)),
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    scaler = None
    if os.path.exists(scaler_path):
        try: scaler = load(scaler_path)
        except Exception: scaler = None
    return model, scaler, meta, spec


def backtest_stream(
    data_dir: str,
    fee_bps: float = 3.0,
    warmup_bars: int = 5000,   # keep this many bars of history for indicators across chunk boundaries
) -> Dict[str, Any]:

    model, scaler, meta, spec = load_artifacts()
    buy_th = float(meta.get("buy_threshold", 0.5))
    sell_th = float(meta.get("sell_threshold", 0.5))

    position = 0  # 0=flat, 1=long
    last_price: Optional[float] = None
    equity = 1.0
    trades = 0
    bars = 0

    # rolling raw buffer
    raw_buffer = pd.DataFrame(columns=["open","high","low","close","volume"])

    print(f"📦 Reading chunks from: {data_dir}")
    for chunk in load_ohlc_chunks(data_dir):
        # append + keep only tail(warmup_bars + len(chunk)) to avoid unbounded memory
        raw_buffer = pd.concat([raw_buffer, chunk])
        raw_buffer = raw_buffer[~raw_buffer.index.duplicated(keep="last")].sort_index()
        if len(raw_buffer) > warmup_bars + len(chunk):
            raw_buffer = raw_buffer.tail(warmup_bars + len(chunk))

        # Build features for the buffer, then process only the NEW time range
        feat = build_features(raw_buffer, compat_inf_to_zero=True)
        new_mask = feat.index >= chunk.index[0]
        feat_new = feat.loc[new_mask]
        if feat_new.empty:
            continue

        for t in feat_new.index:
            # up-to-time window
            sub = feat.loc[:t]
            X, px = make_model_window(sub, spec=spec, scaler=scaler)
            if X is None:
                continue
            with torch.no_grad():
                prob = torch.sigmoid(model(torch.from_numpy(X))).item()
            signal, _ = proba_to_signal(prob, buy_th, sell_th)

            if signal == "BUY" and position == 0:
                position = 1
                last_price = px
                equity *= (1.0 - fee_bps/10000.0)
                trades += 1
            elif signal == "SELL" and position == 1:
                ret = (px - (last_price or px)) / (last_price or px)
                equity *= (1.0 + ret)
                equity *= (1.0 - fee_bps/10000.0)
                position = 0
                last_price = None
                trades += 1
            bars += 1

    # close open pos at the very end
    if position == 1 and last_price is not None and len(raw_buffer):
        final_px = float(raw_buffer["close"].iloc[-1])
        ret = (final_px - last_price) / last_price
        equity *= (1.0 + ret)

    return {
        "equity": float(equity),
        "total_return_pct": float((equity - 1.0) * 100.0),
        "trades": int(trades),
        "bars": int(bars),
    }


def main():
    ap = argparse.ArgumentParser(description="Fast long/flat backtest (chunked)")
    ap.add_argument("-d","--data-dir", default=os.getenv("DATA_DIR","eth_1m_data"))
    ap.add_argument("--fee-bps", type=float, default=float(os.getenv("FEE_BPS","3.0")))
    ap.add_argument("--warmup-bars", type=int, default=int(os.getenv("WARMUP_BARS","5000")))
    args = ap.parse_args()

    res = backtest_stream(args.data_dir, fee_bps=args.fee_bps, warmup_bars=args.warmup_bars)
    print("\n================ Backtest (fast) ================")
    print(f"Bars processed  : {res['bars']}")
    print(f"Trades          : {res['trades']}")
    print(f"Final equity    : {res['equity']:.6f}")
    print(f"Total return    : {res['total_return_pct']:.3f}%")
    print("=================================================\n")


if __name__ == "__main__":
    main()