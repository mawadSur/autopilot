# backtest.py
from __future__ import annotations
import argparse, os
from typing import Dict, Any
import pandas as pd
import torch
from joblib import load

from utils import (
    load_meta, FeatureSpec, load_ohlc_chunks, build_features,
    make_model_window, proba_to_signal,
)

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
        return self.head(hn[-1]).squeeze(-1)


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


def run_backtest_chunked(
    data_dir: str,
    lookahead_steps: int = 10,
    stop_loss_pct: float = 0.5,  # %
    fee_pct: float = 0.075,      # %
    warmup_bars: int = 5000,
) -> Dict[str, Any]:

    model, scaler, meta, spec = load_artifacts()
    buy_th = float(meta.get("buy_threshold", 0.5))
    sell_th = float(meta.get("sell_threshold", 0.5))

    wins = losses = trades = 0
    net_pnl_pct = 0.0

    raw_buffer = pd.DataFrame(columns=["open","high","low","close","volume"])

    print(f"📦 Reading chunks from: {data_dir}")
    for chunk in load_ohlc_chunks(data_dir):
        raw_buffer = pd.concat([raw_buffer, chunk])
        raw_buffer = raw_buffer[~raw_buffer.index.duplicated(keep="last")].sort_index()
        if len(raw_buffer) > warmup_bars + len(chunk):
            raw_buffer = raw_buffer.tail(warmup_bars + len(chunk))

        feat = build_features(raw_buffer, compat_inf_to_zero=True)
        if feat.empty:
            continue
        # we need a simple row-wise loop to align with "lookahead" logic
        for i in range(len(feat) - lookahead_steps):
            sub = feat.iloc[: i+1]
            X, px = make_model_window(sub, spec=spec, scaler=scaler)
            if X is None:
                continue

            with torch.no_grad():
                prob = torch.sigmoid(model(torch.from_numpy(X))).item()
            signal, _ = proba_to_signal(prob, buy_th, sell_th)
            if signal != "BUY":
                continue

            # evaluate outcome over next N bars
            entry_price = float(sub["close"].iloc[-1])
            stop_price = entry_price * (1 - stop_loss_pct/100.0)
            future = feat.iloc[i+1 : i+1+lookahead_steps]
            if future.empty:
                continue

            low_in_period = float(future["low"].min())
            exit_price = float(future["close"].iloc[-1])  # exit at end if no stop
            if low_in_period <= stop_price:
                pnl = -stop_loss_pct - fee_pct  # entry fee assumed included
                losses += 1
            else:
                pnl = ((exit_price - entry_price) / entry_price) * 100.0
                pnl -= (2 * fee_pct)  # entry + exit fees
                if pnl >= 0: wins += 1
                else: losses += 1
            net_pnl_pct += pnl
            trades += 1

    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": (wins / trades * 100.0) if trades else 0.0,
        "net_pnl_pct": net_pnl_pct,
    }


def main():
    ap = argparse.ArgumentParser(description="Lookahead/stop-loss backtest (chunked)")
    ap.add_argument("-d","--data-dir", default=os.getenv("DATA_DIR","eth_1m_data"))
    ap.add_argument("--lookahead", type=int, default=int(os.getenv("LOOKAHEAD_STEPS","10")))
    ap.add_argument("--stop-loss-pct", type=float, default=float(os.getenv("STOP_LOSS_PCT","0.5")))
    ap.add_argument("--fee-pct", type=float, default=float(os.getenv("TRADING_FEE_PCT","0.075")))
    ap.add_argument("--warmup-bars", type=int, default=int(os.getenv("WARMUP_BARS","5000")))
    args = ap.parse_args()

    res = run_backtest_chunked(
        args.data_dir, lookahead_steps=args.lookahead, stop_loss_pct=args.stop_loss_pct,
        fee_pct=args.fee_pct, warmup_bars=args.warmup_bars,
    )
    print("\n================= Backtest (lookahead) =================")
    print(f"Trades          : {res['trades']}")
    print(f"Wins/Losses     : {res['wins']} / {res['losses']}")
    print(f"Win rate        : {res['win_rate_pct']:.2f}%")
    print(f"Net PnL         : {res['net_pnl_pct']:.3f}%")
    print("========================================================\n")


if __name__ == "__main__":
    main()