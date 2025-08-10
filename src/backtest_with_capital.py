# backtest_with_capital.py
from __future__ import annotations
import argparse, os
from typing import Dict, Any, Optional
import numpy as np
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
        last = hn[-1]
        return self.head(last).squeeze(-1)


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


def _max_drawdown(curve: np.ndarray) -> float:
    if curve.size == 0: return 0.0
    peaks = np.maximum.accumulate(curve)
    dd = (curve / peaks) - 1.0
    return float(dd.min())


def backtest_with_capital_stream(
    data_dir: str,
    initial_capital: float = 10_000.0,
    fee_bps: float = 3.0,
    max_leverage: float = 1.0,
    warmup_bars: int = 5000,
) -> Dict[str, Any]:

    model, scaler, meta, spec = load_artifacts()
    buy_th = float(meta.get("buy_threshold", 0.5))
    sell_th = float(meta.get("sell_threshold", 0.5))

    cash = initial_capital
    qty = 0.0
    equity_curve = []

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
        new_mask = feat.index >= chunk.index[0]
        feat_new = feat.loc[new_mask]
        if feat_new.empty:
            continue

        for t in feat_new.index:
            sub = feat.loc[:t]
            X, px = make_model_window(sub, spec=spec, scaler=scaler)
            if X is None:
                equity_curve.append(cash + qty * float(sub["close"].iloc[-1]))
                continue

            with torch.no_grad():
                prob = torch.sigmoid(model(torch.from_numpy(X))).item()
            signal, _ = proba_to_signal(prob, buy_th, sell_th)

            if signal == "BUY" and qty == 0.0:
                notional = min(cash * max_leverage, cash)
                qty = notional / px
                fee = notional * (fee_bps / 10000.0)
                cash -= (notional + fee)
            elif signal == "SELL" and qty > 0.0:
                proceeds = qty * px
                fee = proceeds * (fee_bps / 10000.0)
                cash += (proceeds - fee)
                qty = 0.0

            equity_curve.append(cash + qty * float(sub["close"].iloc[-1]))

    # liquidate at the end if needed
    if qty > 0.0 and len(raw_buffer):
        final_px = float(raw_buffer["close"].iloc[-1])
        proceeds = qty * final_px
        fee = proceeds * (fee_bps / 10000.0)
        cash += (proceeds - fee)
        qty = 0.0
        equity_curve.append(cash)

    curve = np.array(equity_curve, dtype=float)
    total_return = (cash / initial_capital) - 1.0
    mdd = _max_drawdown(curve) if curve.size else 0.0
    return {
        "final_equity": float(cash),
        "total_return_pct": float(total_return * 100.0),
        "max_drawdown_pct": float(mdd * 100.0),
        "points": int(curve.size),
    }


def main():
    ap = argparse.ArgumentParser(description="Backtest with capital & fees (chunked)")
    ap.add_argument("-d","--data-dir", default=os.getenv("DATA_DIR","eth_1m_data"))
    ap.add_argument("--initial", type=float, default=float(os.getenv("INITIAL_CAPITAL","10000")))
    ap.add_argument("--fee-bps", type=float, default=float(os.getenv("FEE_BPS","3.0")))
    ap.add_argument("--max-lev", type=float, default=float(os.getenv("MAX_LEVERAGE","1.0")))
    ap.add_argument("--warmup-bars", type=int, default=int(os.getenv("WARMUP_BARS","5000")))
    args = ap.parse_args()

    res = backtest_with_capital_stream(
        args.data_dir, initial_capital=args.initial, fee_bps=args.fee_bps,
        max_leverage=args.max_lev, warmup_bars=args.warmup_bars,
    )
    print("\n============= Backtest (with capital) =============")
    print(f"Curve points    : {res['points']}")
    print(f"Final equity    : ${res['final_equity']:.2f}")
    print(f"Total return    : {res['total_return_pct']:.3f}%")
    print(f"Max drawdown    : {res['max_drawdown_pct']:.3f}%")
    print("===================================================\n")


if __name__ == "__main__":
    main()
