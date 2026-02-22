import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from utils import compute_features, build_windows_from_flat, FEATURE_COLUMNS
from simulator import Bar, SimulationConfig, PortfolioSimulator
from models import ModelMeta, build_model_from_meta, load_model_state


# Fixtures
@pytest.fixture
def raw_df():
    n = 300
    base = np.linspace(100.0, 101.0, n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="T", tz="UTC"),
        "open": base,
        "high": base * 1.001,
        "low": base * 0.999,
        "close": base * 1.0005,
        "volume": np.linspace(1000, 1200, n),
    })


@pytest.fixture
def features_df(raw_df):
    return compute_features(raw_df)


@pytest.fixture
def windows_exact(features_df):
    arr = features_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False)
    return build_windows_from_flat(arr, seq_len=10)


@pytest.fixture
def bars_list():
    # simple rising market with ATR provided
    bars = []
    for i in range(15):
        price = 100 + i
        bars.append(Bar(open=price, high=price + 1, low=price - 1, close=price + 0.5, atr=0.5))
    return bars


@pytest.fixture
def model_meta_tmp(tmp_path):
    meta = ModelMeta(
        input_size=2,
        hidden_size=4,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        bidirectional=False,
        num_classes=3,
        task="classification",
        model_type="lstm_attention",
        feature_cols=["f1", "f2"],
        window_size=3,
    )
    model = build_model_from_meta(meta)
    state_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), state_path)
    return meta, state_path


# Tests

def test_compute_features_idempotent(features_df):
    df2 = compute_features(features_df.copy())
    assert list(features_df.columns) == list(df2.columns)
    # check a few columns match
    cols = FEATURE_COLUMNS[:5]
    assert np.allclose(features_df[cols].to_numpy(), df2[cols].to_numpy(), equal_nan=True)


def test_build_windows_edge_cases(features_df):
    arr_short = features_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False)[:5]
    windows = build_windows_from_flat(arr_short, seq_len=10)
    assert windows.shape == (0, 10, arr_short.shape[1])
    arr_exact = features_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False)[:10]
    windows2 = build_windows_from_flat(arr_exact, seq_len=10)
    assert windows2.shape == (1, 10, arr_exact.shape[1])


def test_simulator_atr_vs_fixed(bars_list):
    signals = [2] * len(bars_list)
    cfg_atr = SimulationConfig(start_capital=1000, use_atr_stops=True, atr_tp_mult=1.0, atr_sl_mult=0.5, fee_pct=0.0, slippage_pct=0.0)
    cfg_fixed = SimulationConfig(start_capital=1000, use_atr_stops=False, tp_pct=0.01, sl_pct=0.005, fee_pct=0.0, slippage_pct=0.0)
    sim_atr = PortfolioSimulator(cfg_atr)
    sim_fixed = PortfolioSimulator(cfg_fixed)
    for bar, sig in zip(bars_list, signals):
        sim_atr.step(bar, signal=sig)
        sim_fixed.step(bar, signal=sig)
    sim_atr.finalize(bars_list[-1].close)
    sim_fixed.finalize(bars_list[-1].close)
    # equity should differ between ATR and fixed stops
    assert sim_atr.last_equity != sim_fixed.last_equity


def test_meta_round_trip_predictions(model_meta_tmp):
    meta, state_path = model_meta_tmp
    model_a = build_model_from_meta(meta)
    model_b = build_model_from_meta(meta)
    # load saved state into both to ensure consistent init
    load_model_state(model_a, str(state_path))
    load_model_state(model_b, str(state_path))
    x = torch.randn(1, meta.window_size, meta.input_size)
    with torch.no_grad():
        out_a = model_a(x)
        out_b = model_b(x)
    assert torch.allclose(out_a, out_b)


def test_windows_fixture_shape(windows_exact):
    assert windows_exact.ndim == 3
    assert windows_exact.shape[1] == 10
