import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import backtest as backtest_module
from utils import (
    apply_confluence_filter,
    compute_features,
    build_windows_from_flat,
    FEATURE_COLUMNS,
    ProfitOptimizedFeatureEngineer,
)
from simulator import Bar, SimulationConfig, PortfolioSimulator
from models import ModelMeta, WeightedLastStepAttention, build_model_from_meta, load_model_state
from backtest import _bar_has_usable_l2_depth, _raw_signal_from_probs, _warn_on_sparse_l2_depth
from strategy_gate import StrategyGate


# Fixtures
@pytest.fixture
def raw_df():
    n = 300
    base = np.linspace(100.0, 101.0, n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC"),
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


def test_profit_feature_engineer_ranks_nonlinear_signal_above_noise():
    rng = np.random.default_rng(0)
    n = 4000
    nonlinear = rng.uniform(-2.0, 2.0, size=n)
    noise = rng.normal(size=n)
    next_ret = 0.001 * ((nonlinear ** 2) - np.mean(nonlinear ** 2)) + 0.0001 * rng.normal(size=n)
    next_ret = np.clip(next_ret, -0.01, 0.01)

    close = np.empty(n, dtype=np.float64)
    close[0] = 100.0
    for i in range(1, n):
        close[i] = close[i - 1] * (1.0 + next_ret[i - 1])

    df = pd.DataFrame({
        "close": close,
        "nonlinear_signal": nonlinear,
        "noise_signal": noise,
    })
    fwd = df["close"].pct_change().shift(-1)
    assert abs(float(df["nonlinear_signal"].corr(fwd))) < 0.05

    engineer = ProfitOptimizedFeatureEngineer(horizon_bars=1, min_score=0.0, walk_folds=2)
    scores = engineer.rank_features(df, ["nonlinear_signal", "noise_signal"], price_col="close")

    assert scores["nonlinear_signal"] > scores["noise_signal"]


def test_profit_feature_engineer_filter_keeps_nonlinear_signal():
    rng = np.random.default_rng(1)
    n = 3000
    nonlinear = rng.uniform(-2.0, 2.0, size=n)
    weak_linear = rng.normal(scale=0.05, size=n)
    noise = rng.normal(size=n)
    next_ret = 0.0012 * ((nonlinear ** 2) - np.mean(nonlinear ** 2)) + 0.0002 * weak_linear + 0.0001 * rng.normal(size=n)
    next_ret = np.clip(next_ret, -0.01, 0.01)

    close = np.empty(n, dtype=np.float64)
    close[0] = 100.0
    for i in range(1, n):
        close[i] = close[i - 1] * (1.0 + next_ret[i - 1])

    df = pd.DataFrame({
        "close": close,
        "nonlinear_signal": nonlinear,
        "weak_linear_signal": weak_linear,
        "noise_signal": noise,
    })

    engineer = ProfitOptimizedFeatureEngineer(
        horizon_bars=1,
        min_score=0.01,
        walk_folds=2,
        ranking_method="mutual_information",
    )
    selected = engineer.filter_features(
        df,
        ["nonlinear_signal", "weak_linear_signal", "noise_signal"],
        price_col="close",
    )

    assert engineer.selection_summary_["ranking_method"] == "mutual_information"
    assert "nonlinear_signal" in selected[:2]
    assert "noise_signal" not in selected[:1]


def test_apply_confluence_filter_rejects_long_without_liquidity_context():
    sig = apply_confluence_filter(2, {"liq_sweep_low": 0.0, "close_over_avwap_cycle": 0.004})
    assert sig == 1


def test_apply_confluence_filter_allows_long_near_cycle_avwap():
    sig = apply_confluence_filter(2, {"liq_sweep_low": 0.0, "close_over_avwap_cycle": 0.0005})
    assert sig == 2


def test_raw_signal_from_probs_uses_confluence_gate():
    sig = _raw_signal_from_probs(
        np.array([0.05, 0.10, 0.85], dtype=np.float32),
        thr_long=0.70,
        thr_short=0.70,
        margin=0.15,
        feature_row={"liq_sweep_low": 0.0, "close_over_avwap_cycle": 0.003},
    )
    assert sig == 1


def test_strategy_gate_allows_long_after_recent_sweep_low():
    gate = StrategyGate(
        thr_long=0.70,
        thr_short=0.70,
        margin=0.15,
        feature_cols=["liq_sweep_low", "close_over_avwap_cycle", "close_over_avwap_spike", "in_golden_pocket"],
        use_hard_gating=True,
    )
    window = np.array([
        [0.0, 0.0100, 0.0200, 0.0],
        [1.0, 0.0090, 0.0180, 0.0],
        [0.0, 0.0080, 0.0170, 0.0],
        [0.0, 0.0070, 0.0160, 0.0],
        [0.0, 0.0060, 0.0150, 0.0],
        [0.0, 0.0050, 0.0140, 0.0],
    ], dtype=np.float32)

    sig = gate.signal_from_probs(np.array([0.05, 0.10, 0.85], dtype=np.float32), window=window)
    assert sig == 2


def test_strategy_gate_allows_long_near_avwap_or_in_golden_pocket():
    gate = StrategyGate(
        thr_long=0.70,
        thr_short=0.70,
        margin=0.15,
        feature_cols=["liq_sweep_low", "close_over_avwap_cycle", "close_over_avwap_spike", "in_golden_pocket"],
        use_hard_gating=True,
    )
    near_spike_window = np.array([
        [0.0, 0.0100, 0.0050, 0.0],
        [0.0, 0.0100, 0.0040, 0.0],
        [0.0, 0.0100, 0.0030, 0.0],
        [0.0, 0.0100, 0.0020, 0.0],
        [0.0, 0.0100, 0.0014, 0.0],
    ], dtype=np.float32)
    golden_pocket_window = np.array([
        [0.0, 0.0100, 0.0100, 0.0],
        [0.0, 0.0100, 0.0100, 0.0],
        [0.0, 0.0100, 0.0100, 0.0],
        [0.0, 0.0100, 0.0100, 0.0],
        [0.0, 0.0100, 0.0100, 1.0],
    ], dtype=np.float32)

    assert gate.signal_from_probs(np.array([0.05, 0.10, 0.85], dtype=np.float32), window=near_spike_window) == 2
    assert gate.signal_from_probs(np.array([0.05, 0.10, 0.85], dtype=np.float32), window=golden_pocket_window) == 2


def test_strategy_gate_downgrades_long_without_any_confluence():
    gate = StrategyGate(
        thr_long=0.70,
        thr_short=0.70,
        margin=0.15,
        feature_cols=["liq_sweep_low", "close_over_avwap_cycle", "close_over_avwap_spike", "in_golden_pocket"],
        use_hard_gating=True,
    )
    window = np.array([
        [0.0, 0.0100, 0.0100, 0.0],
        [0.0, 0.0090, 0.0090, 0.0],
        [0.0, 0.0080, 0.0080, 0.0],
        [0.0, 0.0070, 0.0070, 0.0],
        [0.0, 0.0060, 0.0060, 0.0],
    ], dtype=np.float32)

    sig = gate.signal_from_probs(np.array([0.05, 0.10, 0.85], dtype=np.float32), window=window)
    assert sig == 1


def test_bar_has_usable_l2_depth_requires_bid_and_ask_curves():
    meta_row = {
        "best_bid": 99.9,
        "best_ask": 100.1,
        "bid_depth_5": 1.0,
        "ask_depth_5": 1.2,
        "vwap_bid_5": 99.85,
        "vwap_ask_5": 100.15,
    }
    assert _bar_has_usable_l2_depth(meta_row)

    meta_row["ask_depth_5"] = np.nan
    assert not _bar_has_usable_l2_depth(meta_row)


def test_warn_on_sparse_l2_depth_only_logs_above_threshold(monkeypatch):
    warnings = []

    def fake_warning(message, *args):
        warnings.append(message % args if args else message)

    monkeypatch.setattr(backtest_module.logger, "warning", fake_warning)

    _warn_on_sparse_l2_depth(total_bars=20, missing_bars=2)
    assert warnings == []

    _warn_on_sparse_l2_depth(total_bars=20, missing_bars=3)
    assert len(warnings) == 1
    assert "15.00% (3/20)" in warnings[0]


def test_simulation_config_enables_hard_gating_by_default():
    assert SimulationConfig().use_hard_gating is True


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


def test_simulator_depth_aware_slippage_scales_with_size():
    bars = [
        Bar(open=100.0, high=100.5, low=99.5, close=100.0),
        Bar(
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.0,
            best_bid=99.9,
            best_ask=100.1,
            ask_depth_5=1.0,
            ask_depth_10=2.0,
            ask_depth_20=4.0,
            bid_depth_5=1.0,
            bid_depth_10=2.0,
            bid_depth_20=4.0,
            vwap_ask_5=100.15,
            vwap_ask_10=100.25,
            vwap_ask_20=100.40,
            vwap_bid_5=99.85,
            vwap_bid_10=99.75,
            vwap_bid_20=99.60,
        ),
    ]
    cfg_small = SimulationConfig(start_capital=50.0, fee_pct=0.0, slippage_pct=0.0, use_market_depth=True)
    cfg_large = SimulationConfig(start_capital=500.0, fee_pct=0.0, slippage_pct=0.0, use_market_depth=True)

    sim_small = PortfolioSimulator(cfg_small)
    sim_large = PortfolioSimulator(cfg_large)
    for sim in (sim_small, sim_large):
        sim.step(bars[0], signal=2)
        sim.step(bars[1], signal=1)

    small_entry = next(t for t in sim_small.trade_log if t["action"] == "enter")
    large_entry = next(t for t in sim_large.trade_log if t["action"] == "enter")
    assert small_entry["price"] > 100.1
    assert large_entry["price"] > small_entry["price"]


def test_simulator_post_only_limit_entry_fills_as_maker():
    bars = [
        Bar(open=100.0, high=100.4, low=99.6, close=100.1),
        Bar(open=100.0, high=100.4, low=99.8, close=100.2, best_bid=99.9, best_ask=100.1),
    ]
    cfg = SimulationConfig(
        start_capital=100.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        post_only_entries=True,
        fallback_to_market_on_missing_book=True,
        fallback_to_market_on_post_only_miss=False,
    )
    sim = PortfolioSimulator(cfg)
    sim.step(bars[0], signal=2)
    sim.step(bars[1], signal=1)

    assert sim.pos == 1
    assert sim.entry_price == pytest.approx(99.9)
    entry = next(t for t in sim.trade_log if t["action"] == "enter")
    assert entry["liquidity_role"] == "maker"
    assert entry["order_type"] == "post_only"


def test_simulator_post_only_miss_skips_entry():
    bars = [
        Bar(open=100.0, high=100.4, low=99.6, close=100.1),
        Bar(open=100.0, high=100.3, low=100.0, close=100.2, best_bid=99.9, best_ask=100.1),
    ]
    cfg = SimulationConfig(
        start_capital=100.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        post_only_entries=True,
        fallback_to_market_on_missing_book=True,
        fallback_to_market_on_post_only_miss=False,
    )
    sim = PortfolioSimulator(cfg)
    sim.step(bars[0], signal=2)
    sim.step(bars[1], signal=1)

    assert sim.pos == 0
    assert sim.missed_entries == 1
    assert not any(t["action"] == "enter" for t in sim.trade_log)


def test_meta_round_trip_predictions(model_meta_tmp):
    meta, state_path = model_meta_tmp
    model_a = build_model_from_meta(meta)
    model_b = build_model_from_meta(meta)
    # load saved state into both to ensure consistent init
    load_model_state(model_a, str(state_path))
    load_model_state(model_b, str(state_path))
    model_a.eval()
    model_b.eval()
    x = torch.randn(1, meta.window_size, meta.input_size)
    with torch.no_grad():
        out_a = model_a(x)
        out_b = model_b(x)
    assert torch.allclose(out_a, out_b)


def test_transformer_classifier_uses_cls_token():
    meta = ModelMeta(
        input_size=4,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        bidirectional=False,
        num_classes=3,
        task="classification",
        model_type="transformer",
        feature_cols=["f1", "f2", "f3", "f4"],
        window_size=6,
        transformer_pooling="cls",
    )
    model = build_model_from_meta(meta)
    assert hasattr(model, "cls_token")
    assert model.save_hyperparameters["pooling"] == "cls"

    x = torch.randn(2, meta.window_size, meta.input_size)
    logits = model(x)
    assert logits.shape == (2, meta.num_classes)


def test_transformer_cls_token_receives_gradient():
    meta = ModelMeta(
        input_size=4,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        bidirectional=False,
        num_classes=3,
        task="classification",
        model_type="transformer",
        feature_cols=["f1", "f2", "f3", "f4"],
        window_size=6,
        transformer_pooling="cls",
    )
    model = build_model_from_meta(meta)
    x = torch.randn(2, meta.window_size, meta.input_size)
    loss = model(x).sum()
    loss.backward()

    assert model.cls_token.grad is not None
    assert torch.count_nonzero(model.cls_token.grad).item() > 0


def test_weighted_last_step_attention_biases_recent_tokens():
    pool = WeightedLastStepAttention(input_dim=4)
    with torch.no_grad():
        pool.query_proj.weight.zero_()
        pool.key_proj.weight.zero_()
        pool.gate_proj.weight.zero_()
        pool.gate_proj.bias.zero_()
        pool.recency_decay.fill_(0.5)

    hidden = torch.ones(1, 5, 4)
    _pooled, weights = pool(hidden)

    assert weights.shape == (1, 5)
    assert torch.argmax(weights, dim=-1).item() == 4
    assert torch.all(weights[0, 1:] > weights[0, :-1]).item()


def test_transformer_classifier_supports_weighted_last_pooling():
    meta = ModelMeta(
        input_size=4,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        bidirectional=False,
        num_classes=3,
        task="classification",
        model_type="transformer",
        feature_cols=["f1", "f2", "f3", "f4"],
        window_size=6,
        transformer_pooling="weighted_last",
    )
    model = build_model_from_meta(meta)
    assert model.save_hyperparameters["pooling"] == "weighted_last"
    assert model.last_step_pool is not None
    assert model.cls_token is None

    x = torch.randn(2, meta.window_size, meta.input_size)
    logits = model(x)
    assert logits.shape == (2, meta.num_classes)


def test_transformer_weighted_last_pool_receives_gradient():
    meta = ModelMeta(
        input_size=4,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        bidirectional=False,
        num_classes=3,
        task="classification",
        model_type="transformer",
        feature_cols=["f1", "f2", "f3", "f4"],
        window_size=6,
        transformer_pooling="weighted_last",
    )
    model = build_model_from_meta(meta)
    x = torch.randn(2, meta.window_size, meta.input_size)
    loss = model(x).sum()
    loss.backward()

    assert model.last_step_pool is not None
    assert model.last_step_pool.recency_decay.grad is not None
    assert torch.count_nonzero(model.last_step_pool.query_proj.weight.grad).item() > 0


def test_windows_fixture_shape(windows_exact):
    assert windows_exact.ndim == 3
    assert windows_exact.shape[1] == 10
