# Pooled multi-asset LSTM pipeline

*Added 2026-06-08. Code under `src/multi_asset/`, tests under `tests/multi_asset_lstm/`.*

## Goal

Train and test **one** LSTM across many crypto coins **and** stocks at once — a
single pooled model rather than one model per symbol. This serves Article 0 of
the Constitution ("anything tradeable — stocks, crypto, …") by giving every new
instrument a model on day one, and lets the network learn cross-asset structure
(e.g. a momentum regime that rhymes across BTC and NVDA).

## The core risk: scale leakage — and how we kill it

Pooling BTC (~$60k) next to a $40 stock is dangerous. If the network can see raw
price or volume **magnitude**, it learns to *identify the asset* from the size of
the numbers instead of learning direction — a shortcut that backtests beautifully
and dies live. Three defenses, all enforced in code:

1. **Scale-invariant features only.**
   `build_pooled_dataset.SCALE_INVARIANT_FEATURES` is a curated subset of
   `utils.FEATURE_COLUMNS_PROFITABLE` containing *only* dimensionless or
   price-relative columns — returns, log-returns, z-scores, candle-geometry
   ratios, `(ema_a-ema_b)/close` spreads, `close/ema` ratios, ATR%, Bollinger
   width/%b, normalized realized vol, cyclical time, ADX. Every raw price / size
   / EMA / MACD / L2 column is excluded. A regression test
   (`ScaleInvariantFeatureTests`) fails if a banned raw-scale column ever leaks
   in.

2. **Vol-normalized labels.** The label is
   `label_forward_return_binary(..., label_kind="vol_normalized")` — "did the
   forward return exceed `k · ATR%`?" So "profitable" means the same number of
   volatility units on every asset, not "moved a lot of dollars." (This is the
   same fix the XGBoost pivot used to kill the vol-confound bug — see
   `XGBOOST_PIVOT_2026_05_12.md`.)

3. **Asset identity via a learned embedding, not magnitude.**
   `model.PooledLSTMClassifier` wraps the project's existing
   `models.LSTMClassifier` (the attention-LSTM — the "same LSTM model") and adds
   an `nn.Embedding(n_assets, embed_dim)`. The embedding is concatenated to every
   timestep, so the network gets asset identity through a *deliberate, learnable*
   channel instead of reading it off the price tag.

## No look-ahead

* **Split** (`train_pooled_lstm.time_split`) is on the **global wall-clock axis**:
  train = oldest, then val, then test, with boundaries chosen so an entire
  instant lands in one split. The scaler is fit on **train only**.
* **Sequences** (`sequences.build_sequences`) are built **per split subframe**, so
  a window never spans a train/val/test boundary; **per asset**, so a window
  never mixes two instruments; and a window is dropped if it straddles a
  **data hole** (a gap ≫ that asset's own median bar spacing). The hole test uses
  per-asset median spacing so weekend/holiday gaps in *daily* equity series do
  **not** shatter windows, while a genuine missing-data hole in a *minute* series
  does. Both behaviors are pinned by tests.

## Components

| File | Role |
|------|------|
| `sources.py` | `DataSource` ABC + `CryptoSource` (ccxt) / `StockSource` (yfinance). All emit the canonical `[timestamp, open, high, low, close, volume]` schema. Third-party libs imported lazily. |
| `universe.py` | Declarative `Universe` of `Instrument(symbol, asset_class, exchange)`; JSON-serializable. `DEFAULT_UNIVERSE` = BTC/ETH/SOL + AAPL/MSFT/SPY/NVDA at `1d`. |
| `backfill.py` | CLI: fetch the universe into `data/<crypto|stocks>/<SYM>/<gran>/ohlcv.csv` (resumable, merge-dedupe). |
| `build_pooled_dataset.py` | Per-asset features + vol-normalized label + `asset_id`, concatenated and globally time-sorted into the pooled table. |
| `sequences.py` | Leakage-safe windowing (see above) + `build_asset_vocab`. |
| `model.py` | `PooledLSTMClassifier` (asset embedding + reused `LSTMClassifier`). |
| `train_pooled_lstm.py` | Global split, scaler-on-train, training loop (class-weighted CE, early stop), threshold sweep + honest **test-gate**, saves `model.pt` + `scaler.joblib` + `meta.json`. |
| `predictor.py` | `PooledLSTMPredictor`: recent OHLCV for a *trained* asset → `(side, confidence)`. SHADOW only. |
| `backtest_pooled.py` | Read-only replay of the held-out tail → per-asset + pooled win-rate at the blessed threshold. No fills, no orders. |

## End-to-end usage

```bash
# 1. Backfill OHLCV for the whole universe (daily bars, 2y).
./.venv/bin/python src/multi_asset/backfill.py --granularity 1d --days 730

# 2. Build the pooled, scale-invariant, vol-normalized dataset.
./.venv/bin/python src/multi_asset/build_pooled_dataset.py \
    --granularity 1d --horizon 1 --out data/pooled/pooled_1d.parquet

# 3. Train ONE LSTM across all assets.
./.venv/bin/python src/multi_asset/train_pooled_lstm.py \
    --dataset data/pooled/pooled_1d.parquet --out model_multi/pooled_1d_v1/

# 4. SHADOW backtest on the held-out tail.
./.venv/bin/python src/multi_asset/backtest_pooled.py \
    --dataset data/pooled/pooled_1d.parquet --model model_multi/pooled_1d_v1/ \
    --out runs/pooled_1d_v1_backtest.json
```

Add an instrument by editing the universe (or a JSON passed via `--universe`),
re-backfilling, and retraining — no code change. A pooled model can only score
assets in its trained vocab; unknown symbols raise (see `predictor.predict`).

## Tests

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest tests/multi_asset_lstm/ -q
```

18 tests: source normalization, universe round-trip, the scale-invariant feature
contract, the sequence leakage guards (cross-asset, data-hole, weekend-safe),
model forward shape, and a full train → save → load → predict round-trip on
synthetic data.

## Known limitations / next steps

* **Granularity must match across the pool.** yfinance intraday is capped at
  ~60 days, so the default universe uses daily bars. Mixing 1m crypto with daily
  stocks in one pooled dataset is intentionally not supported.
* **No live wiring.** This is research/SHADOW: it produces a model and a signal,
  never an order. Integrating `PooledLSTMPredictor` into the live supervisor /
  entry-filter stack is a deliberate, separate step behind the usual paper→live
  shakedown.
* **Calibration.** The trainer reports raw softmax probability and a precision-
  swept threshold with a test-gate; it does not yet apply isotonic/Platt
  calibration the way the XGBoost path does. Candidate follow-up.
* **`pyarrow` not installed** in the current venv, so the dataset builder falls
  back to CSV (same convention as `crypto_training/build_dataset.py`).
```
