# Alpha Lab integration guide

Phase 4 / E2 (CEO plan, 2026-05-07). The alpha lab is the nightly cross-asset
correlation miner that surfaces candidate features for human review and
eventual promotion into the live feature pipeline. The module is purely
additive — `src/alpha_lab/` does not import from the rest of the codebase
beyond the lazy `protocols.tradeable.AssetClass` import in
`feature_sources.py`, and nothing in the rest of the codebase imports
`alpha_lab` yet. This doc describes how a future PR should wire it.

## What ships in this PR (skeleton)

* `correlation_miner.py` — cartesian-product Spearman rank-IC miner.
* `auto_promotion_gate.py` — rolling 30-sample window per pair; emits
  `PromotionCandidate` to a JSONL queue when `|mean(rank_ic)| > 0.05`.
* `nightly_runner.py` — orchestrator entrypoint + CLI.
* `feature_sources.py` — `CryptoFeatureSource` (parquet-backed) and
  `PolymarketFeatureSource` (fetcher-injected) skeletons.

What's NOT yet wired: real Polymarket history fetcher, env-var-driven
`build_default_feature_sources()`, cron job, automated promotion-queue
advancement.

## Wiring `CryptoFeatureSource`

The default behavior is to read from `data/crypto/datasets/<symbol>_v1.parquet`
(produced by `crypto_training/build_dataset.py`):

```python
from alpha_lab.feature_sources import CryptoFeatureSource

src_btc = CryptoFeatureSource("BTC-USD")  # -> data/crypto/datasets/btc_usd_v1.parquet
src_eth = CryptoFeatureSource("ETH-USD", parquet_path=Path("custom/eth_v2.parquet"))
```

For a true live-data path (rolling minute bars from Coinbase REST), follow
the regime-memory backfill pattern: a separate ingest job writes a rolling
parquet snapshot, and the source reads from that snapshot. Putting the
live REST call inside `fetch_window` would burn rate limits on every
nightly run.

## Wiring `PolymarketFeatureSource`

The skeleton accepts an injected fetcher:

```python
from alpha_lab.feature_sources import PolymarketFeatureSource

def my_polymarket_history(market_id, start_utc, end_utc):
    # Hit Polymarket's historical-prices endpoint and return a DataFrame
    # indexed by UTC timestamp with at least a 'midpoint' column.
    ...

src_cpi = PolymarketFeatureSource(
    "0xabc123...",
    fetcher=my_polymarket_history,
    question="CPI > 3% in May 2026?",
)
```

For the production path, plug in the existing `fetcher.fetch_active_markets`
to identify candidate markets, then a new `fetcher.fetch_market_price_history`
(future PR) to pull the time series. The orchestrator already maintains a
`build_scan_results()` cache that could be repurposed as a snapshot DB.

## Wiring `build_default_feature_sources`

Today this returns `[]` so the nightly CLI exits 0 with a "no sources
wired" warning. The recommended production pattern is environment-driven:

```python
# In a deploy-only override of src/alpha_lab/feature_sources.py:
def build_default_feature_sources():
    crypto_symbols = os.environ.get("ALPHA_LAB_CRYPTO_SYMBOLS", "").split(",")
    market_ids = os.environ.get("ALPHA_LAB_POLYMARKET_MARKET_IDS", "").split(",")
    sources = []
    for sym in crypto_symbols:
        if sym.strip():
            sources.append(CryptoFeatureSource(sym.strip()))
    for mid in market_ids:
        if mid.strip():
            sources.append(PolymarketFeatureSource(mid.strip(), fetcher=...))
    return sources
```

## Cron schedule

A nightly run at 05:00 UTC (after Polymarket EOD settles + before US morning
trading desks pick up):

```cron
# /etc/cron.d/alpha_lab
0 5 * * *  autopilot  cd /srv/autopilot && \
    ./.venv/bin/python -m alpha_lab.nightly_runner \
        --output-dir runs/alpha_lab/ \
        --window-days 30 \
        --threshold-rank-ic 0.05 \
        --min-samples 30 \
        --redis-url ${REDIS_URL} \
        >> /var/log/autopilot/alpha_lab.log 2>&1
```

`--redis-url` is critical for production: without it, each cron invocation
starts with an empty rolling window and never reaches the `min_samples`
gate. With Redis, history persists across runs.

## Promoting a candidate into the feature pipeline

Today this is a manual two-step:

1. Operator reviews `runs/alpha_lab/promotion_queue.jsonl` — each line is a
   `PromotionCandidate` payload (pair definition, 30-day mean rank-IC,
   sample count, first/last seen UTC).
2. If approved, the operator adds the feature pair definition to whichever
   downstream consumer wants it. Two likely consumers:
   * `src/utils.compute_features` — for crypto-side cross-asset features.
   * `src/calibration_agent/build_dataset.py` — for prediction-market
     calibration features.

The conservative discipline mirrors `calibration_agent/outcome_weight_adjuster`:
the gate emits, never auto-applies. Future PRs can layer a
`promotion_queue_consumer.py` that reads the JSONL, runs an additional
out-of-sample validation, and only then merges the pair into the pipeline.

## Caveats

* **Spearman is the rank-IC choice.** Robust to outliers, captures
  monotonic non-linear edge that Pearson misses. The miner exposes the
  p-value but does NOT use it for filtering — significance over a noisy
  cartesian product is dominated by multiple-testing concerns that need
  explicit FDR correction (a future enhancement).
* **30-day window assumes one nightly run.** If you bump the cadence,
  bump `--min-samples` proportionally so a high-frequency mining loop
  doesn't promote on a few-hour window.
* **Same-source self-pairs are skipped.** `(feat_a, feat_a)` on a single
  source has trivial rank-IC ≈ 1.0 — the miner short-circuits these to
  keep the ranking interpretable. Cross-source same-name pairs (e.g.
  ``return_5`` on BTC vs ``return_5`` on ETH) are evaluated normally.
* **Asymmetric pair semantics.** The miner treats `feature_a` as the
  signal at time `t` and `feature_b`'s forward return as the target.
  `(A, B)` and `(B, A)` are distinct hypotheses and both end up in the
  ranking. Operators should read each candidate as a directed claim.

## Sanity checks before production wire-up

1. Run the unit tests: `env PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests/prediction_market_scanner -p "test_alpha_lab_*.py"`.
   Should report ~50 tests passing.
2. Smoke-test the CLI: `./.venv/bin/python -m alpha_lab.nightly_runner --output-dir /tmp/alpha_lab_test/`.
   Without sources wired, exits 0 with a warning.
3. Verify a synthetic run with `_SyntheticSource` (see
   `tests/prediction_market_scanner/test_alpha_lab_miner.py`) produces a
   high-rank-IC pair on the strongly-correlated synthetic input.
