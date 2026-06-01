# Regime memory integration guide

This module is purely additive — `src/regime_memory/` does not import from
the rest of the codebase, and nothing in the rest of the codebase imports
`regime_memory` yet. This doc describes how a future PR should wire
`RegimeLookup` into the live trading path without touching the static-config
behavior current callers depend on.

## Quick wiring overview

```
features (DataFrame) ──┐
                       ├─► RegimeLookup.resolve_params(...) ──┐
RegimeStore (.npz) ────┘                                       │
                                                               ▼
                                              {optimal_threshold, kelly_size_pct,
                                               regime_label, realized_sharpe,
                                               _regime_confidence}
                                                               │
                                                               ▼
                                          Apply IFF _regime_confidence >= 0.5
                                          else: keep the existing static config
```

The confidence gate is the safety hatch. If the store is empty, the lookup
returns zero-confidence (defaults), and the predictor / risk manager runs
its current static-threshold path — no regression possible.

## Backward compatibility contract

The integration must be opt-in:

1. `XGBoostPredictor.__init__` (or the supervisor that constructs it)
   gains an optional `regime_lookup: RegimeLookup | None = None` kwarg.
   Default `None` → the current static-threshold behavior is unchanged.
2. `RiskCalculator` similarly takes an optional `regime_lookup` and only
   consults it when non-`None`.
3. The recommended env var is `REGIME_STORE_PATH=path/to/store.npz`,
   resolved per-symbol (e.g., `REGIME_STORE_PATH_ETH_USD=...`). When unset,
   no lookup is constructed and the static path runs.

That keeps the change reversible — flip the env var off and the system
reverts exactly to today's behavior.

## Integration in `XGBoostPredictor._predict`

The right call site is just before the long/short threshold check inside
`_predict`. Pseudocode:

```python
# Inside XGBoostPredictor._predict (after probability inference):
thr_long = self.thr_long
thr_short = self.thr_short

if self.regime_lookup is not None:
    resolved = self.regime_lookup.resolve_params(
        features,                  # the feature DataFrame already in scope
        k=10,
        window_size=self.window_size,
    )
    if resolved.get("_regime_confidence", 0.0) >= 0.5:
        # Override the static threshold with the regime-resolved one. The
        # short-side threshold mirrors at 1 - thr_long for symmetric models;
        # if the codebase carries an asymmetric pair, resolve both.
        thr_long = float(resolved["optimal_threshold"])
        thr_short = float(1.0 - resolved["optimal_threshold"])

# ... existing decision logic using thr_long / thr_short
```

A few things to keep in mind:

- The existing `feature_cols` and `window_size` from the predictor must be
  passed through to the encoder. Mismatches between the encoder used for
  backfill and the one used at inference break cosine semantics silently.
  The integration PR should assert that the predictor's `feature_cols`
  matches the encoder's `feature_cols` at construction.
- Logging: write `_regime_confidence`, `optimal_threshold`, and
  `regime_label` to whatever predict-output struct the supervisor already
  builds. Without those fields in the trade log, regressions in the
  regime store are invisible post-mortem.

## Integration in `RiskCalculator`

Risk sizing is the second place a regime can change behavior:

```python
# Inside RiskCalculator.size(...):
kelly = self.static_kelly  # whatever the current default is

if self.regime_lookup is not None:
    resolved = self.regime_lookup.resolve_params(features, k=10)
    if resolved.get("_regime_confidence", 0.0) >= 0.5:
        kelly = float(resolved["kelly_size_pct"])

# ... existing per-trade size = capital * kelly * other_factors
```

The risk-side override deserves an extra check: cap the regime-resolved
Kelly fraction against whatever the static config would have allowed. The
backfill clamps at 0.25 already, but a defensive `min(kelly, self.max_kelly)`
inside the calculator is cheap insurance for the day someone backfills a
new symbol with a tiny dataset and overshoots.

## Bootstrapping a store

```bash
./.venv/bin/python -m regime_memory.backfill \
    --dataset data/crypto/datasets/eth_usd_v2.parquet \
    --symbol ETH-USD \
    --output regime_store_eth_usd.npz \
    --window-bars 60 \
    --label-horizon-bars 60
```

For 5m datasets pass `--bars-per-year 105120`; for 15m, `35040`. The
backfill streams the parquet so memory stays bounded.

## Caveats

- **Encoder version drift.** `RegimeEncoder.VERSION` will eventually move
  past `v0` (learned encoder upgrade). Stores carry no version stamp today
  — the integration PR should write the encoder version into the npz
  metadata and refuse to load mismatches. Until then, rebuild the store
  whenever the encoder changes.
- **Per-symbol stores only.** Mixing symbols in one store is allowed by
  the data model (each `RegimeWindow.symbol` is preserved) but the lookup
  doesn't filter by symbol. For a multi-symbol bot, run one backfill +
  one store per symbol and pick the right `RegimeLookup` from the
  predictor.
- **Synthetic metadata.** `optimal_threshold` and `regime_label` are
  v0 heuristics (see `backfill.py` module docstring). They give a useful
  prior but should not be treated as validated calibration — pair with a
  real per-symbol threshold sweep before promoting the store to
  production.
