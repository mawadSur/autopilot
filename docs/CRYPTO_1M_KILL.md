# Kill Report — Crypto 1-Minute Directional XGBoost Stack

**Date:** 2026-05-31
**Decision:** FREEZE the crypto 1-minute directional stack until its economics change.
**Why:** It is unprofitable by **arithmetic**, not by tuning. The edge it targets is
smaller than the round-trip cost of the venue it would trade on.

---

## 1. The arithmetic

The tier-2 crypto models are trained to predict a **+10–20 bps** move (the labels were
relabeled to a +20 bps target — see commit `c4c86c1`, and `data/crypto/datasets/*_20bps.csv`).

The repo's own live Coinbase adapter charges the **real** Coinbase Advanced retail fee:

| Side  | Rate  | Source |
|-------|-------|--------|
| taker | 60 bps | `src/exchanges/adapters/coinbase_tradeable.py:54-56` — `FeeModel(maker=0.0040, taker=0.0060)` |
| maker | 40 bps | same |

Round-trip cost of actually executing a trade:

| Path                | Cost      |
|---------------------|-----------|
| taker in + taker out | **~120 bps** |
| taker in + maker out | ~100 bps  |
| maker in + maker out | ~80 bps   |

**A +20 bps target cannot clear a ~120 bps (or even ~80 bps) round-trip.** Every
"winning" trade still loses money. No threshold, calibration, or confluence-gate tweak
fixes a gross edge that is 4–6x smaller than cost. This is a unit-economics failure, full stop.

---

## 2. The bug that hid it: a ~7.5x fee understatement

The real `FeeModel` (60/40 bps) was **defined but never wired into the simulator**. The
backtester charged a fictional fee instead, so every backtest looked far better than reality:

| Location | Before | After |
|----------|--------|-------|
| `trading/simulator.py` `SimulationConfig.fee_pct` default | `0.0008` (8 bps) | unchanged default `0.0008`, but a new honest constructor added (below) |
| `trading/simulator.py` `SimulationConfig.maker_fee_pct` default | `None` → silently fell back to `fee_pct` in `_maker_fee_pct()`, so the maker path was also ~8 bps and **fictional** | now explicitly set by the honest constructor |
| `src/backtest.py:239` | `fee_pct = 0.00075` (**7.5 bps**, hardcoded, even cheaper than the config default) and `maker_fee_pct` never set | removed; backtest now builds its config via `SimulationConfig.from_coinbase_fees()` |

Effective round-trip cost charged by the simulator, measured directly
(`tests/.../test_fee_honesty_kill.py::test_effective_round_trip_cost_is_about_120bps_under_coinbase`
and `::test_old_cheap_default_understated_cost_at_about_16bps`):

| Config | Round-trip cost |
|--------|-----------------|
| old cheap default (`fee_pct=0.0008`) | **16.00 bps** |
| `from_coinbase_fees()` (real 60/40) | **120.00 bps** |

That is a **7.5x** understatement. The old backtests were fictional; they "passed" because
they were charging ~1/7th of real cost. This violated the Constitution's honest-reporting and
no-look-ahead standards.

### The fix (file:line)

- `trading/simulator.py:23-25` — added `COINBASE_TAKER_FEE_PCT = 0.0060` / `COINBASE_MAKER_FEE_PCT = 0.0040`, mirroring the live adapter.
- `trading/simulator.py:90-120` — added `SimulationConfig.from_coinbase_fees(...)`: sets `fee_pct = taker` (60 bps) **and** `maker_fee_pct = maker` (40 bps) so the maker leg is no longer fictional. The legacy `0.0008` default is kept for non-Coinbase/legacy callers.
- `trading/simulator.py:122-138` — added `SimulationConfig.from_fee_model(fee_model)`: duck-typed on `.maker`/`.taker`, accepts the live adapters' `protocols.FeeModel` directly.
- `src/backtest.py:239-264` — deleted the hardcoded `fee_pct = 0.00075`; the simulator config is now built with `SimulationConfig.from_coinbase_fees(...)`, so the backtest charges real Coinbase fees **by default**.

### Optimistic-fill flag (not fixed, just flagged)

`trading/simulator.py:470-479` — the post-only/maker entry path treats a limit posted at
this bar's open as **filled** if the *same* bar's low/high crosses it. That is an optimistic
same-bar look-ahead (real resting limits need price to return to them later in the bar, and
queue position isn't guaranteed). Per scope, the fill engine was not rewritten; the assumption
is now flagged in code so it is not silently relied on. It only inflates maker fill rates — it
does not touch the fee arithmetic above, which already kills the stack on cost alone.

---

## 3. Deterministic evidence (env-independent)

`tests/prediction_market_scanner/test_fee_honesty_kill.py` proves the kill without any data,
model, or network. It constructs a trade that **perfectly hits the +20 bps target** and runs
it through the simulator under real Coinbase taker fees:

```
Start capital            : $10,000.00
Entry (taker, 60 bps)    : -$60.00
Gross move (+20 bps)     : +$20.00
Exit  (taker, 60 bps)    : -$60.00
-----------------------------------
Net PnL                  : -$100.00   (a WINNING +20 bps trade still loses ~100 bps)
```

Key asserts (all passing):
- A perfect +20 bps winner nets **−$100** (negative) under real Coinbase taker fees.
- Effective round-trip cost is **~120 bps** under Coinbase config, vs **~16 bps** under the old cheap default.
- Control: the *same* +20 bps winner is **profitable (+$4)** under the old cheap fees — proving the kill is driven by the fee correction, not a rigged scenario.

Run:
```
env PYTHONPATH=src ./.venv/bin/python -m unittest \
    tests.prediction_market_scanner.test_fee_honesty_kill -v
# Ran 7 tests ... OK
```

---

## 4. Real backtest attempt (timeboxed) — blocked, deterministic test stands in

A real ETH backtest was attempted on `data/crypto/datasets/eth_usd_1m.csv` (129k rows) but
was **blocked by a missing model artifact**: `model_sanity/` ships `model_meta.json` +
`scaler.joblib` but **no `model.pt` weights** (`FileNotFoundError: model_sanity/model.pt`).
Model weights are generated artifacts and are not checked in (per `CLAUDE.md`). The data also
spans only ~3 months from 2026-02-22, short of the 6+3-month walk-forward minimum
(`src/backtest.py` raises `"Not enough data for 6+3 month walk-forward"` for <9 months).

Rather than fight the environment, the **deterministic unit test in §3 is the canonical
evidence**. It is stronger than a single backtest anyway: it isolates the fee arithmetic from
model quality and data noise, and it is reproducible on any machine with no torch, no TA-Lib,
no data, and no network.

The backtest gate itself was confirmed and completed (T2): `src/backtest.py` persists
`profit_report.json` and now records an explicit `gate_passed` / `gate_verdict` from the
PF ≥ 1.8 and max-drawdown ≤ 10% reject gate, computed on the honest fees
(`src/backtest.py:509-535`).

---

## 5. Decision

**FREEZE** the crypto 1-minute directional stack. Do not run it live, and do not treat its
historical backtests as valid — they were computed on a ~7.5x cost understatement.

It may be **unfrozen only** when its economics actually change, i.e. one of:

1. **Longer-horizon labels** whose realized gross edge comfortably exceeds ~120 bps
   round-trip (e.g. multi-hour / daily moves, not 1-minute +20 bps), **or**
2. **A maker-rebate or low-fee venue** where the round-trip cost drops below the model's
   gross edge — and only if the simulator's optimistic same-bar maker fill (§2) is replaced
   with a realistic resting-limit fill model first, since a maker-only strategy lives or
   dies on fill realism.

Until then, capital and engineering effort belong on surfaces where edge > cost — which,
under the mission, means the prediction-market and cross-venue work, not 1-minute crypto
direction.

---

### Files changed by this kill

- `trading/simulator.py` — honest Coinbase fee constants + `from_coinbase_fees()` / `from_fee_model()` constructors; optimistic-fill flagged.
- `src/backtest.py` — backtest now charges real Coinbase fees by default; `profit_report.json` gate verdict persisted.
- `tests/prediction_market_scanner/test_fee_honesty_kill.py` — deterministic kill proof (7 tests).
- `docs/CRYPTO_1M_KILL.md` — this report.
