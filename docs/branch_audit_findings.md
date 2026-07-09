# Post-fee branch audit + regime routing — findings

_Scope: legacy crypto XGBoost stack (`model_crypto/*_20bps_sigmoid`). Task: audit
every decision branch for whether it improves **post-fee expectancy**, and add
regime-aware routing (one model, per-regime thresholds/gating). No retrain._

## What was built

| Artifact | Purpose |
|---|---|
| `src/crypto_training/branch_audit.py` | Execution-aware, post-fee **ablation harness**. Baseline + one-branch-flipped runs + threshold sweep + per-regime EV + combined recommendation. Writes `<model-dir>/branch_audit.json`. |
| `src/regime_router.py` | Deterministic, auditable regime router (trend-up/down/chop from `adx` + `close_over_ema_50`) → per-regime long threshold + enable flag. |
| `src/dynamic_threshold.py` | Cost-aware **dynamic entry threshold**: raises `thr_long` when expected execution cost is high — volatility via `atrp_14` (live), liquidity via spread/depth/imbalance (inert until the book is backfilled). Sign-tunable, clipped; composes on top of the regime/base threshold. |
| `src/predictor.py` (wiring) | `XGBoostPredictor` consults the router when `USE_REGIME_ROUTER=1` and the dynamic threshold when `USE_DYNAMIC_THRESHOLD=1` (applied *after* regime routing; a disabled regime stays blocked). Optional `REGIME_ROUTER_CONFIG` / `DYNAMIC_THRESHOLD_CONFIG` JSON. **Both default OFF.** |
| `tests/test_branch_audit.py` | 11 dependency-light unit tests (classifier, verdicts, honest post-fee reconstruction, consensus). |

### Method / honest-costs notes
- **Data reality:** the featurized datasets carry book columns by name but they are **identically 0** (the L2 book was never backfilled). Real OHLC (with intrabar high/low) is joined from `data/crypto/<SYM>/1m/*.csv` on timestamp. Features are pre-computed, so **no TA-Lib and no retrain** are needed.
- **OOS only:** audited on the **test** slice (last 15%, matching the trainer's time split) — the model never saw it and no threshold was fit on it.
- **Honest fees:** Coinbase Advanced retail — taker 60 bps / maker 40 bps → **~120 bps round-trip** taker.
- **Post-fee expectancy is reconstructed from the equity sequence**, not from `trade_log['ret']`. The simulator books fees against cash but leaves the logged per-trade `ret`/`pnl` **gross**, so `profitability.compute_profitability_metrics` reports a *gross* expectancy (see "Defects surfaced").

## Headline result — the strategy is underwater at retail taker fees

OOS test slice ≈ 2026-05-09 → 2026-05-22 (~13.5 days, ~19.4k bars per symbol).

| Model | Baseline trades | Post-fee EV/trade | Win-rate (net) | Best-thr EV/trade | Any post-fee-positive config? |
|---|---|---|---|---|---|
| ETH `v4_20bps` | 9 | **−115 bps** | 0% | −109 bps @0.40 | **No** |
| BTC `v3_20bps` | 3 | **−114 bps** | 0% | −114 bps @0.30 | **No** |
| SOL `v2_20bps` | 37 | **−128 bps** | 0% | −65 bps @0.80 | **No** |

**Not a single net-winning trade on OOS, for any symbol, at any threshold, in any regime.** The cause is structural, not a tuning miss: the models' label hurdle is **+20 bps** forward return and their per-signal edge is on that order (tens of bps), while round-trip taker cost is **~120 bps**. The edge is ~3–10× smaller than the fee it must clear. No branch toggle closes a gap that large.

## Per-branch findings

- **`consensus` (require N consecutive signals): incompatible with these models — remove.** The models fire on ~0.2% of bars as *isolated* single bars; `consensus=2` needs two consecutive fires, which essentially never happens → it silences the model to **0 trades**. (This is the legacy 3-class-transformer default leaking onto a sparse binary model.) The `XGBoostPredictor` path correctly does not use it; keep it off.
- **`regime_filter` / `regime_routing`: their "improvement" is mostly *trading less*.** On this slice they cut trades toward zero; since every trade is a net loser, not trading trivially raises EV toward 0. The **per-regime EV breakdown shows every regime is negative** (SOL: chop −106, trend_down −129, trend_up −144 bps). So routing **cannot rescue** the strategy on current data — it ships **wired but OFF**.
- **`dynamic_threshold` (cost-aware entry threshold): does not pay on current data — wired but OFF.** Default (cost-aware, `s_vol=+0.06`: demand more conviction in high vol) worsens post-fee EV on all three symbols (ETH −18, BTC −2, SOL −6 bps) by over-thinning to a handful of trades. The volatility-sign sweep confirms neither direction rescues it (ETH: `+0.12→−147bps`/2 trades, `−0.12→−121bps`/32 trades — more trades, still negative). Only the **volatility** term moves today; the **liquidity** term (spread/depth/imbalance) is structurally wired but inert because those columns are all 0. Like regime routing, this is a shaping lever with no post-fee edge to shape until a model clears round-trip cost (or the book is backfilled to activate the liquidity term).
- **`dynamic_slippage` (cost model, always-on): ~25–42 bps/trade.** Turning it off "improves" EV only by fabricating optimistic fills. Reported for cost sensitivity, never auto-cut.
- **`market_depth` (cost model): inert here** — no book data, so it falls back to top-of-book/ATR. Cannot be validated until datasets are rebuilt with L2.
- **`post_only` (maker vs taker): unvalidated** — maker fills need book data we don't have; the harness flags it heuristic. Real leverage exists here (maker-only round-trip ≈ 80 bps vs 120 bps taker) but still exceeds the model's edge.
- **`atr_stops` vs fixed pct: keep ATR stops** (fixed stops were consistently worse across symbols), though the effect is small next to the fee gap.
- **`hard_gate`, `min_atr`, `cooldown`: negligible / low-confidence** at these tiny OOS trade counts.

> Trade counts are small (3–37), so individual branch verdicts are low-confidence. The *direction* is unanimous across three assets and the fee/edge gap is far too large for sample noise to reverse.

## Defects fixed (`trading/simulator.py` — done before any further changes)

Both were fixed together; regression tests in `tests/test_simulator_net_fees.py` (6 tests) lock them in.

1. **Per-trade P/L is now NET of fees.** Previously fees hit `cash`/equity but `trade_log['ret']`/`['pnl']` stayed pre-fee, so `compute_profitability_metrics.expectancy` / `profit_factor` and the sim's own `report()` PF were **gross**. `_exit_position` now attributes both legs' fees (the entry fee is tracked on the open position via `self.entry_fee` and subtracted at exit) into `ret`/`pnl`, and win/loss/profit-factor accounting uses the net figure. Added transparency fields `gross_ret`/`gross_pnl`/`fees` to each exit row. Verified: a +50 bps price win books as a **−50 bps net loss** after ~100 bps round-trip, and per-trade expectancy now equals the net equity return. `train_xgboost` is unaffected — it builds its own already-fee-netted trade log, not the simulator's.
2. **`_normalize_signal` now speaks RAW only (`-1/0/+1`).** It no longer conflates raw with class-style, so a class-style HOLD (`1`) can never again be silently read as a LONG. Class-style callers convert explicitly via the new `class_to_raw()` helper (`0→-1, 1→0, 2→+1`): wired into the `simulate_trades_with_tp_sl*` wrappers and the three `sim.step` sites in `backtest.py`. `live_trader.py` and the branch audit already spoke raw. This also corrected a latent **always-long** bug in `paper_trade.py` (it fed `classes ∈ {1,2}`, both of which the old mapping turned into longs).

## Recommendations (mission: better net execution)

1. **Retrain the label to clear the real cost.** `build_dataset --threshold-bps` currently bakes a **20 bps** hurdle; it must exceed round-trip cost. Use `--threshold-bps ~130` (taker) or `~90` (maker-only) plus a margin so a labeled win is genuinely net-profitable. This is the single highest-impact change — and it is a *retrain* (deferred by the chosen "no-retrain" scope), so it's flagged, not done.
2. **Execution: pursue maker-only.** Round-trip drops 120→80 bps. Requires (a) rebuilding datasets with real L2 book so `post_only`/`market_depth` can be validated, and (b) a fill model that accounts for non-fills.
3. **Add a post-fee expectancy guard.** No model should trade live with audited OOS post-fee EV below a floor. `branch_audit.json` already emits the numbers to gate on (mirror `backtest.py`'s `profit_report.json` gate).
4. **Keep regime routing wired-but-off** until a model exists whose edge > round-trip cost; then re-run the audit — the per-regime EV breakdown will show whether routing then pays.

## How to run

```bash
# Audit one model (writes <model-dir>/branch_audit.json)
./.venv/bin/python src/crypto_training/branch_audit.py --model-dir model_crypto/eth_usd_v4_20bps_sigmoid

# Tests
env PYTHONPATH=src ./.venv/bin/python -m unittest tests.test_branch_audit

# Enable deterministic regime routing live (default off)
USE_REGIME_ROUTER=1 [REGIME_ROUTER_CONFIG=path/to/router.json] ...

# Enable cost-aware dynamic entry threshold live (default off; set per-symbol ref_atrp)
USE_DYNAMIC_THRESHOLD=1 [DYNAMIC_THRESHOLD_CONFIG=path/to/dyn.json] ...
```
