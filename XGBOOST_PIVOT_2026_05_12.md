# XGBoost Pivot — Session Findings (2026-05-12 → 2026-05-13)

## TL;DR

We pivoted from a dead LegacyTransformer (PyTorch) to per-symbol XGBoost models targeting a 65% win rate. Along the way we discovered that **the original failure mode was a volatility-confounded label**, not a model architecture problem. After fixing the label, the **ETH model became a real live-paper candidate**: 60.7% test win-rate at threshold 0.55 on 155 trades, with healthy precision-selectivity calibration. BTC and SOL still need work — they show real edge at low thresholds but anti-predictive calibration in the high-confidence region. A multi-day ETH paper session is now running (PID 21057) to gather real OOS evidence.

---

## The Progression (each step was load-bearing)

### 1. LegacyTransformer was dead
- Full ETH walk-forward backtest with `THR_LONG=0.50`: **Sharpe -10.4, 30% win rate, "UNPROFITABLE - REJECTED"**.
- Lowering thresholds doesn't help — it sweeps in negative-EV trades.
- Memory: `autopilot_threshold_lowering_2026_05_12.md`

### 2. XGBoost wiring worked but val-overfit threshold masked failure
- `XGBoostPredictor` + `MultiSymbolXGBoostPredictor` already existed in `src/predictor.py`.
- Single-tick live smoke confirmed all 3 models load and tick cleanly via `CRYPTO_MODEL_MAP`.
- Pre-existing models (BTC v1, ETH v1, SOL v1) had `optimal_threshold` selected by sweeping val Sharpe — **no test-split verification**.

### 3. Test-split validation collapsed the optimistic picture
- Added `scripts/validate_xgboost_winrate.py` — a parquet-native precision-at-threshold validator.
- Honest OOS test-split results: **BTC** had 2-3 trades only, **ETH was anti-predictive at 28-29% win rate**, **SOL** had all probas collapsed ≤0.30.
- Memory: `autopilot_xgboost_test_split_failure_2026_05_12.md`

### 4. Added test-gate to the trainer, retrained — 6/6 variants failed
- Modified `src/crypto_training/train_xgboost.py` with `_compute_test_gate()`. New fields on every `meta.json`: `test_winrate_at_optimal_threshold`, `test_ntrades_at_optimal_threshold`, `test_gate_reason`. When test fails, `optimal_threshold` is set to `null` and `threshold_status="test_gate_failed"`.
- Retrained BTC/ETH/SOL with both isotonic and sigmoid calibration. **All 6 variants failed the 0.55 winrate floor.**
- This proved the issue wasn't calibration choice — it was deeper.
- Memory: `autopilot_retrain_gate_2026_05_12.md`

### 5. Feature diagnostic exposed the real bug
- Added `scripts/diagnose_features.py` testing 6 hypotheses.
- **Root cause identified**: label was `forward_return_bps > fixed_threshold_bps`. In high-vol periods, more bars trivially cross any fixed threshold regardless of direction. The model learned "high vol → label=1" instead of "direction → label=1".
- Evidence: top-10 features by test AUC across all 3 symbols are 100% volatility measures (`atrp_14`, `atr_14`, `range_ma_20` at 0.64-0.72 AUC). Best directional feature AUC: 0.54-0.59 (near random). Train period had 40-70% higher realised vol than test → label base rate dropped 38-44%.
- Hypotheses ruled out: leakage, feature drift, horizon mismatch, class imbalance.
- Memory: `autopilot_label_bug_2026_05_12.md`

### 6. Vol-normalized rebuild → ETH is shippable
- Modified `src/crypto_training/build_dataset.py` with `label_kind="vol_normalized"` (label = `forward_return_bps > k * atrp_14_bps`, k=0.5).
- Dropped vol features from training feature_cols.
- Rebuilt all 3 parquets. Retrained. Re-validated.
- **ETH test-split precision curve is monotonic and shippable** — see Final State.
- BTC/SOL: anti-predictive at high thresholds. Real edge at low thresholds but calibration is broken in the high-confidence region.
- Memory: `autopilot_voln_rebuild_2026_05_12.md`

---

## Final State of the 3 Models

### ETH (`model_crypto/eth_usd_voln_v1/`) — shippable candidate

| thr | n_trades | test_win_rate |
|---|---|---|
| 0.50 | 7386 | 53.9% |
| 0.52 | 2165 | **56.1%** (clears 55% on 2k trades) |
| **0.55** | **155** | **60.7%** (binom 95% CI ~[0.53, 0.68] — 65% plausibly within reach) |

- Monotonic precision-selectivity — exactly the calibration shape we want
- Sigmoid calibration
- Multi-day paper run live as of 2026-05-13T11:05Z

### BTC (`model_crypto/btc_usd_voln_v1/`) — parked

| thr | n_trades | test_win_rate |
|---|---|---|
| 0.50 | 6701 | 53.2% |
| 0.54 | 1028 | 52.5% |
| **0.55** | 212 | **42.5%** ← drops with selectivity |
| 0.56 | 31 | 38.7% |

- Anti-predictive at high thresholds. The model's high-confidence predictions are *less* accurate than its average. Real signal at low thresholds (53% × thousands) but selectivity backfires.

### SOL (`model_crypto/sol_usd_voln_v1/`) — parked

- Similar non-monotonic pattern to BTC. Small island of 70% win-rate at n=10 (thr=0.58) but trajectory is unstable.

---

## Infrastructure Shipped This Session

| Commit | Component | What it enables |
|---|---|---|
| `e168543` | `raw_max_prob` + `raw_probs` on `SupervisorTick` | Lets observers see *how close* a skipped tick was to threshold (not just "below floor") |
| `8f20967` | Fix loguru format-string in `_warn_on_sparse_l2_depth` | Backtest log warnings now legible (had been printing `%.2f%% (%d/%d)` literally) |
| `d332dc5` | ETH v1 threshold re-sweep (now known to be val-overfit) | Patched v1 meta with threshold_metrics; superseded by voln_v1 |
| `97ef846` | XGBoost inference path in `src/backtest.py` | Backtester now supports both PyTorch and XGBoost via `is_xgboost_model_dir` detection |
| `bf304b4` | `scripts/validate_xgboost_winrate.py` | Parquet-native precision-at-threshold validator — the honest evaluation tool |
| `297681d` | Test-split gate in `train_xgboost.py` | Trainer refuses to write `optimal_threshold` if test_winrate < floor. Procedural fix for val-overfit. |
| `20af728` | `scripts/diagnose_features.py` | 6-hypothesis diagnostic battery — exposed the label bug |
| `04ee268` | Vol-normalized label kind + drop vol features | The actual fix. Enables future correct retrains. |
| `6a62c88` | Gitignore Ruflo runtime artifacts | Cleanup |

All commits on `origin/feature/prediction-market-bot`.

---

## What's Live Right Now

Multi-day ETH paper session, started 2026-05-13T11:05Z:

```
PID: 21057
Log: /Users/mawad/Desktop/autopilot/logs/eth_paper_multiday/eth_voln_v1_20260513T110536Z.log
Run dir: logs/eth_paper_multiday/2026-05-13T11-05-52Z_ETH-USD/
Config: ETH/USD only, thr=0.55, min_conf=0.51, 5s ticks, $10k bankroll
Model: model_crypto/eth_usd_voln_v1/
```

### Monitoring

```bash
# Tail live
tail -f /Users/mawad/Desktop/autopilot/logs/eth_paper_multiday/eth_voln_v1_20260513T110536Z.log

# Just the actionable ticks
grep -E "action=(allowed|errored|halted|force_flatted)" /Users/mawad/Desktop/autopilot/logs/eth_paper_multiday/*.log

# Health check
ps -p 21057 -o pid,etime,rss,command

# Graceful stop
kill -INT 21057
```

### Goal

After ~48 hours, the realised win-rate on allowed ticks will tell us whether the 60.7% historical test number was signal or noise. Binomial CI on 20 fills at 60% is ~[40%, 78%] — wide. 50 fills would tighten to ~[48%, 70%]. Plan for 48+ hours to get a meaningful read.

---

## Open Questions / Next Steps

### Likely-needed work

1. **BTC/SOL calibration fix.** The new label exposed an anti-predictive high-confidence region. Options:
   - Calibration on a held-out calibration set (not val)
   - Try monotonic-constrained isotonic
   - Investigate WHY high-confidence predictions are LESS accurate — could be small-region overfit
2. **Validate the ETH live-paper result.** Once we have 20-50 fills, compare realised win-rate vs the 60.7% historical number.
3. **65% target may need richer features.** Best directional univariate AUC is ~0.55. Reaching 65% precision from that base is ambitious. May need non-price signals: funding rates, options IV skew, on-chain flow, sentiment.

### Procedural improvements landed

- Test-split gate is doing its job. Keep it.
- Diagnostic battery (`scripts/diagnose_features.py`) is reusable for any future label/feature change.
- Win-rate validator (`scripts/validate_xgboost_winrate.py`) is the canonical precision-curve tool.

### Sometime later

- Investigate why `--workers` (`_child_main` in `live_supervisor.py:2920-2972`) is a ticker-poll stub. Single-process `run_loop` is the production path right now.
- Coinbase adapter has no DNS-error retry; Discord notifier has no circuit-breaker. Surfaced by the DNS outage mid-session.
- Untracked `.claude/agents/`, `.claude/commands/`, `.claude/helpers/`, `.claude/settings.json`, `.mcp.json` — operator decision on whether to commit them as the project's Ruflo baseline.

---

## Session Stats

- **9 commits** on `feature/prediction-market-bot`, all on origin
- **1251 tests** passing (was 1235 at session start; +16 from new label tests etc.)
- **6 memory entries** written under `~/.claude/projects/.../memory/`
- **3 sub-agents** spawned (XGBoost backtester refactor, ETH v1 threshold re-sweep, feature diagnostic, vol-normalized rebuild)
- **2 production paper sessions** (45-min XGBoost smoke clean, multi-day ETH live)

## Single Most Important Sentence

**The original problem was a label engineering bug, not a model architecture problem — and the test-split gate plus diagnostic battery are now the procedural defenses against this class of bug recurring.**
