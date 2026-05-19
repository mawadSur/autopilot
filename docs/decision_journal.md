# Decision Journal

Append-only log of operator + agent decisions worth a postmortem later.
One row per decision. Keep entries terse — link out to memory files or
PRs for context.

Columns:

* **date** — ISO date (UTC) the decision landed.
* **kind** — one of `sprint_kickoff`, `model_swap`, `threshold_change`,
  `exit_policy`, `sizing_change`, `kill_switch`, `breaker_change`,
  `ops_change`, `infra`, `experiment`.
* **what** — one-line summary of the decision.
* **hypothesis** — the bet you're making (the thing we'd expect to see
  if this works).
* **metric** — concrete number/series you'll grade it against.
* **result** — `pending` while open; backfill `success` / `failure` /
  `inconclusive` once the metric reads cleanly.

| date | kind | what | hypothesis | metric | result |
|------|------|------|------------|--------|--------|
| 2026-05-18 | sprint_kickoff | Plan B Sprint 1 begun | exits + Kelly + cost-aware thr will convert 60% wr edge to realized P&L | net_pnl per trade on 340 test fires | pending |
| 2026-05-18 | sprint_land | Sprint 1 shipped: 4 commits (7e0054f, 9230896, 31b4f4a, 5a438aa), +69 tests, both trees green. Exit policy + Kelly default ON. | unattended paper run with exits firing produces non-zero `exits_by_reason_total` and at least one closed position within first hour | smoke run pending operator | shipped |
| 2026-05-18 | smoke_run | 45-min all-3 v2 paper smoke (run dir 2026-05-18T20-25-34Z). 261 iterations × 3 symbols = 783 ticks. | exits fire, no positions stuck at breaker cap, no zombies left after SIGINT | 6 entries (all ETH @ thr=0.58), 6 time-stop exits (every one closed), 0 errors, 0 breaker halts, 0 kill_switch, 0 zombies in Redis after shutdown | success — engineering validated |
| 2026-05-18 | smoke_pnl | Realized P&L on the 6 closed trades: 6/6 losses, -0.06% to -0.18% per trade, total ~-$0.70 incl. slippage. ETH drifted down $4 during the 90s entry window. | n=6 too small to grade edge; need ≥30 trades across mixed price regimes | next smoke: longer run + mixed regime | inconclusive — sample size too small |
| 2026-05-18 | sprint_land | Sprint 2 items #5 + #8 shipped (commit a4c96dc): Lane E nightly digest orchestrator (`scripts/run_postmortem.py` + launchd plist 00:05 UTC) and calibration drift monitor (`scripts/diagnose_calibration_drift.py`). +36 tests, 1399 OK. | tooling can run end-to-end against today's smoke fixture | both tools produce correct output but specialists return verdict=unknown and calibration verdict=NO_DATA — supervisor never wrote signal/fill/breaker snapshots or entry_confidence onto closed paper positions | shipped (data gap blocks operational meaning) |
| 2026-05-18 | finding | Lane E + calibration drift both starve on the same gap: supervisor doesn't persist entry_confidence on `position.model_meta` or the signal/fill/breaker snapshots into the closed-position record path that Lane E specialists read from. | a small "Sprint 2.5" supervisor patch unlocks both monitors | snapshot-capture wired on `record_open` + `_drain_pending_paper_fill` | shipped (commit a74d604) |
| 2026-05-18 | bug_fix | Sprint 2.5 (commit a74d604): CLI main() was passing trade_context_store=None to Supervisor. All snapshot helpers silently no-op'd in prod. Tests had been wiring the store post-construction so this masked. Fixed by constructing TradeContextStore in main(); added Position.entry_confidence + resolved_kelly_pct; added PositionStore record_signal/fill/breaker_snapshot writers symmetric to TradeContextStore reader. | Lane E specialists return real verdicts post-patch | deterministic smoke shows confidenced 6/6 (was 0/6); 4 of 5 specialists return innocent (was all 5 unknown) | shipped |
| 2026-05-18 | finding | signal_forensics still emits "feature_buffer empty" + "model_probs empty" advisory bullets per trade. Predictor returns (side, confidence) only — full feature/probs dict requires a `predict_full` extension. | optional Sprint 3 task to close the two advisory bullets in Lane E | adds Mahalanobis check + reliability bin coverage | pending |
| 2026-05-18 | finding | All 3 v2 models show NEGATIVE expected net P&L at every threshold under symmetric-payoff approximation (5+5bps × $50 × 20bps target). | TP-driven wins (0.8% default) produce asymmetric payoff that lifts net P&L positive once live sweep emits rich `threshold_metrics`. | re-run `scripts/select_cost_aware_threshold.py` against real fills | pending |
| 2026-05-18 | sprint_land | Sprint 2 #6 (OutcomeAdjuster) shipped (commit 52a62f6): Redis hash `regime_outcome_adjustment` with per-label streak math (3 losses → +0.01, 5 wins → -0.01, bound ±0.05). RegimeLookup applies delta to optimal_threshold. CLI + launchd plist 00:10 UTC. +41 tests, 1454 OK. | the regime that's been losing trades less; the regime that's been winning trades more | observe hash values move after ≥3-7 days of regime-tagged trades | shipped (forward-compatible) |
| 2026-05-18 | finding | OutcomeAdjuster ran against today's 6 smoke trades: resolved labels 0/6 skipped. Supervisor's `signal_snapshot["regime_label"]` was never populated; adjuster's resolver had nothing to read. | tiny supervisor patch (Sprint 2.6) wires the regime label through the existing snapshot path | next tick produces regime-tagged Position + snapshot | shipped (commit 899bd89) |
| 2026-05-18 | sprint_land | Sprint 2.6 (commit 899bd89): regime_label plumbed through predictor cache → signal snapshot → Position. End-to-end Redis inspection confirms position.regime_label, signal.regime_label, signal.risk_metrics_input.regime_label all carry the label on a fresh tick. +2 tests, 1456 total. | next paper smoke produces regime-tagged trades; OutcomeAdjuster recomputes adjustments daily | regime_outcome_adjustment hash gains non-zero entries within first 30+ trade week | shipped, smoke validation pending |
| 2026-05-18 | smoke_run | 30-min all-3 v2 paper smoke (run dir 2026-05-19T02-23-00Z). 217 iterations × 3 symbols = 651 ticks. | Sprint 2.6 produces regime-tagged trades end-to-end | 0 entries — ETH max P(long)=0.542 vs thr=0.58, BTC max 0.498 vs 0.55, SOL max 0.542 vs 0.58. No signal crossed threshold in this window. 0 exits, 0 errors, 0 zombies. | inconclusive — market was in a non-firing regime for this 30-min window |
| 2026-05-18 | finding | Regime store dim mismatch: 540-dim stores (2026-05-11 backfill, 135 features) vs 480-dim current encoder (120 features after voln-normalized rework). Predictor falls back to static threshold gracefully (651 warnings, 0 crashes). Regime lookup is effectively OFF in prod until stores regenerate. | regenerate regime stores against current encoder; ~15 min per symbol per autopilot-regime-backfill-e2e-2026-05-11 | OutcomeAdjuster + Kelly sizing + signal regime label all engage on next smoke | pending |
