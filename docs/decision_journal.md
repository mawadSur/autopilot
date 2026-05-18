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
| 2026-05-18 | finding | All 3 v2 models show NEGATIVE expected net P&L at every threshold under symmetric-payoff approximation (5+5bps × $50 × 20bps target). | TP-driven wins (0.8% default) produce asymmetric payoff that lifts net P&L positive once live sweep emits rich `threshold_metrics`. | re-run `scripts/select_cost_aware_threshold.py` against real fills | pending |
