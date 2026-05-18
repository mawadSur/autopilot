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
