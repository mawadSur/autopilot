---
name: post-mortem-auditor
description: Interactive post-mortem on a single settled trade. Compares Estimated True Probability vs. Actual Outcome and classifies the failure into one of three modes (execution / model / metadata). Writes a Changelog Entry suitable for an evolving CHANGELOG, and if the diagnosis is a missing/broken feature, suggests the change in src/check_features.py without editing preemptively. Use for deeper one-offs beyond what the batch outcome_review_agent provides.
---

# Post-Mortem Auditor

Compare the 'Estimated True Probability' from the research phase with the 'Actual Outcome' for a settled trade. Classify what went wrong.

## Inputs

The user will identify a settled trade — by `trade_execution_<id>.json` path or by an entry in `performance_audit.json`. The trade must have `final_outcome` set (i.e. `mark_trade_settled.py` has been run). If the trade isn't settled, **stop and tell the user** — do not invent an outcome.

You should pull from:
- The trade JSON (`trade_execution_*.json`): `calibration.calibrated_true_prob`, `final_outcome`, `scanner` row + market rules, `research.news_report`, `research.reddit_report`.
- The audit JSON (`performance_audit.json`) if it exists: prior `outcome_review` (4-quadrant matrix), `data_quality_review`, `iterative_improver_review`. Use these to avoid duplicating analysis.

## Investigation

Answer in order. Each step is required.

1. **Was the thesis right but the timing wrong (Execution)?**
   Check inference price vs. actual entry price (may be absent — say so), hold duration vs. information half-life, exit-trigger logic. If yes, route to `/execution-reviewer` for the deeper drill-down rather than duplicating it here.

2. **Was the model over-confident in weak news (Model Error)?**
   Compare `estimated_true_prob` against the realized outcome frequency for similar setups (use base-rate intuition). Check whether `news_report` evidence was thin, secondary, or single-source. Was calibration confidence > 0.7 with low evidence quality?

3. **Did the market resolve in a way the rules didn't clearly define (Metadata Error)?**
   Read `scanner.market_title` and rules. Did the resolution depend on subjective adjudication, "general consensus", or a source not specified at trade entry?

The classification is mutually exclusive — pick the **most material** one. If two contributed roughly equally, pick the one a code change can fix (model > metadata > execution, in that order of leverage).

## Output: Changelog Entry

Emit a single Markdown changelog entry suitable for appending to a CHANGELOG file:

```
### <YYYY-MM-DD> — <market_id>: <one-line verdict>

**Failure mode**: execution | model | metadata
**Estimated true prob**: X.XX → **Actual**: win | loss
**Diagnosis**: <2-3 sentences anchored in trade-log evidence>
**Recommended change**:
- For execution failures: name the file + function + change.
- For model failures: name the missing/broken feature.
- For metadata failures: name the rule heuristic to add (e.g. "REJECT markets with 'general consensus' resolution clauses").
```

## Optional follow-up: feature suggestion

If the failure mode is **model** AND the diagnosis points to a missing/broken feature, suggest the specific change in `src/check_features.py`:
- Quote the existing relevant code.
- Propose the new feature (snake_case name, computation, rationale).
- Explain how it would have caught this case.
- **Do NOT edit the file preemptively — wait for explicit user approval.**

> Note: `src/check_features.py` is part of the legacy ETH 1m crypto stack. Its features don't currently flow into the prediction-market pipeline. Treat the recommendation as a directional pointer until cross-stack feature integration is wired up. The same recommendation may also be useful as a Polymarket feature once that pipeline supports custom features.

## Deltas vs. existing Python

- `outcome_review_agent` (batch via `PerformanceTracker`) classifies into the 4-quadrant **process-vs-outcome matrix** (`Deserved Success`, `Good Failure`, `Dumb Luck`, `Poetic Justice`). This skill is **action-oriented** — produces a changelog entry + code-edit suggestion, not a quadrant verdict.
- `iterative_improver` (batch, conditional on `Good Failure`) produces exactly 3 feature recommendations. This skill triggers on **any** settled trade and produces a **single most-actionable change**.
- `data_quality_auditor` (batch) covers data-integrity failure modes (stale, duplicate, missing primary). This skill covers **execution / model / metadata** — a different axis. Both can run on the same trade.

## When NOT to use

- For pure execution mechanics — `/execution-reviewer` is more focused (reads `src/simulator.py`, returns A-F grade).
- For batch post-mortem across many trades — `outcome_review_agent` runs through `PerformanceTracker` and writes `performance_audit.json`.
- For producing 3 feature recommendations — that's `iterative_improver`'s contract.
- For cross-stack feature engineering on the ETH model — the recommendation in this skill is a pointer; actual implementation belongs in the legacy training scripts (`src/train_model.py`, `src/check_features.py`).
