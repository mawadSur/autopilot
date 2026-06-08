---
name: risk-gatekeeper
description: Interactive risk + execution gatekeeper for a single market that has already passed calibration. Applies hardcoded liquidity, volume, and clarity thresholds (spread under 2 percent, 50 dollar entry without 0.5 percent market move, time-bounded objective resolution) as binary go/no-go gates. Hard-overrides any prior allow_trade=true verdict if resolution ambiguity is detected. Use as the final 'Dr. No' before paper trade execution.
---

# Risk Management & Execution Gatekeeper

Review the calibrated candidate for Paper Trade readiness.

## Inputs

- The calibrated market verdict (from `/narrative-calibrator` output, or the batch `calibration_agent` output stored in a `trade_execution_*.json` file).
- The market's order book / scanner data — if `bid`, `ask`, `spread`, `volume_24h` are missing, **say so explicitly** rather than guess. The gatekeeper cannot approve trades on absent data.
- The market's resolution rules text — required for the clarity gate.

## Risk constraints

Each constraint must pass for the trade to be approved. Any FAIL → REJECT.

| Constraint | Threshold | Source field |
|---|---|---|
| **Liquidity** | bid-ask spread < 2% | `scanner.spread` or computed from `scanner.best_bid` / `scanner.best_ask` |
| **Volume** | $50 entry should not move the market > 0.5% | `scanner.volume_24h` (proxy) or order book if available |
| **Clarity** | Resolution is time-bounded AND objective (e.g. "Official Government Report" — pass; "General Consensus" — fail) | market rules text |

For the volume gate when only `volume_24h` is available, use this heuristic: a market with > $10K daily volume can typically absorb $50 without > 0.5% impact. Be conservative if liquidity is thin or the order book is unavailable.

## Hard rule (overrides ALL other signals)

If 'Resolution Ambiguity' is detected, you MUST change status to `REJECT` regardless of the 'Edge Estimate', regardless of liquidity, and regardless of any prior `allow_trade=true` verdict from the batch `risk_management_agent`.

Examples that trigger ambiguity rejection:
- "sole discretion" clauses
- "general consensus" / "widely recognized" / "commonly understood" resolution
- Multiple resolution sources without a clear tiebreaker
- Subjective adjudication (e.g., "popular opinion")

## Output

Emit a YAML block with these fields:

```yaml
status: <reject | paper-trade>
constraint_check:
  liquidity: <pass | fail> — <one-line evidence>
  volume: <pass | fail> — <one-line evidence>
  clarity: <pass | fail> — <one-line evidence>
ambiguity_override_triggered: <true | false>
next_step: <string>
```

`next_step` examples:
- `"Execute paper trade at $0.65 for $50"` (status=paper-trade)
- `"Monitor for better liquidity (current spread 3.2%)"` (status=reject, liquidity fail)
- `"REJECT: rules contain 'general consensus' resolution clause"` (status=reject, ambiguity override)

## Deltas vs. existing Python

`src/risk_management_agent/risk_engine.py` applies Kelly-fraction sizing with **soft, continuous penalties** for liquidity and correlation. It produces a `RiskAssessment` with `allow_trade` (bool) and `simulated_position_size_pct` (continuous).

This skill applies the explicit thresholds (2%, $50, 0.5%) as **hard binary gates** — no soft penalties. The Python agent may return `allow_trade=true` while this skill returns `REJECT`. **The skill verdict wins for interactive overrides.**

The skill also carries the explicit ambiguity hard-override that the Python agent currently lacks.

## When NOT to use

- For batch risk sizing across many markets — `risk_management_agent` is fine for that.
- When you need Kelly fraction sizing rather than binary go/no-go — same.
- For execution-mechanics post-mortem on a *settled* trade — use `/execution-reviewer`.
