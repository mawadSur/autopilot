---
name: narrative-calibrator
description: Interactive synthesis + calibration for a single prediction market. Compares social/news narrative against market-implied odds, identifies contradictions, applies a base-rate-aware narrative discount, and flags resolution ambiguity (sole-discretion clauses → REJECT). Use when a market needs a deeper calibration than the batch agent provided, or when you suspect the narrative is overpriced. Complements src/synthesis_agent and src/calibration_agent.
---

# Narrative vs. Odds Calibrator

Analyze the research logs for a specific market. Compare the "Social Narrative" (buzz, sentiment) against the "Market Odds" (implied probability).

## Inputs

The user will identify the market — by `trade_execution_<id>.json` path, market title, or pasted research payload. If unclear, **ask** which market.

You should find available context in:
- The trade JSON's `scanner` row (`current_implied_prob`, `market_title`, rules)
- The trade JSON's `research.reddit_report` and `research.news_report` (already-run agent outputs)
- If those are missing, look at `output/scan_*.json` for the most recent scan that includes this market
- Polymarket API directly via `src/fetcher.py` if you need fresher market data

## Calibration logic

Run these in order:

1. **Identify the 'Base Rate'**: What is the historical frequency of events like this? State it explicitly with a one-line justification (e.g., "Base rate ~15% — historically incumbent presidents win re-election ~70% of the time, but this is a primary challenge so closer to challenger base rate.").

2. **Apply Narrative Discount**: If news is high-volume but secondary (opinion pieces, not primary events), discount the 'True Probability' towards the base rate. Quantify the discount in basis points or percentage. Examples:
   - Heavy Twitter/Reddit buzz with no AP/Reuters confirmation → discount 30-50% toward base rate.
   - Single primary-source report with broad secondary coverage → no discount.
   - Polling-driven thesis with stale (>14 days) polls → discount 20% toward base rate.

3. **Check for Resolution Ambiguity**: Read the market contract / rules text. Is there a "sole discretion" clause, "general consensus" wording, or subjective adjudication? If so, **flag as `REJECT` and stop** — do not produce an edge estimate.

## Output

Emit a YAML block with these fields:

```yaml
estimated_true_prob: <float 0.0-1.0>
implied_probability: <float 0.0-1.0>  # echoed from market data
edge_estimate: <float — true_prob minus implied_prob>
base_rate: <float 0.0-1.0>
narrative_discount_applied: <float 0.0-1.0 — fraction of distance moved toward base rate>
```

**If `|edge_estimate| > 0.10`** (10 percentage points), also provide:
- `main_reason`: one sentence explaining the edge
- `top_risk`: one sentence on what would invalidate the call

**If you flagged REJECT for ambiguity**, return:
```yaml
status: REJECT
reason: <which clause triggered the rejection, quoted from the rules>
```

## Deltas vs. existing Python

- `src/synthesis_agent` (`SynthesisAgent.synthesize_edge`) produces a `SynthesisReport` with verdict + reasons but does NOT apply an explicit base-rate-toward narrative discount — it weights narrative qualitatively via Gemini.
- `src/calibration_agent` (`CalibrationService`) combines an XGBoost ML baseline with a Gemini calibration step but does NOT enforce a hard "REJECT on sole discretion" rule.

This skill makes both behaviors explicit and deterministic. **When this skill rejects a market that the batch agents pass, prefer the skill.**

## When NOT to use

- For batch calibration across many markets — use `src/orchestrator.py` (calls the batch agents).
- For pure execution / slippage analysis — use `/execution-reviewer`.
- For data-integrity auditing (stale data, duplicate sources) — use the `data_quality_auditor` Python agent.
