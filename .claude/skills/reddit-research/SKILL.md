---
name: reddit-research
description: Interactive Reddit research on a single prediction market using the Devvit MCP. Fetches subreddit threads + comments, distills bullish/bearish thesis, sentiment, evidence quality, misinformation risk, and key sources. Use when the user wants Claude-mediated Reddit analysis on a specific market — replaces the legacy PRAW-based Python agent for interactive workflows. Requires Devvit MCP (claude mcp add devvit -- npx -y @devvit/mcp).
---

# Reddit Research (Devvit MCP)

You are a Reddit research analyst for a prediction-market trading system. When invoked, fetch live Reddit discussion via the Devvit MCP, distill the signal, and emit a structured JSON report compatible with the existing `RedditResearchReport` schema (see `src/reddit_research_agent/models.py`).

## Inputs

The user will supply at least:

- `market_title` (string) — the Polymarket question, e.g. "Will the Fed cut rates by July?"
- `market_id` (string) — Polymarket market id, used to namespace any output saved to a `trade_execution_<market_id>.json`
- `current_implied_prob` (float 0.0-1.0) — current YES side market price; needed to assess whether the discussion narrative is consistent with the price
- `subreddit_hints` (optional list of strings) — e.g. `["politics", "neoliberal"]`. If absent, infer from the market category (Politics → r/politics, r/news; Crypto → r/CryptoCurrency, r/ethfinance; Sports → sport-specific subs; etc.).

If any of `market_title`, `market_id`, or `current_implied_prob` is missing, **ask the user** — do not guess.

## Devvit MCP setup (preflight)

Before running any of the steps below, confirm the Devvit MCP tools are loaded. Use `ToolSearch` with a query like `"devvit"` or `"reddit subreddit"` to discover the actual tool surface (the exact tool names depend on the MCP version — likely `mcp__devvit__search_posts`, `mcp__devvit__get_post_comments`, `mcp__devvit__list_subreddit_posts`, or similar).

If `ToolSearch` returns no Devvit tools, stop and tell the user:

> The Devvit MCP doesn't appear to be loaded in this session. Run `claude mcp add devvit -- npx -y @devvit/mcp` and restart Claude Code, then re-invoke `/reddit-research`.

Do not attempt to fall back to PRAW or to the Python `RedditDeepDiver` from inside this skill — the skill exists specifically to replace that path for interactive use.

## Investigation procedure

Run these in order. Keep total fetched context bounded so the analysis fits comfortably in the conversation window.

1. **Pick subreddits.** Start from `subreddit_hints` if given. Otherwise, infer 2-4 candidates from the market category:
   - Politics / elections → `politics`, `news`, `Ask_Politics`, occasionally `neoliberal` or `Conservative` for cross-perspective signal
   - Crypto / tokens → `CryptoCurrency`, `ethfinance`, `Bitcoin`, `solana`
   - Sports / events → the sport-specific sub (`nfl`, `nba`, `soccer`)
   - Tech / company events → `technology`, `business`, plus the company sub if any
   - Geopolitics / conflicts → `geopolitics`, `worldnews`
   Stop at 4 subreddits — broader than that wastes context.

2. **Search posts per subreddit.** Use the Devvit MCP's subreddit-search / post-listing tool (call it via the name `ToolSearch` reveals — likely `mcp__devvit__search_posts` or `mcp__devvit__list_subreddit_posts`). For each subreddit, fetch the top 3-5 recent posts whose title or body mentions the market topic. Use the market_title as the search query (or a tightened version of it, e.g. drop boilerplate like "Will X happen by date Y?" → "X by date Y"). Time-window the search to roughly the past month — older posts are usually stale.

3. **Fetch top comments per relevant post.** For each post that looks substantive (i.e. not a meme thread, not a low-engagement self-post), fetch its top comments via the Devvit MCP's comment-fetch tool. **Cap at ~20 comments per post** to keep context bounded. Prefer comments with high upvotes AND substantive length (>50 words) — short hot takes are noisier than long argued positions.

4. **Distill the signal.** Across all fetched threads + comments, identify:
   - The strongest **bullish thesis** (case for YES) — quote or paraphrase the best-argued version found
   - The strongest **bearish thesis** (case for NO) — same
   - Evidence cited (links, source names, quoted statistics) — this drives `key_evidence` and `evidence_quality_score`
   - Assumptions both sides depend on — drives `key_assumptions`
   - Sentiment balance — count bullish vs. bearish high-quality comments
   - Misinformation risk — check for unsourced confident claims, conspiracy patterns, pump/dump talk, brigading signals (sudden upvote spikes on low-substance posts)

5. **Compare to market price.** The user gave you `current_implied_prob`. If the Reddit consensus and evidence pull strongly toward YES at, say, 80% confidence but the market is at 0.45, that's `underpriced` and should be reflected in `pricing_assessment` + `assessment_reasoning`. Don't anchor too hard on Reddit consensus — Reddit skews young / left-leaning / online-engaged and is not a representative sample. Discount accordingly.

## Scoring guidance

- `conviction_score` (0-10): how strongly does the discussion converge on a single view? 0 = total split / no consensus; 10 = near-unanimous and well-argued.
- `evidence_quality_score` (0-100): are claims grounded in primary sources (official statements, AP/Reuters, government data) or in secondary opinion / Twitter screenshots?
- `misinformation_risk_score` (0-100): higher = more red flags (unsourced claims, conspiracy framing, brigading, sudden tone shifts).
- `sentiment_score` (-100 to +100): -100 strongly bearish (NO), 0 neutral, +100 strongly bullish (YES). Weight by comment quality, not raw upvote counts.

## Output format

Return a single JSON object matching the `RedditResearchReport` schema exactly. No markdown, no commentary outside the JSON — emit it as a fenced ` ```json ` block so it can be copy-pasted into a `trade_execution_<market_id>.json`'s `research.reddit_report` slot or piped into `/risk-gatekeeper` / `/narrative-calibrator`.

```json
{
  "bullish_thesis": "1-3 sentences",
  "bearish_thesis": "1-3 sentences",
  "key_evidence": ["evidence point 1", "evidence point 2"],
  "key_assumptions": ["assumption 1", "assumption 2"],
  "conviction_score": 0,
  "evidence_quality_score": 0,
  "misinformation_risk_score": 0,
  "sentiment_score": 0,
  "key_sources": ["https://reddit.com/r/sub/comments/abc/...", "..."],
  "summary": "2-4 sentence narrative summary",
  "pricing_assessment": "underpriced",
  "assessment_reasoning": "1-3 sentences tying the evidence to the pricing assessment"
}
```

Field constraints (enforced by the Pydantic schema):

- `bullish_thesis`, `bearish_thesis`, `summary`, `assessment_reasoning` — non-empty strings.
- `key_evidence`, `key_assumptions` — list of non-empty strings; aim for ≤ 5 each.
- `conviction_score` — int 0..10.
- `evidence_quality_score`, `misinformation_risk_score` — int 0..100.
- `sentiment_score` — int -100..100.
- `key_sources` — list of up to 10 thread URLs (prefer permalinks, one per top thread cited).
- `pricing_assessment` — exactly one of `"underpriced"`, `"overpriced"`, `"fairly priced"`, `"unclear"`.

If you cannot honestly populate a field (e.g. Reddit had no relevant discussion at all), use `"unclear"` for `pricing_assessment` and explain why in `assessment_reasoning` rather than fabricating a thesis.

## When NOT to use

- **For batch processing of many markets** — the Python orchestrator (`./.venv/bin/python src/orchestrator.py --top N`) handles that. With `RESEARCH_MOCK=true` set, it runs faster than this skill but returns deterministic mock data only — useful for smoke tests, not real signal.
- **Outside an interactive Claude Code session** — the Devvit MCP only exists inside Claude Code. CI/cron/headless scripts must use the Python path (with PRAW credentials, or with `RESEARCH_MOCK=true`).
- **For non-political, non-news, non-crypto markets where Reddit is not a meaningful signal source** — e.g. obscure prop bets, niche scientific outcomes. Use the news agent instead, or skip the Reddit stage entirely.
- **For data-quality auditing** of a settled trade's prior research — that's `data_quality_auditor`'s job.

## Deltas vs. existing Python

`src/reddit_research_agent/` (PRAW + Gemini) runs in batch from `src/orchestrator.py`. It needs `REDDIT_CLIENT_ID`/`SECRET`/`USER_AGENT` env vars and an explicit Gemini call. When those env vars aren't set, the Python fetcher now degrades to mock data (see `src/research_mock.py`) rather than crashing — that change is what makes this skill the recommended interactive path.

This skill differs from the Python agent in three ways:

1. **Live Reddit data via MCP**, no PRAW credential setup required for the user.
2. **Operator-driven subreddit selection** — the user can override category-based defaults inline.
3. **No Gemini round-trip** — the analysis happens directly in this Claude session, so the operator can interactively challenge the verdict before it's saved.

When the skill verdict differs from a prior batch `RedditResearchReport` for the same market, **prefer the skill** — it had access to fresher data and operator judgment.

## Source

Built to replace the PRAW-based interactive path for Reddit research, after the user opted into the Devvit MCP (`claude mcp add devvit -- npx -y @devvit/mcp`). The Python agent stays in the repo for legacy / batch use, with graceful degradation to mock data when no PRAW credentials are present.
