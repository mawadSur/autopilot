# Contributing & Standards

> **The standards in this repo are locked.** They live in the
> [`🔒 PROJECT CONSTITUTION (LOCKED)`](./CLAUDE.md) section of `CLAUDE.md` and may only be
> changed with the secret word **`GETTINGAJET`** (all caps). If you're here to *follow* the
> standards, read on. If you want to *change* them, see [Amending the standards](#amending-the-standards).

This document is the human-facing onboarding guide. `CLAUDE.md` and `AGENTS.md` are the
machine-facing sources of truth — this file just gets a new contributor (human or AI) to the
same way of working fast.

## The mission

Build a **fully automatic trading bot** that continuously finds the best opportunities across
**anything tradeable** — stocks, crypto, prediction markets, or any other instrument — and acts
on them. Every contribution should move toward better signal discovery, calibration, risk
control, or autonomy.

## The way we work: Ruflo

This repo is driven through **[Ruflo](https://www.npmjs.com/package/ruflo)** — an MCP server
plus a set of Claude Code hooks, helpers, agents, and skills that give every contributor the
same memory, routing, and multi-agent coordination. **Use it; don't free-hand around it.**

### 1. One-time setup

```bash
# 1. Python environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Secrets — copy and fill in. NEVER commit .env.
#    See "Environment Variables" in CLAUDE.md for the full key list.
cp .env.example .env   # if present; otherwise create .env

# 3. Ruflo is wired in .mcp.json and .claude/. Open the repo in Claude Code and it
#    auto-loads: the MCP server, the hooks in .claude/settings.json, and project skills.
#    Confirm the toolchain is reachable:
npx -y ruflo@latest --version
```

You don't run Ruflo by hand — Claude Code launches the `ruflo` MCP server (see `.mcp.json`)
and the hooks fire automatically on session start, prompts, edits, and commands.

### 2. The task loop (every non-trivial change)

1. **Recall first.** Search prior knowledge (`memory_search` / `memory_retrieve`) and read any
   `[INTELLIGENCE]` pattern hints before starting. Don't re-solve a solved problem.
2. **Route the task.** Use the agent/skill the router recommends, or pick the closest
   specialized one on purpose.
3. **Coordinate multi-step work through Ruflo** (`swarm_init` + `agent_spawn`, or the project
   skills in `.claude/skills/`) — not ad hoc. The research → calibration → risk → synthesis
   pipeline is the canonical example.
4. **Persist what you learn** with `memory_store` so the next session inherits it.
5. **Let the hooks run.** Don't disable or bypass `.claude/settings.json` hooks.

### Project skills (`.claude/skills/`)

| Skill | Use it for |
|---|---|
| `pipeline-orchestrator` | Running the full scanner → research → calibration → risk → synthesis pipeline |
| `reddit-research` | Social/narrative research on a market or asset |
| `narrative-calibrator` | Turning narrative signal into a calibrated probability |
| `risk-gatekeeper` | Kelly/risk sizing and gating a trade before it's taken |
| `execution-reviewer` | Forensic post-mortem on a single trade execution |
| `post-mortem-auditor` | Broader post-hoc review/auditing of outcomes |

## Engineering standards (non-negotiable)

These are enforced by the Constitution. Summary:

- **Safety first.** Trading defaults to paper mode; live trading is an explicit opt-in. Never
  weaken a risk gate, stop, or position cap to flatter a metric.
- **No look-ahead / no data leakage.** Only use information available at decision time. Record
  real entry prices and real fills.
- **Honest reporting.** Losses, failing tests, and skipped steps get reported with evidence.
- **Reproducible.** Use `./.venv/bin/python`; secrets in `.env`, never in code; don't commit
  generated artifacts (`output/`, `model_sanity/`) as if they were source edits.
- **House style.** Follow `AGENTS.md` (4-space indent, `snake_case`/`PascalCase`, type hints
  where practical, no broad style-only diffs).

## Running the code

```bash
# Polymarket scanner
./.venv/bin/python main.py --top 20 --category Politics

# Multi-agent orchestrator
./.venv/bin/python src/orchestrator.py --top 5 --category Politics --subreddit politics
```

## Tests — three trees, three runners

```bash
# Prediction-market suite (unittest, needs PYTHONPATH=src)
env PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests/prediction_market_scanner

# Legacy core (pytest, needs torch + TA-Lib)
./.venv/bin/python -m pytest tests/test_core.py

# Social narrative agent (unittest, self-contained)
./.venv/bin/python -m unittest discover social-narrative-agent/tests
```

Put new tests in the matching tree; name them `test_*.py`.

## Commits & PRs

Short, imperative, specific subjects (e.g. `Add scanner category filter`). PRs state the
affected surface (scanner, orchestrator, dashboard, legacy trading), list the commands/tests
run, note any new env vars, and include screenshots only for dashboard/UI changes. See
`AGENTS.md` for the full convention.

## Amending the standards

The Constitution in `CLAUDE.md` is locked. To change, remove, or weaken any standard — or the
lock itself — your request must contain the secret word **`GETTINGAJET`** (exactly, all caps).
Without it, the change is refused. When authorized, the change is recorded in an
`## Amendments` log inside the Constitution section.
