# Autopilot

## Disclaimer — Read First

This is a **research, backtesting, and paper-trading** workbench. Nothing in the prediction-market pipeline places real-money trades. Specifically:

- The prediction-market stack only writes decision logs (`trade_execution_*.json`) and runs paper-trade simulations.
- No prediction-market broker connector is wired in.
- The legacy crypto trading stack (`src/live_trader.py`) is independent of the prediction-market pipeline and is not invoked by it.
- This software does not predict the future. Prediction markets are speculative; outcomes are uncertain. Treat every output as a research artifact, not investment advice. There is no guaranteed profit, and there is no claim of one.

If you adapt this for live execution, you are on your own — review every gate, read every log, and assume the system will be wrong.

## Status

See [`features.md`](./features.md) for the full mapping of the prediction-market-bot spec against what's implemented today (✅ done · 🟡 partial · ❌ missing).

Six Claude Code skills live under `.claude/skills/` covering the operator-driven path: `pipeline-orchestrator`, `narrative-calibrator`, `risk-gatekeeper`, `execution-reviewer`, `post-mortem-auditor`, `reddit-research`. The Python agents under `src/` cover the batch path. Both run side by side — see the Architecture section below.

## Crypto Trading Build Progress

The repo is currently being extended for **automated crypto trading on Coinbase + Hyperliquid** (intraday/short-horizon, target $100K bankroll). Build phases tracked here:

| Phase | Component | Status | Notes |
|---|---|---|---|
| 1 | Coinbase connector (`src/exchanges/coinbase.py`) | ✅ done | ccxt-backed; sandbox toggle via `COINBASE_USE_SANDBOX`. Note: ccxt 4.5 has no Coinbase sandbox URL — kill-switch + operator-intent gates still apply, but Phase 5 needs a paper-trade simulator behind the same interface. 15 tests. |
| 2 | Risk circuit breakers (`src/risk/circuit_breakers.py`) | ✅ done | Kill switch (force_flat) > daily loss / drawdown / notional caps (halt_new_entries) > allow. 19 tests. |
| 3 | Redis position state (`src/state/position_store.py`) | ✅ done | `fakeredis` for tests; `redis>=5.0` added to requirements; `reconcile()` drops orphan pending positions > 1h with no exchange-side order. 18 tests. |
| 4 | Alerts pipeline (`src/alerts/notifier.py`) | ✅ done | Discord (info/fills/daily summary) + Telegram (alert/critical with MarkdownV2 escape). Best-effort — never raises to caller. 14 tests. |
| 5 | Live supervisor + 14-day shakedown gate (`src/live_supervisor.py`) | ✅ done | Tick loop wires Phase 1-4. Shakedown resets on uncaught error / kill-switch trip / daily-loss breaker trip. Paper mode synthesizes fills at mid ± 5 bps slippage. 16 tests. |
| 6 | Hyperliquid perps adapter (`src/exchanges/hyperliquid.py`) | ✅ done | Read-only V1 (info / clearinghouseState / userFills). Write methods raise `NotImplementedError` — EIP-712 signing via `eth-account` is intentionally deferred (avoids heavy crypto deps). 14 tests. |
| 7 | Monitoring (Sentry + Prometheus) | 🟡 mostly done | `src/observability/monitoring.py` + supervisor hooks landed (gauges, counters, histogram, sentry capture). Standalone `test_observability.py` is missing (agent timed out); supervisor tests still cover the integration path. Follow-up: add the dedicated unit tests + Grafana dashboard JSON. |

**Test suite:** 403 tests, all green, 0.5s runtime. Six phases of crypto trading infra wired and tested.

**Mandatory gates:**
- No live mode until ≥14 days of clean paper-trade PnL on the supervised loop.
- No friends-and-family money until ≥6 months of personal live trading with positive Sharpe AND a securities lawyer in the loop.
- `KILL_SWITCH_FILE` must be set; `touch $KILL_SWITCH_FILE` halts new entries at the next decision tick.

See `.env.example` Section 4 for the full env-var contract (Coinbase, Hyperliquid, Kraken, Redis, Discord, Telegram, Sentry, circuit-breaker thresholds).

## Overview

Autopilot is currently a Python research and trading workspace, not a dedicated Codex app scaffold.

The repo has three active surfaces:

- `main.py`: a CLI scanner for active Polymarket markets.
- `src/orchestrator.py`: a multi-agent research and calibration pipeline for the top-ranked markets.
- `social-narrative-agent/`: a separate OpenAI-powered CLI for Reddit/social narrative analysis.

The older crypto trading stack is still present under `src/`, but the previous README described the repo as a single FastAPI trading app and no longer matched the code.

## Codex App Status

If you were looking for a Codex/OpenAI app setup, none is wired here yet.

- No `.codex-plugin` manifest was found.
- No `package.json` or frontend app scaffold was found.
- The repo is Python-first and currently organized around CLIs, services, and research agents.

If you want, this repo can be turned into a Codex app later, but that would be a new scaffold rather than documenting something that already exists.

## What Lives Here

### 1. Market Scanner

`main.py` fetches active Polymarket markets, filters low-quality setups, runs an LLM-based clarity check, scores research priority, prints a ranked table, and exports JSON to `output/scan_*.json`.

Data sources and logic:

- Polymarket Gamma API via `src/fetcher.py`
- market heuristics via `src/analyzer.py`
- ranking via `src/ranker.py`
- Gemini-based clarity and narrative scoring via `src/llm_judge.py`

### 2. Multi-Agent Orchestrator

`src/orchestrator.py` takes the top scanner results and adds deeper research:

- Reddit discussion context via `src/reddit_research_agent/`
- Google News RSS context via `src/news_research_agent/`
- Gemini-based Reddit/news/calibration agents under `src/*_agent/`
- an ML baseline from `src/calibration_agent/ml_baseline.py`

The output is a calibrated probability plus an action such as `paper-trade candidate` or `monitor`.

### 3. Social Narrative Agent

`social-narrative-agent/main.py` is a separate CLI that:

- pulls Reddit posts and comments with `requests`
- uses the OpenAI API for structured claim extraction and narrative analysis
- compares the crowd narrative with current market odds

### 4. Legacy Trading Stack

The original crypto-trading code still exists and appears to be maintained separately from the newer prediction-market flow:

- `src/main.py`: FastAPI control API
- `src/dashboard_server.py`: telemetry/state server
- `src/dashboard_app.py`: Streamlit dashboard
- `src/history.py`, `src/train_model.py`, `src/backtest.py`, `src/live_trader.py`, `src/deploy.py`: data, training, backtest, and deployment scripts

## System Architecture

The prediction-market pipeline runs as a 7-stage chain. Stages 1-6 happen at decision time; stage 7 happens after the trade settles.

```
                Polymarket Gamma API
                         |
                         v
  [1. SCAN]      main.py  -> output/scan_*.json
                         |   src/fetcher.py + analyzer.py + ranker.py + llm_judge.py
                         v
  [2. RESEARCH]  reddit_research_agent  +  news_research_agent
                         |
                         v
  [3. SYNTHESIZE] synthesis_agent         (narrative vs. odds)
                         |
                         v
  [4. CALIBRATE]  calibration_agent       (XGBoost baseline + Gemini)
                         |
                         v
  [5. RISK]       risk_management_agent   (Kelly + liquidity + correlation penalties)
                         |
                         v
  [6. EXECUTE]    orchestrator -> trade_execution_<market_id>.json
                         |   (status="open"; source="orchestrator"; no broker connector)
                         |
                         |   Two additional writers produce the same canonical schema
                         |   so the calibration dataset doesn't depend on
                         |   real-decision throughput:
                         |     - shadow_capture.py    -> source="shadow"   (full fidelity, bulk)
                         |     - backfill_from_polymarket.py -> source="backfill" (DEGRADED, smoke-test only)
                         |   Downstream consumers don't care about the source —
                         |   only build_dataset.py filters on it (orchestrator+shadow by default).
                         |
                  mark_trade_settled.py   (flips status, fills final_outcome)
                         |
                         v
  [7. AUDIT]      PerformanceTracker
                         |--> outcome_review_agent     (4-quadrant matrix)
                         |--> data_quality_auditor     (7 failure modes)
                         |--> iterative_improver       (Good Failure -> 3 features)
                         '--> performance_audit.json
                                       |
                                       v
                  analytics_dashboard.py CLI
```

Two parallel control surfaces drive the same agents:
- **Batch (Python):** `src/orchestrator.py` runs the full pipeline non-interactively. Use for many markets at once.
- **Interactive (Claude Code skills under `.claude/skills/`):** `/pipeline-orchestrator`, `/narrative-calibrator`, `/risk-gatekeeper`, `/execution-reviewer`, `/post-mortem-auditor`. Use for deep one-off analysis, overrides, and debugging. Skills can override batch verdicts.

## Setup

Python `3.10` is the safest assumption here. The checked-in `Dockerfile` also uses Python 3.10.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- `TA-Lib` is required by the dependency list and may need a system install or `conda-forge`.
- `requirements.txt` includes the full stack: ML, AWS, FastAPI, Streamlit, Gemini, OpenAI, Reddit, and data tooling.

## Environment Variables

`src/config.py` loads `.env` from either the repo root or `src/.env`.

Example minimum setup:

```env
# Recommended for the scanner and required for the multi-agent Gemini flows
GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-2.5-flash

# OPTIONAL — only needed if you want src/orchestrator.py to fetch live Reddit
# data via PRAW. When unset, the Python fetcher logs a one-time INFO notice
# and returns deterministic mock data. For interactive Reddit research,
# use the /reddit-research skill (Devvit MCP) instead.
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=autopilot-reddit-research-agent/1.0

# Required only for social-narrative-agent/
OPENAI_API_KEY=your_openai_key

# Optional legacy trading / AWS settings
COINDESK_API_KEY=your_coindesk_key
ENDPOINT_NAME=your_sagemaker_endpoint
SAGEMAKER_ROLE_ARN=your_role_arn
AWS_REGION=us-east-1
```

Important behavior:

- `main.py` can still run without `GEMINI_API_KEY`; it falls back to a neutral judge result when the LLM call fails.
- `src/orchestrator.py` requires Gemini. Reddit credentials are **optional** — when missing, the Reddit fetcher degrades to deterministic mock data and emits a one-time INFO log pointing at the `/reddit-research` skill (Devvit MCP) for the live interactive path.
- `social-narrative-agent/` requires `OPENAI_API_KEY`.

## Common Commands

### Scan active markets

```bash
python main.py --top 20 --category Politics
```

Useful flags:

- `--top`: number of rows to print
- `--category`: case-insensitive category filter
- `--min-volume-24h`: liquidity floor
- `--max-pages`: cap Polymarket pagination

### Run the multi-agent orchestrator

```bash
python src/orchestrator.py --top 5 --category Politics --subreddit politics
```

This pulls ranked markets, Reddit context, news context, and calibration output for the top candidates.

### Interactive Reddit research (Devvit MCP)

Replaces the PRAW-based Python Reddit agent for one-off, operator-driven analysis on a single market. Live Reddit data is fetched through the Devvit MCP from inside a Claude Code session — no PRAW credentials required.

```bash
# One-time setup
claude mcp add devvit -- npx -y @devvit/mcp

# Then in a Claude Code session
/reddit-research
```

The skill emits a JSON `RedditResearchReport` payload that can be saved into a `trade_execution_<market_id>.json`'s `research.reddit_report` slot or piped into `/risk-gatekeeper` / `/narrative-calibrator`. For batch processing of many markets, keep using `src/orchestrator.py` (with PRAW credentials, or with `RESEARCH_MOCK=true` for deterministic mock data).

### Run the social narrative agent

```bash
python social-narrative-agent/main.py --topic "OpenAI GPT-5 release" --current-odds 0.42
```

### Building a calibration dataset

The XGBoost calibration baseline is trained from `trade_execution_<id>.json` logs. Three writers produce these logs (see `features.md` → "Data Sources" for the full table):

- `src/orchestrator.py` — `source="orchestrator"`, full fidelity, low volume (per real decision).
- `src/calibration_agent/shadow_capture.py` — `source="shadow"`, full fidelity, high volume (every active market per scan). Run as a daemon to accumulate clean data fast.
- `src/calibration_agent/backfill_from_polymarket.py` — `source="backfill"`, **degraded fidelity** (post-resolution snapshots). Smoke-test only; opt in with `--include-backfill`.

Typical end-to-end:

```bash
# 1) Accumulate data — pick one or both
./.venv/bin/python src/calibration_agent/shadow_capture.py \
    --output-dir ./shadow_logs --interval-seconds 3600
./.venv/bin/python src/calibration_agent/backfill_from_polymarket.py \
    --output-dir ./shadow_logs --limit 200

# 2) Wait for the underlying markets to resolve, then mark settled
./.venv/bin/python src/mark_trade_settled.py ./shadow_logs

# 3) Assemble the training table (orchestrator + shadow only by default)
./.venv/bin/python src/calibration_agent/build_dataset.py ./shadow_logs \
    --output ./datasets/calibration.parquet

# 3b) Or include backfill rows for a smoke-test of the training pipeline
./.venv/bin/python src/calibration_agent/build_dataset.py ./shadow_logs \
    --output ./datasets/calibration_smoke.parquet --include-backfill

# 4) Train the XGBoost calibration baseline
./.venv/bin/python src/calibration_agent/train_xgboost.py \
    ./datasets/calibration.parquet --output ./models/calibration.joblib

# 5) Evaluate the saved model against the dataset
./.venv/bin/python src/calibration_agent/evaluate_xgboost.py \
    ./models/calibration.joblib ./datasets/calibration.parquet
```

### Prediction-market FastAPI

The prediction-market scanner + agent pipeline is exposed over HTTP via
`src/api/main.py`. This is **distinct** from the legacy crypto-trading
FastAPI at `src/main.py` and runs on its own port.

```bash
./.venv/bin/uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8001
```

Endpoints (see `src/api/models.py` for request/response shapes):

```bash
# Liveness probe.
curl -s http://127.0.0.1:8001/health

# Run the Polymarket scanner.
curl -s -X POST http://127.0.0.1:8001/scan \
    -H 'content-type: application/json' \
    -d '{"top_n": 5, "category": "Politics", "min_volume_24h": 5000}'

# Reddit + News research for one market id.
curl -s -X POST http://127.0.0.1:8001/research \
    -H 'content-type: application/json' \
    -d '{"market_id": "0x...", "subreddits": ["politics"]}'

# Calibration agent (XGBoost baseline + Gemini) for one market id.
curl -s -X POST http://127.0.0.1:8001/predict \
    -H 'content-type: application/json' \
    -d '{"market_id": "0x..."}'

# Risk engine + assessment, given a calibration payload from /predict.
curl -s -X POST http://127.0.0.1:8001/risk \
    -H 'content-type: application/json' \
    -d '{"market_id": "0x...", "calibration": {...}, "bankroll": 10000}'

# End-to-end pipeline: scan → research → calibrate → risk → write
# trade_execution_<id>.json. The log lives under $AUTOPILOT_TRADE_STORE
# (defaults to the repo root).
curl -s -X POST http://127.0.0.1:8001/paper-trade \
    -H 'content-type: application/json' \
    -d '{"market_id": "0x...", "top_n": 5, "bankroll": 10000}'

# Mark a trade log settled (in-place mutation of trade_execution_<id>.json).
curl -s -X POST http://127.0.0.1:8001/settle \
    -H 'content-type: application/json' \
    -d '{"market_id": "0x...", "outcome": "win", "market_outcome": "yes",
         "exit_price": 0.78, "realized_pnl_usd": 120.0,
         "news": "Resolution headline."}'

# List trade logs filtered by status and/or source.
curl -s 'http://127.0.0.1:8001/trades?status=open&source=orchestrator&limit=50'

# List post-mortem reviews from performance_audit.json.
curl -s 'http://127.0.0.1:8001/postmortems?limit=50'
```

Environment variables read by this app:

- `AUTOPILOT_TRADE_STORE` — directory for `trade_execution_<id>.json`
  logs and `performance_audit.json` (default: repo root).
- `APP_VERSION` — version string surfaced by `GET /health` (default `0.1.0`).
- `RESEARCH_MOCK` — when truthy, the research agents return deterministic
  mock data instead of calling Reddit/News/Gemini.

### SQLite mirror (optional)

The trade execution logs (`trade_execution_*.json`) and post-mortem reviews
(`performance_audit.json`) remain the canonical write path. As an *additive*
secondary index, the SQLite mirror lets the FastAPI `/trades` + `/postmortems`
endpoints (and `build_dataset.py`) query rows fast without walking the
filesystem on every call.

The mirror is opt-in via `AUTOPILOT_SQLITE_PATH`; when unset, all behavior is
unchanged.

```bash
# One-time: enable SQLite mirroring
export AUTOPILOT_SQLITE_PATH=$(pwd)/autopilot.sqlite

# Sync existing JSON files into SQLite
./.venv/bin/python src/storage/sync.py
```

The sync CLI accepts `--trade-store-dir`, `--audit-path`, and `--db-path`
overrides for non-default layouts. With the env var set, `build_dataset.py`
reads from SQLite by default; pass `--source files` to force the legacy
filesystem walk or `--source sqlite` to require it.

### Run the legacy FastAPI trading API

```bash
python src/main.py
```

### Run the legacy dashboard state server

```bash
python src/dashboard_server.py
```

### Run the legacy Streamlit dashboard

```bash
streamlit run src/dashboard_app.py
```

## Tests

The repo uses `unittest`.

Example targeted runs:

```bash
env PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests/prediction_market_scanner -p 'test_main.py'
env PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests/prediction_market_scanner -p 'test_orchestrator.py'
./.venv/bin/python -m unittest discover social-narrative-agent/tests
```

## Repository Map

```text
.
├── main.py                         # Polymarket scanner CLI
├── social-narrative-agent/         # Separate OpenAI-based social analysis CLI
├── src/
│   ├── orchestrator.py             # Multi-agent research + calibration runner
│   ├── llm_judge.py                # Gemini-based market clarity/narrative judge
│   ├── reddit_research_agent/      # Reddit fetch + analysis
│   ├── news_research_agent/        # Google News RSS fetch + analysis
│   ├── calibration_agent/          # ML baseline + calibration agent
│   ├── main.py                     # Legacy FastAPI trading API
│   ├── dashboard_server.py         # Legacy telemetry server
│   └── dashboard_app.py            # Legacy Streamlit dashboard
└── tests/                          # Prediction-market and agent tests
```

## Safety Notes

- **Prediction markets are speculative.** Calibrated probability estimates from this system are research outputs, not investment advice. Adverse selection is real (the people on the other side of a contract may know more than the model).
- **No execution path by default.** The prediction-market pipeline does not connect to any broker or exchange. If you wire one in, do so behind explicit feature flags + manual approval gates, and add tests before the first run.
- **All decisions are auditable.** Every trade decision writes `trade_execution_<market_id>.json`; every post-mortem appends to `performance_audit.json`. Do not delete these without backups; they are the only record of system behavior.
- **API keys live in `.env`** (never committed): `GEMINI_API_KEY`, `REDDIT_CLIENT_ID`/`SECRET`/`USER_AGENT`, `OPENAI_API_KEY`. The legacy stack also reads `COINDESK_API_KEY` and AWS credentials.
- **Resolution risk.** Many prediction markets resolve via subjective adjudication (e.g. "general consensus", "sole discretion"). The interactive `/risk-gatekeeper` skill flags these as auto-REJECT; the batch `risk_management_agent` does not yet — track this in [`features.md`](./features.md).

## Known Gaps

For the full status mapping (what's done, partial, missing) against the prediction-market-bot spec, see [`features.md`](./features.md). Headline items:

- The checked-in `Dockerfile` still points at an older `main.py --step trade` interface that does not match the current root CLI.
- No `pyproject.toml`, no `.env.example`, no `ruff` config, no LLM-provider abstraction — see features.md sections "Toolchain" and "Documentation".
- ~~No FastAPI app for the prediction-market pipeline~~ — implemented in `src/api/main.py`. Endpoints `/health`, `/scan`, `/research`, `/predict`, `/risk`, `/paper-trade`, `/settle`, `/trades`, `/postmortems` are live; see "Prediction-market FastAPI" under Common Commands.
- ~~Storage is filesystem JSON; the spec calls for SQLite.~~ — additive SQLite mirror lives at `src/storage/`. JSON files remain canonical; SQLite is opt-in via `AUTOPILOT_SQLITE_PATH`. See "SQLite mirror (optional)" under Common Commands.
- Twitter/X research adapter is missing entirely (Reddit + News/RSS only).
- Paper-trade tracking captures decisions but not entry/exit prices or PnL.
