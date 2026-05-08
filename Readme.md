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
| 7 | Monitoring (Sentry + Prometheus) | ✅ done | `src/observability/monitoring.py` + supervisor hooks landed (gauges, counters, histograms, Sentry capture + breadcrumbs). Phase-16 ops work added: tick duration, model confidence, order latency, daily PnL by symbol, shakedown clean-days, kill-switch state, reconciliation orphans/ghosts/drift, auto-pause/auto-trip counters. 10 dedicated tests in `test_observability_metrics.py`. Follow-up: Grafana dashboard JSON. |
| 8 | Real model wired into supervisor (`src/predictor.py`) | ✅ done | `LegacyTransformerPredictor` loads `model_sanity/`, fetches Coinbase 1m candles via REST, computes 36 features, runs the transformer, returns `(side, confidence)`. Env-controlled by `LEGACY_MODEL_DIR`. Falls back to neutral placeholder on any load failure so the supervisor never crashes on model issues. 11 tests including end-to-end against the real bundle. |
| 9 | Operator tools — paper-session monitor (`src/paper_session_monitor.py`) | ✅ done | Read-only CLI that parses supervisor tick log lines and prints rolling per-symbol stats (action distribution, confidence percentiles, time-since-last-signal). Run alongside a paper session: `tee paper.log` then `paper_session_monitor.py paper.log --follow`. 16 tests. |
| 10 | Multi-symbol orchestration — per-symbol shakedown | ✅ done | `ShakedownState` now carries a `per_symbol: Dict[str, SymbolShakedownState]` map. Each symbol has its own `paper_days_clean` counter and `live_unlocked_at_utc`. Per-symbol errors only reset that symbol's streak; account-level events (kill switch trip, daily-loss breaker trip) reset every symbol. `is_live_unlocked(symbol)` for per-symbol gating; `is_live_unlocked()` (no args) returns the most-restrictive aggregate. Legacy single-counter shakedown JSON files migrate to per-symbol on load and are rewritten to disk in the new layout. 8 new tests; 24 supervisor tests total. |
| 11 | Crypto training pipeline (`src/crypto_training/`) | ✅ done | USD-native model training stack. `backfill_ohlcv.py` paginates Coinbase `/products/{id}/candles` (350-bar/req limit) into one CSV per UTC day, idempotent + resumable. `build_dataset.py` loads + concats day CSVs, computes 36 features via `utils.compute_features`, applies forward-return labels (binary above/below threshold OR three-class). `train_xgboost.py` does time-based train/val/test split, fits XGBClassifier, isotonic-calibrates with `FrozenEstimator + CalibratedClassifierCV`, persists `model.joblib + meta.json`. 31 new tests (13 backfill + 9 dataset + 9 trainer). |
| 11.4 | XGBoost predictor adapter (`src/predictor.py`) | ✅ done | `XGBoostPredictor` loads `model_crypto/<v>/{model.joblib,meta.json}`, maintains per-symbol Coinbase 1m candle buffer, computes features via `utils.compute_features`, runs `predict_proba`, returns `("buy", proba)` when `proba >= thr_long` else neutral. Tunable `thr_long` (high values = high precision, fewer trades). 6 new tests including end-to-end with a tiny trained model. |
| 11.5 | Multi-symbol predictor + raw-proba logging | ✅ done | `MultiSymbolXGBoostPredictor` maps each symbol to its own model + threshold, since per-symbol prob distributions differ wildly (live probe: ETH peaks at 0.67, BTC at 0.34, SOL at 0.28 even on the same minute — one global threshold is useless). Configured via `CRYPTO_MODEL_MAP="ETH/USD=path:0.50,BTC/USD=path:0.30,..."`. Supervisor's `--log-dir DIR` auto-saves each run to `<DIR>/<UTC-ts>_<symbols>/{supervisor.log,ticks.json,summary.json}`. Both predictors now log raw class probabilities at INFO level so threshold tuning is data-driven. 7 new tests (model-map parser + multi-symbol routing + dir saving). |
| 12 | Phase 0 safety hardening | ✅ done | Eight items shipped after a CEO + eng review pass. Per-symbol equity peak in `ShakedownState` (one symbol's drawdown no longer halts siblings). `fcntl.flock` + atomic temp+rename on `.shakedown.json` writes AND boot reads. Per-symbol error counter moved to Redis (HASH `errors:by_symbol:{date}`, 48h TTL — process-safe under multiprocessing). Defer-to-next-tick paper-fill state machine (paper Sharpe is no longer fictitious). NaN/inf guard on model confidence (predictor + supervisor). `--symbols` deduping with exit 2 on empty. UTC midnight auto-trigger for `daily_close` in `run_loop`. Per-symbol model isolation + `scaler.feature_names_in_` order assertion at boot. ~38 new tests. |
| 13 | Phase 1 profitability + math | ✅ done | Polymarket fee deduction in Kelly (`POLYMARKET_FEE_BPS=200` in `config.py` — stops systematic 2-3% over-sizing). Extreme-price filter `[0.02, 0.98]` in `passes_market_filters()` (Kelly explosion at p≥0.98 prevented pre-emptively). Walk-forward CV (anchored rolling windows replace single 70/15/15 split). Sharpe-weighted threshold sweep on validation set, persisted as `optimal_threshold` in `meta.json`. Backtest PnL in trainer (reuses `profitability.py`, returns Sharpe + max-DD + win_rate alongside AUC/Brier). Bayesian fusion (Beta-posterior) replacing additive XGB+LLM blend. `scale_pos_weight` class weighting. `OutcomeWeightAdjuster` per-trade synchronous + EMA decay + `[0.5, 2.0]` bounds + audit JSONL. DRY: `extract_json_object` consolidated into `src/utils.py`. ~40 new tests. |
| 14 | Lane E foundation: trade-context snapshots | ✅ done | `src/state/trade_context_store.py` — Redis-backed snapshot store keyed by trade_id × phase (`signal`, `fill`, `breaker`, `close`), 30-day TTL, NaN→None sanitisation. Trigger gate inside `position_store.record_close()` enqueues a postmortem job iff `realized_pnl < 0 AND (|loss| ≥ 0.5% bankroll OR forced_flat)`. `ForensicsFinding` dataclass + `BaseForensicsAgent` protocol with `_with_timeout` (60s) + `_safe_run` (catches all exceptions → verdict="unknown"). Supervisor wires snapshot capture at signal/fill/breaker phases. ~46 new tests. |
| 15 | Lane E swarm: 5 specialists + synthesizer | ✅ done | Five forensics agents under `src/loss_postmortem/`, one per root-cause hypothesis: `signal_forensics.py` (Mahalanobis OOD, threshold margin, calibration at this prob bin), `execution_forensics.py` (signal→fill latency, slippage actual vs expected, partial fills, stale ticker, rejection trail, stop-loss drift), `sizing_forensics.py` (recompute drift, fee-deduction audit, correlation cluster, liquidity penalty, % bankroll), `context_forensics.py` (1h news window via `news_research_agent`, Polymarket macro shifts, vol spikes, optional Gemini summarization), `process_integrity.py` (breaker log coherence, kill-switch state, race-condition trail, paper-vs-live divergence). `synthesizer.py` orchestrates all 5 in parallel via `multiprocessing.Pool` with spawn context, classifies root cause, writes `runs/postmortems/{trade_id}.{json,md}`, drives 4 feedback channels (OutcomeAdjuster delta + retrain queue + risk recommender + daily digest via `alerts/notifier`). ~70 new tests. |
| 16 | Operational hardening for unattended runtime | ✅ done | Position reconciliation script + CLI (`src/ops/reconciliation.py` + `reconciliation_cli.py`) — detects orphans (in store, not on exchange), ghosts (on exchange, not in store), and size drift; emits Prometheus metrics + Sentry breadcrumbs. `AutoPauseGate` (`src/risk/auto_pause.py`) — combined daily-loss + low-confidence (mean < baseline-2σ) trip writes `~/.autopilot_auto_paused` marker. Confidence baseline rolling window in Redis (`src/state/confidence_history.py`). Auto-trip kill switch on N consecutive errors per symbol (latched per-process to prevent alert storms). `position_store.orphan_count()` surface. Full Prometheus metric coverage (tick duration, model confidence, order latency, daily PnL by symbol, shakedown clean-days). GitHub Actions workflow at `.github/workflows/tests.yml` runs the prediction-market test suite on every PR + push. ~42 new tests. |

**Test suite:** 772 tests, all green, ~3s runtime. Sixteen phases wired across two CEO/eng review rounds. ETH/USD, BTC/USD, SOL/USD models all trained on 90 days of Coinbase OHLCV data with isotonic calibration (test AUC 0.70 / 0.67 / 0.63 respectively). The supervisor is multi-symbol-safe (per-symbol risk isolation, multiprocessing-ready), the trainer evaluates real PnL not just AUC, the calibration agent fuses XGB + LLM via a Bayesian posterior, and every losing trade ≥0.5% of bankroll triggers a 5-agent forensics swarm.

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

### Run the multi-symbol crypto supervisor

The supervisor is the live tick loop for crypto trading on Coinbase. It runs paper mode by default; promotion to live requires per-symbol shakedown gates (≥14 days clean PnL).

```bash
# Paper-mode session, all three symbols, log to a timestamped run directory.
./.venv/bin/python src/live_supervisor.py \
    --mode paper \
    --symbols ETH/USD,BTC/USD,SOL/USD \
    --log-dir ./runs \
    --tick-interval-s 60

# Operator stops via Ctrl-C OR by tripping the kill switch:
touch "$KILL_SWITCH_FILE"   # forces flat, blocks new entries

# After ≥14 days of clean per-symbol paper PnL, eligible symbols can flip to live:
./.venv/bin/python src/live_supervisor.py --mode live --symbols ETH/USD ...
```

Required env vars (see `.env.example` Section 4 for the full contract): `COINBASE_API_KEY`, `COINBASE_API_SECRET`, `COINBASE_USE_SANDBOX`, `KILL_SWITCH_FILE`, `REDIS_URL` (defaults to `redis://localhost:6379/0`), `CRYPTO_MODEL_MAP="ETH/USD=path:0.50,BTC/USD=path:0.30,SOL/USD=path:0.40"`. Optional: `DISCORD_WEBHOOK_URL`, `TELEGRAM_BOT_TOKEN`, `SENTRY_DSN`, `PROMETHEUS_PUSHGATEWAY_URL`.

What happens per tick (per symbol, in parallel under multiprocessing):
1. Fetch ticker + features from Coinbase
2. Capture **signal-phase snapshot** to Redis (`trade_ctx:{trade_id}:signal`)
3. Run predictor → `(side, confidence)`
4. NaN guard + confidence floor
5. Risk engine sizes via Kelly (with Polymarket fee deduction if applicable)
6. Circuit breakers gate (kill switch, daily-loss, drawdown, notional caps)
7. Place order (live) or simulate fill at *next-bar* open (paper, deferred to next tick)
8. Capture **fill-phase snapshot**
9. Update per-symbol equity peak, increment Redis error counter on failure
10. Auto-trip kill switch if 3+ consecutive errors on the symbol

At UTC midnight `daily_close()` auto-fires: rolls per-symbol PnL into shakedown clean-day counters, emits the daily summary alert, scans the day's postmortems and dispatches the digest, evaluates `AutoPauseGate` (pauses if daily loss > 2% AND mean confidence < baseline − 2σ).

### Read live supervisor output

```bash
# Read-only paper-session monitor (rolling per-symbol stats)
./.venv/bin/python src/paper_session_monitor.py runs/<UTC-ts>_*/supervisor.log --follow
```

### Operational tools

```bash
# Reconcile our position store against the exchange's truth.
# Flags orphans (in store, not on exchange), ghosts (on exchange, not in store),
# and size drift. Run as cron, or manually after a restart.
./.venv/bin/python -m src.ops.reconciliation_cli --symbol ETH/USD

# Inspect the auto-pause marker (presence = supervisor halted itself today)
ls -la ~/.autopilot_auto_paused 2>/dev/null

# Clear the auto-pause after operator review
rm ~/.autopilot_auto_paused
```

#### Retrain all crypto models

`scripts/retrain_all_crypto_models.sh` retrains the BTC/ETH/SOL XGBoost
calibration bundles in-place and verifies that each emitted `meta.json`
populates the `feature_means` + `feature_stds` fields required by A1
SignalForensics' Mahalanobis OOD check. Estimated runtime: ~30-60 minutes
on a developer machine.

```bash
# One-time after checkout (git tracks the executable bit but a fresh
# clone from some hosts may need this).
chmod +x scripts/retrain_all_crypto_models.sh

# Run all three retrains; prints a per-symbol summary table at the end
# (status / feature_means populated / feature_stds populated / AUC / slope).
bash scripts/retrain_all_crypto_models.sh
```

The script does NOT push to origin; the operator commits the regenerated
`model_crypto/<symbol>/` artefacts after spot-checking AUC and
`reliability_slope` against the previous bundle.

#### Grafana dashboard

`observability/grafana_dashboard.json` is a v0 Grafana dashboard
covering the Phase-16 metrics (tick duration, model confidence, order
latency, daily PnL by symbol, shakedown clean-days, kill switch /
auto-trip / auto-pause, reconciliation orphans/ghosts/drift, open
positions). Default data-source UID is `prometheus`; rename via the
Grafana UI or before import if your setup uses a different label.

```bash
# Import via the Grafana CLI (8.x+; replace --insecure with --token for
# auth-aware setups).
grafana-cli --homepath /usr/share/grafana dashboard import \
    observability/grafana_dashboard.json

# Or via the HTTP API.
curl -s -X POST 'http://admin:admin@localhost:3000/api/dashboards/db' \
    -H 'content-type: application/json' \
    -d @observability/grafana_dashboard.json
```

Operator should sanity-check the panel queries against their Prometheus
instance after import — the v0 dashboard ships with assumed query labels
(`symbol`, `reason`) that may need adjusting.

### Commit history notes

A small audit-trail wart lives at commit `5ffbb91`. Its commit message
reads "Lane E: loss-postmortem integration tests + 5 fixture scenarios"
but the actual diff is **W1C Task 1** — auto-pause + confidence-history
CLI wiring (3 files, +318 lines on `src/live_supervisor.py` and 2 test
files). The mistitling was caused by a staging race during W1A's first
commit attempt.

The actual Lane E integration tests + 5 simulated-loss fixtures landed
at commit `b8ed10b` ("Lane E: integration tests + 5 simulated-loss
fixtures (W1A recovery)").

Future archaeologists reading `git log` should treat `5ffbb91` as the
canonical auto-pause CLI commit and `b8ed10b` as the canonical Lane E
integration tests commit. The commits are functionally correct; the
title swap was not rewritten because:
1. The harness forbids `git rebase -i` (interactive flag).
2. The branch is already pushed to origin, so a non-interactive
   `git filter-branch` rewrite would force-push and is riskier than
   the wart it would clean up.

### Loss Postmortem Swarm

Every losing trade ≥0.5% of bankroll (or any breaker-forced-flat exit) triggers a 5-agent forensics swarm. The swarm runs as a separate process (or batch job) that drains the Redis queue `postmortem:queue` populated by the supervisor.

```bash
# Manual trigger for one trade (debugging)
./.venv/bin/python -c "
from src.loss_postmortem.synthesizer import LossPostmortemSynthesizer
s = LossPostmortemSynthesizer.from_env()
report = s.process_one('your-trade-id-here')
print(report.summary)
"

# Drain the queue (production: run as cron / sidecar)
./.venv/bin/python -c "
from src.loss_postmortem.synthesizer import LossPostmortemSynthesizer
LossPostmortemSynthesizer.from_env().drain(max_items=10)
"

# Read a postmortem
cat runs/postmortems/<trade-id>.md

# Inspect the retrain queue (auto-populated when 3+ Signal-cause losses cluster on a symbol within 24h)
cat runs/retrain_queue.jsonl

# Inspect risk recommendations (auto-populated when 5+ Sizing-cause losses cluster — never auto-applied; human review only)
cat runs/risk_recommendations.jsonl
```

How the swarm classifies a loss:

```
                     Loss event (≥0.5% bankroll OR forced_flat)
                                       │
                                       ▼
                          Trigger gate in record_close()
                                       │
                                       │ (LPUSH postmortem:queue)
                                       ▼
                     Synthesizer.process_one(trade_id)
                                       │
                              spawn 5 specialists
                                  in parallel
                                  via multiprocessing
                                       │
        ┌────────────┬────────────┼────────────┬────────────┐
        ▼            ▼            ▼            ▼            ▼
   A1 Signal     A2 Execution  A3 Sizing   A4 Context   A5 Process
   (Mahalanobis  (slippage,    (Kelly      (news 1h,    (breaker
    OOD,          latency,      recompute,  Polymarket   logs,
    threshold     stale         fee audit,  macro,       kill-switch
    margin,       ticker,       correlation vol spike,   coherence,
    calibration)  partial       cluster)    Gemini       race-cond)
                  fills)                    summary)
        │            │            │            │            │
        └────────────┴─────┬──────┴────────────┴────────────┘
                           ▼
                       Synthesizer
                  classifies root cause
            (Signal/Execution/Sizing/Context/Process/Mixed/Unknown)
                           │
                ┌──────────┼──────────┬──────────────┐
                ▼          ▼          ▼              ▼
      runs/postmortems/  Outcome   retrain_       risk_
      {trade_id}.json   Adjuster   queue.jsonl   recommendations
      {trade_id}.md     (weight    (3+ Signal-   .jsonl
                         delta:    cause cluster (5+ Sizing-
                         Signal/    in 24h →     cause cluster
                         Sizing→    auto-queued)  → human-review
                         -0.05;                    only)
                         Mixed→
                         -0.03)
                           │
                           ▼
                  Daily digest via
                  alerts/notifier.py
                  (top-3 losses + root-cause
                   distribution + new actions)
```

Each specialist is `BaseForensicsAgent`-derived, returns a `ForensicsFinding` with `verdict ∈ {innocent, contributing, primary_cause, unknown}`, evidence bullets, suggested action, severity (1-5), runtime. Per-agent timeout 60s. A crashing agent never blocks the swarm — its `verdict="unknown"` with `error=<exc text>`.

Adding a new specialist: subclass `BaseForensicsAgent`, set `agent_name`, implement `investigate(trade_id) -> ForensicsFinding`, register in the synthesizer's `agent_factories`. See `src/loss_postmortem/signal_forensics.py` for the canonical example.

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

The repo uses `unittest`. There are three test trees with different runner conventions — don't assume one command runs them all.

```bash
# Full prediction-market + crypto trading + Lane E swarm suite (772 tests, ~3s)
env PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests/prediction_market_scanner

# Single test module
env PYTHONPATH=src ./.venv/bin/python -m unittest tests.prediction_market_scanner.test_synthesizer

# Single test method
env PYTHONPATH=src ./.venv/bin/python -m unittest tests.prediction_market_scanner.test_synthesizer.SynthesizerTest.test_root_cause_classification

# Social narrative agent (separate, no PYTHONPATH)
./.venv/bin/python -m unittest discover social-narrative-agent/tests

# Legacy core tests (pytest, requires torch + TA-Lib)
./.venv/bin/python -m pytest tests/test_core.py
```

CI: every push and PR to `main` or `feature/prediction-market-bot` triggers `.github/workflows/tests.yml` which installs TA-Lib + the full requirements.txt and runs the prediction-market suite + `test_compute_features`.

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
