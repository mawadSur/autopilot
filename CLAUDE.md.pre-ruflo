# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo Shape

This is a Python research workspace, not a packaged app. Two largely independent stacks live side by side under one repo and share the same `.venv` / `requirements.txt`:

1. **Prediction-market stack** (newer, active) — Polymarket scanner + multi-agent research pipeline. Entry points: `main.py`, `src/orchestrator.py`, `social-narrative-agent/main.py`.
2. **Legacy crypto trading stack** — ETH 1m model training, backtesting, live/paper trading, FastAPI control plane, Streamlit dashboard. Lives under `src/` with `main.py` (FastAPI), `train_model.py`, `backtest.py`, `live_trader.py`, `paper_trade.py`, `dashboard_server.py`, `dashboard_app.py` as the main entry points.

The Dockerfile's `CMD ["python", "main.py", "--step", "trade"]` is stale — `main.py` is now the Polymarket scanner CLI and has no `--step` flag.

## Common Commands

Use the venv interpreter directly (`./.venv/bin/python`) — there's no console-script wrapper.

```bash
# Polymarket scanner (top-level main.py)
./.venv/bin/python main.py --top 20 --category Politics

# Multi-agent orchestrator (scanner + Reddit + News + calibration + risk)
./.venv/bin/python src/orchestrator.py --top 5 --category Politics --subreddit politics

# Social narrative agent (separate OpenAI-based CLI)
./.venv/bin/python social-narrative-agent/main.py --topic "..." --current-odds 0.42

# Legacy FastAPI control API
./.venv/bin/python src/main.py

# Legacy Streamlit dashboard
streamlit run src/dashboard_app.py
```

### Tests

There are **three test trees, each with different runner conventions** — don't assume one command runs them all:

- `tests/prediction_market_scanner/` — `unittest`, requires `PYTHONPATH=src` (imports flat from `src/`).
- `tests/test_core.py` — `pytest`, requires the full legacy stack (`torch`, `TA-Lib`, etc.).
- `social-narrative-agent/tests/` — `unittest`, no `PYTHONPATH` needed (self-contained CLI).

```bash
# Prediction-market suite
env PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests/prediction_market_scanner

# Single test module
env PYTHONPATH=src ./.venv/bin/python -m unittest tests.prediction_market_scanner.test_orchestrator

# Single test method
env PYTHONPATH=src ./.venv/bin/python -m unittest tests.prediction_market_scanner.test_orchestrator.SomeTest.test_x

# Social narrative agent suite
./.venv/bin/python -m unittest discover social-narrative-agent/tests

# Legacy core tests (pytest, requires torch + TA-Lib)
./.venv/bin/python -m pytest tests/test_core.py
```

There is no formatter, linter, pre-commit hook, or CI workflow configured (no `pyproject.toml`, no `.pre-commit-config.yaml`, no `.github/workflows`). The `Makefile` only has a `reinstall` target that pins `protobuf==5.28.3` before reinstalling `requirements.txt` — needed when TF/protobuf get out of sync.

### Generated artifacts vs. user work

- `output/` — written by `main.py` (`scan_*.json` exports). Safe to delete.
- `model_sanity/` — canonical model artifacts dir. `best_live_config.json`, `training_summary.json`, `model_meta.json`, `scaler.joblib`, and per-fold subdirs (`fold_*/metrics.json`, `fold_*/model_meta.json`) are all *generated* by `src/train_model.py` and friends. If you see them as untracked in `git status`, that's a fresh training run, not user edits — don't reflexively commit them.

### `.gitignore` footgun

`.gitignore` has `*.txt` — meaning new `.txt` files (notes, scratchpads, even some configs) silently won't appear in `git status`. `requirements.txt` is tracked because it predates the rule. If a `.txt` file mysteriously isn't being seen, that's why.

## Architecture

### Path manipulation is deliberate

Both `main.py` and `src/orchestrator.py` mutate `sys.path` at import time to add `src/` (and the repo root) so imports like `from fetcher import ...` work without a package install. Don't "clean this up" by converting to relative imports — it would break the test runs above and the legacy modules that import each other flat.

### Prediction-market data flow

`main.py` → fetches Polymarket via `src/fetcher.py` → filters/scores via `src/analyzer.py` + `src/ranker.py` → calls `src/llm_judge.py` (Gemini, with a neutral fallback if the call fails) → exports JSON to `output/scan_*.json`.

`src/orchestrator.py` calls `build_scan_results()` from `main.py`, then for the top-N markets fans out to a fixed set of agents under `src/`:

- `reddit_research_agent/` — PRAW-style fetch + Gemini analysis
- `news_research_agent/` — Google News RSS + Gemini analysis (the orchestrator probes two import paths, see `_NEWS_IMPORT_CANDIDATES`)
- `calibration_agent/` — combines an XGBoost ML baseline (`ml_service.get_xgboost_probability`) with a Gemini calibration step
- `risk_management_agent/` — Kelly/risk sizing on top of the calibrated probability
- `synthesis_agent/` — final action recommendation
- `outcome_review_agent/` — post-hoc review/logging

Each agent owns its own `models.py` (pydantic/dataclass result types), `analyzer.py` (LLM-driven), and usually a `fetcher.py`. The orchestrator uses `_call_with_supported_kwargs` and `_resolve_method` to tolerate small signature drift across agents — keep that pattern when wiring new agents.

### Legacy trading config

`src/config.py` exposes a `cfg` singleton built from `pydantic-settings` (with a v1 fallback). It loads `.env` from either the repo root *or* `src/.env`, and covers data paths, model thresholds (`thr_long`, `thr_short`, `margin`, `consensus`), ATR stop multipliers, live trading params, AWS/SageMaker, and CoinDesk WS settings. `FEATURE_VERSION` is exposed as `cfg.FEATURE_VERSION` but is **not** currently checked at model-load time — `ModelMeta` (in `src/models.py`) stores `feature_cols` directly rather than a version tag. Treat `FEATURE_VERSION` as informational/documentary unless you wire it into the load path yourself.

The legacy stack reads CSVs from `eth_1m_data/`, writes artifacts into `model_sanity/`, and the FastAPI app under `src/main.py` lazy-imports `boto3`/`sagemaker` inside endpoint bodies (verified across `/predict`, `/train/aws`, `/deploy`, `/sagemaker/*`) so `uvicorn main:app --reload` doesn't pay the AWS-SDK import cost on every reload.

## Environment Variables

`src/config.py` looks at `.env` in repo root *and* `src/.env`. Common keys:

- `GEMINI_API_KEY`, `GEMINI_MODEL` (default `gemini-2.5-flash`) — required for orchestrator; scanner falls back to a neutral judge if missing.
- `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` — required for `src/orchestrator.py` Reddit research.
- `OPENAI_API_KEY` — required for `social-narrative-agent/`.
- `COINDESK_API_KEY`, `ENDPOINT_NAME`, `SAGEMAKER_ROLE_ARN`, `AWS_REGION` — legacy trading / SageMaker.
- All `TradingConfig` fields can be overridden by env (`THR_LONG`, `CAPITAL`, `SYMBOL`, etc.) — see `src/config.py` for the full list.

## Conventions

See `AGENTS.md` for style and PR conventions (the source of truth). Headline: avoid broad style-only edits since no formatter is configured, and PR descriptions should call out which surface changed (scanner, orchestrator, dashboard, legacy trading) plus any new env vars.
