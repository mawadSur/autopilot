# Autopilot

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

# Required for Reddit research inside src/orchestrator.py
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
- `src/orchestrator.py` is not meaningfully usable without Gemini and Reddit credentials.
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

### Run the social narrative agent

```bash
python social-narrative-agent/main.py --topic "OpenAI GPT-5 release" --current-odds 0.42
```

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

## Known Gaps

- The checked-in `Dockerfile` still points at an older `main.py --step trade` interface that does not match the current root CLI.
- This README documents the current Python workflows only; it does not represent a packaged Codex app because this repo does not contain one yet.
