# AI Crypto Trading Bot

This repository contains a Python trading stack plus a Vite/React dashboard for monitoring live signals. The backend covers historical data ingestion, model training, FastAPI serving, Redis-backed live pipeline services, and SageMaker deployment helpers.

## Project Layout

```text
.
├── src/                      # Python application code
│   ├── main.py               # FastAPI API server
│   ├── train_model.py        # Local LSTM training entry point
│   ├── backtest.py           # Historical inference / backtesting
│   ├── history.py            # Binance historical candle ingestion
│   ├── inference_live.py     # Redis-driven live inference service
│   ├── feature_engine_live.py# Redis-driven feature engineering service
│   ├── live_data_stream.py   # Live Binance stream ingestion
│   ├── live_trading_engine.py# Live trading execution service
│   └── pipeline/             # Async pipeline helpers and websocket utilities
├── dashboard/                # React + Vite frontend
├── model/                    # Saved local model artifacts
├── requirements.txt          # Python dependencies
├── run_live_pipeline.py      # Multi-process live pipeline orchestrator
└── test_pipeline.py          # Basic live pipeline smoke checks
```

## Requirements

- Python 3.10+
- Node.js 20.19+ for the dashboard
- Redis running locally on `localhost:6379` for the live pipeline
- AWS credentials configured if you use SageMaker deployment/training flows

## Python Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the repo root for any credentials or runtime overrides you need:

```env
BINANCE_KEY=...
BINANCE_SECRET=...
REDIS_HOST=localhost
REDIS_PORT=6379
ENDPOINT_NAME=...
APP_VERSION=0.1.0
```

## Dashboard Setup

```bash
cd dashboard
npm install
npm run dev
```

## Common Workflows

Fetch historical Binance candles:

```bash
python src/history.py --help
```

Train a local model:

```bash
python src/train_model.py --data-path eth_1m_data --output-dir model
```

Run the FastAPI server:

```bash
uvicorn src.main:app --reload
```

Start the live Redis-based pipeline:

```bash
python run_live_pipeline.py
```

Run the live pipeline smoke tests:

```bash
python test_pipeline.py
```

## Notes

- The live pipeline depends on Redis and local model artifacts under `model/`.
- SageMaker helper scripts live under `src/` and expect valid AWS credentials plus a configured IAM role / bucket.
- This project is for experimentation and education. It is not financial advice.
