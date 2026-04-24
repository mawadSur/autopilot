# AI Crypto Trading Bot - Project Context

## Project Overview
This project is an end-to-end automated trading bot for ETH/USDT, leveraging deep learning (LSTM) to predict short-term price movements. It integrates real-time data ingestion from Binance, feature engineering, model training (both local and via AWS SageMaker), and a live trading execution pipeline.

### Core Technologies
- **Language:** Python 3.10+
- **Machine Learning:** PyTorch (LSTM), scikit-learn (scaling), ONNX (optional inference)
- **API & Streaming:** FastAPI, WebSockets, Redis (Streams/Cache)
- **Data Storage:** HDF5 (large-scale training), SQLite (local cache), Redis (real-time)
- **Cloud/DevOps:** AWS SageMaker, S3, Docker

## Project Structure
- `src/`: Main source directory containing the logic for the bot.
    - `pipeline/`: Modularized pipeline components (ingestion, inference, signals).
    - `main.py`: FastAPI entry point for serving predictions and monitoring.
    - `history.py`: Script for fetching historical 1-minute k-lines from Binance.
    - `feature_engine_live.py`: Real-time computation of technical indicators.
    - `inference_live.py`: Service for running model predictions on live data.
    - `live_trading_engine.py`: Orchestrates trade execution based on signals.
    - `launch_sagemaker_job.py`: Orchestrates AWS SageMaker training and deployment.
- `model/`: Directory for model artifacts (`model.pt`, `scaler.joblib`, `model_meta.json`).
- `eth_1m_data/`: Local storage for raw historical CSV files.
- `labeled_chunks/`: Processed training data in `.npz` format.
- `run_live_pipeline.py`: Root orchestrator for the live trading services.

## Architecture & Data Flow
1. **Ingestion:** `live_data_stream.py` consumes 1m candles from Binance WebSocket and pushes to Redis Stream `stream:ETHUSDT:1m`.
2. **Features:** `feature_engine_live.py` reads from Redis, computes 15-17 technical indicators (RSI, MACD, etc.), and maintains a 192-candle sliding window.
3. **Inference:** `inference_live.py` loads the LSTM model and predicts the probability of a price increase.
4. **Trading:** `live_trading_engine.py` executes trades based on prediction confidence (default threshold 0.6).
5. **Logging:** `data_logger_live.py` persists results for performance analysis and future retraining.

## Building and Running

### Setup
1. **Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Environment:**
   Create a `.env` file with `BINANCE_KEY`, `BINANCE_SECRET`, and AWS credentials/role if using SageMaker.

### Key Commands
- **Run Live Pipeline:**
  ```bash
  python run_live_pipeline.py
  ```
  *(Requires a running Redis server: `redis-server`)*
- **Start API Server:**
  ```bash
  uvicorn src.main:app --reload
  ```
- **Fetch Historical Data:**
  ```bash
  python src/history.py
  ```
- **Train Locally:**
  ```bash
  python src/train_model.py
  ```
- **SageMaker Training/Deploy:**
  ```bash
  python src/launch_sagemaker_job.py
  ```

## Development Conventions
- **Feature Engineering:** Uses a standard set of 15 features including OHLC basics and technical indicators like RSI, MACD, and Bollinger Band width.
- **Model Window:** The LSTM expects a sequence of **192 candles** (3.2 hours of 1m data) as input.
- **Inter-Process Communication:** Services are decoupled and communicate via **Redis Streams** for low-latency data passing.
- **Device Management:** Scripts automatically detect and utilize CUDA if available, falling back to CPU.
- **Data Format:** Large datasets are consolidated into **HDF5** for efficient indexed access during training.

## Roadmap & Status
- **Phase 1 (Complete):** Data storage, feature engineering, and SageMaker training infrastructure.
- **Phase 2 (Complete):** Live streaming, Redis integration, and real-time inference pipeline.
- **Phase 3 (Complete):** Multi-symbol support, automated daily retraining, and production readiness with state persistence.

## New Phase 3 Features
- **Multi-Symbol Orchestration:** `run_live_pipeline.py` now supports multiple symbols (e.g., `ETHUSDT,BTCUSDT,SOLUSDT`) via the `SYMBOLS` environment variable or `--symbols` flag.
- **Automated Retraining:** `src/auto_retrain.py` can be scheduled to retrain models daily using data captured in the `market_data_store.h5` HDF5 store.
- **Model Hot-Reload:** `inference_live.py` automatically detects and reloads updated model artifacts without downtime.
- **State Persistence:** `live_trading_engine.py` persists position state to local JSON files, allowing the bot to recover and manage open positions across restarts.
- **Enhanced Monitoring:** `src/main.py` provides symbol-specific WebSocket streams and a `/status` endpoint for real-time monitoring of all active pairs.
