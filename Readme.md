# 🤖 AI Crypto Trading Bot

An end‑to‑end crypto trading system focused on **1‑minute ETH/USDT** data. It includes:
- data ingestion (Binance or CoinDesk),
- feature engineering + labeling,
- model training (local or SageMaker),
- backtesting,
- live inference + optional paper/real execution.

This README explains **how to get started** and **how the system works**.

## Table of Contents
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Workflow](#workflow)
- [API Endpoints](#api-endpoints)
- [Disclaimer](#disclaimer)

## Project Overview

The core of this project is a predictive model that analyzes 1‑minute OHLCV (and optionally microstructure) data for ETH/USDT. It learns short‑term patterns and outputs a **3‑class signal**:
- **short_win** (-1),
- **timeout/hold** (0),
- **long_win** (+1).

The project is designed with a clear separation of concerns:
1.  **Data Engineering**: Scripts to fetch and prepare large datasets.
2.  **ML Training**: A robust training and deployment pipeline using AWS SageMaker.
3.  **Inference**: A real-time API built with FastAPI to serve live predictions and stream data.

## Architecture

The project follows a standard MLOps workflow:

1.  **📥 Data Ingestion**
    - Binance OHLCV: `history.py`
    - CoinDesk data via reusable client: `coindesk_client.py`
2.  **🛠️ Feature Engineering + Labeling**
    - `utils.compute_features` builds the full feature set.
    - `train_model.py` applies dynamic triple‑barrier labels (cost‑aware, ATR‑aware).
3.  **🧠 Model Training**
    - Local: `python src/train_model.py`
    - SageMaker: `python src/launch_sagemaker_job.py`
4.  **📊 Backtesting**
    - `python src/backtest.py --model-dir model/seq_90`
5.  **📡 Live / Paper Trading**
    - CoinDesk live collector + inference: `python src/live_coindesk_collector.py --paper`

## Features

- **Automated Data Fetching**: Pulls years of 1-minute ETH/USDT data from Binance or CoinDesk.
- **Advanced Feature Engineering**: Robust bucketed features (OHLCV + stationarity + microstructure).
- **Cloud-Based Training**: Leverages AWS SageMaker for scalable, powerful GPU-based model training.
- **Real-time Inference API**: A non-blocking API built with FastAPI to deliver signals with low latency.
- **Live Signal Streaming**: A WebSocket endpoint streams predictions every 5 seconds, perfect for a live dashboard.
- **Paper Trading Simulation**: Endpoints to run and monitor a simulated trading strategy in the background.
- **Backtesting Support**: `backtest.py` evaluates full-history strategies.

## Project Structure

```
.
├── eth_1m_data/              # Stores raw historical data from Binance
├── data/coindesk/...         # Unified CoinDesk datasets (1 CSV per month)
├── backtest.py               # Generates signals over historical data for backtesting.
├── history.py                # Fetches historical 1-minute data from Binance.
├── coindesk_client.py        # Reusable CoinDesk Data API client.
├── live_coindesk_collector.py# Live CoinDesk collector + inference + (paper) execution.
├── inference.py              # Contains the code for the SageMaker real-time inference endpoint.
├── label.py                  # Performs feature engineering and creates labels for the dataset.
├── launch_sagemaker_job.py   # Orchestrates the AWS SageMaker training and deployment process.
├── main.py                   # The main FastAPI application for serving live signals.
├── paper_trade.py            # A script to run a simple paper trading simulation.
├── model_meta.json           # Stores model metadata, like the decision threshold.
├── requirements.txt          # A list of all Python dependencies for the project.
└── README.md                 # This documentation file.
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mawadSur/autopilot.git
    cd autopilot
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (TA-Lib is required; if pip fails on wheels, install the conda-forge package: `conda install -c conda-forge ta-lib`)

4.  **Set up environment variables**
    Create a `.env` file in the root directory and add your Binance API keys (if using Binance data) and CoinDesk API key (if using CoinDesk data).
    ```env
    # For history.py (fetching data)
    BINANCE_KEY="your_binance_api_key"
    BINANCE_SECRET="your_binance_api_secret"

    # For CoinDesk Data API (used by coindesk_client.py)
    COINDESK_API_KEY="your_coindesk_api_key"

    # Optional: live execution via Coinbase (ccxt)
    COINBASE_API_KEY="..."
    COINBASE_API_SECRET="..."
    COINBASE_PASSPHRASE="..."
    ```

5.  **Configure AWS Credentials**:
    Ensure your environment is configured with AWS credentials. The simplest way is to use the AWS CLI:
    ```bash
    aws configure
    ```
    You will also need to **provide the correct IAM Role ARN** in `launch_sagemaker_job.py`.

## Getting Started (Quickstart)

1) **Fetch data**
   - Binance:  
     ```bash
     python src/history.py
     ```
   - CoinDesk via client:  
     ```bash
     python src/coindesk_client.py --help
     ```

2) **Train locally**
   ```bash
   python src/train_model.py --data-path data/coindesk/ETH-USDT/1m --output-dir model
   ```

3) **Backtest**
   ```bash
   python src/backtest.py --model-dir model/seq_90
   ```

4) **Run live collector (paper mode)**
   ```bash
   python src/live_coindesk_collector.py --paper
   ```

## Workflow (Detailed)

Follow these steps to run the project from end to end:

1.  **Fetch Data**:
    - Binance (OHLCV only): `python src/history.py`
    - CoinDesk (via client): `python src/coindesk_client.py --help`
    ```bash
    python src/history.py
    # or (see options)
    python src/coindesk_client.py --help
    ```

2.  **Train and Deploy on AWS**: Execute the `launch_sagemaker_job.py` script. This will handle uploading data, training the model, and deploying it to a live endpoint.
    ```bash
    python src/launch_sagemaker_job.py
    ```
    After this step is complete, you will have a trained model in S3 and a live SageMaker endpoint.

3.  **Run the Live API Server**: To run the API locally, you must first download the model artifacts from S3.

    **How to Download Model Files from S3**

    After the SageMaker training job is complete, it saves a `model.tar.gz` file to your S3 bucket. You can find the exact path in the console output of the training job.

    Use the AWS CLI to download and extract the files:
    ```bash
    # Replace with the actual S3 path from your training job output
aws s3 cp s3://sagemaker-pytorch-2025-07-17-03-15-00-123/output/model.tar.gz .

    # This will extract best_model.pth and scaler.pkl into your current directory
    tar -xzvf model.tar.gz
    ```
    
    Now that the model files are local, start the FastAPI server:
    ```bash
    uvicorn main:app --reload
    ```


4.  **Connect a Frontend**: The API is now running on `http://127.0.0.1:8000`. You can connect a client to the WebSocket at `ws://127.0.0.1:8000/ws/signal-stream`.

## API Endpoints

The interactive API documentation (via Swagger UI) is available at `http://127.0.0.1:8000/docs` when the server is running.

- **`GET /`**: Welcome message.
- **`GET /signal/latest`**: Fetches the latest market data and returns a single "BUY" or "HOLD" signal.
- **`POST /papertrade/start`**: Starts a background paper trading simulation.
- **`GET /papertrade/status/{job_id}`**: Retrieves the status and log of a running simulation.
- **`WS /ws/signal-stream`**: WebSocket endpoint that streams the latest signal, confidence, and price every 5 seconds.

## Disclaimer

**This project is for educational purposes only and is not financial advice.** The predictions from this model are not guaranteed to be accurate. Trading cryptocurrencies involves significant risk, and you should never trade with money you cannot afford to lose. The author is not responsible for any financial losses incurred by using this software.
