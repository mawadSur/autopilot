import uvicorn
from fastapi import FastAPI, WebSocket, BackgroundTasks, HTTPException
from pydantic import BaseModel
import asyncio
import uuid
from typing import List, Dict

# --- Local Imports (for data processing and simulation) ---
# Ensure these files are in the same directory or a sub-directory
from utils import load_ohlc_chunks, compute_rsi, compute_atr
from aws_train_model import preprocess_data, LSTMModel # Re-using your preprocessing and model class

# --- PyTorch and ML Imports ---
import torch
import joblib
import numpy as np
import pandas as pd

# --- Binance Client for Live Data ---
from binance.client import Client

# ==============================================================================
# 1. APPLICATION SETUP & MODEL LOADING (SINGLETON PATTERN)
# ==============================================================================

app = FastAPI(title="AI Trading API", version="1.0.0")

# --- Global State & Model Cache ---
# This dictionary will hold our loaded model and scaler to avoid reloading them.
model_cache = {}
# This dictionary will store the state of background paper trading jobs.
paper_trade_jobs = {}

@app.on_event("startup")
def load_model_and_scaler():
    """
    This function runs once when the API server starts.
    It loads the trained model and scaler into memory.
    """
    print("API starting up. Loading model and scaler...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define the model architecture (must match the trained model)
        model = LSTMModel(input_size=17, hidden_size=64, num_layers=2, output_size=1, dropout_rate=0.3)
        
        # Load the saved weights
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        model.to(device).eval() # Set model to evaluation mode
        
        # Load the scaler
        scaler = joblib.load("scaler.pkl")
        
        # Store in our global cache
        model_cache["model"] = model
        model_cache["scaler"] = scaler
        model_cache["device"] = device
        
        print("✅ Model and scaler loaded successfully.")
    except FileNotFoundError:
        print("⚠️ WARNING: 'best_model.pth' or 'scaler.pkl' not found. Some endpoints will not work.")
        model_cache["model"] = None


# ==============================================================================
# 2. DATA MODELS (for API request/response validation)
# ==============================================================================

class SignalResponse(BaseModel):
    signal: str # "BUY" or "HOLD"
    confidence: float
    current_price: float

class PaperTradeRequest(BaseModel):
    starting_amount: float = 1000.0
    duration_minutes: int = 60
    take_profit_pct: float = 1.5 # e.g., 1.5%
    stop_loss_pct: float = 1.0   # e.g., 1.0%

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class PaperTradeStatus(BaseModel):
    status: str
    progress: float # Percentage complete
    total_profit: float
    trades_made: int
    log: List[str]

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================

def get_live_prediction(klines: List) -> dict:
    """
    Takes a list of kline data, preprocesses it, and returns a prediction.
    """
    if not model_cache.get("model"):
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    # Convert klines to DataFrame
    df = pd.DataFrame(klines, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df.set_index('date', inplace=True)
    df = df.astype(float)

    feature_cols = [
        'open', 'high', 'low', 'close', 'body', 'range', 'upper_wick', 'lower_wick', 'return',
        'sma_ratio', 'ema_20', 'macd', 'rsi_14', 'vol_change', 'atr', 'price_vs_hourly_trend', 'bb_width'
    ]
    window_size = 150

    # This is a simplified preprocessing for a single prediction.
    # It assumes enough historical klines are passed in for feature calculation.
    # In a real system, you'd manage a persistent history buffer.
    # For now, we just calculate features on the provided data.
    
    # Feature Engineering (copied from your training script)
    df['body'] = df['close'] - df['open']; df['range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['return'] = df['close'].pct_change()
    df['sma_10'] = df['close'].rolling(10).mean(); df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_ratio'] = df['sma_10'] / df['sma_50'] - 1; df['ema_20'] = df['close'].ewm(span=20).mean()
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['rsi_14'] = compute_rsi(df['close'], 14); df['vol_change'] = df['volume'].pct_change()
    df['atr'] = compute_atr(df); df_hourly = df['close'].resample('1h').mean()
    hourly_ema = df_hourly.ewm(span=20).mean()
    df['hourly_ema_20'] = hourly_ema.reindex(df.index, method='ffill')
    df['price_vs_hourly_trend'] = (df['close'] - df['hourly_ema_20']) / df['hourly_ema_20']
    df['bb_std'] = df['close'].rolling(20).std(); df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_width'] = ((df['bb_mid'] + 2 * df['bb_std']) - (df['bb_mid'] - 2 * df['bb_std'])) / df['bb_mid']
    df.replace([np.inf, -np.inf], 0, inplace=True); df.dropna(inplace=True)
    
    if len(df) < window_size:
        return {"signal": "HOLD", "confidence": 0.0, "current_price": df.iloc[-1]['close']}

    # Prepare input for the model
    last_sequence = df.tail(window_size)[feature_cols].values
    last_sequence_scaled = model_cache["scaler"].transform(last_sequence)
    
    input_tensor = torch.from_numpy(last_sequence_scaled).unsqueeze(0).to(torch.float32)
    input_tensor = input_tensor.to(model_cache["device"])

    # Get prediction
    with torch.no_grad():
        output = model_cache["model"](input_tensor)
        confidence = torch.sigmoid(output).item()

    signal = "BUY" if confidence > 0.5 else "HOLD" # Using a 0.5 threshold for now
    
    return {
        "signal": signal,
        "confidence": confidence,
        "current_price": df.iloc[-1]['close']
    }

async def run_paper_trade_simulation(job_id: str, params: PaperTradeRequest):
    """
    This function runs in the background to simulate paper trading.
    """
    client = Client() # Testnet or live client can be configured here
    job = paper_trade_jobs[job_id]
    
    position_open = False
    entry_price = 0
    balance = params.starting_amount
    
    for i in range(params.duration_minutes):
        try:
            # Fetch last 200 minutes of data for feature calculation
            klines = client.get_klines(symbol='ETHUSDT', interval=Client.KLINE_INTERVAL_1MINUTE, limit=200)
            
            prediction = get_live_prediction(klines)
            current_price = prediction['current_price']
            
            if position_open:
                # Check for take profit or stop loss
                if current_price >= entry_price * (1 + params.take_profit_pct / 100):
                    pnl = (current_price - entry_price) * (balance / entry_price)
                    balance += pnl
                    log_msg = f"SELL (TP): at {current_price:.2f} | PnL: {pnl:.2f} | New Balance: {balance:.2f}"
                    job["log"].append(log_msg)
                    position_open = False
                elif current_price <= entry_price * (1 - params.stop_loss_pct / 100):
                    pnl = (current_price - entry_price) * (balance / entry_price)
                    balance += pnl
                    log_msg = f"SELL (SL): at {current_price:.2f} | PnL: {pnl:.2f} | New Balance: {balance:.2f}"
                    job["log"].append(log_msg)
                    position_open = False
                else:
                    job["log"].append(f"HOLDING: Price at {current_price:.2f}")
            
            else: # No position open, look for a buy signal
                if prediction['signal'] == 'BUY':
                    entry_price = current_price
                    position_open = True
                    log_msg = f"BUY: at {entry_price:.2f} | Confidence: {prediction['confidence']:.3f}"
                    job["log"].append(log_msg)
                    job["trades_made"] += 1
                else:
                    job["log"].append(f"HOLD: Price at {current_price:.2f}, Confidence: {prediction['confidence']:.3f}")

            job["progress"] = ((i + 1) / params.duration_minutes) * 100
            await asyncio.sleep(60) # Wait for 1 minute

        except Exception as e:
            job["log"].append(f"ERROR: {e}")
            await asyncio.sleep(60)

    job["status"] = "complete"
    job["total_profit"] = balance - params.starting_amount


# ==============================================================================
# 4. API ENDPOINTS
# ==============================================================================

@app.get("/", summary="API Root")
def read_root():
    return {"message": "Welcome to the AI Trading API. See /docs for available endpoints."}

@app.get("/signal/latest", response_model=SignalResponse, summary="Get Latest Signal")
async def get_latest_signal():
    """
    Fetches the latest market data for ETH/USDT and returns a single
    BUY or HOLD signal based on the trained model.
    """
    client = Client()
    # Fetch enough historical data for feature calculation (e.g., last 200 minutes)
    klines = client.get_klines(symbol='ETHUSDT', interval=Client.KLINE_INTERVAL_1MINUTE, limit=200)
    
    prediction = get_live_prediction(klines)
    return prediction

@app.websocket("/ws/signal-stream")
async def websocket_signal_stream(websocket: WebSocket):
    """
    Streams the latest signal and confidence level every 5 seconds.
    """
    await websocket.accept()
    client = Client()
    try:
        while True:
            klines = client.get_klines(symbol='ETHUSDT', interval=Client.KLINE_INTERVAL_1MINUTE, limit=200)
            prediction = get_live_prediction(klines)
            await websocket.send_json(prediction)
            await asyncio.sleep(5) # Stream data every 5 seconds
    except Exception:
        print("Client disconnected from WebSocket.")

@app.post("/papertrade/start", response_model=JobResponse, summary="Start Paper Trading Simulation")
async def start_paper_trade(params: PaperTradeRequest, background_tasks: BackgroundTasks):
    """
    Starts a paper trading simulation in the background. The simulation will run
    for the specified duration, making trades based on the model's signals.
    """
    job_id = str(uuid.uuid4())
    paper_trade_jobs[job_id] = {
        "status": "running",
        "progress": 0.0,
        "total_profit": 0.0,
        "trades_made": 0,
        "log": [f"Starting paper trade simulation for {params.duration_minutes} minutes..."]
    }
    background_tasks.add_task(run_paper_trade_simulation, job_id, params)
    return {"job_id": job_id, "status": "started", "message": "Paper trading simulation has begun."}

@app.get("/papertrade/status/{job_id}", response_model=PaperTradeStatus, summary="Get Paper Trading Status")
async def get_paper_trade_status(job_id: str):
    """
    Retrieves the current status, progress, and log of a paper trading simulation.
    """
    job = paper_trade_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    return job

# ==============================================================================
# 5. SERVER EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # To run this server: uvicorn api_server:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)