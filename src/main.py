# main.py (FastAPI)
from __future__ import annotations

import os
import signal
import subprocess
import asyncio
import json
import redis.asyncio as redis
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

# NOTE: We intentionally avoid heavy AWS / ML imports at module import time.
#       They will be imported inside endpoints to keep `uvicorn main:app --reload` reliable.

APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
BASE_DIR = Path(__file__).resolve().parent

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", os.getenv("TRADE_SYMBOL", "ETHUSDT")).split(",") if s.strip()]

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                # Connection might be closed
                pass

manager = ConnectionManager()

async def candle_broadcaster(symbol: str):
    """Broadcast the latest cached candle for a symbol."""
    r = redis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}", decode_responses=True)
    cache_key = f"cache:{symbol}:latest_candle"
    print(f"📡 Candle broadcaster started for {cache_key}")
    last_time: Optional[str] = None

    while True:
        try:
            data = await r.hgetall(cache_key)
            if data and data.get("time") != last_time:
                last_time = data.get("time")
                await manager.broadcast({"type": "candle", "symbol": symbol, "data": data})
            await asyncio.sleep(1)
        except Exception as e:
            print(f"❌ Candle broadcaster error [{symbol}]: {e}")
            await asyncio.sleep(5)

async def signal_broadcaster(symbol: str):
    """Poll Redis list for new signals for a symbol and broadcast."""
    r = redis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}", decode_responses=True)
    list_key = f"list:{symbol}:signals"
    print(f"📡 Signal broadcaster started for {list_key}")
    
    while True:
        try:
            res = await r.brpop(list_key, timeout=5)
            if res:
                _, data = res
                await manager.broadcast({"type": "signal", "symbol": symbol, "data": json.loads(data)})
        except Exception as e:
            print(f"❌ Signal broadcaster error [{symbol}]: {e}")
            await asyncio.sleep(5)

async def trade_broadcaster(symbol: str):
    """Poll Redis list for new trades for a symbol and broadcast."""
    r = redis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}", decode_responses=True)
    list_key = f"list:{symbol}:trades"
    print(f"📡 Trade broadcaster started for {list_key}")
    
    while True:
        try:
            res = await r.brpop(list_key, timeout=5)
            if res:
                _, data = res
                await manager.broadcast({"type": "trade", "symbol": symbol, "data": json.loads(data)})
        except Exception as e:
            print(f"❌ Trade broadcaster error [{symbol}]: {e}")
            await asyncio.sleep(5)

def create_app() -> FastAPI:
    app = FastAPI(title="AI Crypto Trading API", version=APP_VERSION)

    # CORS (loose by default; tighten for production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # simple process registry for long-running scripts
    app.state.procs: Dict[str, subprocess.Popen] = {}

    @app.on_event("startup")
    async def startup_event():
        # Start background broadcasters for each symbol
        for symbol in SYMBOLS:
            asyncio.create_task(candle_broadcaster(symbol))
            asyncio.create_task(signal_broadcaster(symbol))
            asyncio.create_task(trade_broadcaster(symbol))

    @app.websocket("/ws/signal-stream")
    async def signal_stream(websocket: WebSocket):
        await manager.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception:
            manager.disconnect(websocket)

    @app.get("/")
    async def root():
        return {"message": "AI Crypto Trading API is running", "version": APP_VERSION, "symbols": SYMBOLS}

    @app.get("/status")
    async def status():
        """Get status of all active symbols."""
        r = redis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}", decode_responses=True)
        results = {}
        for symbol in SYMBOLS:
            # Get latest candle
            candle = await r.hgetall(f"cache:{symbol}:latest_candle")
            
            # Check for open position (reading state file)
            state_file = Path(f"state_{symbol}.json")
            position = None
            if state_file.exists():
                try:
                    with open(state_file, "r") as f:
                        position = json.load(f)
                except:
                    pass
            
            results[symbol] = {
                "active": True,
                "latest_price": candle.get("close"),
                "last_update": candle.get("timestamp"),
                "position": position
            }
        return results

    @app.post("/retrain")
    async def trigger_retrain():
        """Manually trigger automated retraining."""
        script = BASE_DIR / "auto_retrain.py"
        if not script.exists():
            raise HTTPException(404, f"Missing script: {script}")
        
        cmd = [os.getenv("PYTHON", "python"), str(script)]
        try:
            proc = subprocess.Popen(cmd, cwd=str(BASE_DIR.parent))
            app.state.procs[f"retrain:{proc.pid}"] = proc
            return {"status": "started", "pid": proc.pid}
        except Exception as e:
            raise HTTPException(500, f"Failed to start retraining: {e}")

    @app.get("/health")
    async def health():
        # check presence of key env vars and process registry health
        issues = []
        if not os.getenv("ENDPOINT_NAME"):
            issues.append("ENDPOINT_NAME not set")
        return {"status": "ok" if not issues else "degraded", "issues": issues, "running": list(app.state.procs.keys())}

    # ---------------------------
    # Prediction (SageMaker)
    # ---------------------------
    class PredictBody(BaseModel):
        inputs: list[list[float]] = Field(..., description="2D array: [window_size x feature_count]")

    @app.post("/predict")
    async def predict(body: PredictBody):
        endpoint_name = os.getenv("ENDPOINT_NAME")
        if not endpoint_name:
            raise HTTPException(500, "ENDPOINT_NAME env var is not set")

        # Lazy import sagemaker runtime
        try:
            import boto3
            from botocore.config import Config
            import sagemaker
            from sagemaker.predictor import Predictor
            from sagemaker.serializers import JSONSerializer
            from sagemaker.deserializers import JSONDeserializer
        except Exception as e:
            raise HTTPException(500, f"Failed to import boto3/sagemaker: {e}")

        try:
            config = Config(read_timeout=180, connect_timeout=180, retries={"max_attempts": 0})
            sm_runtime = boto3.client("sagemaker-runtime", config=config)
            session = sagemaker.Session(sagemaker_runtime_client=sm_runtime)
            predictor = Predictor(
                endpoint_name=endpoint_name,
                sagemaker_session=session,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer()
            )
            result = predictor.predict({"inputs": body.inputs})
            # Expecting {"probability": float, "signal": int}
            return {"endpoint": endpoint_name, **result}
        except Exception as e:
            raise HTTPException(500, f"Inference failed: {e}")

    # ---------------------------
    # Local prediction (optional; uses best_model.pth + inference.LSTMModel)
    # ---------------------------
    @app.post("/predict/local")
    async def predict_local(body: PredictBody):
        try:
            import torch
            import numpy as np
            import json
            from inference import LSTMModel
        except Exception as e:
            raise HTTPException(500, f"Failed to import local model deps: {e}")

        model_path = BASE_DIR / "best_model.pth"
        if not model_path.exists():
            raise HTTPException(404, f"Missing local model weights: {model_path}")

        # Hyperparams must match your trained model; feel free to adjust or load from a meta file
        input_size = len(body.inputs[0]) if body.inputs else 17
        model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2, output_size=1, dropout_rate=0.3)
        try:
            state = torch.load(str(model_path), map_location="cpu")
            model.load_state_dict(state)
        except Exception as e:
            raise HTTPException(500, f"Failed to load model: {e}")
        model.eval()
        x = np.asarray(body.inputs, dtype="float32")[None, ...]  # [1, T, F]
        with torch.no_grad():
            logits = model(torch.from_numpy(x))
            prob = float(torch.sigmoid(logits).item())
        return {"probability": prob, "signal": 1 if prob > 0.5 else 0}

    # ---------------------------
    # S3 / SageMaker utilities
    # ---------------------------
    class TrainAWSBody(BaseModel):
        # match aws_train_model.py args
        pair: str = "ETHUSDT"
        interval: str = "1m"
        window_size: int = 150
        epochs: int = 100
        batch_size: int = 1024
        learning_rate: float = 0.001
        patience: int = 5
        lookahead_period: int = 10
        risk_reward_ratio: float = 2.0
        profit_threshold_pct: float = 0.5

    @app.post("/train/aws")
    async def train_aws(body: TrainAWSBody):
        """
        Starts the torch training defined in aws_train_model.py as a subprocess.
        Returns a process id you can poll/stop.
        """
        script = BASE_DIR / "aws_train_model.py"
        if not script.exists():
            raise HTTPException(404, f"Missing script: {script}")

        cmd = [
            os.getenv("PYTHON", "python"),
            str(script),
            "--pair", body.pair,
            "--interval", body.interval,
            "--window-size", str(body.window_size),
            "--epochs", str(body.epochs),
            "--batch-size", str(body.batch_size),
            "--learning-rate", str(body.learning_rate),
            "--patience", str(body.patience),
            "--lookahead-period", str(body.lookahead_period),
            "--risk-reward-ratio", str(body.risk_reward_ratio),
            "--profit-threshold-pct", str(body.profit_threshold_pct),
        ]
        try:
            proc = subprocess.Popen(cmd, cwd=str(BASE_DIR))
            app.state.procs[f"train-aws:{proc.pid}"] = proc
            return {"status": "started", "pid": proc.pid, "cmd": cmd}
        except Exception as e:
            raise HTTPException(500, f"Failed to start training: {e}")

    class DeployBody(BaseModel):
        model_s3_path: str
        endpoint_name: str
        instance_type: str = "ml.t2.medium"

    @app.post("/deploy")
    async def deploy(body: DeployBody):
        """
        Deploy a model artifact to a SageMaker endpoint.
        """
        try:
            import sagemaker
            from sagemaker.pytorch import PyTorchModel
        except Exception as e:
            raise HTTPException(500, f"Failed to import sagemaker: {e}")

        try:
            sm_session = sagemaker.Session()
            role = os.getenv("SAGEMAKER_ROLE") or sagemaker.get_execution_role()
            model = PyTorchModel(
                model_data=body.model_s3_path,
                role=role,
                entry_point="inference.py",
                framework_version="2.4",
                py_version="py310",
                sagemaker_session=sm_session,
            )
            predictor = model.deploy(
                initial_instance_count=1,
                instance_type=body.instance_type,
                endpoint_name=body.endpoint_name,
            )
            return {"status": "deployed", "endpoint_name": predictor.endpoint_name}
        except Exception as e:
            raise HTTPException(500, f"Deploy failed: {e}")

    @app.post("/sagemaker/cleanup")
    async def cleanup():
        """
        Deletes SageMaker endpoints and trial components using delete.py helpers.
        """
        try:
            from delete import delete_trial_components, delete_endpoints
        except Exception as e:
            raise HTTPException(500, f"Import error: {e}")
        try:
            delete_trial_components()
            delete_endpoints()
            return {"status": "ok"}
        except Exception as e:
            raise HTTPException(500, f"Cleanup failed: {e}")

    @app.get("/sagemaker/endpoints")
    async def list_endpoints():
        try:
            import boto3
            sm = boto3.client("sagemaker")
            out = sm.list_endpoints(MaxResults=100)
            names = [e["EndpointName"] for e in out.get("Endpoints", [])]
            return {"endpoints": names}
        except Exception as e:
            raise HTTPException(500, f"List endpoints failed: {e}")

    # ---------------------------
    # Data / history
    # ---------------------------
    class HistoryBody(BaseModel):
        start_days_ago: int = 1000
        symbol: str = "ETHUSDT"
        interval: str = "1m"

    @app.post("/history/fetch")
    async def history_fetch(body: HistoryBody):
        """
        Runs history.py as a subprocess to fetch OHLCV data. Long-running.
        """
        script = BASE_DIR / "history.py"
        if not script.exists():
            raise HTTPException(404, f"Missing script: {script}")
        cmd = [os.getenv("PYTHON", "python"), str(script)]
        try:
            # Pass params via env vars for now (history.py reads BINANCE_* and uses defaults)
            proc = subprocess.Popen(cmd, cwd=str(BASE_DIR))
            app.state.procs[f"history:{proc.pid}"] = proc
            return {"status": "started", "pid": proc.pid}
        except Exception as e:
            raise HTTPException(500, f"Failed to start history fetch: {e}")

    # ---------------------------
    # Backtests
    # ---------------------------
    class BacktestBody(BaseModel):
        # optional; backtest.py loads from 'eth_1m_data' folder by default
        data_folder: str = "eth_1m_data"

    @app.post("/backtest")
    async def start_backtest(body: BacktestBody):
        script = BASE_DIR / "backtest.py"
        if not script.exists():
            raise HTTPException(404, f"Missing script: {script}")
        cmd = [os.getenv("PYTHON", "python"), str(script)]
        try:
            proc = subprocess.Popen(cmd, cwd=str(BASE_DIR))
            app.state.procs[f"backtest:{proc.pid}"] = proc
            return {"status": "started", "pid": proc.pid}
        except Exception as e:
            raise HTTPException(500, f"Failed to start backtest: {e}")

    # ---------------------------
    # Paper trading (testnet)
    # ---------------------------
    @app.post("/paper-trade/start")
    async def paper_trade_start():
        script = BASE_DIR / "paper_trade.py"
        if not script.exists():
            raise HTTPException(404, f"Missing script: {script}")
        # Ensure it points to the 'model' directory for meta and weights
        cmd = [os.getenv("PYTHON", "python"), str(script), "--model-dir", "model"]
        try:
            proc = subprocess.Popen(cmd, cwd=str(BASE_DIR.parent))
            app.state.procs[f"paper-trade:{proc.pid}"] = proc
            return {"status": "started", "pid": proc.pid}
        except Exception as e:
            raise HTTPException(500, f"Failed to start paper trading: {e}")

    class StopBody(BaseModel):
        pid: int

    @app.post("/jobs/stop")
    async def stop_job(body: StopBody):
        proc = next((p for k, p in app.state.procs.items() if p.pid == body.pid), None)
        if not proc:
            raise HTTPException(404, f"No tracked process with pid {body.pid}")
        try:
            proc.send_signal(signal.SIGTERM)
            return {"status": "stopping", "pid": body.pid}
        except Exception as e:
            raise HTTPException(500, f"Failed to stop process {body.pid}: {e}")

    @app.get("/jobs")
    async def list_jobs():
        out = []
        for key, proc in list(app.state.procs.items()):
            code = proc.poll()
            out.append({"key": key, "pid": proc.pid, "returncode": code})
            if code is not None:
                # process finished; cleanup
                app.state.procs.pop(key, None)
        return {"jobs": out}

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
