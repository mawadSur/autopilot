# main.py (FastAPI)
from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from config import cfg

# NOTE: We intentionally avoid heavy AWS / ML imports at module import time.
#       They will be imported inside endpoints to keep `uvicorn main:app --reload` reliable.

APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
BASE_DIR = Path(__file__).resolve().parent

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

    @app.get("/")
    async def root():
        return {"message": "AI Crypto Trading API is running", "version": APP_VERSION}

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
    # Removed /predict/local (legacy path). Use /predict with a deployed endpoint.

    # ---------------------------
    # S3 / SageMaker utilities
    # ---------------------------
    class TrainAWSBody(BaseModel):
        bucket: Optional[str] = None
        prefix: str = "eth-training"
        local_train_dir: str = cfg.data_dir
        train_s3_uri: Optional[str] = None

    @app.post("/train/aws")
    async def train_aws(body: TrainAWSBody):
        """
        Starts the torch training defined in train_model.py as a subprocess.
        Returns a process id you can poll/stop.
        """
        script = BASE_DIR / "deploy.py"
        if not script.exists():
            raise HTTPException(404, f"Missing script: {script}")

        cmd = [
            os.getenv("PYTHON", "python"),
            str(script),
            "--mode", "training",
            "--bucket", body.bucket or os.getenv("S3_BUCKET_NAME", ""),
            "--prefix", body.prefix,
            "--local-train-dir", body.local_train_dir,
        ]
        if body.train_s3_uri:
            cmd.extend(["--train-s3-uri", body.train_s3_uri])
        try:
            proc = subprocess.Popen(cmd, cwd=str(BASE_DIR))
            app.state.procs[f"train-aws:{proc.pid}"] = proc
            return {"status": "started", "pid": proc.pid, "cmd": cmd}
        except Exception as e:
            raise HTTPException(500, f"Failed to start training: {e}")

    class DeployBody(BaseModel):
        model_s3_path: str
        endpoint_name: str
        serverless_memory: int = cfg.serverless_memory_mb
        serverless_max_concurrency: int = cfg.serverless_max_concurrency

    @app.post("/deploy")
    async def deploy(body: DeployBody):
        """
        Deploy a model artifact to a SageMaker endpoint.
        """
        script = BASE_DIR / "deploy.py"
        if not script.exists():
            raise HTTPException(404, f"Missing script: {script}")

        cmd = [
            os.getenv("PYTHON", "python"),
            str(script),
            "--mode", "serverless",
            "--model-s3", body.model_s3_path,
            "--endpoint-name", body.endpoint_name,
            "--serverless-memory", str(body.serverless_memory),
            "--serverless-max-concurrency", str(body.serverless_max_concurrency),
        ]
        try:
            proc = subprocess.Popen(cmd, cwd=str(BASE_DIR))
            app.state.procs[f"deploy:{proc.pid}"] = proc
            return {"status": "started", "pid": proc.pid, "cmd": cmd}
        except Exception as e:
            raise HTTPException(500, f"Deploy failed: {e}")

    @app.post("/sagemaker/cleanup")
    async def cleanup():
        """
        Deletes SageMaker endpoints and trial components using delete.py helpers.
        """
        try:
            from delete import delete_experiments, delete_endpoints
        except Exception as e:
            raise HTTPException(500, f"Import error: {e}")
        try:
            delete_experiments(boto3.client("sagemaker"), dry_run=False)
            delete_endpoints(boto3.client("sagemaker"), dry_run=False)
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
        cmd = [os.getenv("PYTHON", "python"), str(script)]
        try:
            proc = subprocess.Popen(cmd, cwd=str(BASE_DIR))
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
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
