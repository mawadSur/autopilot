from __future__ import annotations

from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

app = FastAPI(title="Autopilot Trading State Server")

# ----------------------------
# Data Models
# ----------------------------

class TelemetryUpdate(BaseModel):
    timestamp: Any
    price: float
    equity: float
    p_short: float
    p_hold: float
    p_long: float
    signal: int
    position: int # -1, 0, 1
    recent_trades: List[Dict[str, Any]] = []

class StateResponse(BaseModel):
    last_update: float
    price: float
    equity: float
    p_short: float
    p_hold: float
    p_long: float
    signal: int
    position: int
    equity_curve: List[Dict[str, Any]] # {"ts": ..., "val": ...}
    recent_trades: List[Dict[str, Any]]

# ----------------------------
# In-Memory Store
# ----------------------------

global_state = {
    "last_update": 0.0,
    "price": 0.0,
    "equity": 0.0,
    "p_short": 0.0,
    "p_hold": 0.0,
    "p_long": 0.0,
    "signal": 0,
    "position": 0,
    "equity_curve": [], # max 1000 points
    "recent_trades": []  # max 100 entries
}

# ----------------------------
# Endpoints
# ----------------------------

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/state", response_model=StateResponse)
def get_state():
    return global_state

@app.post("/api/telemetry")
def update_telemetry(update: TelemetryUpdate):
    global_state["last_update"] = time.time()
    global_state["price"] = update.price
    global_state["equity"] = update.equity
    global_state["p_short"] = update.p_short
    global_state["p_hold"] = update.p_hold
    global_state["p_long"] = update.p_long
    global_state["signal"] = update.signal
    global_state["position"] = update.position
    
    # Update equity curve
    ts = str(update.timestamp)
    global_state["equity_curve"].append({"ts": ts, "val": update.equity})
    if len(global_state["equity_curve"]) > 1000:
        global_state["equity_curve"].pop(0)
        
    # Sync trades
    if update.recent_trades:
        # Simple merge/replace for now
        global_state["recent_trades"] = update.recent_trades[-100:]
        
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
