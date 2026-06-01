"""Prediction-market FastAPI app.

Distinct from ``src/main.py`` (the legacy crypto-trading FastAPI). This app
exposes the Polymarket scanner + multi-agent research / calibration / risk
pipeline as HTTP endpoints. See ``src/api/main.py`` for the app factory and
endpoint handlers.
"""

from src.api.main import app, create_app

__all__ = ["app", "create_app"]
