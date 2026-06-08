"""Crypto training pipeline: OHLCV backfill, dataset assembly, XGBoost training.

Mirrors the prediction-market calibration_agent layout but consumes
Coinbase OHLCV (not Polymarket trade-execution logs). The trained model
is intended as a USD-native replacement for the legacy 2023-2025
USDT-trained transformer in ``model_sanity/``.
"""
