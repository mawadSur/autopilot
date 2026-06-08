"""Stocks-stack test package.

P3 stocks adapter (Alpaca). Hermetic — every HTTP call is mocked so the
suite runs without network access. Runner:

    env PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests/stocks

A parallel suite under ``tests/prediction_market_scanner/`` covers the
same adapter from the prediction-market test tree; the duplication is
intentional so both runners exercise the wiring.
"""
