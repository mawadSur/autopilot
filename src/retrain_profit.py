#!/usr/bin/env python3
"""
One-click profit-optimized retraining.

Usage:
  python src/retrain_profit.py --data-path ... --output-dir ...
"""

from __future__ import annotations

import os
import sys

from train_model import main as train_main


def main() -> None:
    os.environ["PROFIT_MODE"] = "1"
    train_main()


if __name__ == "__main__":
    main()
