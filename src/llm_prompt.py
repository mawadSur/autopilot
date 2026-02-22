#!/usr/bin/env python3
"""LLM prompt utilities used for numeric next-close prediction."""

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np

NUMBER_RE = re.compile(r"[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?")


def build_prompt_from_window(
    feature_names: List[str],
    window: np.ndarray,
    prompt_template: Optional[str] = None,
) -> str:
    """Build a compact prompt from the latest feature row in a window."""
    last = window[-1]
    pairs = [f"{name}: {float(val):.8f}" for name, val in zip(feature_names, last)]
    features_text = "\\n".join(pairs)
    if prompt_template is None:
        prompt = (
            "You are given the latest feature values for an asset's 1-minute bar.\\n"
            "Using these features, predict the next bar's closing price (a single numeric value).\\n"
            "Output ONLY the predicted close as a plain number (no extra words).\\n\\n"
            "Features:\\n"
            f"{features_text}\\n\\n"
            "Answer:"
        )
    else:
        prompt = prompt_template.replace("{features}", features_text)
    return prompt


def parse_number_from_text(text: str) -> Optional[float]:
    """Extract the first floating number from LLM output."""
    if not isinstance(text, str):
        return None
    m = NUMBER_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


__all__ = ["build_prompt_from_window", "parse_number_from_text", "NUMBER_RE"]
