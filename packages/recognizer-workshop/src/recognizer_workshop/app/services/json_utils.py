"""Helpers for converting application values to strict JSON-compatible values."""

import math
from typing import Any

import numpy as np


def to_json_safe(value: Any) -> Any:
    """Convert common NumPy/Python values to strict JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, (np.floating, np.integer)):
        scalar = value.item()
        if isinstance(scalar, float) and not math.isfinite(scalar):
            return None
        return scalar
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    return value
