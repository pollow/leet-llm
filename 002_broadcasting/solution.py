"""002 — Broadcasting — reference solution (TODO: fill in)."""

from __future__ import annotations

import numpy as np


def add_bias(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x + b


def standardize(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)
