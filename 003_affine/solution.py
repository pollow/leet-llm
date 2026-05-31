"""003 — Affine Transform — reference solution (TODO: fill in)."""

from __future__ import annotations

import numpy as np


def affine(x: np.ndarray, W: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    y = x @ W.T
    if b is not None:
        y = y + b
    return y
