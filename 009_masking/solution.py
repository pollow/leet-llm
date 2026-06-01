"""009 — Masked Fill & Triangular Mask — reference solution (TODO: fill in)."""

from __future__ import annotations

import numpy as np


def masked_fill(x: np.ndarray, mask: np.ndarray, value: float) -> np.ndarray:
    return np.where(mask, value, x)


def triangular_mask(n: int) -> np.ndarray:
    return np.triu(np.ones((n, n), dtype=bool), k=1)
