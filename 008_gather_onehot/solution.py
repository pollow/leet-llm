"""008 — Gather Rows & One-Hot — reference solution (TODO: fill in)."""

from __future__ import annotations

import numpy as np


def gather_rows(table: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return table[idx]


def one_hot(idx: np.ndarray, n: int) -> np.ndarray:
    return np.eye(n)[idx]
