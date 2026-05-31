"""004 — Batched Matmul & einsum — reference solution (TODO: fill in)."""

from __future__ import annotations

import numpy as np


def batched_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("...ij,...jk->...ik", a, b)


def outer_product(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.einsum("...i,...j->...ij", u, v)


def batched_trace(a: np.ndarray) -> np.ndarray:
    return np.einsum("...ii", a)
