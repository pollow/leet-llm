"""007 — Top-k & Argmax — reference solution (TODO: fill in)."""

from __future__ import annotations

import numpy as np


def argmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.argmax(x, axis=axis)


def top_k(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argpartition(x, -k, axis=-1)[..., -k:]
    vals = np.take_along_axis(x, idx, axis=-1)

    order = np.argsort(vals, axis=-1)[..., ::-1]
    vals = np.take_along_axis(vals, order, axis=-1)
    idx = np.take_along_axis(idx, order, axis=-1)
    return (vals, idx)
