"""006 — Log-Sum-Exp & Log-Softmax — reference solution (TODO: fill in)."""

from __future__ import annotations

import numpy as np


def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    se = np.exp(x - m).sum(axis=axis)
    lse = np.log(se) + m.squeeze(axis=axis)
    return lse


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x - np.expand_dims(logsumexp(x, axis), axis=axis)
