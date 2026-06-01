"""010 — Categorical sampling — reference solution (TODO: fill in)."""

from __future__ import annotations

import numpy as np


def sample_categorical(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cdf = probs.cumsum(axis=-1) # [..., K]
    u = rng.random(probs.shape[:-1] + (1,)) # [..., 1]
    return (u < cdf).argmax(axis=-1) # [..., 1]
