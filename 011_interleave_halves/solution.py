"""011 — Interleave & Halves.

Implement the four functions below. See README.md for the full explanation.
Run `uv run grade 011` to check your work.
"""

from __future__ import annotations

import numpy as np


def interleave(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Interleave two `(..., m)` arrays into `(..., 2m)`: a0, b0, a1, b1, ..."""
    return np.stack((a, b), axis=-1).reshape(*a.shape[:-1], -1)


def deinterleave(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of interleave: split `(..., 2m)` into (evens, odds), each `(..., m)`."""
    m2 = x.shape[-1]
    return x[..., :m2:2], x[..., 1:m2:2]


def split_halves(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split `(..., 2m)` into its front and back halves, each `(..., m)`."""
    m = x.shape[-1]
    return x[..., : m // 2], x[..., m//2 :]
    


def join_halves(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Inverse of split_halves: concatenate two `(..., m)` arrays along the last axis."""
    return np.concatenate((a, b), axis=-1)
