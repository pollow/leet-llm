"""011 — Interleave & Halves.

Implement the four functions below. See README.md for the full explanation.
Run `uv run grade 011` to check your work.
"""

from __future__ import annotations

import numpy as np


def interleave(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Interleave two `(..., m)` arrays into `(..., 2m)`: a0, b0, a1, b1, ..."""
    raise NotImplementedError("Implement interleave — see 011_interleave_halves/README.md")


def deinterleave(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of interleave: split `(..., 2m)` into (evens, odds), each `(..., m)`."""
    raise NotImplementedError("Implement deinterleave — see 011_interleave_halves/README.md")


def split_halves(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split `(..., 2m)` into its front and back halves, each `(..., m)`."""
    raise NotImplementedError("Implement split_halves — see 011_interleave_halves/README.md")


def join_halves(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Inverse of split_halves: concatenate two `(..., m)` arrays along the last axis."""
    raise NotImplementedError("Implement join_halves — see 011_interleave_halves/README.md")
