"""008 — Gather Rows & One-Hot. See README.md. Run `uv run grade 008`."""

from __future__ import annotations

import numpy as np


def gather_rows(table: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Select rows of a ``(N, F)`` table by integer indices ``idx`` (shape S).
    Returns shape ``S + (F,)``."""
    raise NotImplementedError("Implement gather_rows — see 008_gather_onehot/README.md")


def one_hot(idx: np.ndarray, n: int) -> np.ndarray:
    """One-hot encode integer labels ``idx`` (shape S) into floats of shape
    ``S + (n,)``."""
    raise NotImplementedError("Implement one_hot — see 008_gather_onehot/README.md")
