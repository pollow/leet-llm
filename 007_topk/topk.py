"""007 — Top-k & Argmax. See README.md. Run `uv run grade 007`."""

from __future__ import annotations

import numpy as np


def argmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Index of the maximum value along ``axis`` (the axis is removed)."""
    raise NotImplementedError("Implement argmax — see 007_topk/README.md")


def top_k(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The ``k`` largest values and their indices along the last axis, sorted
    descending. Returns ``(values, indices)``, each of shape ``(..., k)``."""
    raise NotImplementedError("Implement top_k — see 007_topk/README.md")
