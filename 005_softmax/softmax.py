"""005 — Softmax (numerically stable). See README.md. Run `uv run grade 005`."""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along ``axis`` (subtract the max first).
    Output has the same shape as ``x`` and sums to 1 along ``axis``."""
    raise NotImplementedError("Implement softmax — see 005_softmax/README.md")
