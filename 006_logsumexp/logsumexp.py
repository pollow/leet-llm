"""006 — Log-Sum-Exp & Log-Softmax. See README.md. Run `uv run grade 006`."""

from __future__ import annotations

import numpy as np


def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable ``log(sum(exp(x)))`` along ``axis`` (the axis is reduced)."""
    raise NotImplementedError("Implement logsumexp — see 006_logsumexp/README.md")


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """``x - logsumexp(x, axis)``; same shape as ``x``."""
    raise NotImplementedError("Implement log_softmax — see 006_logsumexp/README.md")
