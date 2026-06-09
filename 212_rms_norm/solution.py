"""212 — RMSNorm.

Implement ``rms_norm``. See README.md for the full explanation.
Run `uv run grade 212` to check your work.
"""

from __future__ import annotations

import numpy as np


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """RMSNorm over the last axis: ``x / sqrt(mean(x**2) + eps) * weight``."""
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    norm = (x / rms) * weight
    return norm
