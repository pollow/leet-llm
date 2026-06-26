"""202 — Activations (GELU, sigmoid & SiLU).

Implement ``gelu``, ``sigmoid`` and ``silu``. See README.md for details.
Run `uv run grade 202` to check your work.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU: ``x · ½ · (1 + erf(x / √2))``."""
    return x * 0.5 * (1 + erf(x / np.sqrt(2)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic sigmoid: ``1 / (1 + exp(-x))``."""
    return 1.0 / (1.0 + np.exp(-x))


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / swish: ``x · sigmoid(x)``."""
    return x * sigmoid(x)
