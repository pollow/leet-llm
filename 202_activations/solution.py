"""202 — Activations (GELU & SiLU).

Implement ``gelu`` and ``silu``. See README.md for the full explanation.
Run `uv run grade 202` to check your work.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU: ``x · ½ · (1 + erf(x / √2))``."""
    return x * 0.5 * (1 + erf(x / np.sqrt(2)))


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / swish: ``x · sigmoid(x)``."""
    sig = 1 / (1 + np.exp(-x))
    return x * sig
