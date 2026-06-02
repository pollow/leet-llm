"""202 — Activations (GELU & SiLU).

Implement ``gelu`` and ``silu``. See README.md for the full explanation.
Run `uv run grade 202` to check your work.
"""

from __future__ import annotations

import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU: ``x · ½ · (1 + erf(x / √2))``."""
    raise NotImplementedError("Implement gelu — see 202_activations/README.md")


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / swish: ``x · sigmoid(x)``."""
    raise NotImplementedError("Implement silu — see 202_activations/README.md")
