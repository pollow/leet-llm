"""208 — Residual Connections & Norm Placement.

Implement ``add_residual``. See README.md for the full explanation (incl. pre- vs
post-norm placement, which the block tasks 209-211/216 build on).
Run `uv run grade 208` to check your work.
"""

from __future__ import annotations

import numpy as np


def add_residual(x: np.ndarray, sublayer_out: np.ndarray) -> np.ndarray:
    """Residual connection: ``x + sublayer_out``."""
    assert x.shape == sublayer_out.shape
    return x + sublayer_out
