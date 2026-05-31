"""002 — Broadcasting. See README.md. Run `uv run grade 002`."""

from __future__ import annotations

import numpy as np


def add_bias(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Add a length-F vector ``b`` to ``x`` of shape ``(..., F)``, broadcasting
    over all leading axes. Returns a new array."""
    raise NotImplementedError("Implement add_bias — see 002_broadcasting/README.md")


def standardize(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Standardize over the last axis: ``(x - mean) / sqrt(var + eps)``, with the
    mean and (population) variance taken over the last axis only."""
    raise NotImplementedError("Implement standardize — see 002_broadcasting/README.md")
