"""203 — Layer Normalization.

Implement ``layer_norm``. See README.md for the full explanation.
Run `uv run grade 203` to check your work.

Hint: you may reuse ``from leet_llm import standardize`` (002).
"""

from __future__ import annotations

import numpy as np


def layer_norm(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """LayerNorm over the last axis: ``gamma * (x - mean) / sqrt(var + eps) + beta``."""
    raise NotImplementedError("Implement layer_norm — see 203_layer_norm/README.md")
