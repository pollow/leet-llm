"""213 — RoPE (Rotary Position Embedding).

Implement ``rope`` using the rotate-half convention. See README.md for details.
Run `uv run grade 213` to check your work.
"""

from __future__ import annotations

import numpy as np


def rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """Apply rotary position embedding to ``x`` (..., L, d) over the last axis.

    Rotate-half convention: ``x * cos(angle) + rotate_half(x) * sin(angle)``.
    """
    raise NotImplementedError("Implement rope — see 213_rope/README.md")
