"""204 — Sinusoidal Positional Encoding.

Implement ``sinusoidal_pe``. See README.md for the full explanation.
Run `uv run grade 204` to check your work.
"""

from __future__ import annotations

import numpy as np


def sinusoidal_pe(seq_len: int, dim: int) -> np.ndarray:
    """Fixed (seq_len, dim) sinusoidal positional encoding (even=sin, odd=cos)."""
    raise NotImplementedError(
        "Implement sinusoidal_pe — see 204_positional_encoding/README.md"
    )
