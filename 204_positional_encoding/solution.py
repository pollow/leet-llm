"""204 — Sinusoidal Positional Encoding.

Implement ``sinusoidal_pe``. See README.md for the full explanation.
Run `uv run grade 204` to check your work.
"""

from __future__ import annotations

import numpy as np
from leet_llm import interleave


def sinusoidal_pe(seq_len: int, dim: int) -> np.ndarray:
    """Fixed (seq_len, dim) sinusoidal positional encoding (even=sin, odd=cos)."""
    """
        PE[pos, 2i]   = sin( pos / 10000^{2i/d} )
        PE[pos, 2i+1] = cos( pos / 10000^{2i/d} )
    """

    ids = np.arange(dim // 2)
    pw = np.pow(10000, -2 * ids / dim)
    pos = np.arange(seq_len)
    base = np.outer(pos, pw)

    return interleave(np.sin(base), np.cos(base))
