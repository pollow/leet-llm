"""112 — Position Indices.

Implement ``position_ids``. See README.md for the full explanation.
Run `uv run grade 112` to check your work.
"""

from __future__ import annotations

import numpy as np


def position_ids(seqs: list[list[int]], max_len: int | None = None) -> np.ndarray:
    """Return a ``(B, L)`` array of positions ``0..n-1`` per row, 0 in the padding region."""
    raise NotImplementedError(
        "Implement position_ids — see 112_position_indices/README.md"
    )
