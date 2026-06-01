"""111 — Padding & Mask.

Implement the two functions below. See README.md for the full explanation.
Run `uv run grade 111` to check your work.
"""

from __future__ import annotations

import numpy as np


def pad_batch(
    seqs: list[list[int]], pad_id: int = 0, max_len: int | None = None
) -> np.ndarray:
    """Pad ``seqs`` to a rectangular ``(B, L)`` int array (truncating to ``max_len``)."""
    raise NotImplementedError(
        "Implement pad_batch — see 111_padding_and_mask/README.md"
    )


def padding_mask(seqs: list[list[int]], max_len: int | None = None) -> np.ndarray:
    """Return a ``(B, L)`` array: 1 for real positions, 0 for padding."""
    raise NotImplementedError(
        "Implement padding_mask — see 111_padding_and_mask/README.md"
    )
