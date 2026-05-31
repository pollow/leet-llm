"""009 — Masked Fill & Triangular Mask. See README.md. Run `uv run grade 009`."""

from __future__ import annotations

import numpy as np


def masked_fill(x: np.ndarray, mask: np.ndarray, value: float) -> np.ndarray:
    """Return a copy of ``x`` with positions where ``mask`` is True set to ``value``.
    ``mask`` broadcasts against ``x``; ``x`` is not modified."""
    raise NotImplementedError("Implement masked_fill — see 009_masking/README.md")


def triangular_mask(n: int) -> np.ndarray:
    """Boolean ``(n, n)`` mask, True strictly above the diagonal (``j > i``)."""
    raise NotImplementedError("Implement triangular_mask — see 009_masking/README.md")
