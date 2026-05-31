"""003 — Affine Transform. See README.md. Run `uv run grade 003`."""

from __future__ import annotations

import numpy as np


def affine(x: np.ndarray, W: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Compute ``y = x @ W.T + b``.

    x: ``(..., F_in)``, W: ``(F_out, F_in)``, b: ``(F_out,)`` or None.
    Returns ``(..., F_out)``.
    """
    raise NotImplementedError("Implement affine — see 003_affine/README.md")
