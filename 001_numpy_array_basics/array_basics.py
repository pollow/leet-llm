"""001 — Array Basics: Reshape & Transpose.

Implement the two functions below. See README.md for the full explanation.
Run `uv run grade 001` to check your work.
"""

from __future__ import annotations

import numpy as np


def group_last_axis(x: np.ndarray, n_groups: int) -> np.ndarray:
    """Split the last axis of ``x`` into ``n_groups`` groups and bring the group
    axis to the front.

    ``(B, L, F)`` -> ``(B, n_groups, L, F // n_groups)``.
    """
    raise NotImplementedError(
        "Implement group_last_axis — see 001_numpy_array_basics/README.md"
    )


def ungroup_last_axis(x: np.ndarray) -> np.ndarray:
    """Inverse of :func:`group_last_axis`.

    ``(B, G, L, f)`` -> ``(B, L, G * f)``.
    """
    raise NotImplementedError(
        "Implement ungroup_last_axis — see 001_numpy_array_basics/README.md"
    )
