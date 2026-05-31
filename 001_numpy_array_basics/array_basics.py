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
    b, length, features = x.shape
    per_group = features // n_groups
    # (B, L, F) -> (B, L, G, f) -> (B, G, L, f)
    grouped = x.reshape(b, length, n_groups, per_group)
    return grouped.transpose(0, 2, 1, 3)


def ungroup_last_axis(x: np.ndarray) -> np.ndarray:
    """Inverse of :func:`group_last_axis`.

    ``(B, G, L, f)`` -> ``(B, L, G * f)``.
    """
    b, n_groups, length, per_group = x.shape
    # (B, G, L, f) -> (B, L, G, f) -> (B, L, G*f)
    return x.transpose(0, 2, 1, 3).reshape(b, length, n_groups * per_group)
