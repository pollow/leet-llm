"""001 — Array Basics: Reshape & Transpose — reference solution."""

from __future__ import annotations

import numpy as np


def group_last_axis(x: np.ndarray, n_groups: int) -> np.ndarray:
    b, length, features = x.shape
    per_group = features // n_groups
    # (B, L, F) -> (B, L, G, f) -> (B, G, L, f)
    return x.reshape(b, length, n_groups, per_group).transpose(0, 2, 1, 3)


def ungroup_last_axis(x: np.ndarray) -> np.ndarray:
    b, n_groups, length, per_group = x.shape
    # (B, G, L, f) -> (B, L, G, f) -> (B, L, G*f)
    return x.transpose(0, 2, 1, 3).reshape(b, length, n_groups * per_group)
