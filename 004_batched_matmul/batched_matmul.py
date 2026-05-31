"""004 — Batched Matmul & einsum. See README.md. Run `uv run grade 004`."""

from __future__ import annotations

import numpy as np


def batched_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix-multiply the last two axes, batched over leading axes.
    ``(..., M, K) , (..., K, N) -> (..., M, N)``. Use np.einsum."""
    raise NotImplementedError(
        "Implement batched_matmul — see 004_batched_matmul/README.md"
    )


def outer_product(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Outer product of two vectors, batched over leading axes.
    ``(..., M) , (..., N) -> (..., M, N)``. Use np.einsum."""
    raise NotImplementedError(
        "Implement outer_product — see 004_batched_matmul/README.md"
    )


def batched_trace(a: np.ndarray) -> np.ndarray:
    """Trace of the last two (square) axes. ``(..., N, N) -> (...)``. Use np.einsum."""
    raise NotImplementedError(
        "Implement batched_trace — see 004_batched_matmul/README.md"
    )
