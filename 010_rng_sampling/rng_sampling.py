"""010 — Categorical sampling. See README.md. Run `uv run grade 010`."""

from __future__ import annotations

import numpy as np


def sample_categorical(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one index per distribution along the last axis of ``probs``.

    ``probs`` has shape ``(..., K)`` (each last-axis vector sums to 1); returns an
    integer array of shape ``probs.shape[:-1]``. Use ``rng`` for all randomness.
    """
    raise NotImplementedError(
        "Implement sample_categorical — see 010_rng_sampling/README.md"
    )
