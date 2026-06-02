"""207 — Feed-Forward Network (classic MLP).

Implement ``ffn`` (and the ``FFNParams`` container). See README.md for details.
Run `uv run grade 207` to check your work.

Hint: reuse ``from leet_llm import affine, gelu`` (003, 202).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FFNParams:
    """Two-layer MLP weights, applied as ``x @ W.T + b``."""

    W1: np.ndarray  # (d_ff, d)
    b1: np.ndarray  # (d_ff,)
    W2: np.ndarray  # (d, d_ff)
    b2: np.ndarray  # (d,)


def ffn(x: np.ndarray, params: FFNParams) -> np.ndarray:
    """Classic FFN: ``gelu(x @ W1.T + b1) @ W2.T + b2``."""
    raise NotImplementedError("Implement ffn — see 207_feed_forward/README.md")
