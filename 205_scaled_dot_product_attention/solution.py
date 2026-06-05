"""205 — Scaled Dot-Product Attention.

Implement ``sdpa``. See README.md for the full explanation.
Run `uv run grade 205` to check your work.

Hint: you may reuse ``from leet_llm import softmax, masked_fill`` (005, 009).
"""

from __future__ import annotations

import numpy as np

from leet_llm import softmax, masked_fill


def sdpa(
    q: np.ndarray, # [..., seq_len, dim_head]
    k: np.ndarray, # [..., seq_len, dim_head]
    v: np.ndarray, # [..., seq_len, dim_head]
    mask: np.ndarray | None = None, # [..., seq_len, seq_len]
) -> np.ndarray:
    """Scaled dot-product attention: ``softmax(QKᵀ/√d_k + mask) · V``.

    ``mask`` is boolean with ``True`` marking positions to hide (set to −∞ before softmax).
    """
    d_k = q.shape[-1]
    score = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(d_k)
    if mask is not None:
        score = masked_fill(score, mask, -np.inf)
    A = softmax(score)
    return A @ v
