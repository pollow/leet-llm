"""205 — Scaled Dot-Product Attention.

Implement ``sdpa``. See README.md for the full explanation.
Run `uv run grade 205` to check your work.

Hint: you may reuse ``from leet_llm import softmax, masked_fill`` (005, 009).
"""

from __future__ import annotations

import numpy as np


def sdpa(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Scaled dot-product attention: ``softmax(QKᵀ/√d_k + mask) · V``.

    ``mask`` is boolean with ``True`` marking positions to hide (set to −∞ before softmax).
    """
    raise NotImplementedError(
        "Implement sdpa — see 205_scaled_dot_product_attention/README.md"
    )
