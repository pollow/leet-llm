"""215 — Grouped-Query Attention (GQA).

Implement ``gqa``. See README.md for the full explanation.
Run `uv run grade 215` to check your work.

Hint: reuse ``from leet_llm import sdpa, group_last_axis, affine, AttnParams`` (205, 001,
003, 206). Repeat each K/V head ``n_heads // n_kv_heads`` times to match the query heads.
"""

from __future__ import annotations

import numpy as np


def gqa(
    x: np.ndarray,
    params: "AttnParams",  # noqa: F821 — from leet_llm import AttnParams (206)
    n_heads: int,
    n_kv_heads: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Grouped-query attention; reduces to MHA when ``n_kv_heads == n_heads``."""
    raise NotImplementedError("Implement gqa — see 215_gqa/README.md")
