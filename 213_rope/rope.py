"""213 — RoPE (Rotary Position Embedding), two conventions + the relative-position checker.

Implement ``rope_interleaved``, ``rope_half`` and ``rope_qk_dot``. See README.md.
Run `uv run grade 213` to check your work.

Hint: you may reuse ``from leet_llm import interleave, deinterleave, split_halves,
join_halves`` (011). The interleaved form is the one L3 / 216 use.
"""

from __future__ import annotations

import numpy as np


def rope_interleaved(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """RoPE, interleaved (Meta) convention: rotate adjacent pairs (x_2i, x_2i+1)."""
    raise NotImplementedError("Implement rope_interleaved — see 213_rope/README.md")


def rope_half(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """RoPE, rotate-half (HF) convention: out = x*cos + [-x2, x1]*sin."""
    raise NotImplementedError("Implement rope_half — see 213_rope/README.md")


def rope_qk_dot(q: np.ndarray, k: np.ndarray, m: int, n: int, base: float = 10000.0) -> np.ndarray:
    """Return <RoPE(q, m), RoPE(k, n)> over the last axis (interleaved convention).

    Used to verify RoPE's defining property: this depends only on the relative position
    (n - m), and equals <q, k> when m == n.
    """
    raise NotImplementedError("Implement rope_qk_dot — see 213_rope/README.md")
