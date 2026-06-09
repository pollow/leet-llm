"""213 — RoPE (Rotary Position Embedding), two conventions + the relative-position checker.

Implement ``rope_interleaved``, ``rope_half`` and ``rope_qk_dot``. See README.md.
Run `uv run grade 213` to check your work.

Hint: you may reuse ``from leet_llm import interleave, deinterleave, split_halves,
join_halves`` (011). The interleaved form is the one L3 / 216 use.
"""

from __future__ import annotations

import numpy as np

from leet_llm import deinterleave, interleave, split_halves


def calc_angle(dim_head: int, positions: np.ndarray, base: float = 10000.0):
    """Helper class for calculating angle tensor for RoPE."""
    idx = np.arange(0, dim_head, 2) # [0, 2, 4, ..., dim_head]
    inv_freq = np.pow(base, -idx / dim_head) #  [dim_head/2, ]
    return positions[..., None] * inv_freq # [batch, seq_len, dim_head / 2]

def rope_interleaved(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """RoPE, interleaved (Meta) convention: rotate adjacent pairs (x_2i, x_2i+1)."""
    a, b = deinterleave(x) # [batch, seq_len, dim_head / 2]

    dim_head = x.shape[-1]
    angle = calc_angle(dim_head, positions, base)

    out_a = a * np.cos(angle) - b * np.sin(angle)
    out_b = a * np.sin(angle) + b * np.cos(angle)

    return interleave(out_a, out_b)


def rope_half(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """RoPE, rotate-half (HF) convention: out = x*cos + [-x2, x1]*sin."""
    a, b = split_halves(x)
    rotate_half = np.concatenate([-b, a], axis = -1)

    dim_head = x.shape[-1]
    angle = calc_angle(dim_head, positions, base)
    angle = np.concatenate([angle, angle], axis=-1) # [batch, seq_len, dim_head]

    return x * np.cos(angle) + rotate_half * np.sin(angle)



def rope_qk_dot(q: np.ndarray, k: np.ndarray, m: int, n: int, base: float = 10000.0) -> np.ndarray:
    """Return <RoPE(q, m), RoPE(k, n)> over the last axis (interleaved convention).

    Used to verify RoPE's defining property: this depends only on the relative position
    (n - m), and equals <q, k> when m == n.
    """
    rope_q_m = rope_interleaved(q, np.array(m))
    rope_k_n = rope_interleaved(k, np.array(n))
    return np.sum(rope_q_m * rope_k_n, axis=-1)
