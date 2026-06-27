"""215 — Grouped-Query Attention (GQA).

Implement ``gqa``. See README.md for the full explanation.
Run `uv run grade 215` to check your work.

Hint: reuse ``from leet_llm import sdpa, group_last_axis, affine, AttnParams`` (205, 001,
003, 206). Repeat each K/V head ``n_heads // n_kv_heads`` times to match the query heads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from leet_llm import (
    AttnParams,
    affine,
    sdpa,
    group_last_axis,
    ungroup_last_axis,
    rope_scaled_freqs,
    rope_from_freqs,
)


@dataclass(frozen=True)
class RopeParams:
    base: float = 10000.0
    pair_type: str = "interleaved"  # "interleaved" | "half"
    scaling: dict | None = None


def gqa(
    x: np.ndarray,  # [batch, seq_len, d_model]
    params: "AttnParams",
    n_heads: int,
    n_kv_heads: int,
    mask: np.ndarray | None = None,
    positions: np.ndarray | None = None,
    rope_params: RopeParams | None = None,
    af: float = 1.0,
    sink_logits: np.ndarray | None = None,
) -> np.ndarray:
    """Grouped-query attention; reduces to MHA when ``n_kv_heads == n_heads``."""
    num_group = n_heads // n_kv_heads

    Q = affine(x, params.Wq, params.bq)  # [batch_size, seq_len, d_model]
    K = affine(x, params.Wk, params.bk)  # [batch_size, seq_len, n_kv_heads * dim_head]
    V = affine(x, params.Wv, params.bv)  # [batch_size, seq_len, n_kv_heads * dim_head]

    Q = group_last_axis(Q, n_heads)  # [batch_size, n_heads, seq_len, dim_head]
    K = group_last_axis(K, n_kv_heads)  # [batch_size, n_kv_heads, seq_len, dim_head]
    V = group_last_axis(V, n_kv_heads)  # [batch_size, n_kv_heads, seq_len, dim_head]

    origin_shape = Q.shape
    shape = [Q.shape[0], n_kv_heads, -1] + list(Q.shape[2:])
    Q = Q.reshape(shape)
    K = K[:, :, None, ...]
    V = V[:, :, None, ...]

    if rope_params is not None and positions is not None:
        inv_freqs = rope_scaled_freqs(
            origin_shape[-1], rope_params.base, rope_params.scaling
        )
        Q = rope_from_freqs(Q, positions, inv_freqs, rope_params.pair_type)
        K = rope_from_freqs(K, positions, inv_freqs, rope_params.pair_type)
        Q = Q * af
        K = K * af

    sink = None
    if sink_logits is not None:
        sink = np.asarray(sink_logits).reshape(1, n_kv_heads, num_group, 1, 1)
    out = sdpa(Q, K, V, mask, sink_logits=sink)

    gqa = ungroup_last_axis(out.reshape(origin_shape))

    return affine(gqa, params.Wo, params.bo)
