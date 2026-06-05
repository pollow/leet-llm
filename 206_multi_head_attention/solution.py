"""206 — Multi-Head Attention.

Implement ``mha`` (and the ``AttnParams`` container). See README.md for details.
Run `uv run grade 206` to check your work.

Hint: reuse ``from leet_llm import sdpa, group_last_axis, affine`` (205, 001, 003).
Pass ``x_kv`` for cross-attention; leave it None for self-attention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from leet_llm import affine, group_last_axis, sdpa, ungroup_last_axis


@dataclass(frozen=True)
class AttnParams:
    """Projection weights for attention.

    ``Wq/Wk/Wv/Wo`` are ``(out, in)`` matrices applied as ``x @ W.T`` (reuse L0 003 affine).
    The four biases are optional: ``None`` ⇒ bias-free (Llama / GQA). The classic
    Transformer and GPT-2-style models pass real biases (L3 opus-mt capstone).
    """

    Wq: np.ndarray
    Wk: np.ndarray
    Wv: np.ndarray
    Wo: np.ndarray
    bq: np.ndarray | None = None
    bk: np.ndarray | None = None
    bv: np.ndarray | None = None
    bo: np.ndarray | None = None


def mha(
    x_q: np.ndarray,
    params: AttnParams,
    n_heads: int,
    x_kv: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Multi-head attention. ``x_kv`` defaults to ``x_q`` (self-attention).

    Apply the optional ``params.bq/bk/bv/bo`` to the q/k/v/out projections when present (treat ``None`` as zero).
    """
    if x_kv is None:
        x_kv = x_q

    Q = affine(x_q, params.Wq, params.bq)  # [batch_size, seq_len, d_model]
    K = affine(x_kv, params.Wk, params.bk)
    V = affine(x_kv, params.Wv, params.bv)

    Q = group_last_axis(Q, n_heads)  # [batch_size, n_heads, seq_len, dim_head]
    K = group_last_axis(K, n_heads)
    V = group_last_axis(V, n_heads)

    mha = ungroup_last_axis(sdpa(Q, K, V, mask))

    return affine(mha, params.Wo, params.bo)
