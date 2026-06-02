"""206 — Multi-Head Attention.

Implement ``mha`` (and the ``AttnParams`` container). See README.md for details.
Run `uv run grade 206` to check your work.

Hint: reuse ``from leet_llm import sdpa, group_last_axis, affine`` (205, 001, 003).
Pass ``x_kv`` for cross-attention; leave it None for self-attention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    raise NotImplementedError(
        "Implement mha — see 206_multi_head_attention/README.md"
    )
