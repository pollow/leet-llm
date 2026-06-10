"""209 — Encoder Block (BERT-style).

Implement ``encoder_block`` (and the ``EncoderBlockParams`` container). See README.md.
Run `uv run grade 209` to check your work.

Hint: reuse ``from leet_llm import mha, ffn, layer_norm, add_residual, AttnParams, FFNParams``.
Post-norm, bidirectional self-attention (no causal mask).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from leet_llm import AttnParams, FFNParams, mha, add_residual, ffn, layer_norm


@dataclass(frozen=True)
class EncoderBlockParams:
    """Weights for one encoder block: attention + FFN + two LayerNorms."""

    attn: AttnParams  # noqa: F821 — from leet_llm import AttnParams (206)
    ffn: FFNParams  # noqa: F821 — from leet_llm import FFNParams (207)
    norm1_gamma: np.ndarray
    norm1_beta: np.ndarray
    norm2_gamma: np.ndarray
    norm2_beta: np.ndarray


def encoder_block(
    x: np.ndarray,
    params: EncoderBlockParams,
    n_heads: int,
    mask: np.ndarray | None = None,
    activation: str = "gelu",
) -> np.ndarray:
    """One post-norm encoder block: LN(x + SelfAttn(x)) then LN(a + FFN(a, activation))."""
    a = mha(x, params.attn, n_heads, mask=mask)
    a = layer_norm(add_residual(x, a), params.norm1_gamma, params.norm1_beta)
    y = ffn(a, params.ffn, activation=activation)
    y = layer_norm(add_residual(a, y), params.norm2_gamma, params.norm2_beta)
    return y
