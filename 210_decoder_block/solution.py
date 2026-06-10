"""210 — Decoder Block (original seq2seq).

Implement ``decoder_block`` (and the ``DecoderBlockParams`` container). See README.md.
Run `uv run grade 210` to check your work.

Hint: reuse ``from leet_llm import mha, ffn, layer_norm, add_residual, triangular_mask,
AttnParams, FFNParams``. Cross-attention = ``mha(a, cross_attn, x_kv=enc_out)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from leet_llm import (
    AttnParams,
    FFNParams,
    mha,
    ffn,
    layer_norm,
    add_residual,
    triangular_mask,
)


@dataclass(frozen=True)
class DecoderBlockParams:
    """Weights for one seq2seq decoder block: masked self-attn + cross-attn + FFN."""

    self_attn: AttnParams  # noqa: F821 — from leet_llm import AttnParams (206)
    cross_attn: AttnParams  # noqa: F821
    ffn: FFNParams  # noqa: F821 — from leet_llm import FFNParams (207)
    norm1_gamma: np.ndarray
    norm1_beta: np.ndarray
    norm2_gamma: np.ndarray
    norm2_beta: np.ndarray
    norm3_gamma: np.ndarray
    norm3_beta: np.ndarray


def decoder_block(
    x: np.ndarray,
    enc_out: np.ndarray,
    params: DecoderBlockParams,
    n_heads: int,
    self_mask: np.ndarray | None = None,
    cross_mask: np.ndarray | None = None,
    activation: str = "gelu",
) -> np.ndarray:
    """One post-norm decoder block: masked self-attn -> cross-attn -> FFN(activation)."""
    a = mha(x, params.self_attn, n_heads, mask=self_mask)
    a = layer_norm(add_residual(x, a), params.norm1_gamma, params.norm1_beta)
    b = mha(a, params.cross_attn, n_heads, mask=cross_mask, x_kv=enc_out)
    b = layer_norm(add_residual(a, b), params.norm2_gamma, params.norm2_beta)
    y = ffn(b, params.ffn, activation=activation)
    y = layer_norm(add_residual(b, y), params.norm3_gamma, params.norm3_beta)
    return y
