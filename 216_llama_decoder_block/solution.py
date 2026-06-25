"""216 — Llama Decoder Block.

Implement ``llama_decoder_block`` (and the ``LlamaBlockParams`` container). See README.md.
Run `uv run grade 216` to check your work.

Hint: reuse ``from leet_llm import rms_norm, rope_interleaved, sdpa, swiglu_ffn,
add_residual, group_last_axis, affine, triangular_mask, AttnParams, SwiGLUParams``. Apply
RoPE (interleaved convention, as L3 uses) to the per-head q/k projections before the
attention scores; pre-norm RMSNorm placement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from leet_llm import (
    AttnParams,
    SwiGLUParams,
    rms_norm,
    rope_from_freqs,
    rope_scaled_freqs,
    sdpa,
    swiglu_ffn,
    add_residual,
    group_last_axis,
    affine,
    triangular_mask,
    ungroup_last_axis,
    RopeParams,
    gqa,
)


@dataclass(frozen=True)
class LlamaBlockParams:
    """Weights for one Llama decoder block: RoPE-GQA + SwiGLU, two RMSNorms, bias-free."""

    attn: AttnParams
    ffn: SwiGLUParams
    attn_norm: np.ndarray  # RMSNorm weight (d,)
    ffn_norm: np.ndarray  # RMSNorm weight (d,)


def llama_decoder_block(
    x: np.ndarray,
    params: LlamaBlockParams,
    n_heads: int,
    n_kv_heads: int,
    positions: np.ndarray,
    mask: np.ndarray | None = None,
    eps: float = 1e-5,
    rope_params: RopeParams = RopeParams(),
) -> np.ndarray:
    """One pre-norm Llama block: RMSNorm(eps) -> RoPE-GQA -> residual -> RMSNorm(eps) -> SwiGLU -> residual."""
    if mask is None:
        L = x.shape[-2]
        mask = triangular_mask(L)
    a = rms_norm(x, params.attn_norm, eps=eps)
    attn = gqa(a, params.attn, n_heads, n_kv_heads, mask, positions, rope_params)
    h = add_residual(x, attn)
    f = rms_norm(h, params.ffn_norm, eps=eps)
    ffn = swiglu_ffn(f, params.ffn)
    return add_residual(h, ffn)
