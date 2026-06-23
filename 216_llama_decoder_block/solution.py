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

from leet_llm import AttnParams, SwiGLUParams, rms_norm, rope_from_freqs, rope_scaled_freqs, sdpa, swiglu_ffn, add_residual, group_last_axis, affine, triangular_mask, ungroup_last_axis


@dataclass(frozen=True)
class RopeParams:
    base: float = 10000.0
    pair_type: str = "interleaved"  # "interleaved" | "half"
    scaling: dict | None = None


@dataclass(frozen=True)
class LlamaBlockParams:
    """Weights for one Llama decoder block: RoPE-GQA + SwiGLU, two RMSNorms, bias-free."""
    attn: AttnParams
    ffn: SwiGLUParams
    attn_norm: np.ndarray  # RMSNorm weight (d,)
    ffn_norm: np.ndarray  # RMSNorm weight (d,)


def _rope_gqa(
    x: np.ndarray,  # [batch, seq_len, d_model]
    params: AttnParams,
    n_heads: int,
    n_kv_heads: int,
    positions: np.ndarray,
    mask: np.ndarray | None = None,
    rope_params: RopeParams = RopeParams(),
) -> np.ndarray:
    """Grouped-query attention with interleaved RoPE applied to QK"""
    n_g = n_heads // n_kv_heads  # group size

    Q = affine(x, params.Wq, params.bq)  # [batch_size, seq_len, d_model]
    # [batch_size, seq_len, n_kv_heads * dim_head]
    K = affine(x, params.Wk, params.bk)
    # [batch_size, seq_len, n_kv_heads * dim_head]
    V = affine(x, params.Wv, params.bv)

    Q = group_last_axis(Q, n_heads)  # [batch_size, n_heads, seq_len, dim_head]
    # [batch_size, n_kv_heads, seq_len, dim_head]
    K = group_last_axis(K, n_kv_heads)
    # [batch_size, n_kv_heads, seq_len, dim_head]
    V = group_last_axis(V, n_kv_heads)

    q_shape = Q.shape
    grouped_shape = [q_shape[0], n_kv_heads, n_g] + list(q_shape[2:])
    Q = Q.reshape(grouped_shape)
    K = K[:, :, None, ...]
    V = V[:, :, None, ...]

    inv_freqs = rope_scaled_freqs(
        q_shape[-1], rope_params.base, rope_params.scaling)

    Q_rope = rope_from_freqs(Q, positions, inv_freqs, rope_params.pair_type)
    K_rope = rope_from_freqs(K, positions, inv_freqs, rope_params.pair_type)

    gqa = sdpa(Q_rope, K_rope, V, mask)
    gqa = gqa.reshape(q_shape)
    gqa = ungroup_last_axis(gqa)

    return affine(gqa, params.Wo, params.bo)


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
    attn = _rope_gqa(a, params.attn, n_heads,
                     n_kv_heads, positions, mask, rope_params)
    h = add_residual(x, attn)
    f = rms_norm(h, params.ffn_norm, eps=eps)
    ffn = swiglu_ffn(f, params.ffn)
    return add_residual(h, ffn)
