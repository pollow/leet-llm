"""Task 310 reference implementation for Gemma-2 forward wiring.

This file defines ``softcap``, ``geglu_ffn``, ``load_gemma``, and
``gemma_forward``. Keep docstrings focused on API contracts; see ``README.md``
for model rationale and detailed delta explanations.
"""

from __future__ import annotations

from dataclasses import dataclass
from token import tok_name

import numpy as np

from leet_llm import (
    AttnParams,
    RopeParams,
    add_residual,
    affine,
    embedding,
    gelu,
    group_last_axis,
    masked_fill,
    rms_norm,
    rope_from_freqs,
    rope_scaled_freqs,
    softmax,
    triangular_mask,
    ungroup_last_axis,
    sliding_window_mask,
)


def softcap(x: np.ndarray, cap: float) -> np.ndarray:
    """Apply tanh soft-capping: ``cap * tanh(x / cap)``.

    Args:
        x: Input tensor of any shape.
        cap: Positive scalar cap value.
    Returns:
        Tensor with the same shape as ``x``.
    """
    return cap * np.tanh(x / cap)


@dataclass(frozen=True)
class GeGLUParams:
    """GeGLU FFN weights: gate/up ``(intermediate, d)``, down ``(d, intermediate)``."""

    gate: np.ndarray  # (intermediate_size, d)
    up: np.ndarray  # (intermediate_size, d)
    down: np.ndarray  # (d, intermediate_size)


def _gelu_tanh(z: np.ndarray):
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * np.pow(z, 3.0))))


def geglu_ffn(x: np.ndarray, params: GeGLUParams) -> np.ndarray:
    """Compute Gemma GeGLU FFN output.

    Formula: ``down( gelu_tanh(x @ gate.T) * (x @ up.T) )``.
    Use GELU tanh approximation (not SiLU). Input and output shapes are ``(B, L, d)``.
    """
    h = x @ params.gate.T
    h = _gelu_tanh(h)
    h = h * (x @ params.up.T)
    h = h @ params.down.T
    return h


def gemma_rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return rms_norm(x, 1.0 + w, eps=eps)


# ---------------------------------------------------------------------------
# Gemma-2 whole-model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GemmaConfig:
    """Gemma-2 config values used by loading and forward pass."""

    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    intermediate_size: int
    norm_eps: float = 1e-6
    rope_base: float = 10000.0
    query_pre_attn_scalar: int = 256
    final_logit_softcapping: float = 30.0
    attn_logit_softcapping: float = 50.0
    sliding_window: int = 4096
    max_seq_len: int = 8192


@dataclass(frozen=True)
class GemmaParams:
    """Packed Gemma-2 weights: tied token embedding, per-layer tensors, final norm."""

    tok_embed: np.ndarray  # (V, d)
    layers: list  # list of per-layer dicts
    final_norm: np.ndarray  # (d,)


def load_gemma(weights: dict, cfg: GemmaConfig) -> GemmaParams:
    """Map HF Gemma-2 tensors into ``GemmaParams``.

    Keep tensor shapes consistent with ``cfg`` and preserve tied embeddings:
    ``lm_head`` reuses ``model.embed_tokens.weight``.
    See ``README.md`` for the full key map.
    """
    tok_embed = weights["model.embed_tokens.weight"]
    final_norm = weights["model.norm.weight"]

    layers: list[dict[str, np.ndarray]] = []
    for i in range(cfg.n_layers):
        prefix = f"model.layers.{i}"
        layers.append(
            {
                "input_norm": weights[f"{prefix}.input_layernorm.weight"],
                "post_attn_norm": weights[f"{prefix}.post_attention_layernorm.weight"],
                "pre_ffn_norm": weights[f"{prefix}.pre_feedforward_layernorm.weight"],
                "post_ffn_norm": weights[f"{prefix}.post_feedforward_layernorm.weight"],
                "Wq": weights[f"{prefix}.self_attn.q_proj.weight"],
                "Wk": weights[f"{prefix}.self_attn.k_proj.weight"],
                "Wv": weights[f"{prefix}.self_attn.v_proj.weight"],
                "Wo": weights[f"{prefix}.self_attn.o_proj.weight"],
                "W_gate": weights[f"{prefix}.mlp.gate_proj.weight"],
                "W_up": weights[f"{prefix}.mlp.up_proj.weight"],
                "W_down": weights[f"{prefix}.mlp.down_proj.weight"],
            }
        )

    return GemmaParams(tok_embed=tok_embed, layers=layers, final_norm=final_norm)


def gemma_sdpa(
    q: np.ndarray,  # [..., seq_len, dim_head]
    k: np.ndarray,  # [..., seq_len, dim_head]
    v: np.ndarray,  # [..., seq_len, dim_head]
    mask: np.ndarray,  # [..., seq_len, seq_len]
    query_pre_attn_scalar: float,
    cap: float,
) -> np.ndarray:
    """Scaled dot-product attention: ``softmax(QKᵀ/√d_k + mask) · V``.

    ``mask`` is boolean with ``True`` marking positions to hide (set to −∞ before softmax).
    """
    score = (q @ np.swapaxes(k, -1, -2)) * (query_pre_attn_scalar**-0.5)
    score = softcap(score, cap)
    score = masked_fill(score, mask, -np.inf)

    A = softmax(score)
    return A @ v


def gemma_gqa(
    x: np.ndarray,  # [batch, seq_len, d_model]
    params: AttnParams,
    n_heads: int,
    n_kv_heads: int,
    mask: np.ndarray,
    positions: np.ndarray,
    rope_params: RopeParams,
    query_pre_attn_scalar: float,
    cap: float,
) -> np.ndarray:
    """Grouped-query attention; reduces to MHA when ``n_kv_heads == n_heads``."""
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

    inv_freqs = rope_scaled_freqs(
        origin_shape[-1], rope_params.base, rope_params.scaling
    )
    Q = rope_from_freqs(Q, positions, inv_freqs, rope_params.pair_type)
    K = rope_from_freqs(K, positions, inv_freqs, rope_params.pair_type)

    out = gemma_sdpa(Q, K, V, mask, query_pre_attn_scalar, cap)

    gqa = ungroup_last_axis(out.reshape(origin_shape))

    return affine(gqa, params.Wo, params.bo)


def gemma_decoder_block(
    x: np.ndarray,
    layer: dict,
    cfg: GemmaConfig,
    positions: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    a = gemma_rms_norm(x, layer["input_norm"], eps=cfg.norm_eps)
    attn_params = AttnParams(
        Wq=layer["Wq"],
        Wk=layer["Wk"],
        Wv=layer["Wv"],
        Wo=layer["Wo"],
    )
    attn = gemma_gqa(
        a,
        attn_params,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        mask=mask,
        positions=positions,
        rope_params=RopeParams(
            base=cfg.rope_base,
            pair_type="half",
        ),
        query_pre_attn_scalar=cfg.query_pre_attn_scalar,
        cap=cfg.attn_logit_softcapping,
    )
    attn = gemma_rms_norm(attn, layer["post_attn_norm"], eps=cfg.norm_eps)
    h = add_residual(x, attn)

    f = gemma_rms_norm(h, layer["pre_ffn_norm"], eps=cfg.norm_eps)
    ffn = geglu_ffn(
        f,
        GeGLUParams(
            layer["W_gate"],
            layer["W_up"],
            layer["W_down"],
        ),
    )
    ffn = gemma_rms_norm(ffn, layer["post_ffn_norm"], eps=cfg.norm_eps)

    return add_residual(h, ffn)


def gemma_forward(
    input_ids: np.ndarray,
    params: GemmaParams,
    cfg: GemmaConfig,
    start_pos: int = 0,
) -> np.ndarray:
    """Run Gemma-2 forward pass and return logits ``(B, L, V)``.

    Required wiring details:
    - Start with ``embedding(input_ids, tok_embed) * sqrt(dim)``.
    - Use ``query_pre_attn_scalar ** -0.5`` for score scaling.
    - Even layers use sliding-window causal mask; odd layers use full causal mask.
    - Final projection is tied: ``logits = h @ tok_embed.T``, then final ``softcap``.

    ``start_pos`` offsets RoPE positions for decode continuation.
    """
    h = embedding(input_ids, params.tok_embed) * np.sqrt(cfg.dim)
    L = input_ids.shape[-1]
    positions = np.arange(start_pos, start_pos + L)
    full_mask = triangular_mask(L)
    sliding_mask = sliding_window_mask(L, cfg.sliding_window)

    for i, layer in enumerate(params.layers):
        mask = sliding_mask if i % 2 == 0 else full_mask
        h = gemma_decoder_block(h, layer, cfg, positions, mask)

    h = gemma_rms_norm(h, params.final_norm, eps=cfg.norm_eps)
    logits = h @ params.tok_embed.T
    logits = softcap(logits, cfg.final_logit_softcapping)
    return logits
