"""306 — Per-Head Q/K RMSNorm (Qwen3 / OLMo-2 delta) + Qwen3 whole-model forward.

Two tasks in one module (the loader finds exactly one stub .py per folder):

1. ``qk_norm`` — the Qwen3 delta operator (per-head RMSNorm on Q and K before RoPE).
2. ``Qwen3Config`` / ``Qwen3Params`` / ``load_qwen3`` / ``qwen3_forward``
   — the full Qwen3 decoder-only model, composing L2 primitives.

See README.md. Run ``uv run grade 306`` to check your work.

Hints:
- ``qk_norm``: reuse ``from leet_llm import rms_norm`` (212), applied per ``head_dim``
  to the Q and K head vectors before RoPE. The classic Llama block skips this; Qwen3
  adds learned ``q_norm``/``k_norm`` weights.
- ``qwen3_forward``: compose ``embedding`` (201) → per layer [``rms_norm`` (212) →
  q/k/v ``affine`` (003) + ``group_last_axis`` (001) (pass ``n_heads``/``n_kv_heads``
  as n_groups — group_last_axis already returns (B, n_groups, L, head_dim)) →
  **``qk_norm``** on q,k → ``rope_half`` (213) on q,k → repeat-kv + ``sdpa`` (205)
  causal → merge + o_proj → ``add_residual`` (208) → ``rms_norm`` → ``swiglu_ffn``
  (214) → residual] → final ``rms_norm`` → ``@ lm_head.T``.  Use rotate-half RoPE
  (``rope_half``), NOT ``llama_decoder_block`` (216, interleaved-only).  No QKV bias.
  Qwen3 has an explicit ``head_dim`` config field — use ``cfg.head_dim``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from random import triangular

import numpy as np

os.environ.setdefault("LEET_LLM_TARGET", "solution")

from leet_llm import (
    AttnParams,
    LlamaBlockParams,
    SwiGLUParams,
    add_residual,
    affine,
    embedding,
    group_last_axis,
    rms_norm,
    rope_half,
    sdpa,
    swiglu_ffn,
    triangular_mask,
    ungroup_last_axis,
)


def qk_norm(
    q: np.ndarray,
    k: np.ndarray,
    q_weight: np.ndarray,
    k_weight: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply per-head RMSNorm to Q and K before RoPE/attention.

    Normalises every head vector over ``head_dim`` (the last axis) using a
    per-head learned scale (``q_weight`` / ``k_weight``).  This is the sole
    architectural delta of Qwen3 / OLMo-2 over the Llama-3 attention block;
    it precedes RoPE so the rotation operates on normalised vectors.

    Parameters
    ----------
    q:
        Query tensor, shape ``(..., n_q_heads, L, head_dim)``.
    k:
        Key tensor, shape ``(..., n_kv_heads, L, head_dim)``.
    q_weight:
        Per-head Q scale, shape ``(head_dim,)``.
    k_weight:
        Per-head K scale, shape ``(head_dim,)``.
    eps:
        Small constant for numerical stability (Qwen3 default: ``1e-6``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(q_normed, k_normed)`` with the same shapes as the inputs.
    """
    return (
        rms_norm(q, q_weight, eps),
        rms_norm(k, k_weight, eps),
    )


# ---------------------------------------------------------------------------
# Qwen3 whole-model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Qwen3Config:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int  # explicit in Qwen3 config; NOT necessarily dim // n_heads
    vocab_size: int
    max_seq_len: int = 4096
    norm_eps: float = 1e-6
    qk_norm_eps: float = 1e-6
    rope_base: float = 10000.0


@dataclass(frozen=True, kw_only=True)
class Qwen3AttnParams(AttnParams):
    """Traditional AttnParams with qk_norm weights."""

    q_norm: np.ndarray
    k_norm: np.ndarray


@dataclass(frozen=True)
class Qwen3BlockParams:
    """Weights for one Llama decoder block: RoPE-GQA + SwiGLU, two RMSNorms, bias-free."""

    attn: Qwen3AttnParams
    ffn: SwiGLUParams
    attn_norm: np.ndarray  # RMSNorm weight (d,)
    ffn_norm: np.ndarray  # RMSNorm weight (d,)


@dataclass(frozen=True)
class Qwen3Params:
    tok_embed: np.ndarray  # (V, d)
    layers: list[Qwen3BlockParams]  # list of per-layer dicts (see load_qwen3)
    final_norm: np.ndarray  # (d,) RMSNorm weight
    # (V, d)  [tied: same as tok_embed when tie_word_embeddings=True]
    lm_head: np.ndarray


def load_qwen3(weights: dict, cfg: Qwen3Config) -> Qwen3Params:
    """Map HF-named arrays into Qwen3Params.

    HF weight names (no un-permute — rotate-half layout as-is):
      model.embed_tokens.weight                   (V, d)
      model.norm.weight                           (d,)
      lm_head.weight                              (V, d)  [absent when tie_word_embeddings=True → use embed]
      model.layers.{i}.input_layernorm.weight              (d,)
      model.layers.{i}.post_attention_layernorm.weight     (d,)
      model.layers.{i}.self_attn.q_proj.weight             (n_heads*head_dim, d)
      model.layers.{i}.self_attn.k_proj.weight             (n_kv_heads*head_dim, d)
      model.layers.{i}.self_attn.v_proj.weight             (n_kv_heads*head_dim, d)
      model.layers.{i}.self_attn.o_proj.weight             (d, n_heads*head_dim)
      model.layers.{i}.self_attn.q_norm.weight             (head_dim,)
      model.layers.{i}.self_attn.k_norm.weight             (head_dim,)
      model.layers.{i}.mlp.gate_proj.weight                (ffn_dim, d)
      model.layers.{i}.mlp.up_proj.weight                  (ffn_dim, d)
      model.layers.{i}.mlp.down_proj.weight                (d, ffn_dim)
    """
    tok_embed = weights["model.embed_tokens.weight"]
    final_norm = weights["model.norm.weight"]
    lm_head = weights["lm_head.weight"]

    layers: list[Qwen3BlockParams] = []
    for i in range(cfg.n_layers):
        prefix = f"model.layers.{i}"
        attn_norm = weights[f"{prefix}.input_layernorm.weight"]
        ffn_norm = weights[f"{prefix}.post_attention_layernorm.weight"]

        Wq = weights[f"{prefix}.self_attn.q_proj.weight"]
        Wk = weights[f"{prefix}.self_attn.k_proj.weight"]
        Wv = weights[f"{prefix}.self_attn.v_proj.weight"]
        Wo = weights[f"{prefix}.self_attn.o_proj.weight"]
        q_norm = weights[f"{prefix}.self_attn.q_norm.weight"]
        k_norm = weights[f"{prefix}.self_attn.k_norm.weight"]
        attn = Qwen3AttnParams(
            Wq=Wq,
            Wk=Wk,
            Wv=Wv,
            Wo=Wo,
            bq=None,
            bk=None,
            bv=None,
            bo=None,
            q_norm=q_norm,
            k_norm=k_norm,
        )

        W1 = weights[f"{prefix}.mlp.gate_proj.weight"]  # gate
        W3 = weights[f"{prefix}.mlp.up_proj.weight"]  # up
        W2 = weights[f"{prefix}.mlp.down_proj.weight"]  # down
        ffn = SwiGLUParams(W1=W1, W3=W3, W2=W2)

        layers.append(
            LlamaBlockParams(attn=attn, ffn=ffn, attn_norm=attn_norm, ffn_norm=ffn_norm)
        )

    return Qwen3Params(
        tok_embed=tok_embed, layers=layers, final_norm=final_norm, lm_head=lm_head
    )


def _rope_gqa_qk_norm(
    x: np.ndarray,
    params: Qwen3AttnParams,
    n_heads: int,
    n_kv_heads: int,
    positions: np.ndarray,
    mask: np.ndarray | None = None,
    eps: float = 1e-6,
    rope_base: float = 10000.0,
):
    """Grouped-query attention with rotate-half rope and QK-Norm"""
    n_g = n_heads // n_kv_heads  # group_size

    Q = affine(x, params.Wq, params.bq)
    K = affine(x, params.Wk, params.bk)
    V = affine(x, params.Wv, params.bv)

    Q = group_last_axis(Q, n_heads)
    K = group_last_axis(K, n_kv_heads)
    V = group_last_axis(V, n_kv_heads)

    q_shape = Q.shape
    grouped_shape = [q_shape[0], n_kv_heads, n_g] + list(q_shape[2:])
    Q = Q.reshape(grouped_shape)
    K = K[:, :, None, ...]
    V = V[:, :, None, ...]

    Q, K = qk_norm(Q, K, params.q_norm, params.k_norm, eps)

    Q_rope = rope_half(Q, positions, base=rope_base)
    K_rope = rope_half(K, positions, base=rope_base)

    gqa = sdpa(Q_rope, K_rope, V, mask)
    gqa = gqa.reshape(q_shape)
    gqa = ungroup_last_axis(gqa)

    return affine(gqa, params.Wo, params.bo)


def qwen3_decoder_block(
    x: np.ndarray,
    params: Qwen3BlockParams,
    n_heads: int,
    n_kv_heads: int,
    positions: np.ndarray,
    mask: np.ndarray | None = None,
    eps: float = 1e-5,
    rope_base: float = 10000.0,
):
    if mask is None:
        L = x.shape[-2]
        mask = triangular_mask(L)

    a = rms_norm(x, params.attn_norm, eps=eps)
    attn = _rope_gqa_qk_norm(
        a, params.attn, n_heads, n_kv_heads, positions, mask, eps, rope_base
    )

    h = add_residual(x, attn)
    f = rms_norm(h, params.ffn_norm, eps=eps)
    ffn = swiglu_ffn(f, params.ffn)

    return add_residual(h, ffn)


def qwen3_forward(
    input_ids: np.ndarray,
    params: Qwen3Params,
    cfg: Qwen3Config,
) -> np.ndarray:
    """Token embed → N Qwen3 blocks (causal) → final RMSNorm → lm_head logits.

    Returns logits of shape ``(B, L, V)``.

    Qwen3 = rotate-half Llama with per-head qk-norm before RoPE, no QKV bias,
    and an explicit ``head_dim`` config field. Compose from granular L2 primitives
    (NOT ``llama_decoder_block`` which uses interleaved RoPE):
      ``embedding`` → per layer [``rms_norm`` → q/k/v ``affine`` + head-split →
      **``qk_norm``** on q & k → ``rope_half`` on q & k → repeat-kv + ``sdpa``
      (causal mask) → merge + o_proj → ``add_residual`` → ``rms_norm`` →
      ``swiglu_ffn`` → residual] → final ``rms_norm`` → ``@ lm_head.T``.
    """
    h = embedding(input_ids, params.tok_embed)
    L = input_ids.shape[-1]
    positions = np.arange(0, L)
    mask = triangular_mask(L)

    for blockParam in params.layers:
        h = qwen3_decoder_block(
            h,
            blockParam,
            cfg.n_heads,
            cfg.n_kv_heads,
            positions=positions,
            mask=mask,
            eps=cfg.norm_eps,
            rope_base=cfg.rope_base,
        )

    h = rms_norm(h, params.final_norm, cfg.norm_eps)
    logits = h @ params.lm_head.T

    return logits


if __name__ == "__main__":
    from utils import run_qwen3_cli

    run_qwen3_cli(
        module_name="306_qk_norm/solution.py",
        load_fn=load_qwen3,
        forward_fn=qwen3_forward,
        config_cls=Qwen3Config,
    )
