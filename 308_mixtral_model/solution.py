"""308 — Sparse MoE FFN (moe_ffn) + Mixtral whole-model forward.

Two tasks in one module (the loader finds exactly one stub .py per folder):

1. ``moe_ffn`` — the Mixtral delta operator (sparse top-k mixture-of-experts FFN).
2. ``MixtralConfig`` / ``MixtralParams`` / ``load_mixtral`` / ``mixtral_forward``
   — the full Mixtral decoder-only model, composing L2 primitives.

See README.md. Run ``uv run grade 308`` to check your work.

Hints:
- ``moe_ffn``: reuse ``from leet_llm import softmax`` (005), ``top_k`` (007),
  ``swiglu_ffn`` / ``SwiGLUParams`` (214). The MoE routing is:
  softmax over ALL experts first, then select top-k, then renormalise the
  selected weights to sum to 1, then dispatch each token through its selected
  experts and accumulate the weighted sum.
- ``mixtral_forward``: compose ``embedding`` (201) → per layer [``rms_norm`` (212) →
  q/k/v ``affine`` (003) + ``group_last_axis`` (001) → ``rope_half`` (213) on q/k →
  repeat-kv + ``sdpa`` (205) causal → merge + o_proj → ``add_residual`` (208) →
  ``rms_norm`` → **``moe_ffn``** → residual] → final ``rms_norm`` → ``@ lm_head.T``.
  Use rotate-half RoPE (``rope_half``), NOT ``llama_decoder_block`` (216,
  interleaved-only). No QKV bias.
"""

from __future__ import annotations

from leet_llm import embedding
from leet_llm import ungroup_last_axis
from leet_llm import sdpa
from leet_llm import rope_from_freqs
from leet_llm import rope_scaled_freqs
from leet_llm import group_last_axis
from leet_llm import affine
from leet_llm import top_k
from leet_llm import softmax
from leet_llm import swiglu_ffn
from leet_llm import add_residual
from leet_llm import rms_norm
from leet_llm import triangular_mask
from leet_llm import RopeParams
from leet_llm import SwiGLUParams

from dataclasses import dataclass

import numpy as np

from leet_llm import AttnParams


def moe_ffn(
    x: np.ndarray,
    router_weight: np.ndarray,
    experts: list,
    num_active_experts: int,
) -> np.ndarray:
    """Sparse mixture-of-experts FFN layer (Mixtral convention).

    Applies a top-k gating over ``num_local_experts`` SwiGLU experts, returning
    the renormalised weighted sum of the selected experts' outputs.

    Parameters
    ----------
    x:
        Input activations, shape ``(B, L, d)`` or ``(T, d)`` (tokens × dim).
    router_weight:
        Router linear weight, shape ``(num_experts, d)``.
        ``router_logits = x @ router_weight.T``.
    experts:
        ``list[SwiGLUParams]`` of length ``num_experts``.  Each entry holds
        ``(W1, W3, W2)`` for the gate/up/down projections (reuse
        ``swiglu_ffn`` from 214).
    top_k:
        Number of experts each token routes to.

    Returns
    -------
    np.ndarray
        Same shape as ``x``.
    """
    x_shape = x.shape
    d_model = x_shape[-1]
    x = x.reshape((-1, d_model))  # [T, d]
    router_logits = x @ router_weight.T  # [T, E]
    routing_weights = softmax(router_logits)
    weights, idx = top_k(routing_weights, num_active_experts)  # [T, k]
    weights /= weights.sum(axis=-1, keepdims=True)  # [T, k]

    out = np.zeros_like(x)
    T = x.shape[0]

    for t in range(T):
        for k in range(num_active_experts):
            out[t] += weights[t, k] * swiglu_ffn(x[t], experts[int(idx[t, k])])

    out = out.reshape(x_shape)
    return out


# ---------------------------------------------------------------------------
# Mixtral whole-model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MixtralConfig:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    num_local_experts: int
    num_experts_per_tok: int
    max_seq_len: int = 4096
    norm_eps: float = 1e-5
    rope_base: float = 10000.0


@dataclass(frozen=True)
class MixtralBlockParams:
    """Weights for one Llama decoder block: RoPE-GQA + SwiGLU, two RMSNorms, bias-free."""

    attn: AttnParams
    moe: list[SwiGLUParams]
    moe_router: np.ndarray  # [num_expoerts, d]
    attn_norm: np.ndarray  # RMSNorm weight (d,)
    ffn_norm: np.ndarray  # RMSNorm weight (d,)


@dataclass(frozen=True)
class MixtralParams:
    tok_embed: np.ndarray  # (V, d)
    layers: list[MixtralBlockParams]  # list of per-layer dicts (see load_mixtral)
    final_norm: np.ndarray  # (d,) RMSNorm weight
    lm_head: np.ndarray  # (V, d)


def load_mixtral(weights: dict, cfg: MixtralConfig) -> MixtralParams:
    """Map HF-named arrays into MixtralParams.

    HF weight names (no un-permute — rotate-half layout as-is):
      model.embed_tokens.weight                               (V, d)
      model.norm.weight                                       (d,)
      lm_head.weight                                          (V, d)  [absent → use embed]
      model.layers.{i}.input_layernorm.weight                 (d,)
      model.layers.{i}.post_attention_layernorm.weight        (d,)
      model.layers.{i}.self_attn.q_proj.weight                (d, d)
      model.layers.{i}.self_attn.k_proj.weight                (n_kv_heads*head_dim, d)
      model.layers.{i}.self_attn.v_proj.weight                (n_kv_heads*head_dim, d)
      model.layers.{i}.self_attn.o_proj.weight                (d, d)
      model.layers.{i}.mlp.gate.weight                        (num_experts, d)  router
      model.layers.{i}.mlp.experts.gate_up_proj               (num_experts, 2*ffn_dim, d)
      model.layers.{i}.mlp.experts.down_proj                  (num_experts, d, ffn_dim)
    """
    tok_embed = weights["model.embed_tokens.weight"]
    final_norm = weights["model.norm.weight"]
    lm_head = weights["lm_head.weight"]

    layers: list[MixtralBlockParams] = []
    for i in range(cfg.n_layers):
        prefix = f"model.layers.{i}"
        attn_norm = weights[f"{prefix}.input_layernorm.weight"]
        ffn_norm = weights[f"{prefix}.post_attention_layernorm.weight"]

        Wq = weights[f"{prefix}.self_attn.q_proj.weight"]
        Wk = weights[f"{prefix}.self_attn.k_proj.weight"]
        Wv = weights[f"{prefix}.self_attn.v_proj.weight"]
        Wo = weights[f"{prefix}.self_attn.o_proj.weight"]
        attn = AttnParams(
            Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo, bq=None, bk=None, bv=None, bo=None
        )

        router = weights[f"{prefix}.mlp.gate.weight"]  # gate
        gate_up = weights[f"{prefix}.mlp.experts.gate_up_proj"]  # up
        down = weights[f"{prefix}.mlp.experts.down_proj"]  # down
        ffn_dim = down.shape[-1]
        moe = [
            SwiGLUParams(
                W1=gate_up[i, :ffn_dim, ...], W2=down[i], W3=gate_up[i, ffn_dim:, ...]
            )
            for i in range(router.shape[0])
        ]

        layers.append(
            MixtralBlockParams(
                attn=attn,
                moe=moe,
                moe_router=router,
                attn_norm=attn_norm,
                ffn_norm=ffn_norm,
            )
        )

    return MixtralParams(
        tok_embed=tok_embed, layers=layers, final_norm=final_norm, lm_head=lm_head
    )


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

    inv_freqs = rope_scaled_freqs(q_shape[-1], rope_params.base, rope_params.scaling)

    Q_rope = rope_from_freqs(Q, positions, inv_freqs, rope_params.pair_type)
    K_rope = rope_from_freqs(K, positions, inv_freqs, rope_params.pair_type)

    gqa = sdpa(Q_rope, K_rope, V, mask)
    gqa = gqa.reshape(q_shape)
    gqa = ungroup_last_axis(gqa)

    return affine(gqa, params.Wo, params.bo)


def mixtral_decoder_block(
    x: np.ndarray,  # [B, L, d]
    params: MixtralBlockParams,
    n_heads: int,
    n_kv_heads: int,
    positions: np.ndarray,
    mask: np.ndarray | None = None,
    eps: float = 1e-5,
    rope_params: RopeParams = RopeParams(),
    num_experts_per_tok: int = 1,
):
    if mask is None:
        L = x.shape[-2]
        mask = triangular_mask(L)
    a = rms_norm(x, params.attn_norm, eps=eps)  # [B, L, d]
    attn = _rope_gqa(
        a, params.attn, n_heads, n_kv_heads, positions, mask, rope_params
    )  # [B, L, d]
    h = add_residual(x, attn)
    f = rms_norm(h, params.ffn_norm, eps=eps)
    ffn = moe_ffn(f, params.moe_router, params.moe, num_experts_per_tok)
    return add_residual(h, ffn)


def mixtral_forward(
    input_ids: np.ndarray,
    params: MixtralParams,
    cfg: MixtralConfig,
    start_pos: int = 0,
) -> np.ndarray:
    """Token embed → N Mixtral blocks (causal) → final RMSNorm → lm_head logits.

    Returns logits of shape ``(B, L, V)``.

    Mixtral = rotate-half Llama with the per-layer FFN replaced by ``moe_ffn``.
    Compose from granular L2 primitives (NOT ``llama_decoder_block``):
      ``embedding`` → per layer [``rms_norm`` → q/k/v ``affine`` + head-split →
      ``rope_half`` on q & k → repeat-kv + ``sdpa`` (causal mask) → merge +
      o_proj → ``add_residual`` → ``rms_norm`` → **``moe_ffn``** → residual] →
      final ``rms_norm`` → ``@ lm_head.T``.
    """
    h = embedding(input_ids, params.tok_embed)
    L = input_ids.shape[-1]
    # start_pos: ignore for now — only used by L4 KV-cache decoding
    positions = np.arange(start_pos, start_pos + L)
    mask = triangular_mask(L)

    for blockParam in params.layers:
        h = mixtral_decoder_block(
            h,
            blockParam,
            cfg.n_heads,
            cfg.n_kv_heads,
            positions=positions,
            mask=mask,
            eps=cfg.norm_eps,
            rope_params=RopeParams(base=cfg.rope_base, pair_type="half"),
            num_experts_per_tok=cfg.num_experts_per_tok,
        )

    h = rms_norm(h, params.final_norm, cfg.norm_eps)
    logits = h @ params.lm_head.T

    return logits
