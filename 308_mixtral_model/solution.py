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

from leet_llm import (
    RopeParams,
    SwiGLUParams,
    add_residual,
    embedding,
    gqa,
    rms_norm,
    softmax,
    swiglu_ffn,
    top_k,
    triangular_mask,
)

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
        Input activations, shape ``(B, L, d)``.
        Internally flattened to token-major ``(T, d)`` for routing, so passing
        pre-flattened ``(T, d)`` also works and is returned in the same shape.
    router_weight:
        Router linear weight, shape ``(num_experts, d)``.
        ``router_logits = x @ router_weight.T``.
    experts:
        ``list[SwiGLUParams]`` of length ``num_experts``.  Each entry holds
        ``(W1, W3, W2)`` for the gate/up/down projections (reuse
        ``swiglu_ffn`` from 214).
    num_active_experts:
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
        for expert_rank in range(num_active_experts):
            out[t] += weights[t, expert_rank] * swiglu_ffn(
                x[t], experts[int(idx[t, expert_rank])]
            )

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
    moe_router: np.ndarray  # [num_experts, d]
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
    attn = gqa(
        a, params.attn, n_heads, n_kv_heads, mask, positions, rope_params
    )  # [B, L, d]
    h = add_residual(x, attn)
    f = rms_norm(h, params.ffn_norm, eps=eps)
    ffn = moe_ffn(f, params.moe_router, params.moe, num_experts_per_tok)
    return add_residual(h, ffn)


def mixtral_forward(
    input_ids: np.ndarray,
    params: MixtralParams,
    cfg: MixtralConfig,
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
    positions = np.arange(0, L)
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
