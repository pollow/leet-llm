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

from dataclasses import dataclass

import numpy as np


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
    raise NotImplementedError("Implement moe_ffn — see 308_mixtral_model/README.md")


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
class MixtralParams:
    tok_embed: np.ndarray   # (V, d)
    layers: list            # list of per-layer dicts (see load_mixtral)
    final_norm: np.ndarray  # (d,) RMSNorm weight
    lm_head: np.ndarray     # (V, d)


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
    raise NotImplementedError("Implement load_mixtral — see 308_mixtral_model/README.md")


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
    raise NotImplementedError("Implement mixtral_forward — see 308_mixtral_model/README.md")
