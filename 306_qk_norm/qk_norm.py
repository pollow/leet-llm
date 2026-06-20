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

from dataclasses import dataclass

import numpy as np


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
        Per-head Q scale, shape ``(head_dim,)``.  Qwen3 HF weight name:
        ``self_attn.q_norm.weight``.
    k_weight:
        Per-head K scale, shape ``(head_dim,)``.  Qwen3 HF weight name:
        ``self_attn.k_norm.weight``.
    eps:
        Small constant for numerical stability (Qwen3 default: ``1e-6``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(q_normed, k_normed)`` with the same shapes as the inputs.
    """
    raise NotImplementedError("Implement qk_norm — see 306_qk_norm/README.md")


# ---------------------------------------------------------------------------
# Qwen3 whole-model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Qwen3Config:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int          # explicit in Qwen3 config; NOT necessarily dim // n_heads
    vocab_size: int
    max_seq_len: int = 4096
    norm_eps: float = 1e-6
    qk_norm_eps: float = 1e-6
    rope_base: float = 10000.0


@dataclass(frozen=True)
class Qwen3Params:
    tok_embed: np.ndarray   # (V, d)
    layers: list            # list of per-layer dicts (see load_qwen3)
    final_norm: np.ndarray  # (d,) RMSNorm weight
    lm_head: np.ndarray     # (V, d)  [tied: same as tok_embed when tie_word_embeddings=True]


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
    raise NotImplementedError("Implement load_qwen3 — see 306_qk_norm/README.md")


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
    raise NotImplementedError("Implement qwen3_forward — see 306_qk_norm/README.md")
