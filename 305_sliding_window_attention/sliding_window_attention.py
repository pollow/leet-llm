"""305 — Sliding-Window (Banded Causal) Mask + Mistral whole-model forward.

Two tasks in one module (the loader finds exactly one stub .py per folder):

1. ``sliding_window_mask`` — the Mistral delta operator (band causal mask).
2. ``MistralConfig`` / ``MistralParams`` / ``load_mistral`` / ``mistral_forward``
   — the full Mistral decoder-only model, composing L2 primitives.

See README.md. Run ``uv run grade 305`` to check your work.

Hints:
- ``sliding_window_mask``: reuse ``from leet_llm import triangular_mask`` (009).
  The band is ``(i - W, i]``.
- ``mistral_forward``: compose ``embedding`` (201) → per layer [``rms_norm`` (212) →
  q/k/v ``affine`` (003) + ``group_last_axis`` (001) → ``rope_half`` (213) on q/k →
  repeat-kv + ``sdpa`` (205) with ``sliding_window_mask`` → merge + o_proj →
  ``add_residual`` (208) → ``rms_norm`` → ``swiglu_ffn`` (214) → residual] →
  final ``rms_norm`` → ``@ lm_head.T``. Use rotate-half RoPE (``rope_half``), NOT
  ``llama_decoder_block`` (216, which is interleaved-only). No QKV bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sliding_window_mask(seq_len: int, window: int) -> np.ndarray:
    """Return a boolean ``(seq_len, seq_len)`` causal sliding-window mask.

    The mask is ``0.0`` where query ``i`` may attend to key ``j`` (i.e. ``j``
    is within the causal window ``(i − window, i]``) and ``-inf`` elsewhere.

    Parameters
    ----------
    seq_len:
        Sequence length ``L``.
    window:
        Number of past tokens each query can attend to (Mistral
        ``sliding_window`` config field). When ``window >= seq_len`` the mask
        reduces to the standard causal (lower-triangular) mask.

    Returns
    -------
    np.ndarray, shape ``(L, L)``, dtype bool
        Boolean mask where ``True`` means masked and ``False`` means attended.
    """
    raise NotImplementedError("Implement sliding_window_mask — see 305_sliding_window_attention/README.md")


# ---------------------------------------------------------------------------
# Mistral whole-model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MistralConfig:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    sliding_window: int
    max_seq_len: int = 4096
    norm_eps: float = 1e-5
    rope_base: float = 10000.0


@dataclass(frozen=True)
class MistralParams:
    tok_embed: np.ndarray   # (V, d)
    layers: list            # list of per-layer dicts (see load_mistral)
    final_norm: np.ndarray  # (d,) RMSNorm weight
    lm_head: np.ndarray     # (V, d)


def load_mistral(weights: dict, cfg: MistralConfig) -> MistralParams:
    """Map HF-named arrays into MistralParams.

    HF weight names (no un-permute — rotate-half layout as-is):
      model.embed_tokens.weight         (V, d)
      model.norm.weight                 (d,)
      lm_head.weight                    (V, d)  [untied in Mistral; tied when absent → use embed]
      model.layers.{i}.input_layernorm.weight          (d,)
      model.layers.{i}.post_attention_layernorm.weight (d,)
      model.layers.{i}.self_attn.q_proj.weight         (d, d)
      model.layers.{i}.self_attn.k_proj.weight         (n_kv_heads*head_dim, d)
      model.layers.{i}.self_attn.v_proj.weight         (n_kv_heads*head_dim, d)
      model.layers.{i}.self_attn.o_proj.weight         (d, d)
      model.layers.{i}.mlp.gate_proj.weight            (ffn_dim, d)
      model.layers.{i}.mlp.up_proj.weight              (ffn_dim, d)
      model.layers.{i}.mlp.down_proj.weight            (d, ffn_dim)
    """
    raise NotImplementedError("Implement load_mistral — see 305_sliding_window_attention/README.md")


def mistral_forward(
    input_ids: np.ndarray,
    params: MistralParams,
    cfg: MistralConfig,
) -> np.ndarray:
    """Token embed → N Mistral blocks (band-causal) → final RMSNorm → lm_head logits.

    Returns logits of shape ``(B, L, V)``.

    Mistral = rotate-half Llama with the sliding-window band mask. Compose from
    granular L2 primitives (NOT ``llama_decoder_block`` which uses interleaved RoPE):
      ``embedding`` → per layer [``rms_norm`` → q/k/v ``affine`` + head-split →
      ``rope_half`` on q & k → repeat-kv + ``sdpa`` with ``sliding_window_mask`` →
      merge + o_proj → ``add_residual`` → ``rms_norm`` → ``swiglu_ffn`` →
      residual] → final ``rms_norm`` → ``@ lm_head.T``.
    """
    raise NotImplementedError("Implement mistral_forward — see 305_sliding_window_attention/README.md")
