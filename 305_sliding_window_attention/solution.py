"""305 — Sliding-Window (Banded Causal) Mask + Mistral whole-model forward.

Two tasks in one module (the loader finds exactly one stub .py per folder):

1. ``sliding_window_mask`` — the Mistral delta operator (band causal mask).
2. ``MistralConfig`` / ``MistralParams`` / ``load_mistral`` / ``mistral_forward``
   — the full Mistral decoder-only model, composing L2 primitives.

See README.md. Run ``uv run grade 305`` to check your work.

Hints:
- ``sliding_window_mask``: reuse ``from leet_llm import triangular_mask`` (009).
  The band is ``(i - W, i]``.
- ``mistral_forward``: token ``embedding`` (201) → per layer ``llama_decoder_block``
  (216) with ``RopeParams(pair_type="half")`` under a ``sliding_window_mask`` band →
  final ``rms_norm`` (212) → ``@ lm_head.T``. Mistral's block has no block-level delta,
  so it **reuses** ``llama_decoder_block`` directly; the only Mistral-specific piece is
  the sliding-window mask passed in. No QKV bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from leet_llm import AttnParams, LlamaBlockParams, RopeParams, SwiGLUParams, embedding, llama_decoder_block, rms_norm


def sliding_window_mask(seq_len: int, window: int) -> np.ndarray:
    """Return a boolean ``(seq_len, seq_len)`` causal sliding-window mask.

    The mask is ``False`` where query ``i`` may attend to key ``j`` (i.e. ``j``
    is within the causal window ``(i − window, i]``) and ``True`` elsewhere.

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
    # Future positions: j > i
    future_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    # Too-old positions: j <= i - window
    too_old_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool), k=-window)
    return future_mask | too_old_mask


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

    tok_embed = weights["model.embed_tokens.weight"]
    final_norm = weights["model.norm.weight"]
    lm_head = weights["lm_head.weight"]

    layers: list[LlamaBlockParams] = []
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

        W1 = weights[f"{prefix}.mlp.gate_proj.weight"]  # gate
        W3 = weights[f"{prefix}.mlp.up_proj.weight"]  # up
        W2 = weights[f"{prefix}.mlp.down_proj.weight"]  # down
        ffn = SwiGLUParams(W1=W1, W3=W3, W2=W2)

        layers.append(
            LlamaBlockParams(attn=attn, ffn=ffn,
                             attn_norm=attn_norm, ffn_norm=ffn_norm)
        )

    return MistralParams(
        tok_embed=tok_embed, layers=layers, final_norm=final_norm, lm_head=lm_head
    )


def mistral_forward(
    input_ids: np.ndarray,
    params: MistralParams,
    cfg: MistralConfig,
) -> np.ndarray:
    """Token embed → N Mistral blocks (band-causal) → final RMSNorm → lm_head logits.

    Returns logits of shape ``(B, L, V)``.

    Mistral's decoder block is a plain rotate-half Llama block (RMSNorm → GQA with
    rotate-half RoPE → SwiGLU), so it **reuses** ``llama_decoder_block`` (216) directly
    with ``RopeParams(pair_type="half")``; the only Mistral delta is the
    ``sliding_window_mask`` band passed as the attention mask.
    """
    h = embedding(input_ids, params.tok_embed)
    L = input_ids.shape[-1]

    positions = np.arange(0, L)
    mask = sliding_window_mask(L, cfg.sliding_window)

    for blockParam in params.layers:
        h = llama_decoder_block(h, blockParam, cfg.n_heads, cfg.n_kv_heads,
                                positions=positions, mask=mask, eps=cfg.norm_eps,
                                rope_params=RopeParams(base=cfg.rope_base, pair_type="half"))

    h = rms_norm(h, params.final_norm, cfg.norm_eps)
    logits = h @ params.lm_head.T

    return logits
