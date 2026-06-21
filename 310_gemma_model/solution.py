"""310 — GeGLU + softcap + sandwich norm: Gemma-2 whole-model forward.

Two new operators plus a full Gemma-2 decoder assembly:

1. ``softcap(x, cap)`` — tanh soft-capping: ``cap * tanh(x / cap)``.
   Applied to attention logits (before softmax) and to the final output logits.
2. ``geglu_ffn(x, params)`` — GeGLU feed-forward with GELU(tanh) activation:
   ``down_proj( gelu_tanh(gate_proj(x)) * up_proj(x) )``.
3. ``GemmaConfig`` / ``GemmaParams`` / ``load_gemma`` / ``gemma_forward`` —
   the Gemma-2 decoder-only model composing L2 primitives.

Gemma-2 wrinkles vs the Llama-3 baseline (303):
- **√d embedding scale**: after embedding, multiply hidden by ``sqrt(hidden_size)``.
- **``(1+w)`` RMSNorm**: ``(1 + weight) * rms_normed(x)`` (cast to float32 internally).
- **Sandwich norm**: four RMSNorm calls per decoder layer (input → attn-out → residual;
  pre-FFN → FFN-out → residual).
- **GeGLU FFN** with GELU(tanh), not SiLU.
- **Attention logit soft-cap** (``attn_logit_softcapping``) applied before softmax.
- **Final logit soft-cap** (``final_logit_softcapping``) applied after lm_head.
- **``query_pre_attn_scalar``** for the attention scale (``scalar ** -0.5``), which may
  differ from ``head_dim ** -0.5``.
- **Alternating SWA/full layers**: even-indexed (0, 2, …) use sliding-window causal
  attention (``sliding_window_mask``), odd-indexed use full causal.
- **Tied embeddings**: ``lm_head = embed_tokens`` (no separate ``lm_head.weight``).

See README.md. Run ``uv run grade 310`` to check your work.

Hints:
- ``softcap``: ``cap * np.tanh(x / cap)``. Large |x| saturates at ±cap.
- ``geglu_ffn``: ``(x @ gate.T)`` → gelu_tanh → multiply by ``(x @ up.T)`` → ``@ down.T``.
  GELU(tanh): ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))``.
- ``gemma_rms_norm(x, w, eps)``: cast x to float64 internally, compute RMS, return
  ``(1 + w) * (x / rms)`` cast back to input dtype.
- ``gemma_forward``: ``embedding`` → ``* sqrt(d)`` → per-layer [``gemma_rms_norm`` (input) →
  GQA-attn with softcap → ``gemma_rms_norm`` (post-attn) → residual add; then
  ``gemma_rms_norm`` (pre-FFN) → ``geglu_ffn`` → ``gemma_rms_norm`` (post-FFN) → residual add]
  → final ``gemma_rms_norm`` → lm_head logits → ``softcap(logits, final_logit_softcapping)``.
- Reuse ``from leet_llm import rope_half, softmax, embedding, sliding_window_mask``.
- Attention: q_proj (B,L,H*head_dim) → reshape (B,H,L,head_dim) → rope_half;
  k_proj (B,L,KVH*head_dim) → (B,KVH,L,head_dim) → rope_half; repeat KV to H heads;
  scores = (q @ k.T) * scalar**-0.5; softcap scores; add causal mask; softmax; @ v; o_proj.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def softcap(x: np.ndarray, cap: float) -> np.ndarray:
    """Tanh soft-capping: ``cap * tanh(x / cap)``.

    Compresses large logit magnitudes without hard clipping. As |x| → ∞ the output
    saturates toward ±cap; for |x| ≪ cap the function is approximately linear.

    Parameters
    ----------
    x:
        Input array of any shape (float64 recommended).
    cap:
        Soft-cap magnitude. Output lies in ``(-cap, cap)``.

    Returns
    -------
    np.ndarray
        Same shape as ``x``.

    Math
    ----
    .. math::
        \\text{softcap}(x, c) = c \\cdot \\tanh\\!\\left(\\frac{x}{c}\\right)
    """
    raise NotImplementedError("Implement softcap — see 310_gemma_model/README.md")


@dataclass(frozen=True)
class GeGLUParams:
    """Weight matrices for a GeGLU feed-forward block.

    Attributes
    ----------
    gate:
        Gate projection weight, shape ``(intermediate_size, d)``.
    up:
        Up projection weight, shape ``(intermediate_size, d)``.
    down:
        Down projection weight, shape ``(d, intermediate_size)``.
    """

    gate: np.ndarray  # (intermediate_size, d)
    up: np.ndarray    # (intermediate_size, d)
    down: np.ndarray  # (d, intermediate_size)


def geglu_ffn(x: np.ndarray, params: GeGLUParams) -> np.ndarray:
    """GeGLU feed-forward block with GELU(tanh) activation (Gemma-2 MLP).

    Applies ``down_proj( gelu_tanh(gate_proj(x)) * up_proj(x) )``.
    The GELU variant is the tanh approximation (``gelu_pytorch_tanh`` in HuggingFace),
    NOT SiLU.

    Parameters
    ----------
    x:
        Input activations, shape ``(B, L, d)``.
    params:
        ``GeGLUParams`` holding gate/up/down projection weights.

    Returns
    -------
    np.ndarray
        Output shape ``(B, L, d)``.

    Math
    ----
    .. math::
        \\text{GELU}_{\\text{tanh}}(z) = 0.5 z \\left(1 + \\tanh\\!\\left(
            \\sqrt{\\tfrac{2}{\\pi}} (z + 0.044715 z^3)
        \\right)\\right)

    .. math::
        \\text{GeGLU-FFN}(x) = \\text{down}\\!\\left(
            \\text{GELU}_{\\text{tanh}}(x W_{\\text{gate}}^\\top)
            \\odot (x W_{\\text{up}}^\\top)
        \\right)
    """
    raise NotImplementedError("Implement geglu_ffn — see 310_gemma_model/README.md")


# ---------------------------------------------------------------------------
# Gemma-2 whole-model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GemmaConfig:
    """Configuration for a Gemma-2 decoder-only model.

    Attributes
    ----------
    dim:
        Hidden size (``hidden_size`` in HF config).
    n_layers:
        Number of decoder layers.
    n_heads:
        Number of query attention heads.
    n_kv_heads:
        Number of key/value heads (GQA; ``n_kv_heads <= n_heads``).
    head_dim:
        Per-head dimension for Q, K, V (Gemma-2 sets this explicitly).
    vocab_size:
        Vocabulary size.
    intermediate_size:
        FFN intermediate width.
    norm_eps:
        RMSNorm epsilon.
    rope_base:
        RoPE base frequency.
    query_pre_attn_scalar:
        Scalar used to compute the attention scale as ``scalar ** -0.5``.
        May differ from ``head_dim``.
    final_logit_softcapping:
        Cap applied to the final lm_head logits.
    attn_logit_softcapping:
        Cap applied to attention scores before softmax.
    sliding_window:
        Sliding-window size for sliding-window-attention layers.
    max_seq_len:
        Maximum sequence length (used for position indices).
    """

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
    """Packed weights for a Gemma-2 model.

    Attributes
    ----------
    tok_embed:
        Token embedding table, shape ``(V, d)``.  Also used as ``lm_head``
        (Gemma-2 ties embeddings).
    layers:
        List of per-layer dicts (see ``load_gemma`` for key names).
    final_norm:
        Final RMSNorm weight, shape ``(d,)``.
    """

    tok_embed: np.ndarray   # (V, d)
    layers: list            # list of per-layer dicts
    final_norm: np.ndarray  # (d,)


def load_gemma(weights: dict, cfg: GemmaConfig) -> GemmaParams:
    """Map HF-named weight arrays into ``GemmaParams``.

    HF weight names
    ---------------
    Global::

        model.embed_tokens.weight       (V, d)
        model.norm.weight               (d,)
        lm_head.weight                  absent — tied to embed_tokens

    Per layer ``model.layers.{i}``::

        .input_layernorm.weight             (d,)
        .post_attention_layernorm.weight    (d,)
        .pre_feedforward_layernorm.weight   (d,)
        .post_feedforward_layernorm.weight  (d,)
        .self_attn.q_proj.weight            (n_heads * head_dim, d)
        .self_attn.k_proj.weight            (n_kv_heads * head_dim, d)
        .self_attn.v_proj.weight            (n_kv_heads * head_dim, d)
        .self_attn.o_proj.weight            (d, n_heads * head_dim)
        .mlp.gate_proj.weight               (intermediate_size, d)
        .mlp.up_proj.weight                 (intermediate_size, d)
        .mlp.down_proj.weight               (d, intermediate_size)

    Gemma-2 **ties** embeddings: ``lm_head.weight`` is absent in the state dict;
    use ``model.embed_tokens.weight`` as the output projection.
    """
    raise NotImplementedError("Implement load_gemma — see 310_gemma_model/README.md")


def gemma_forward(
    input_ids: np.ndarray,
    params: GemmaParams,
    cfg: GemmaConfig,
    start_pos: int = 0,
) -> np.ndarray:
    """Token embed → N Gemma-2 decoder blocks → final RMSNorm → logits.

    Returns logits of shape ``(B, L, V)``.

    Gemma-2 assembly (each wrinkle matters — see README.md):

    1. ``embedding(input_ids, tok_embed) * sqrt(dim)``
    2. Per decoder layer (index ``i``):

       a. ``gemma_rms_norm(h, input_layernorm)``  [``(1+w)`` norm]
       b. GQA attention with rotate-half RoPE and attention logit soft-cap
       c. ``gemma_rms_norm(attn_out, post_attention_layernorm)``
       d. ``h = h + post_attn_normed``  [first residual]
       e. ``gemma_rms_norm(h, pre_feedforward_layernorm)``
       f. ``geglu_ffn(pre_ffn_normed, ...)``
       g. ``gemma_rms_norm(ffn_out, post_feedforward_layernorm)``
       h. ``h = h + post_ffn_normed``  [second residual]

       Even-indexed layers (0, 2, …) use sliding-window causal attention
       (``sliding_window_mask``); odd-indexed layers use full causal.

    3. Final ``gemma_rms_norm(h, final_norm)``
    4. ``logits = h @ tok_embed.T``  (tied lm_head)
    5. ``softcap(logits, final_logit_softcapping)``
    """
    raise NotImplementedError("Implement gemma_forward — see 310_gemma_model/README.md")
