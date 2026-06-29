"""Task 310 stubs for Gemma-2 operators and full forward wiring.

This file defines ``softcap``, ``geglu_ffn``, ``load_gemma``, and
``gemma_forward``. Keep docstrings focused on API contracts; see ``README.md``
for model rationale and detailed delta explanations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def softcap(x: np.ndarray, cap: float) -> np.ndarray:
    """Apply tanh soft-capping: ``cap * tanh(x / cap)``.

    Args:
        x: Input tensor of any shape.
        cap: Positive scalar cap value.
    Returns:
        Tensor with the same shape as ``x``.
    """
    raise NotImplementedError("Implement softcap — see 310_gemma_model/README.md")


@dataclass(frozen=True)
class GeGLUParams:
    """GeGLU FFN weights: gate/up ``(intermediate, d)``, down ``(d, intermediate)``."""

    gate: np.ndarray  # (intermediate_size, d)
    up: np.ndarray    # (intermediate_size, d)
    down: np.ndarray  # (d, intermediate_size)


def geglu_ffn(x: np.ndarray, params: GeGLUParams) -> np.ndarray:
    """Compute Gemma GeGLU FFN output.

    Formula: ``down( gelu_tanh(x @ gate.T) * (x @ up.T) )``.
    Use GELU tanh approximation (not SiLU). Input and output shapes are ``(B, L, d)``.
    """
    raise NotImplementedError("Implement geglu_ffn — see 310_gemma_model/README.md")


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

    tok_embed: np.ndarray   # (V, d)
    layers: list            # list of per-layer dicts
    final_norm: np.ndarray  # (d,)


def load_gemma(weights: dict, cfg: GemmaConfig) -> GemmaParams:
    """Map HF Gemma-2 tensors into ``GemmaParams``.

    Keep tensor shapes consistent with ``cfg`` and preserve tied embeddings:
    ``lm_head`` reuses ``model.embed_tokens.weight``.
    See ``README.md`` for the full key map.
    """
    raise NotImplementedError("Implement load_gemma — see 310_gemma_model/README.md")


def gemma_forward(
    input_ids: np.ndarray,
    params: GemmaParams,
    cfg: GemmaConfig,
) -> np.ndarray:
    """Run Gemma-2 forward pass and return logits ``(B, L, V)``.

    Required wiring details:
    - Start with ``embedding(input_ids, tok_embed) * sqrt(dim)``.
    - Use ``query_pre_attn_scalar ** -0.5`` for score scaling.
    - Even layers use sliding-window causal mask; odd layers use full causal mask.
    - Final projection is tied: ``logits = h @ tok_embed.T``, then final ``softcap``.
    """
    raise NotImplementedError("Implement gemma_forward — see 310_gemma_model/README.md")
