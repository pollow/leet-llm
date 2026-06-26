"""307 — Long-context RoPE scaling: the Llama-3.1 whole-model forward.

Llama-3.1 is 303's Llama with one localized delta: the RoPE inverse
frequencies are **rescaled** by a ``rope_scaling`` schedule so the same
pretrained weights generalise to far longer contexts.  Everything else —
GQA, SwiGLU, RMSNorm, rotate-half RoPE — is unchanged from the baseline.

This task REUSES RoPE helpers from 213_rope:
``rope_scaled_freqs`` / ``rope_from_freqs``.
It also consumes the GQA RoPE hook surface from 215 (``RopeParams`` + ``positions`` wiring).

Refactor 303_llama_model to support new RoPE strategy:
1. ``llama31_forward`` the Llama-3.1 decoder-only model wired through 303's forward.

See README.md. Run ``uv run grade 307`` to check your work.

Hints:
- ``llama31_forward``: call ``llama_forward`` (303) directly first so you can observe
  the default behavior; then refactor the reused path to inject the 307 RoPE schedule.
- TODO(213): ensure long-context RoPE helpers are exposed via ``leet_llm``.
- TODO(215): ensure GQA accepts ``positions``/``rope_params`` without breaking 215's
  original tests and behavior when RoPE hook args are omitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from leet_llm import (
    LlamaConfig,
    LlamaParams,
    llama_forward,
)


# ---------------------------------------------------------------------------
# Llama-3.1 whole-model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Llama31Config:
    """Configuration for a Llama-3.1 decoder-only model.

    Attributes
    ----------
    dim:
        Hidden size (``hidden_size``).
    n_layers:
        Number of decoder layers.
    n_heads:
        Number of query attention heads.
    n_kv_heads:
        Number of key/value heads (GQA).
    vocab_size:
        Vocabulary size.
    max_seq_len:
        Maximum sequence length (position indices).
    norm_eps:
        RMSNorm epsilon.
    rope_base:
        RoPE base frequency (``rope_theta``).
    rope_scaling:
        The ``rope_scaling`` schedule passed to ``rope_scaled_freqs`` (``None``
        → default RoPE).  Llama-3.1 uses ``rope_type="llama3"``.
    """

    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    max_seq_len: int = 131072
    norm_eps: float = 1e-5
    rope_base: float = 500000.0
    rope_scaling: dict | None = None


Llama31Params = LlamaParams


def load_llama31(weights: dict, cfg: Llama31Config) -> Llama31Params:
    """Reuse 303's Llama weight mapping; 307 changes only RoPE frequencies."""
    raise NotImplementedError(
        "Implement load_llama31 — see 307_llama31_model/README.md"
    )


def llama31_forward(
    input_ids: np.ndarray,
    params: Llama31Params,
    cfg: Llama31Config,
    start_pos: int = 0,
) -> np.ndarray:
    """Call 303's forward directly as the baseline.

    This intentionally shows the default 303 RoPE path first. The 307 task then asks for
    refactoring the reused block/forward path so scaled rotate-half RoPE can be injected
    through the existing GQA RoPE hook surface.
    Returns logits of shape ``(B, L, V)``.
    """
    cfg303 = LlamaConfig(
        dim=cfg.dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        vocab_size=cfg.vocab_size,
        max_seq_len=cfg.max_seq_len,
        norm_eps=cfg.norm_eps,
        rope_base=cfg.rope_base,
    )
    # TODO(307): inject scaled rotate-half RoPE for llama3 parity.
    return llama_forward(
        input_ids,
        params,
        cfg303,
        start_pos=start_pos,
    )
