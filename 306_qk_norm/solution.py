"""306 — Per-Head Q/K RMSNorm (Qwen3 / OLMo-2 delta).

Implement ``qk_norm``. See README.md for the full explanation.
Run `uv run grade 306` to check your work.

Hint: reuse ``from leet_llm import rms_norm`` (212), applied per ``head_dim`` to the
Q and K head vectors before RoPE. The classic Llama block skips this; Qwen3 adds
learned ``q_norm``/``k_norm`` weights.
"""

from __future__ import annotations

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
