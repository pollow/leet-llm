"""211 — GPT Block (decoder-only).

Implement ``gpt_block`` (and the ``GPTBlockParams`` container). See README.md.
Run `uv run grade 211` to check your work.

Hint: reuse ``from leet_llm import mha, ffn, layer_norm, add_residual, triangular_mask,
AttnParams, FFNParams``. Pre-norm, masked self-attention, no cross-attention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GPTBlockParams:
    """Weights for one decoder-only (GPT) block: masked self-attn + FFN."""

    attn: AttnParams  # noqa: F821 — from leet_llm import AttnParams (206)
    ffn: FFNParams  # noqa: F821 — from leet_llm import FFNParams (207)
    norm1_gamma: np.ndarray
    norm1_beta: np.ndarray
    norm2_gamma: np.ndarray
    norm2_beta: np.ndarray


def gpt_block(
    x: np.ndarray,
    params: GPTBlockParams,
    n_heads: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """One pre-norm GPT block: h = x + Attn(LN(x)); y = h + FFN(LN(h))."""
    raise NotImplementedError("Implement gpt_block — see 211_gpt_block/README.md")
