"""216 — Llama Decoder Block.

Implement ``llama_decoder_block`` (and the ``LlamaBlockParams`` container). See README.md.
Run `uv run grade 216` to check your work.

Hint: reuse ``from leet_llm import rms_norm, rope_interleaved, sdpa, swiglu_ffn,
add_residual, group_last_axis, affine, triangular_mask, AttnParams, SwiGLUParams``. Apply
RoPE (interleaved convention, as L3 uses) to the per-head q/k projections before the
attention scores; pre-norm RMSNorm placement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LlamaBlockParams:
    """Weights for one Llama decoder block: RoPE-GQA + SwiGLU, two RMSNorms, bias-free."""

    attn: AttnParams  # noqa: F821 — from leet_llm import AttnParams (206)
    ffn: SwiGLUParams  # noqa: F821 — from leet_llm import SwiGLUParams (214)
    attn_norm: np.ndarray  # RMSNorm weight (d,)
    ffn_norm: np.ndarray  # RMSNorm weight (d,)


def llama_decoder_block(
    x: np.ndarray,
    params: LlamaBlockParams,
    n_heads: int,
    n_kv_heads: int,
    positions: np.ndarray,
    mask: np.ndarray | None = None,
    eps: float = 1e-5,
) -> np.ndarray:
    """One pre-norm Llama block: RMSNorm(eps) -> RoPE-GQA -> residual -> RMSNorm(eps) -> SwiGLU -> residual."""
    raise NotImplementedError(
        "Implement llama_decoder_block — see 216_llama_decoder_block/README.md"
    )
