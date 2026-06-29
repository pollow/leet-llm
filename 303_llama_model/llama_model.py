"""303 — Whole decoder-only Llama (rebuild of llama3.np on stories15M).

Stack the L2 Llama blocks into a full model and emit vocab logits matching llama3.np.
See README.md. Run ``uv run grade 303`` to check your work.

Reuse: ``from leet_llm import llama_decoder_block, rms_norm, triangular_mask, AttnParams,
SwiGLUParams, LlamaBlockParams``. HF weight-layout facts are GIVEN in the README.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LlamaConfig:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    max_seq_len: int = 2048
    norm_eps: float = 1e-6
    rope_base: float = 10000.0


@dataclass(frozen=True)
class LlamaParams:
    tok_embed: np.ndarray            # (V, d)
    layers: list                     # list[LlamaBlockParams] (from leet_llm import LlamaBlockParams)
    final_norm: np.ndarray           # (d,) RMSNorm weight
    lm_head: np.ndarray              # (V, d)


def load_llama(weights: dict, cfg: LlamaConfig) -> LlamaParams:
    """Map a dict of HF-named arrays (see README table) into LlamaParams."""
    raise NotImplementedError("Implement load_llama — see 303_llama_model/README.md")


def llama_forward(input_ids: np.ndarray, params: LlamaParams, cfg: LlamaConfig) -> np.ndarray:
    """Token embed → N Llama blocks (causal) → final RMSNorm → lm_head.
    Returns logits (B, L, V)."""
    raise NotImplementedError("Implement llama_forward — see 303_llama_model/README.md")
