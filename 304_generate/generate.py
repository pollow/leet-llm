"""304 — Sampling + autoregressive generation for the decoder-only Llama.

``sample`` turns logits into a next-token id (greedy / temperature / top-k / top-p);
``generate`` runs the stateless autoregressive loop until eos. See README.md.
Run ``uv run grade 304``.

Reuse: ``from leet_llm import softmax, top_k, sample_categorical, llama_forward``.
"""

from __future__ import annotations

import numpy as np


def sample(logits: np.ndarray, rng: np.random.Generator | None = None, *,
           temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> int:
    """Pick a next-token id from 1-D ``logits`` (V,).
    ``temperature==0`` ⇒ greedy argmax. Otherwise apply temperature, optional top-k
    truncation, optional top-p (nucleus) truncation, renormalize, then sample with ``rng``."""
    raise NotImplementedError("Implement sample — see 304_generate/README.md")


def generate(input_ids: np.ndarray, params, cfg, *, max_new_tokens: int = 256,
             rng: np.random.Generator | None = None, temperature: float = 1.0,
             top_k: int = 0, top_p: float = 1.0, eos_id: int | None = None) -> list[int]:
    """Stateless autoregressive decode: each step recomputes the full prefix via
    ``llama_forward`` (no KV-cache — that is L4), samples the last-position logits,
    appends, and stops at ``eos_id``. Returns the full id list (prompt + generated)."""
    raise NotImplementedError("Implement generate — see 304_generate/README.md")
