"""304 — Sampling + autoregressive generation for the decoder-only Llama.

``sample`` turns logits into a next-token id (greedy / temperature / top-k / top-p);
``generate`` runs the stateless autoregressive loop until eos. See README.md.
Run ``uv run grade 304``.

Reuse: ``from leet_llm import softmax, top_k, sample_categorical, llama_forward``.
"""

from __future__ import annotations

import numpy as np

from leet_llm import softmax, top_k as topk_fn, sample_categorical, llama_forward

def sample(logits: np.ndarray, rng: np.random.Generator | None = None, *,
           temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> int:
    """Pick a next-token id from 1-D ``logits`` (V,).
    ``temperature==0`` ⇒ greedy argmax. Otherwise apply temperature, optional top-k
    truncation, optional top-p (nucleus) truncation, renormalize, then sample with ``rng``."""
    if temperature == 0:
        return np.argmax(logits)

    V = logits.shape[0]
    z = logits / temperature
    if top_k > 0 and top_k < V:
        vals, idx = topk_fn(logits, top_k)
        masked = np.full_like(z, -np.inf)
        masked[idx] = vals
        z = masked
    elif top_p < 1.0:
        p = softmax(z)
        order = np.argsort(p)[::-1]
        sorted_probs = p[order]
        cum = np.cumsum(sorted_probs)
        remove = (cum - sorted_probs) >= top_p
        z[order[remove]] = -np.inf

    p = softmax(z)

    return sample_categorical(p, rng)



def generate(input_ids: np.ndarray, params, cfg, *, max_new_tokens: int = 256,
             rng: np.random.Generator | None = None, temperature: float = 1.0,
             top_k: int = 0, top_p: float = 1.0, eos_id: int | None = None) -> list[int]:
    """Stateless autoregressive decode: each step recomputes the full prefix via
    ``llama_forward`` (no KV-cache — that is L4), samples the last-position logits,
    appends, and stops at ``eos_id``. Returns the full id list (prompt + generated)."""
    ids = input_ids[0].tolist()  # (1, S) -> list[int]; batch size 1 by design
    for _ in range(max_new_tokens):
        logits = llama_forward(np.array([ids]), params, cfg)  # (1, t, V)
        idx = int(sample(logits[0, -1], rng, temperature=temperature,
                         top_k=top_k, top_p=top_p))
        ids.append(idx)
        if idx == eos_id:
            break

    return ids
