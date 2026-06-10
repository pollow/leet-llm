"""302 — Greedy translation with the encoder-decoder Transformer.

Encode the source once, then autoregressively greedy-decode target tokens until EOS.
Stateless recompute (no KV-cache — that is L4). See README.md.
Run ``uv run grade 302`` to check your work.

Reuse: ``from leet_llm import transformer_logits, TransformerConfig, load_marian``.
"""

from __future__ import annotations

import numpy as np

from leet_llm import transformer_logits, TransformerConfig, load_marian

def translate(src_ids: np.ndarray, params, cfg, max_new_tokens: int = 64) -> list[int]:
    """Greedy-decode a single source sequence (shape (1, S)) → list[int] of target ids
    (including the leading decoder_start_id and the trailing eos_id if produced)."""
    ids = [cfg.decoder_start_id]
    for _ in range(max_new_tokens):
        tgt_ids = np.array([ids], dtype=np.int64)
        logits = transformer_logits(src_ids, tgt_ids, params, cfg) # [B, L, V]
        next_id = int(np.argmax(logits[0, -1]))
        ids.append(next_id)
        if next_id == cfg.eos_id:
            break
    
    return ids
