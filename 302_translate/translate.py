"""302 — Greedy translation with the encoder-decoder Transformer.

Encode the source once, then autoregressively greedy-decode target tokens until EOS.
Stateless recompute (no KV-cache — that is L4). See README.md.
Run ``uv run grade 302`` to check your work.

Reuse: ``from leet_llm import transformer_logits, TransformerConfig, load_marian``.
"""

from __future__ import annotations

import numpy as np


def translate(src_ids: np.ndarray, params, cfg, max_new_tokens: int = 64) -> list[int]:
    """Greedy-decode a single source sequence (shape (1, S)) → list[int] of target ids
    (including the leading decoder_start_id and the trailing eos_id if produced)."""
    raise NotImplementedError("Implement translate — see 302_translate/README.md")
