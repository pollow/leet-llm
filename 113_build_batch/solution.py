"""113 — Build Batch — reference solution.

Composes the learner's own 111/112 pieces through the leet_llm facade.
"""

from __future__ import annotations

import numpy as np

from leet_llm import pad_batch, padding_mask, position_ids


def build_batch(
    seqs: list[list[int]], pad_id: int = 0, max_len: int | None = None
) -> dict[str, np.ndarray]:
    return {
        "input_ids": pad_batch(seqs, pad_id, max_len),
        "pad_mask": padding_mask(seqs, max_len),
        "position_ids": position_ids(seqs, max_len),
    }
