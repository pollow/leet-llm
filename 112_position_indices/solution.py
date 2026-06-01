"""112 — Position Indices — reference solution."""

from __future__ import annotations

import numpy as np


def position_ids(seqs: list[list[int]], max_len: int | None = None) -> np.ndarray:
    length = max(len(s) for s in seqs) if max_len is None else max_len
    out = np.zeros((len(seqs), length), dtype=np.int64)
    for r, s in enumerate(seqs):
        n = min(len(s), length)
        out[r, :n] = np.arange(n)
    return out
