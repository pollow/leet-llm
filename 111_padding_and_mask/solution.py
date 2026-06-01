"""111 — Padding & Mask — reference solution."""

from __future__ import annotations

import numpy as np


def _target_len(seqs: list[list[int]], max_len: int | None) -> int:
    return max(len(s) for s in seqs) if max_len is None else max_len


def pad_batch(
    seqs: list[list[int]], pad_id: int = 0, max_len: int | None = None
) -> np.ndarray:
    length = _target_len(seqs, max_len)
    out = np.full((len(seqs), length), pad_id, dtype=np.int64)
    for r, s in enumerate(seqs):
        clipped = s[:length]
        out[r, : len(clipped)] = clipped
    return out


def padding_mask(seqs: list[list[int]], max_len: int | None = None) -> np.ndarray:
    length = _target_len(seqs, max_len)
    out = np.zeros((len(seqs), length), dtype=np.int64)
    for r, s in enumerate(seqs):
        out[r, : min(len(s), length)] = 1
    return out
