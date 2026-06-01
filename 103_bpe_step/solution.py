"""103 — BPE Primitives: count pairs & apply one merge — reference solution."""

from __future__ import annotations


def count_pairs(seq: list[int]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for a, b in zip(seq, seq[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def apply_merge(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    n = len(seq)
    while i < n:
        if i < n - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out
