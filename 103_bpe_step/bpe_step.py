"""103 — BPE Primitives: count pairs & apply one merge.

Implement the two functions below. See README.md for the full explanation.
Run `uv run grade 103` to check your work.
"""

from __future__ import annotations


def count_pairs(seq: list[int]) -> dict[tuple[int, int], int]:
    """Return a map from each adjacent ``(seq[i], seq[i+1])`` pair to its count."""
    raise NotImplementedError("Implement count_pairs — see 103_bpe_step/README.md")


def apply_merge(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` in ``seq`` with ``new_id``."""
    raise NotImplementedError("Implement apply_merge — see 103_bpe_step/README.md")
