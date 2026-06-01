"""104 — BPE Training Loop.

Implement ``bpe_train``. See README.md for the full explanation.
Run `uv run grade 104` to check your work.
"""

from __future__ import annotations


def bpe_train(text: str, vocab_size: int) -> tuple[list[str], list[float]]:
    """Learn a BPE vocabulary of up to ``vocab_size`` pieces from ``text``.

    Returns ``(tokens, scores)``: parallel lists where ``tokens[i]`` is the piece
    string with id ``i`` and ``scores[i]`` is its merge priority (``0`` for the base
    characters, then ``-1, -2, …`` in merge order). Ties on pair frequency are broken
    by choosing the lexicographically smallest ``(left, right)`` piece pair.
    """
    raise NotImplementedError("Implement bpe_train — see 104_bpe_train/README.md")
