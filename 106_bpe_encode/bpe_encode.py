"""106 — BPE Encode (score-greedy).

Implement ``bpe_encode``. See README.md for the full explanation.
Run `uv run grade 106` to check your work.
"""

from __future__ import annotations


def bpe_encode(text: str, tokens: list[str], scores: list[float]) -> list[int]:
    """Encode ``text`` to token ids using the loaded ``(tokens, scores)`` tokenizer.

    Greedily merge the adjacent pair whose concatenation has the highest score, until no
    adjacent pair forms a known piece. Characters absent from the vocabulary are dropped.
    """
    raise NotImplementedError("Implement bpe_encode — see 106_bpe_encode/README.md")
