"""110 — tiktoken-style Load & Encode.

Implement the two functions below. See README.md for the full explanation.
Run `uv run grade 110` to check your work.

Hint: reuse your pre-tokenizer — ``from leet_llm import regex_split`` — to split into
chunks before applying the byte-pair merges within each chunk.
"""

from __future__ import annotations


def tiktoken_encode(text: str, ranks: dict[bytes, int]) -> list[int]:
    """Encode ``text`` to ids using a tiktoken-style ``ranks`` table (rank-greedy merge)."""
    raise NotImplementedError(
        "Implement tiktoken_encode — see 110_tiktoken_load_encode/README.md"
    )


def tiktoken_decode(ids: list[int], ranks: dict[bytes, int]) -> str:
    """Decode ids back to text by inverting ``ranks`` and UTF-8 decoding the bytes."""
    raise NotImplementedError(
        "Implement tiktoken_decode — see 110_tiktoken_load_encode/README.md"
    )
