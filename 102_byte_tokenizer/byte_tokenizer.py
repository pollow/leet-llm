"""102 — Byte Tokenizer.

Implement the two functions below. See README.md for the full explanation.
Run `uv run grade 102` to check your work.
"""

from __future__ import annotations


def text_to_byte_ids(text: str) -> list[int]:
    """Encode ``text`` as a list of UTF-8 byte values (each in ``0..255``)."""
    raise NotImplementedError(
        "Implement text_to_byte_ids — see 102_byte_tokenizer/README.md"
    )


def byte_ids_to_text(ids: list[int]) -> str:
    """Decode a list of UTF-8 byte values back to a string."""
    raise NotImplementedError(
        "Implement byte_ids_to_text — see 102_byte_tokenizer/README.md"
    )
