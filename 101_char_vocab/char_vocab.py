"""101 — Character Vocabulary.

Implement the three functions below. See README.md for the full explanation.
Run `uv run grade 101` to check your work.
"""

from __future__ import annotations


def build_char_vocab(text: str) -> tuple[dict[str, int], list[str]]:
    """Build a character vocabulary from ``text``.

    Returns ``(stoi, itos)`` where ``itos`` is the **sorted** list of unique
    characters and ``stoi`` maps each character to its index in ``itos``.
    """
    raise NotImplementedError("Implement build_char_vocab — see 101_char_vocab/README.md")


def char_encode(text: str, stoi: dict[str, int]) -> list[int]:
    """Encode ``text`` to a list of ids via per-character lookup in ``stoi``."""
    raise NotImplementedError("Implement char_encode — see 101_char_vocab/README.md")


def char_decode(ids: list[int], itos: list[str]) -> str:
    """Decode ``ids`` back to a string via ``itos``."""
    raise NotImplementedError("Implement char_decode — see 101_char_vocab/README.md")
