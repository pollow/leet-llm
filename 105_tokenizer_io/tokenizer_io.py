"""105 — Tokenizer I/O.

Implement the two functions below. See README.md for the full explanation.
Run `uv run grade 105` to check your work.
"""

from __future__ import annotations


def save_tokenizer(tokens: list[str], scores: list[float], path: str) -> None:
    """Write ``tokens`` and ``scores`` to ``path`` as JSON ``{"tokens", "scores"}``."""
    raise NotImplementedError(
        "Implement save_tokenizer — see 105_tokenizer_io/README.md"
    )


def load_tokenizer(path: str) -> tuple[list[str], list[float]]:
    """Read a tokenizer JSON written by :func:`save_tokenizer`; return ``(tokens, scores)``."""
    raise NotImplementedError(
        "Implement load_tokenizer — see 105_tokenizer_io/README.md"
    )
