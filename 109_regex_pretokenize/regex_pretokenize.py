"""109 — Regex Pre-tokenizer (tiktoken-style).

Implement ``regex_split``. See README.md for the full explanation.
Run `uv run grade 109` to check your work.

Hint: use the third-party ``regex`` module (``import regex``) — Python's built-in ``re``
lacks the Unicode ``\\p{L}`` / ``\\p{N}`` property classes the GPT-2 pattern needs.
"""

from __future__ import annotations


def regex_split(text: str) -> list[str]:
    """Split ``text`` into tiktoken-style pre-token chunks (GPT-2 pattern)."""
    raise NotImplementedError(
        "Implement regex_split — see 109_regex_pretokenize/README.md"
    )
