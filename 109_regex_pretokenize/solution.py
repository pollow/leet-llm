"""109 — Regex Pre-tokenizer (tiktoken-style) — reference solution."""

from __future__ import annotations

import regex

# The GPT-2 / tiktoken pre-tokenization pattern.
_PAT = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def regex_split(text: str) -> list[str]:
    return _PAT.findall(text)
