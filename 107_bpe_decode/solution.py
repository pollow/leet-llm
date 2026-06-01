"""107 — BPE Decode — reference solution."""

from __future__ import annotations


def bpe_decode(ids: list[int], tokens: list[str]) -> str:
    return "".join(tokens[i] for i in ids)
