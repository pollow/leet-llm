"""101 — Character Vocabulary — reference solution."""

from __future__ import annotations


def build_char_vocab(text: str) -> tuple[dict[str, int], list[str]]:
    itos = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(itos)}
    return stoi, itos


def char_encode(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[ch] for ch in text]


def char_decode(ids: list[int], itos: list[str]) -> str:
    return "".join(itos[i] for i in ids)
