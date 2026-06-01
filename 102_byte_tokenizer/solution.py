"""102 — Byte Tokenizer — reference solution."""

from __future__ import annotations


def text_to_byte_ids(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def byte_ids_to_text(ids: list[int]) -> str:
    return bytes(ids).decode("utf-8")
