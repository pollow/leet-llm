"""110 — tiktoken-style Load & Encode — reference solution.

Reuses the learner's own 109 pre-tokenizer through the leet_llm facade.
"""

from __future__ import annotations

from leet_llm import regex_split


def _merge(piece: bytes, ranks: dict[bytes, int]) -> list[bytes]:
    parts = [bytes([b]) for b in piece]
    while len(parts) > 1:
        best_rank: int | None = None
        best_i = -1
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and (best_rank is None or r < best_rank):
                best_rank, best_i = r, i
        if best_i == -1:
            break
        parts[best_i : best_i + 2] = [parts[best_i] + parts[best_i + 1]]
    return parts


def tiktoken_encode(text: str, ranks: dict[bytes, int]) -> list[int]:
    ids: list[int] = []
    for chunk in regex_split(text):
        piece = chunk.encode("utf-8")
        if piece in ranks:
            ids.append(ranks[piece])
            continue
        for part in _merge(piece, ranks):
            ids.append(ranks[part])
    return ids


def tiktoken_decode(ids: list[int], ranks: dict[bytes, int]) -> str:
    inv = {r: p for p, r in ranks.items()}
    return b"".join(inv[i] for i in ids).decode("utf-8")
