"""105 — Tokenizer I/O — reference solution."""

from __future__ import annotations

import json


def save_tokenizer(tokens: list[str], scores: list[float], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"tokens": tokens, "scores": scores}, f)


def load_tokenizer(path: str) -> tuple[list[str], list[float]]:
    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)
    return model["tokens"], model["scores"]
