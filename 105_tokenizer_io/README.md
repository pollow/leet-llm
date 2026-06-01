# 105 — Tokenizer I/O (save & load)

**Level 1 · Tokenization & Batching**

## Description

A trained tokenizer is just data: a list of piece strings and their scores. Persist it to
disk and read it back. We use the **same JSON shape as the real `stories15M` artifact** —
`{"tokens": [...], "scores": [...]}` — so the encoder you write next (106) can load *your*
trained tokenizer and the real one with identical code.

This is the seam that makes "load a real tokenizer" free: once `encode` consumes a loaded
`(tokens, scores)`, it doesn't care whether you trained it or downloaded it.

## The Math

No math — a serialization round-trip. `load_tokenizer(save_tokenizer(t, s)) == (t, s)`.
Note JSON numbers come back as floats, which is exactly what scores are.

## Function Signature

```python
def save_tokenizer(tokens: list[str], scores: list[float], path: str) -> None: ...
def load_tokenizer(path: str) -> tuple[list[str], list[float]]: ...
```

## Read More

- The real artifact format: `llama3.np/tokenizer.model.np` is this exact JSON.
- Python [`json`](https://docs.python.org/3/library/json.html)

## How to Test

```bash
uv run grade 105
```
