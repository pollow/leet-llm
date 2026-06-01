# 101 — Character Vocabulary

**Level 1 · Tokenization & Batching**

## Description

The simplest tokenizer maps each distinct character to an integer id. Build the
vocabulary from a corpus, then encode text to ids and decode ids back to text.
This is the tokenizer *interface* — `encode`/`decode` over a fixed vocabulary — that
every later tokenizer (BPE included) reuses.

## The Math

Given a corpus string, the vocabulary is the **sorted set** of its unique characters.
`itos[i]` is the character with id `i`; `stoi` is its inverse. Encoding is a
per-character lookup; decoding joins the looked-up characters. For any text drawn from
the vocabulary, `char_decode(char_encode(text)) == text`.

## Function Signature

```python
def build_char_vocab(text: str) -> tuple[dict[str, int], list[str]]: ...   # (stoi, itos)
def char_encode(text: str, stoi: dict[str, int]) -> list[int]: ...
def char_decode(ids: list[int], itos: list[str]) -> str: ...
```

## Read More

- Karpathy, "Let's build GPT" — char tokenizer: https://www.youtube.com/watch?v=kCc8FmEb1nY
- Python [`sorted`](https://docs.python.org/3/library/functions.html#sorted) /
  [`set`](https://docs.python.org/3/library/stdtypes.html#set)

## How to Test

```bash
uv run grade 101
```
