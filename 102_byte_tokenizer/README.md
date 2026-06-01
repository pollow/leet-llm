# 102 — Byte Tokenizer

**Level 1 · Tokenization & Batching**

## Description

A character vocabulary breaks on any character it has never seen. The fix every modern
tokenizer uses underneath: fall back to **bytes**. There are only 256 possible bytes, so
a 256-entry vocabulary can represent *any* text — no unknowns, ever. Encode text to its
UTF-8 byte values and back.

This is why real tokenizer vocabularies reserve their first 256 slots for raw bytes (you
saw them as ids 3–258 in the Llama tokenizer): they are the universal fallback.

## The Math

UTF-8 encodes each character as 1–4 bytes; each byte is an integer in `0..255`. ASCII
characters are one byte; accented Latin letters are two; CJK characters three; many emoji
four. Encoding is `text → utf-8 bytes → list[int]`; decoding is the exact inverse, so
`byte_ids_to_text(text_to_byte_ids(text)) == text` for **all** text.

## Function Signature

```python
def text_to_byte_ids(text: str) -> list[int]: ...   # each in 0..255
def byte_ids_to_text(ids: list[int]) -> str: ...
```

## Read More

- UTF-8: https://en.wikipedia.org/wiki/UTF-8
- Python [`str.encode`](https://docs.python.org/3/library/stdtypes.html#str.encode) /
  [`bytes.decode`](https://docs.python.org/3/library/stdtypes.html#bytes.decode)

## How to Test

```bash
uv run grade 102
```
