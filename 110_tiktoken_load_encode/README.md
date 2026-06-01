# 110 — tiktoken-style Load & Encode

**Level 1 · Tokenization & Batching**

## Description

Load a real **tiktoken-format** tokenizer and encode/decode with it. The artifact is a
*rank table*: a dict mapping a byte string to an integer that is **both its merge priority
and its token id** (lower rank = merged earlier). This is the format Llama-3.2, Qwen, and
GPT-4 ship. Reusing your pre-tokenizer from 109, this same function will tokenize any of
them — pointing it at a different rank table is all it takes.

> Load-only: we never train this tokenizer; we reproduce its encoding exactly.

## The Math

For each chunk from `regex_split` (109):

1. UTF-8 encode the chunk to bytes; start with one single-byte piece per byte.
2. Repeatedly find the adjacent pair whose concatenation is in the rank table with the
   **lowest rank**; merge it (ties → earliest position). Stop when no adjacent pair is in
   the table.
3. Emit `ranks[piece]` for each final piece.

Decoding inverts the table (`id → bytes`), concatenates, and UTF-8 decodes. Note this is
**rank-greedy** (lowest first), the mirror image of 106's score-greedy (highest first) —
same algorithm, opposite sign convention.

## Function Signature

```python
def tiktoken_encode(text: str, ranks: dict[bytes, int]) -> list[int]: ...
def tiktoken_decode(ids: list[int], ranks: dict[bytes, int]) -> str: ...
```

You can build a real `ranks` table with `tiktoken.get_encoding("gpt2")._mergeable_ranks`.

## Read More

- tiktoken `_byte_pair_merge`: https://github.com/openai/tiktoken/blob/main/src/lib.rs
- Llama-3 uses this format (vocab 128,256): https://github.com/meta-llama/llama-models

## How to Test

```bash
uv run grade 110
```
