# 107 — BPE Decode

**Level 1 · Tokenization & Batching**

## Description

The inverse of 106: turn token ids back into text. For a BPE tokenizer this is delightfully
simple — each id maps to a piece string, and you concatenate them. The invariant that
matters: `decode(encode(text)) == text` for any text built from the vocabulary.

## The Math

`bpe_decode(ids, tokens) = "".join(tokens[i] for i in ids)`. Because the pieces already
carry their spacing (e.g. `" ab"` includes the leading space), no separator or post-
processing is needed — the concatenation reproduces the original text exactly.

## Function Signature

```python
def bpe_decode(ids: list[int], tokens: list[str]) -> str: ...
```

## Read More

- Reference: `llama3.np/tokenizer.py` (`decode`).

## How to Test

```bash
uv run grade 107
```
