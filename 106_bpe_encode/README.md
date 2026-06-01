# 106 — BPE Encode (score-greedy)

**Level 1 · Tokenization & Batching**

## Description

The keystone: turn text into token ids using a trained tokenizer. This is the **exact
algorithm** the real `stories15M` tokenizer uses, so the encoder you write here drives the
L3 model later. It consumes a *loaded* `(tokens, scores)` — your own from 104, or the real
artifact — with identical code.

## The Math

1. Map each character of `text` to its id. Characters not in the vocabulary are **dropped**
   (the base tokenizer reserves bytes for these; our toy char vocab simply skips them).
2. Repeatedly, scan all adjacent id pairs. For each, look up the concatenated piece string
   in the vocabulary; among those that exist, pick the one with the **highest score**
   (ties keep the **earliest** position — strict `>`). Merge that single position into the
   one new id. Stop when no adjacent pair forms a known piece.

Because score = merge order, this greedily rebuilds the pieces training discovered, highest
priority first.

## Function Signature

```python
def bpe_encode(text: str, tokens: list[str], scores: list[float]) -> list[int]: ...
```

(No BOS/EOS here — special tokens are 108.)

## Read More

- Reference implementation: `llama3.np/tokenizer.py` (`encode`) — the oracle this matches.
- Karpathy, *Let's build the GPT Tokenizer*: https://www.youtube.com/watch?v=zduSFxRajkE

## How to Test

```bash
uv run grade 106
```
