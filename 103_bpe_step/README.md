# 103 — BPE Primitives: Count Pairs & Apply One Merge

**Level 1 · Tokenization & Batching**

## Description

Byte-Pair Encoding (BPE) is built from one idea repeated many times: find the most
common adjacent pair of symbols and fuse it into a single new symbol. This task builds
the two primitives that the training loop (104) drives:

1. `count_pairs` — tally every adjacent pair in a sequence.
2. `apply_merge` — replace every occurrence of one chosen pair with a new id.

A "sequence" here is just a list of integer symbol ids (characters at the start, then
progressively merged pieces).

## The Math

For a sequence `s`, the adjacent pairs are `(s[0],s[1]), (s[1],s[2]), …`. `count_pairs`
returns a map `pair -> count`. `apply_merge(s, (a,b), new)` scans left to right and
replaces each **non-overlapping** `a,b` with `new`: e.g. `apply_merge([0,0,0],(0,0),7)`
= `[7,0]` (the first pair is consumed, leaving a lone `0`).

## Function Signature

```python
def count_pairs(seq: list[int]) -> dict[tuple[int, int], int]: ...
def apply_merge(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]: ...
```

## Read More

- Sennrich et al. 2016, *Neural Machine Translation of Rare Words with Subword Units*
  (the BPE-for-NLP paper): https://arxiv.org/abs/1508.07909
- Wikipedia — Byte pair encoding: https://en.wikipedia.org/wiki/Byte_pair_encoding

## How to Test

```bash
uv run grade 103
```
