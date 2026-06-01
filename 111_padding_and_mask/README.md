# 111 — Padding & Mask

**Level 1 · Tokenization & Batching**

## Description

Sequences in a batch have different lengths, but a tensor must be rectangular. **Pad** the
short ones to a common length, and build a **mask** that marks which positions are real and
which are filler — so later computation can ignore the padding.

## The Math

Given `B` sequences, let `L` be the longest (or a fixed `max_len`). `pad_batch` returns a
`(B, L)` integer array: each row is its sequence followed by `pad_id` up to `L`, truncated
to `L` if longer. `padding_mask` returns a `(B, L)` array of `1` for real positions and `0`
for padding (derived from the original lengths, so it's unambiguous even if a real id
equals `pad_id`).

## Function Signature

```python
def pad_batch(seqs: list[list[int]], pad_id: int = 0,
              max_len: int | None = None) -> np.ndarray: ...   # (B, L) int
def padding_mask(seqs: list[list[int]], max_len: int | None = None) -> np.ndarray: ...  # (B, L) 0/1
```

## Read More

- NumPy [`full`](https://numpy.org/doc/stable/reference/generated/numpy.full.html) /
  [`zeros`](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html)
- You can reuse `from leet_llm import masked_fill` (009) ideas, though here you build the
  mask directly.

## How to Test

```bash
uv run grade 111
```
