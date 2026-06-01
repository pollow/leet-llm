# 112 — Position Indices

**Level 1 · Tokenization & Batching**

## Description

A transformer has no built-in notion of order — it needs to be *told* each token's position
in the sequence. Generate those position indices for a padded batch: `0, 1, 2, …` along
each real sequence, and `0` in the padding region (padding positions are masked out later,
so their value just needs to be valid).

## The Math

For a batch padded to `(B, L)`, `position_ids[r]` is `[0, 1, …, n_r-1, 0, 0, …]` where
`n_r` is the (possibly truncated) length of row `r`. The result is a `(B, L)` integer array
matching the shapes from 111.

## Function Signature

```python
def position_ids(seqs: list[list[int]], max_len: int | None = None) -> np.ndarray: ...
```

## Read More

- NumPy [`arange`](https://numpy.org/doc/stable/reference/generated/numpy.arange.html)
- Positions feed the positional encodings you'll build in Level 2.

## How to Test

```bash
uv run grade 112
```
