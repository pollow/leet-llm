# 201 — Embedding

**Level 2 · Operators & Layers**

## Description

A model can't do math on token *ids* — it needs vectors. An **embedding table** is a
learned matrix with one row per vocabulary entry; embedding a sequence of ids just looks up
the corresponding rows. This is the first layer of every transformer.

## The Math

Given a table `E ∈ ℝ^{V×d}` (one `d`-dimensional row per vocab id) and integer ids of any
shape `(…)`, the output has shape `(…, d)` with

```
out[…] = E[ids[…]]
```

i.e. row `ids[…]` of the table. No arithmetic — a pure gather.

## Function Signature

```python
def embedding(ids: np.ndarray, table: np.ndarray) -> np.ndarray: ...
#   ids: (...) int   table: (V, d)   ->   (..., d)
```

## Read More

- *Attention Is All You Need*, Vaswani et al. 2017 — §3.4 (embeddings): https://arxiv.org/abs/1706.03762
- You may reuse `from leet_llm import gather_rows` (008).

## How to Test

```bash
uv run grade 201
```
