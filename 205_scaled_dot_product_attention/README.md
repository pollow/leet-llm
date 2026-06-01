# 205 — Scaled Dot-Product Attention

**Level 2 · Operators & Layers**

## Description

The heart of the transformer. Each **query** compares itself against every **key** via a
dot product, the scores become a probability distribution over the **values**, and the
output is the weighted average of those values. An optional mask hides positions a query is
not allowed to attend to (causal future, or padding).

## The Math

For queries `Q ∈ ℝ^{Lq×d_k}`, keys `K ∈ ℝ^{Lk×d_k}`, values `V ∈ ℝ^{Lk×d_v}`:

```
scores = Q Kᵀ / √d_k                      # (Lq, Lk)
scores[mask] = −∞                          # where mask is True (positions to hide)
A = softmax(scores)   (over the last axis) # (Lq, Lk), each row sums to 1
out = A V                                  # (Lq, d_v)
```

The `1/√d_k` scaling keeps the scores from growing with dimension and saturating the
softmax. `mask` is boolean with `True` marking positions to hide (matching `triangular_mask`
/ `masked_fill` from 009).

## Function Signature

```python
def sdpa(q: np.ndarray, k: np.ndarray, v: np.ndarray,
         mask: np.ndarray | None = None) -> np.ndarray: ...
#   q: (..., Lq, d_k)  k: (..., Lk, d_k)  v: (..., Lk, d_v)
#   mask: (..., Lq, Lk) bool, True ⇒ hide   ->   (..., Lq, d_v)
```

## Read More

- *Attention Is All You Need*, Vaswani et al. 2017 — §3.2.1: https://arxiv.org/abs/1706.03762
- You may reuse `from leet_llm import softmax, masked_fill, triangular_mask` (005, 009).

## How to Test

```bash
uv run grade 205
```
