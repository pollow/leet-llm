# 215 — Grouped-Query Attention (GQA)

**Level 2 · Operators & Layers**

## Description

The Llama upgrade to multi-head attention (206). In MHA every query head has its own key/value
head; at inference that's a lot of K/V to store and read. **GQA** shrinks that cost by giving
several query heads a *shared* K/V head — `n_kv_heads < n_heads`, with each K/V head repeated
to cover its group. It interpolates between full MHA (`n_kv_heads = n_heads`) and
multi-query attention (`n_kv_heads = 1`), and is what Llama-3 and most large models use.

This task also defines the reusable RoPE hook surface (`RopeParams`, optional `positions` /
`rope_params` inputs on `gqa`) used by later long-context whole-model tasks.

## The Math

With `n_heads` query heads, `n_kv_heads` key/value heads, group size `g = n_heads / n_kv_heads`,
head dim `d_k`:

```
Q = x W_qᵀ  -> n_heads heads             # W_q: (d, d)
K = x W_kᵀ, V = x W_vᵀ -> n_kv_heads heads  # W_k, W_v: (n_kv_heads·d_k, d)
repeat_kv: each K/V head is repeated g times to align with the query heads
out = Concat( SDPA(Q_i, K_{i//g}, V_{i//g}, mask) )_i  W_oᵀ
```

When `n_kv_heads == n_heads` this is exactly MHA (a useful invariant to test).

## Function Signature

```python
def gqa(x: np.ndarray, params: AttnParams, n_heads: int, n_kv_heads: int,
        mask: np.ndarray | None = None,
        positions: np.ndarray | None = None,
        rope_params: RopeParams | None = None) -> np.ndarray: ...
#   reuses AttnParams (206); here Wk, Wv project to n_kv_heads·d_k   ->   (..., L, d)
```

`AttnParams` carries optional `bq/bk/bv/bo` biases (default `None` = bias-free, as Llama uses); the classic Transformer passes real biases.

## Read More

- *GQA: Training Generalized Multi-Query Transformer Models*, Ainslie et al. 2023: https://arxiv.org/abs/2305.13245
- Reuse `from leet_llm import sdpa, group_last_axis, affine, AttnParams` (205, 001, 003, 206).

## How to Test

```bash
uv run grade 215
```
