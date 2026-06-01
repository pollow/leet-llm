# 206 — Multi-Head Attention

**Level 2 · Operators & Layers**

## Description

One attention is myopic — it can only average values one way. **Multi-head attention** runs
`h` attentions in parallel in lower-dimensional subspaces, letting different heads specialize
(syntax, position, coreference…), then concatenates and projects the result. The *same*
function does **self-attention** (queries, keys, values all from one sequence) and
**cross-attention** (queries from one sequence, keys/values from another — pass `x_kv`).

## The Math

With model dim `d`, `h` heads, head dim `d_k = d/h`, learned projections
`W_q, W_k, W_v, W_o ∈ ℝ^{d×d}` (applied as `x Wᵀ`, bias-free):

```
Q = x_q W_qᵀ,   K = x_kv W_kᵀ,   V = x_kv W_vᵀ        # x_kv defaults to x_q
split Q, K, V into h heads of width d_k
head_i = SDPA(Q_i, K_i, V_i, mask)                    # task 205
MHA = Concat(head_1, …, head_h) W_oᵀ                  # (..., Lq, d)
```

Splitting/merging heads is the `(…, L, d) ↔ (…, h, L, d_k)` reshape+transpose from 001.

## Function Signature

```python
@dataclass(frozen=True)
class AttnParams:
    Wq: np.ndarray; Wk: np.ndarray; Wv: np.ndarray; Wo: np.ndarray   # each (d, d)

def mha(x_q: np.ndarray, params: AttnParams, n_heads: int,
        x_kv: np.ndarray | None = None,
        mask: np.ndarray | None = None) -> np.ndarray: ...
#   x_q: (..., Lq, d)   x_kv: (..., Lk, d) or None   ->   (..., Lq, d)
```

## Read More

- *Attention Is All You Need*, Vaswani et al. 2017 — §3.2.2: https://arxiv.org/abs/1706.03762
- Reuse `from leet_llm import sdpa, group_last_axis, affine` (205, 001, 003).

## How to Test

```bash
uv run grade 206
```
