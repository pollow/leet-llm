# 004 — Batched Matmul & `einsum`

**Level 0 · NumPy Foundations**

## Description

`np.einsum` ("Einstein summation") is a compact, powerful way to express almost any
product-and-sum over array axes: matrix products, batched matrix products, outer products,
traces, transposes, and more — all from a tiny index string. Learning to read and write
`einsum` makes a lot of later code dramatically clearer.

Implement three operations **using `np.einsum`**:

1. **`batched_matmul(a, b)`** — matrix-multiply the last two axes, batched over the
   leading axes.
2. **`outer_product(u, v)`** — the outer product of two vectors, batched over leading axes.
3. **`batched_trace(a)`** — the trace (sum of the diagonal) of the last two axes.

## The Math

With `einsum`, repeated indices are summed and the output indices are whatever you put
after `->`.

- **`batched_matmul(a, b)`**: `a` is `(..., M, K)`, `b` is `(..., K, N)` →  `(..., M, N)`.
  `out[..., m, n] = Σ_k a[..., m, k] * b[..., k, n]`.

- **`outer_product(u, v)`**: `u` is `(..., M)`, `v` is `(..., N)` → `(..., M, N)`.
  `out[..., m, n] = u[..., m] * v[..., n]`.

- **`batched_trace(a)`**: `a` is `(..., N, N)` → `(...)`.
  `out[...] = Σ_i a[..., i, i]`.

## Function Signature

```python
def batched_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray: ...
def outer_product(u: np.ndarray, v: np.ndarray) -> np.ndarray: ...
def batched_trace(a: np.ndarray) -> np.ndarray: ...
```

## Read More

- NumPy — [`einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
- A visual intro to einsum: https://rockt.github.io/2018/04/30/einsum

## How to Test

```bash
uv run grade 004
```
