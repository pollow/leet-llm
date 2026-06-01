# 011 — Interleave & Halves

**Level 0 · NumPy Foundations**

## Description

There are two natural ways to break an even-length last axis into two equal parts and put it
back together:

- **Interleaving** — the two parts occupy *alternating* positions: `a₀, b₀, a₁, b₁, …`
- **Halving** — the two parts are contiguous *blocks*: the front half and the back half.

Both are pure reshape/indexing — no arithmetic. They show up whenever a vector is treated as
a set of pairs (interleave) or as two stacked blocks (halves).

## The Math

For arrays whose **last axis has even length**:

- **`interleave(a, b)`** with `a, b` of shape `(…, m)` → `(…, 2m)`:
  ```
  out[..., 2i]   = a[..., i]
  out[..., 2i+1] = b[..., i]
  ```
- **`deinterleave(x)`** is its inverse: from `(…, 2m)` return `(evens, odds)`, each `(…, m)`,
  i.e. `evens = x[..., 0::2]`, `odds = x[..., 1::2]`.
- **`split_halves(x)`** from `(…, 2m)` → `(x[..., :m], x[..., m:])`.
- **`join_halves(a, b)`** is its inverse: concatenate along the last axis → `(…, 2m)`.

## Function Signature

```python
def interleave(a: np.ndarray, b: np.ndarray) -> np.ndarray: ...      # (...,m),(...,m) -> (...,2m)
def deinterleave(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ... # (...,2m) -> (...,m),(...,m)
def split_halves(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ... # (...,2m) -> (...,m),(...,m)
def join_halves(a: np.ndarray, b: np.ndarray) -> np.ndarray: ...      # (...,m),(...,m) -> (...,2m)
```

## Read More

- NumPy — [`stack`](https://numpy.org/doc/stable/reference/generated/numpy.stack.html),
  [`reshape`](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html),
  [slicing with steps](https://numpy.org/doc/stable/user/basics.indexing.html),
  [`concatenate`](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html)

## How to Test

```bash
uv run grade 011
```
