# 009 — Masked Fill & Triangular Mask

**Level 0 · NumPy Foundations**

## Description

A *mask* is a boolean array that marks which positions of another array to act on.
Masking lets you selectively overwrite or hide values without loops.

You'll implement:

1. **`masked_fill(x, mask, value)`** — return a copy of `x` with every position where
   `mask` is `True` replaced by `value`. The mask broadcasts against `x`.
2. **`triangular_mask(n)`** — build an `n × n` boolean mask that is `True` strictly
   *above* the diagonal. (A common use is to mark "later" positions that a position is not
   allowed to see.)

## The Math

- **`masked_fill(x, mask, value)`**: with `mask` broadcastable to `x`,
  ```
  out[p] = value   if mask[p] else x[p]
  ```
  The operation returns a new array (it does **not** modify `x`).

- **`triangular_mask(n)`**: a boolean `(n, n)` array `M` with
  ```
  M[i, j] = True  if j > i  else False
  ```
  i.e. the strictly upper triangle.

## Function Signature

```python
def masked_fill(x: np.ndarray, mask: np.ndarray, value: float) -> np.ndarray: ...
def triangular_mask(n: int) -> np.ndarray: ...   # bool, shape (n, n)
```

## Read More

- NumPy — [`where`](https://numpy.org/doc/stable/reference/generated/numpy.where.html),
  [`triu`](https://numpy.org/doc/stable/reference/generated/numpy.triu.html),
  [boolean masking](https://numpy.org/doc/stable/user/basics.indexing.html#boolean-array-indexing)

## How to Test

```bash
uv run grade 009
```
