# 008 — Gather Rows & One-Hot

**Level 0 · NumPy Foundations**

## Description

Two indexing fundamentals:

1. **`gather_rows(table, idx)`** — pick out whole rows of a 2-D table by their integer
   index. Given a `(N, F)` table and an array of indices, return the selected rows,
   preserving the index array's shape.
2. **`one_hot(idx, n)`** — convert integer labels into one-hot vectors: index `i` becomes
   a length-`n` vector that is `1` at position `i` and `0` elsewhere.

These are pure indexing exercises — no loops needed.

## The Math

- **`gather_rows(table, idx)`**: `table` is `(N, F)`, `idx` is an integer array of any
  shape `S` with values in `[0, N)`. The result has shape `S + (F,)`:
  ```
  out[s, :] = table[idx[s], :]
  ```

- **`one_hot(idx, n)`**: `idx` is an integer array of shape `S` with values in `[0, n)`.
  The result has shape `S + (n,)` and dtype float:
  ```
  out[s, c] = 1.0 if c == idx[s] else 0.0
  ```

## Function Signature

```python
def gather_rows(table: np.ndarray, idx: np.ndarray) -> np.ndarray: ...
def one_hot(idx: np.ndarray, n: int) -> np.ndarray: ...
```

## Read More

- NumPy — [integer array indexing](https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing)
- NumPy — [`eye`](https://numpy.org/doc/stable/reference/generated/numpy.eye.html),
  [`take`](https://numpy.org/doc/stable/reference/generated/numpy.take.html)

## How to Test

```bash
uv run grade 008
```
