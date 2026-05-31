# 007 — Top-k & Argmax

**Level 0 · NumPy Foundations**

## Description

Selecting the largest values of an array — the single largest (`argmax`) or the `k`
largest (`top_k`) — is a fundamental reduction. The trick is doing it efficiently and
returning results in a predictable order.

You'll implement:

1. **`argmax(x, axis)`** — the index of the maximum value along an axis.
2. **`top_k(x, k)`** — the `k` largest values **and their indices** along the last axis,
   sorted from largest to smallest.

## The Math

- **`argmax(x, axis)`** → integer indices, with the chosen `axis` removed. (This is exactly
  what `np.argmax` does; implement it with NumPy.)

- **`top_k(x, k)`** along the **last axis** returns a pair `(values, indices)`, each of
  shape `(..., k)`:
  - `values[..., 0] ≥ values[..., 1] ≥ ... ≥ values[..., k-1]` (descending),
  - `values[..., j] == x[..., indices[..., j]]`.

  A fast approach: use [`np.argpartition`](https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html)
  to grab the top `k` indices in any order, then sort just those `k` by value (descending).
  Tests use inputs with no ties, so the ordering is unambiguous.

## Function Signature

```python
def argmax(x: np.ndarray, axis: int = -1) -> np.ndarray: ...
def top_k(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...   # (values, indices)
```

## Read More

- NumPy — [`argmax`](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html),
  [`argpartition`](https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html),
  [`argsort`](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html),
  [`take_along_axis`](https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html)

## How to Test

```bash
uv run grade 007
```
