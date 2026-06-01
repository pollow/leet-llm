# 003 — Affine Transform

**Level 0 · NumPy Foundations**

## Description

An *affine transform* is the workhorse of numerical computing: take a vector, multiply it
by a weight matrix, and add an offset. Stacked together, these are the bulk of the
arithmetic in almost every model you'll build later.

You'll implement `affine(x, W, b)` computing `y = x · Wᵀ + b`, where the matrix multiply
applies to the **last axis** and broadcasts over all leading axes. The bias is optional.

No Python `for`-loops — use a single matrix multiply (`@` / `np.matmul`) plus broadcasting.

## The Math

Given:
- `x` of shape `(..., F_in)` — any number of leading axes,
- `W` of shape `(F_out, F_in)` — the weight matrix (note: rows are outputs),
- `b` of shape `(F_out,)` or `None`,

compute `y` of shape `(..., F_out)`:
`y[..., o] = sum_i x[..., i] * W[o, i] + b[o]` (when bias is provided).

Storing `W` as `(F_out, F_in)` and transposing in the multiply is the convention used by
most ML frameworks, so we follow it here.

## Function Signature

```python
def affine(x: np.ndarray, W: np.ndarray, b: np.ndarray | None = None) -> np.ndarray: ...
```

## Read More

- NumPy — [`matmul` / `@`](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)
  and how it broadcasts over leading (batch) axes
- NumPy — [`transpose` / `.T`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.T.html)

## How to Test

```bash
uv run grade 003
```
