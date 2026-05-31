# 002 — Broadcasting

**Level 0 · NumPy Foundations**

## Description

*Broadcasting* is how NumPy applies an operation between arrays of different shapes
without writing loops or copying data: it stretches a smaller array across a larger one
along axes of size 1 or missing axes. Mastering it lets you write fast, readable
array code.

Two small exercises:

1. **`add_bias`** — add a length-`F` vector to every position of an array whose last axis
   has size `F`, no matter how many leading axes there are.
2. **`standardize`** — rescale each vector along the last axis to have zero mean and unit
   variance.

Use broadcasting and reductions only — **no Python `for`-loops** over the data.

## The Math

- **`add_bias(x, b)`** with `x` of shape `(..., F)` and `b` of shape `(F,)`:
  `out[..., j] = x[..., j] + b[j]`. The `(F,)` vector broadcasts across all leading axes.

- **`standardize(x, eps)`** over the **last axis** only. For each vector `v = x[..., :]`:

  ```
  mean = v.mean()
  var  = v.var()                       # population variance (ddof = 0)
  out  = (v - mean) / sqrt(var + eps)
  ```

  `eps` is a small constant for numerical stability (so we never divide by ~0). Compute
  `mean`/`var` with `keepdims=True` so they broadcast back against `x`.

## Function Signature

```python
def add_bias(x: np.ndarray, b: np.ndarray) -> np.ndarray: ...
def standardize(x: np.ndarray, eps: float = 1e-5) -> np.ndarray: ...
```

Neither function should modify its inputs.

## Read More

- NumPy — [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- NumPy — [`mean`](https://numpy.org/doc/stable/reference/generated/numpy.mean.html),
  [`var`](https://numpy.org/doc/stable/reference/generated/numpy.var.html),
  [`keepdims`](https://numpy.org/doc/stable/reference/generated/numpy.sum.html) argument

## How to Test

```bash
uv run grade 002
```
