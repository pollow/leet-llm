# 005 — Softmax (numerically stable)

**Level 0 · NumPy Foundations**

## Description

`softmax` turns a vector of real-valued scores into a probability distribution: every
output is positive and the outputs sum to 1, with larger scores getting exponentially
more weight. It appears all over numerical computing.

The naïve formula: `exp(x) / sum(exp(x))`. Mind the **overflow**.

## The Math

For a vector `x` along the chosen `axis`:

```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

For numerical stability, use an equivalent shifted form with `m = max(x)` along `axis`.
The output has the **same shape** as the input and sums to 1 along `axis`.

## Function Signature

```python
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray: ...
```

## Read More

- The softmax function: https://en.wikipedia.org/wiki/Softmax_function
- Numerical stability: https://cs231n.github.io/linear-classify/#softmax
- NumPy — [`max`](https://numpy.org/doc/stable/reference/generated/numpy.amax.html) /
  [`sum`](https://numpy.org/doc/stable/reference/generated/numpy.sum.html) with `axis` and `keepdims`

## How to Test

```bash
uv run grade 005
```
