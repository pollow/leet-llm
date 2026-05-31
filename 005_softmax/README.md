# 005 — Softmax (numerically stable)

**Level 0 · NumPy Foundations**

## Description

`softmax` turns a vector of real-valued scores into a probability distribution: every
output is positive and the outputs sum to 1, with larger scores getting exponentially
more weight. It appears all over numerical computing.

The naïve formula `exp(x) / sum(exp(x))` **overflows** for large inputs (`exp(1000)` is
`inf`). The fix is a one-line trick: subtract the max before exponentiating. Your job is
the stable version, along an arbitrary axis.

## The Math

For a vector `x` along the chosen `axis`:

```
m   = max(x)                 # along the axis, keepdims
e   = exp(x - m)             # shift first → largest exponent is exp(0) = 1
out = e / sum(e)             # sum along the axis, keepdims
```

Subtracting `m` doesn't change the result mathematically (`exp(x-m)/Σexp(x-m)` equals
`exp(x)/Σexp(x)`), but it keeps every exponent in a safe range. The output has the **same
shape** as the input.

## Function Signature

```python
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray: ...
```

## Read More

- The softmax function: https://en.wikipedia.org/wiki/Softmax_function
- Why subtract the max (numerical stability): https://cs231n.github.io/linear-classify/#softmax
- NumPy — [`max`](https://numpy.org/doc/stable/reference/generated/numpy.amax.html) /
  [`sum`](https://numpy.org/doc/stable/reference/generated/numpy.sum.html) with `axis` and `keepdims`

## How to Test

```bash
uv run grade 005
```
