# 006 — Log-Sum-Exp & Log-Softmax

**Level 0 · NumPy Foundations**

## Description

Working in *log-space* is the standard way to keep probabilities numerically stable —
multiplying many small numbers underflows to 0, but adding their logs is safe. The key
primitive is **log-sum-exp**: computing `log(Σ exp(x))` without overflowing.

You'll implement:

1. **`logsumexp(x, axis)`** — the stable `log(Σ exp(x))`.
2. **`log_softmax(x, axis)`** — `log(softmax(x))`, built cleanly on top of `logsumexp`.

## The Math

Naïvely, `log(Σ exp(x))` overflows when any `x` is large. Factor out the max `m`:

```
log Σ_i exp(x_i) = m + log Σ_i exp(x_i - m),   where m = max(x)
```

Now the largest exponent is `exp(0) = 1`, so nothing overflows. (`m` is taken along
`axis`.)

**Log-softmax** falls right out of it — no division needed:

```
log_softmax(x) = x - logsumexp(x)
```

This is both more stable and faster than `log(softmax(x))`.

## Function Signature

```python
def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray: ...      # axis is reduced away
def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray: ...    # same shape as x
```

## Read More

- The log-sum-exp trick: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
- NumPy — [`max`](https://numpy.org/doc/stable/reference/generated/numpy.amax.html),
  [`log`](https://numpy.org/doc/stable/reference/generated/numpy.log.html),
  [`exp`](https://numpy.org/doc/stable/reference/generated/numpy.exp.html)

## How to Test

```bash
uv run grade 006
```
