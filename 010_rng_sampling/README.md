# 010 — Sampling from a Categorical Distribution

**Level 0 · NumPy Foundations**

## Description

Given a probability distribution over `K` choices, *sampling* means randomly drawing one
choice so that choice `i` comes up with probability `p_i`. You'll implement a **batched,
reproducible** categorical sampler.

Reproducibility matters: the function takes an explicit `np.random.Generator`, so the same
seed always yields the same draws (essential for testable, debuggable code).

## The Math

`probs` has shape `(..., K)` where each length-`K` vector along the last axis is a valid
distribution (non-negative, sums to 1). For each such vector, draw one index in `[0, K)`.
The result has the **leading shape** `(...)` — the last axis is consumed.

One standard approach is **inverse-CDF sampling**:

```
P(sample = i) = probs[i]
```

Any equivalent vectorized implementation is acceptable.

> Use the passed-in `rng` for **all** randomness (don't call the global `np.random.*`),
> so results are reproducible.

## Function Signature

```python
def sample_categorical(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...
```

Returns an integer array of shape `probs.shape[:-1]`.

## Read More

- NumPy — [random `Generator`](https://numpy.org/doc/stable/reference/random/generator.html),
  [`cumsum`](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html)
- Inverse transform sampling: https://en.wikipedia.org/wiki/Inverse_transform_sampling

## How to Test

```bash
uv run grade 010
```
