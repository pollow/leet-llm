# 203 — Layer Normalization

**Level 2 · Operators & Layers**

## Description

Deep stacks train and run more stably when each layer's input is **normalized**.
LayerNorm standardizes every feature vector to zero mean and unit variance over its last
axis, then applies a learned per-feature scale `γ` and shift `β`. It's the normalization of
the original transformer and of GPT.

## The Math

For a feature vector `x ∈ ℝ^d` (the last axis), with mean `μ` and variance `σ²` taken over
that axis:

```
μ = mean(x)
σ² = mean((x − μ)²)
LayerNorm(x) = γ ⊙ (x − μ) / √(σ² + ε) + β
```

`γ, β ∈ ℝ^d` are learned; `ε` is a small constant for numerical stability. The statistics
are computed **per vector**, independently for every position in the batch.

## Function Signature

```python
def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
               eps: float = 1e-5) -> np.ndarray: ...
#   x: (..., d)   gamma, beta: (d,)   ->   (..., d)
```

## Read More

- *Layer Normalization*, Ba, Kiros & Hinton 2016: https://arxiv.org/abs/1607.06450
- You may reuse `from leet_llm import standardize` (002).

## How to Test

```bash
uv run grade 203
```
