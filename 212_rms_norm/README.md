# 212 — RMSNorm

**Level 2 · Operators & Layers**

## Description

The Llama upgrade to LayerNorm (203). **RMSNorm** drops the mean-subtraction and the bias —
it only rescales each vector by its root-mean-square — and works just as well while being
cheaper. It's the normalization used by Llama, Mistral, Qwen, and most modern LLMs.

## The Math

For a feature vector `x ∈ ℝ^d` (the last axis), learned scale `w ∈ ℝ^d`:

```
RMS(x) = √( mean(x²) + ε )
RMSNorm(x) = x / RMS(x) ⊙ w
```

**Contrast with LayerNorm (203):** no `− μ` (no re-centering), no `+ β` (no bias) — just
scale by the RMS and apply the learned gain `w`.

## Function Signature

```python
def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray: ...
#   x: (..., d)   weight: (d,)   ->   (..., d)
```

## Read More

- *Root Mean Square Layer Normalization*, Zhang & Sennrich 2019: https://arxiv.org/abs/1910.07467
- *Llama 2*, Touvron et al. 2023 (uses pre-norm RMSNorm): https://arxiv.org/abs/2307.09288

## How to Test

```bash
uv run grade 212
```
