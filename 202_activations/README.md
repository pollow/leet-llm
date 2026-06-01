# 202 — Activations (GELU & SiLU)

**Level 2 · Operators & Layers**

## Description

Between linear layers a transformer applies a smooth, non-linear **activation**. You'll
build the two that matter for this course: **GELU** (the classic GPT/BERT choice) and
**SiLU** (a.k.a. swish, the gate inside Llama's SwiGLU). Both are smooth, near-linear for
large positive inputs, and saturate toward 0 for large negative inputs.

## The Math

With the standard normal CDF `Φ` and the logistic sigmoid `σ(x) = 1 / (1 + e^{−x})`:

```
GELU(x) = x · Φ(x) = x · ½ · (1 + erf(x / √2))      # exact form
SiLU(x) = x · σ(x)
```

Use the **exact** GELU (via `erf`), not the tanh approximation.

## Function Signature

```python
def gelu(x: np.ndarray) -> np.ndarray: ...
def silu(x: np.ndarray) -> np.ndarray: ...
```

## Read More

- *Gaussian Error Linear Units (GELUs)*, Hendrycks & Gimpel 2016: https://arxiv.org/abs/1606.08415
- *Sigmoid-Weighted Linear Units (SiLU)*, Elfwing et al. 2017: https://arxiv.org/abs/1702.03118
- `scipy.special.erf` / `math.erf` for the exact GELU.

## How to Test

```bash
uv run grade 202
```
