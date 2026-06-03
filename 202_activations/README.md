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

### Why erf? The intuition behind GELU

GELU uses the Gaussian CDF `Φ(x)` as a smooth "gate" that determines how much of the
input to pass through:

- **For large positive x**: `Φ(x) → 1`, so `GELU(x) ≈ x` (identity, passes through unchanged)
- **For large negative x**: `Φ(x) → 0`, so `GELU(x) ≈ 0` (blocks negative values)
- **Near zero**: smooth S-shaped transition, unlike ReLU's sharp kink at 0

This probabilistic gating (from the normal distribution) gives GELU three key advantages
over ReLU:

1. **Smoothness everywhere**: Differentiable at 0 (ReLU has a non-differentiable kink),
   which helps gradient-based optimization
2. **Non-zero gradients for negatives**: Small negative inputs get small but non-zero
   gradients, preventing "dead neurons" that can occur with ReLU
3. **Stochastic interpretation**: Can be viewed as randomly dropping inputs with
   probability `1 - Φ(x)` — inputs are kept with probability proportional to their
   magnitude, providing a form of adaptive regularization

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
