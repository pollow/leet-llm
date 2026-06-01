# 213 — RoPE (Rotary Position Embedding)

**Level 2 · Operators & Layers**

## Description

The Llama upgrade to additive positional encoding (204). Instead of *adding* a position
signal to embeddings, **RoPE** *rotates* each query and key vector by an angle proportional
to its position. Because a dot product between two rotated vectors depends only on the
*difference* of their angles, attention scores become naturally **relative** — token `m`
attending to token `n` sees a function of `m − n`. Applied to q/k inside attention.

## The Math

For head dimension `d` (even), base `θ = 10000`, frequency index `i ∈ {0,…,d/2−1}`:

```
inv_freq_i = θ^{−2i/d}
angle(p)   = p · inv_freq            # for position p, shape (d/2,)
cos, sin   = cos(angle), sin(angle)  # broadcast to width d as [angle, angle]
```

Using the **rotate-half** convention (HuggingFace / Llama), split `x = [x₁, x₂]` into its
two halves and:

```
rotate_half(x) = [ −x₂, x₁ ]
RoPE(x, p) = x ⊙ cos(p) + rotate_half(x) ⊙ sin(p)
```

(This is the half-split convention, **not** the paper's interleaved adjacent pairs — the
tests pin this form.)

## Function Signature

```python
def rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray: ...
#   x: (..., L, d) with d even (the L axis is -2)   positions: (L,) int   ->   (..., L, d)
```

## Read More

- *RoFormer: Enhanced Transformer with Rotary Position Embedding*, Su et al. 2021: https://arxiv.org/abs/2104.09864

## How to Test

```bash
uv run grade 213
```
