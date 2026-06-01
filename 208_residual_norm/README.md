# 208 — Residual Connections & Norm Placement

**Level 2 · Operators & Layers**

## Description

Transformers are deep, and deep stacks vanish gradients and forget their input. The fix is
the **residual connection**: every sublayer *adds* its output back to its input, so there's
always a clean path for information to flow. *Where* the normalization sits relative to that
add — **post-norm** (original transformer) vs **pre-norm** (GPT-2, Llama) — is the other
half of this task, and it's why modern models train stably at depth.

## The Math

A residual connection around a `Sublayer` (attention or FFN), with `Norm` = LayerNorm (203):

```
add_residual(x, s) = x + s

post-norm:   y = Norm(x + Sublayer(x))      # original transformer
pre-norm:    y = x + Sublayer(Norm(x))      # GPT-2, Llama — norm INSIDE the residual
```

Pre-norm keeps the residual path un-normalized end to end, which is what lets very deep
stacks train without careful warmup.

## Function Signature

```python
def add_residual(x: np.ndarray, sublayer_out: np.ndarray) -> np.ndarray: ...
#   both (..., d)   ->   (..., d)
```

(The blocks in 209–211 and 216 use this plus `layer_norm`/`rms_norm` in the placement noted
above.)

## Read More

- *Deep Residual Learning*, He et al. 2015: https://arxiv.org/abs/1512.03385
- *On Layer Normalization in the Transformer Architecture* (pre- vs post-norm), Xiong et al. 2020: https://arxiv.org/abs/2002.04745

## How to Test

```bash
uv run grade 208
```
