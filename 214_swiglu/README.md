# 214 — SwiGLU Feed-Forward

**Level 2 · Operators & Layers**

## Description

The Llama upgrade to the classic FFN (207). **SwiGLU** replaces the single
expand→activate→project path with a **gated** one: two parallel projections, where a
SiLU-activated "gate" multiplies an "up" projection element-wise before the down projection.
Gated FFNs consistently beat plain MLPs at the same parameter count, and Llama uses
bias-free linears throughout.

## The Math

With SiLU (202), three bias-free weight matrices (`W₁` gate, `W₃` up, `W₂` down):

```
SwiGLU(x) = ( SiLU(x W₁ᵀ) ⊙ (x W₃ᵀ) ) W₂ᵀ
#   W₁, W₃: (d_ff, d)     W₂: (d, d_ff)     no biases
```

**Contrast with the classic FFN (207):** two input projections instead of one, an
element-wise gate instead of a plain activation, and no biases.

## Function Signature

```python
@dataclass(frozen=True)
class SwiGLUParams:
    W1: np.ndarray; W3: np.ndarray; W2: np.ndarray   # gate, up, down — bias-free

def swiglu_ffn(x: np.ndarray, params: SwiGLUParams) -> np.ndarray: ...
#   x: (..., d)   ->   (..., d)
```

## Read More

- *GLU Variants Improve Transformer*, Shazeer 2020: https://arxiv.org/abs/2002.05202
- Reuse `from leet_llm import silu, affine` (202, 003).

## How to Test

```bash
uv run grade 214
```
