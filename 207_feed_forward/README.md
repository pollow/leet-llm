# 207 — Feed-Forward Network (classic MLP)

**Level 2 · Operators & Layers**

## Description

After attention mixes information *across* positions, the **feed-forward network** processes
each position *independently*, expanding to a larger hidden width, applying a non-linearity,
and projecting back. It's the classic two-layer MLP used in the original transformer and GPT.

## The Math

With `x ∈ ℝ^d`, hidden width `d_ff` (usually `4d`), weights applied as `x Wᵀ + b`:

```
FFN(x) = GELU(x W₁ᵀ + b₁) W₂ᵀ + b₂
#   W₁: (d_ff, d)  b₁: (d_ff,)   W₂: (d, d_ff)  b₂: (d,)
```

The same weights are applied at every position.

## Function Signature

```python
@dataclass(frozen=True)
class FFNParams:
    W1: np.ndarray; b1: np.ndarray; W2: np.ndarray; b2: np.ndarray

def ffn(x: np.ndarray, params: FFNParams) -> np.ndarray: ...
#   x: (..., d)   ->   (..., d)
```

## Read More

- *Attention Is All You Need*, Vaswani et al. 2017 — §3.3 (position-wise FFN): https://arxiv.org/abs/1706.03762
- Reuse `from leet_llm import affine, gelu` (003, 202).

## How to Test

```bash
uv run grade 207
```
