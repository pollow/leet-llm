# 209 — Encoder Block (BERT-style)

**Level 2 · Operators & Layers**

## Description

The first complete transformer layer: one **encoder block**. It runs **bidirectional**
self-attention (every position sees every other — no mask), then a position-wise FFN, each
wrapped in a residual connection and normalization. Stacking these is exactly the BERT
encoder. This is the "understand the classic architecture" half of the level.

## The Math

With self-attention (206) and FFN (207), **post-norm** placement (original transformer):

```
a = LayerNorm₁( x + MHA(x, x) )         # bidirectional self-attention, no causal mask
y = LayerNorm₂( a + FFN(a) )
```

(An optional padding mask may be passed to the attention; there is no causal mask.)

## Function Signature

```python
@dataclass(frozen=True)
class EncoderBlockParams:
    attn: AttnParams           # 206
    ffn: FFNParams             # 207
    norm1_gamma: np.ndarray; norm1_beta: np.ndarray
    norm2_gamma: np.ndarray; norm2_beta: np.ndarray

def encoder_block(x: np.ndarray, params: EncoderBlockParams, n_heads: int,
                  mask: np.ndarray | None = None) -> np.ndarray: ...
#   x: (..., L, d)   ->   (..., L, d)
```

## Read More

- *Attention Is All You Need*, Vaswani et al. 2017 — §3.1 (encoder): https://arxiv.org/abs/1706.03762
- *BERT*, Devlin et al. 2018: https://arxiv.org/abs/1810.04805
- Reuse `from leet_llm import mha, ffn, layer_norm, add_residual, AttnParams, FFNParams`.

## How to Test

```bash
uv run grade 209
```
