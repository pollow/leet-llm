# 210 — Decoder Block (original seq2seq)

**Level 2 · Operators & Layers**

## Description

The original transformer **decoder block** — the encoder's counterpart for generation. It
adds two things to the encoder: its self-attention is **causally masked** (a position may
only attend to itself and earlier positions), and a **cross-attention** sublayer lets the
decoder read the encoder's output (queries from the decoder, keys/values from the encoder).
This is the block used in translation / T5-style seq2seq models.

## The Math

With masked self-attention, cross-attention, and FFN, **post-norm** placement:

```
a = LayerNorm₁( x + MHA(x, x, causal_mask) )        # masked self-attention
b = LayerNorm₂( a + MHA(a, enc_out) )               # cross-attention: q=a, kv=enc_out
y = LayerNorm₃( b + FFN(b) )
```

Cross-attention is just `mha` (206) with a different `x_kv` — no new attention code.

## Function Signature

```python
@dataclass(frozen=True)
class DecoderBlockParams:
    self_attn: AttnParams           # 206
    cross_attn: AttnParams          # 206
    ffn: FFNParams                  # 207
    norm1_gamma: np.ndarray; norm1_beta: np.ndarray
    norm2_gamma: np.ndarray; norm2_beta: np.ndarray
    norm3_gamma: np.ndarray; norm3_beta: np.ndarray

def decoder_block(x: np.ndarray, enc_out: np.ndarray, params: DecoderBlockParams,
                  n_heads: int, self_mask: np.ndarray | None = None,
                  cross_mask: np.ndarray | None = None,
                  activation: str = "gelu") -> np.ndarray: ...
#   x: (..., L, d)   enc_out: (..., L_enc, d)   ->   (..., L, d)
#   activation passed through to ffn; default "gelu"
```

## Read More

- *Attention Is All You Need*, Vaswani et al. 2017 — §3.1 (decoder): https://arxiv.org/abs/1706.03762
- Reuse `from leet_llm import mha, ffn, layer_norm, add_residual, triangular_mask, AttnParams, FFNParams`.

## How to Test

```bash
uv run grade 210
```
