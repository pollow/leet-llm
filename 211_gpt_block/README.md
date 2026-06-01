# 211 — GPT Block (decoder-only)

**Level 2 · Operators & Layers**

## Description

The block modern LLMs are built from: the **GPT block**, a *decoder-only* transformer layer.
Take the seq2seq decoder block (210) and **remove cross-attention** — with no encoder, there's
nothing to read from — leaving masked self-attention + FFN. GPT-2 also moved the
normalization **inside** the residual (**pre-norm**) for stable deep training. Stack these and
you have GPT.

## The Math

With masked self-attention (206) and FFN (207), **pre-norm** placement (GPT-2):

```
h = x + MHA( LayerNorm₁(x), causal_mask )      # masked self-attention
y = h + FFN( LayerNorm₂(h) )
```

The causal mask = `triangular_mask(L)` (009): position `t` may attend only to `≤ t`.

## Function Signature

```python
@dataclass(frozen=True)
class GPTBlockParams:
    attn: AttnParams           # 206
    ffn: FFNParams             # 207
    norm1_gamma: np.ndarray; norm1_beta: np.ndarray
    norm2_gamma: np.ndarray; norm2_beta: np.ndarray

def gpt_block(x: np.ndarray, params: GPTBlockParams, n_heads: int,
              mask: np.ndarray | None = None) -> np.ndarray: ...
#   x: (..., L, d)   ->   (..., L, d)   (mask defaults to causal)
```

## Read More

- *Improving Language Understanding by Generative Pre-Training* (GPT), Radford et al. 2018.
- *Language Models are Unsupervised Multitask Learners* (GPT-2, pre-norm), Radford et al. 2019.
- Reuse `from leet_llm import mha, ffn, layer_norm, add_residual, triangular_mask, AttnParams, FFNParams`.

## How to Test

```bash
uv run grade 211
```
