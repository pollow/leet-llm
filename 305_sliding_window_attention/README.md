# 305 — Sliding-Window (Banded Causal) Mask

**Level 3 · Track C — OSS-Zoo Architecture Deltas**

## Description

The Llama baseline uses a full causal mask: query `i` can attend to every past
token `j ≤ i`. Mistral 7B swaps this for a **sliding-window (banded) causal
mask**: query `i` can only attend to the most recent `W` tokens.  Tokens before
the window (`j ≤ i − W`) are masked out.

This single change is the Mistral delta over the Llama decoder stack — everything
else (RoPE, RMSNorm, SwiGLU, GQA) is identical.

**GIVEN (Mistral HF config):** the `sliding_window` field carries `W`.  During
the forward pass, HF constructs a `(1, 1, L, L)` boolean mask (`True` = attended)
and passes it to each attention layer.

→ **L4 connection:** the sliding-window mask is only half the story.  Efficient
inference also requires **windowed KV eviction** — discarding KV cache entries
that have left the window — a ring-buffer or rolling cache that is an L4 serving
concern.

## The Math

With sequence length `L` and window size `W`, the additive pre-softmax mask is:

```
mask[i, j] = 0.0   if  i − W < j ≤ i   (causal window: j is recent enough)
             -inf   otherwise            (future or too-old token)
```

The mask is added to the scaled dot-product scores `Q Kᵀ / √d_k` before
softmax, sending masked logits to −∞ (weight → 0).

When `W ≥ L` no token is ever too old and the mask reduces to the standard
causal (lower-triangular) mask.

## Function Signature

```python
def sliding_window_mask(seq_len: int, window: int) -> np.ndarray:
    """Return an additive (seq_len, seq_len) causal sliding-window mask.

    0.0  where query i may attend to key j  (i - window < j <= i)
    -inf elsewhere (future or outside the band)
    """
```

`seq_len` — sequence length `L`.  
`window` — Mistral `sliding_window` config field; number of past positions each
query can see.

Returns a `(L, L)` float64 array.  Add it to attention scores before softmax.

Hint: reuse `from leet_llm import triangular_mask` (009) as a building block.

## Read More

- *Mistral 7B* (Jiang et al. 2023): <https://arxiv.org/abs/2310.06825>

## How to Test

```bash
uv run grade 305
```
