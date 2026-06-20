# 306 — Per-head Q/K RMSNorm

**Level 3 · Track C — OSS-Zoo Architecture Deltas**

## Description

The Llama-3 attention block passes raw Q and K vectors (after the linear
projections) straight into RoPE and then into dot-product attention.  Qwen3
and OLMo-2 add a single extra step: **apply an RMSNorm over each head vector,
independently for Q and K, before RoPE**.

This per-head normalisation keeps the Q and K magnitudes bounded regardless of
projection weight scale, which stabilises training at large model sizes.
Because it runs before RoPE, the rotation operates on already-normalised
vectors — the norm and position encoding are fully decoupled.

**GIVEN (Qwen3 HF config):** each `Qwen3Attention` module contains two
`Qwen3RMSNorm` instances — `q_norm` and `k_norm` — each with a learned
`weight` vector of shape `(head_dim,)`.  Their HF weight names are
`self_attn.q_norm.weight` and `self_attn.k_norm.weight`.  The `eps` value
(`variance_epsilon`) defaults to `1e-6`.

## The Math

Let `v` be a single head vector of dimension `d_head`.  The per-head RMSNorm
is:

```
rms(v)      = sqrt( mean(v**2) + eps )
qk_norm(v)  = (v / rms(v)) * w
```

where `w` is the learned scale vector (shape `(d_head,)`).

This is applied **independently** to every `(head, position)` vector in `q`
and `k`.  The norm operates over `head_dim` (the last axis) only — it does not
mix information across heads or sequence positions.

After normalisation the forward pass continues with RoPE (`rope_half` or
`rope_interleaved`), GQA, and scaled dot-product attention, unchanged from the
Llama-3 baseline.

## Function Signature

```python
def qk_norm(
    q: np.ndarray,
    k: np.ndarray,
    q_weight: np.ndarray,
    k_weight: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply per-head RMSNorm to Q and K before RoPE/attention.

    q, k      — shapes (..., n_heads, L, head_dim)
    q_weight  — shape (head_dim,)   [self_attn.q_norm.weight]
    k_weight  — shape (head_dim,)   [self_attn.k_norm.weight]
    eps       — variance epsilon (Qwen3 default: 1e-6)

    Returns (q_normed, k_normed) with the same shapes as the inputs.
    """
```

Hint: reuse `from leet_llm import rms_norm` (212) as a building block.

## Read More

- *Qwen3 Technical Report* (Qwen Team 2025): <https://arxiv.org/abs/2505.09388>
- *OLMo-2* (AI2 2024): <https://arxiv.org/abs/2501.00656>

## How to Test

```bash
uv run grade 306
```
