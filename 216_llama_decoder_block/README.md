# 216 — Llama Decoder Block

**Level 2 · Operators & Layers**

## Description

The capstone of the level: assemble one **Llama decoder block** — the unit the L3 model
stacks `n_layers` times. It's the GPT block (211) with every classic part swapped for its
Llama upgrade: **pre-norm RMSNorm** (212) instead of LayerNorm, **RoPE** (213) on the
queries/keys instead of additive positions, **grouped-query attention** (215) instead of
plain MHA, and a **SwiGLU** FFN (214) instead of the GELU MLP — all bias-free.

## The Math

With pre-norm RMSNorm placement, causal mask, RoPE on q/k, grouped-query attention, SwiGLU:

```
a   = RMSNorm(x, w_attn)
attn = GroupedQueryAttention( a, with RoPE(positions) applied to q and k, causal_mask )
h   = x + attn                                  # residual
y   = h + SwiGLU( RMSNorm(h, w_ffn) )           # residual
```

Implementation requirement: **extend your task-215 `gqa` implementation with RoPE** (interleaved
convention), and use that RoPE-augmented GQA inside this block. Do not write a separate
attention algorithm from scratch for this task.

RoPE is applied to the per-head query and key projections *inside* the attention, after
projection and before the scores.

## Function Signature

```python
@dataclass(frozen=True)
class LlamaBlockParams:
    attn: AttnParams           # 206 (Wk/Wv are n_kv_heads-sized, as in GQA)
    ffn: SwiGLUParams          # 214
    attn_norm: np.ndarray      # RMSNorm weight (d,)
    ffn_norm: np.ndarray       # RMSNorm weight (d,)

def llama_decoder_block(x: np.ndarray, params: LlamaBlockParams, n_heads: int,
                        n_kv_heads: int, positions: np.ndarray,
                        mask: np.ndarray | None = None,
                        eps: float = 1e-5) -> np.ndarray: ...
#   x: (..., L, d)   positions: (L,)   ->   (..., L, d)   (mask defaults to causal)
```

`llama_decoder_block` takes an optional `eps` (default `1e-5`) used by both RMSNorms; stories15M / L3 uses `1e-6`.

## Read More

- *Llama 2*, Touvron et al. 2023: https://arxiv.org/abs/2307.09288
- *Llama 3*, Grattafiori et al. 2024: https://arxiv.org/abs/2407.21783
- Required composition: start from `215_gqa`, add interleaved RoPE on q/k, then use that
  extended attention inside `llama_decoder_block`.
- Reuse `from leet_llm import rms_norm, rope_interleaved, sdpa, swiglu_ffn, add_residual,
  group_last_axis, affine, triangular_mask, AttnParams, SwiGLUParams`. (RoPE uses the
  **interleaved** convention here, matching the L3 capstone.)

## How to Test

```bash
uv run grade 216
```
