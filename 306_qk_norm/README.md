# 306 — Per-head Q/K RMSNorm + Qwen3 Whole-Model Forward

**Level 3 · Track C — OSS-Zoo Architecture Deltas**

## Description

### 1. `qk_norm` operator

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

### 2. Qwen3 whole-model forward

**Qwen3 = Llama with per-head qk-norm before rotate-half RoPE, no QKV bias.**

The architecture is otherwise 303's Llama assembly: token embedding → N decoder
blocks → final RMSNorm → lm_head.  Each block uses rotate-half RoPE (`rope_half`,
task 213), GQA (`sdpa` with repeat-kv), SwiGLU FFN (SiLU gate), and RMSNorm.
The sole delta is `qk_norm` applied to Q and K head tensors *before* `rope_half`.

**GIVEN (Qwen3-0.6B HF facts):**

Config fields:
- `hidden_size` → `dim`
- `num_attention_heads` → `n_heads`
- `num_key_value_heads` → `n_kv_heads`
- `head_dim` — **explicit field** in Qwen3 (NOT `hidden_size // num_attention_heads`
  in general); use `cfg.head_dim` everywhere
- `num_hidden_layers` → `n_layers`
- `vocab_size`
- `rms_norm_eps` → `norm_eps` (also used for `qk_norm_eps`, default `1e-6`)
- `rope_theta` → `rope_base`
- `tie_word_embeddings = True` — `lm_head.weight` is **absent** from the
  checkpoint; `load_qwen3` must use `model.embed_tokens.weight` as `lm_head`

HF weight names (no un-permute — rotate-half layout as-is):
```
model.embed_tokens.weight                              (V, d)
model.norm.weight                                      (d,)
lm_head.weight                                         absent (tied embeddings)
model.layers.{i}.input_layernorm.weight                (d,)
model.layers.{i}.post_attention_layernorm.weight       (d,)
model.layers.{i}.self_attn.q_proj.weight               (n_heads*head_dim, d)
model.layers.{i}.self_attn.k_proj.weight               (n_kv_heads*head_dim, d)
model.layers.{i}.self_attn.v_proj.weight               (n_kv_heads*head_dim, d)
model.layers.{i}.self_attn.o_proj.weight               (d, n_heads*head_dim)
model.layers.{i}.self_attn.q_norm.weight               (head_dim,)
model.layers.{i}.self_attn.k_norm.weight               (head_dim,)
model.layers.{i}.mlp.gate_proj.weight                  (ffn_dim, d)
model.layers.{i}.mlp.up_proj.weight                    (ffn_dim, d)
model.layers.{i}.mlp.down_proj.weight                  (d, ffn_dim)
```

## The Math

### qk_norm

Let `v` be a single head vector of dimension `d_head`.  The per-head RMSNorm
is:

```
rms(v)      = sqrt( mean(v**2) + eps )
qk_norm(v)  = (v / rms(v)) * w
```

where `w` is the learned scale vector (shape `(d_head,)`).

Applied **independently** to every `(head, position)` vector in `q` and `k`.
The norm operates over `head_dim` (the last axis) only.

### qwen3_forward (block-level assembly)

```
x = embedding(input_ids, tok_embed)                  # (B, L, d)
for each layer i:
    a = rms_norm(x, attn_norm[i])
    q = affine(a, q_proj[i]) reshaped → (B, n_heads,  L, head_dim)
    k = affine(a, k_proj[i]) reshaped → (B, n_kv_heads, L, head_dim)
    v = affine(a, v_proj[i]) reshaped → (B, n_kv_heads, L, head_dim)
    q, k = qk_norm(q, k, q_norm[i], k_norm[i])      # per-head RMSNorm
    q, k = rope_half(q, k, positions, rope_base)     # rotate-half RoPE
    k, v = repeat-kv(k, v, n_heads // n_kv_heads)
    o = sdpa(q, k, v, causal_mask)                   # scaled dot-product
    o = merge + affine(o, o_proj[i])
    x = add_residual(x, o)
    f = rms_norm(x, ffn_norm[i])
    x = add_residual(x, swiglu_ffn(f, gate[i], up[i], down[i]))
x = rms_norm(x, final_norm)
logits = x @ lm_head.T                               # (B, L, V)
```

## Function Signatures

```python
def qk_norm(q, k, q_weight, k_weight, eps=1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Apply per-head RMSNorm to Q and K before RoPE/attention.

    q, k      — shapes (..., n_heads, L, head_dim)
    q_weight  — shape (head_dim,)   [self_attn.q_norm.weight]
    k_weight  — shape (head_dim,)   [self_attn.k_norm.weight]
    eps       — variance epsilon (Qwen3 default: 1e-6)

    Returns (q_normed, k_normed) with the same shapes as the inputs.
    """

def load_qwen3(weights: dict, cfg: Qwen3Config) -> Qwen3Params:
    """Map HF-named arrays into Qwen3Params.
    When 'lm_head.weight' is absent (tie_word_embeddings=True), use
    'model.embed_tokens.weight' as lm_head.
    """

def qwen3_forward(input_ids: np.ndarray, params: Qwen3Params, cfg: Qwen3Config) -> np.ndarray:
    """Token embed → N Qwen3 blocks (causal) → final RMSNorm → lm_head logits.
    Returns logits shape (B, L, V).
    """
```

## Read More

- *Qwen3 Technical Report* (Qwen Team 2025): <https://arxiv.org/abs/2505.09388>
- *OLMo-2* (AI2 2024): <https://arxiv.org/abs/2501.00656>

## How to Test

```bash
# Grade the operator + whole-model hermetic fixture (always-on):
uv run grade 306

# Real-weights test against Qwen/Qwen3-0.6B (~1.2 GB download):
bash 306_qk_norm/download.sh
uv run grade 306
```
