# 305 — Sliding-Window (Banded Causal) Mask + Mistral Whole-Model

**Level 3 · Track C — OSS-Zoo Architecture Deltas**

## Description

This task bundles two things:

1. **`sliding_window_mask`** — the Mistral delta operator that replaces Llama's full
   causal mask with a band-causal mask.

2. **`MistralConfig` / `MistralParams` / `load_mistral` / `mistral_forward`** — the
   full Mistral decoder-only forward pass, identical to the Llama forward (303) with
   one change: the **sliding-window band mask** in place of the full causal mask.

### The Mistral architecture

Mistral = rotate-half Llama with the band mask.  Every other component is the
same: **RMSNorm** pre-norm, **rotate-half RoPE** (`rope_half`, 213), **GQA**
(`n_kv_heads < n_heads`), **SwiGLU FFN**, no QKV bias, untied lm_head.

The Llama baseline uses a full causal mask: query `i` can attend to every past
token `j ≤ i`. Mistral 7B swaps this for a **sliding-window (banded) causal
mask**: query `i` can only attend to the most recent `W` tokens.  Tokens before
the window (`j ≤ i − W`) are masked out.

**GIVEN (Mistral HF config):**
- `sliding_window` — window size `W` (e.g. 4096 in Mistral 7B).
- `hidden_size` → `dim`, `num_hidden_layers` → `n_layers`,
  `num_attention_heads` → `n_heads`, `num_key_value_heads` → `n_kv_heads`,
  `vocab_size`, `max_position_embeddings` → `max_seq_len`,
  `rms_norm_eps` → `norm_eps`, `rope_theta` → `rope_base`.

→ **L4 connection:** the sliding-window mask is only half the story.  Efficient
inference also requires **windowed KV eviction** — discarding KV cache entries
that have left the window — a ring-buffer or rolling cache that is an L4 serving
concern.

---

## The Math

### Band causal mask

With sequence length `L` and window size `W`, the boolean mask is:

```
mask[i, j] = False   if  i − W < j ≤ i   (causal window: j is recent enough)
             True    otherwise            (future or too-old token)
```

This mask is passed to `sdpa` where `True` positions are masked out.

When `W ≥ L` no token is ever too old and the mask reduces to the standard
causal (lower-triangular) mask.

### Mistral forward (per layer)

For each of the `n_layers` decoder layers:

```
a  = RMSNorm(h, attn_norm, eps)
Q  = a @ Wq.T    split → (B, n_heads, L, head_dim)   then rope_half
K  = a @ Wk.T    split → (B, n_kv_heads, L, head_dim) then rope_half → repeat n_heads/n_kv_heads times
V  = a @ Wv.T    split → (B, n_kv_heads, L, head_dim)               → repeat
o  = sdpa(Q, K, V, mask=band_mask_bool) → merge → @ Wo.T
h  = h + o
f  = RMSNorm(h, ffn_norm, eps)
h  = h + SwiGLU(f)
```

Final: `RMSNorm(h, final_norm) @ lm_head.T → logits (B, L, V)`.

**RoPE convention:** rotate-half (`rope_half`, 213) — HF weight layout
as-is, **no un-permute** (unlike 303/304 which use interleaved + un-permute).

---

## Function Signatures

```python
def sliding_window_mask(seq_len: int, window: int) -> np.ndarray:
    """Return a bool (seq_len, seq_len) causal sliding-window mask.

    False where query i may attend to key j  (i - window < j <= i)
    True  elsewhere (future or outside the band)
    Returns bool array of shape (L, L), where True means masked.
    """

@dataclass(frozen=True)
class MistralConfig:
    dim: int; n_layers: int; n_heads: int; n_kv_heads: int
    vocab_size: int; sliding_window: int
    max_seq_len: int = 4096; norm_eps: float = 1e-5; rope_base: float = 10000.0

def load_mistral(weights: dict, cfg: MistralConfig) -> MistralParams:
    """Map HF-named arrays → MistralParams (no un-permute)."""

def mistral_forward(
    input_ids: np.ndarray,      # (B, L) int
    params: MistralParams,
    cfg: MistralConfig,
) -> np.ndarray:                # (B, L, V) float64 logits
```

### HF weight names (GIVEN, no un-permute)

| HF key | shape | usage |
|---|---|---|
| `model.embed_tokens.weight` | `(V, d)` | token embedding |
| `model.norm.weight` | `(d,)` | final RMSNorm |
| `lm_head.weight` | `(V, d)` | output projection (untied) |
| `model.layers.{i}.input_layernorm.weight` | `(d,)` | pre-attn norm |
| `model.layers.{i}.post_attention_layernorm.weight` | `(d,)` | pre-FFN norm |
| `model.layers.{i}.self_attn.q_proj.weight` | `(d, d)` | Q projection |
| `model.layers.{i}.self_attn.k_proj.weight` | `(n_kv_heads·head_dim, d)` | K projection |
| `model.layers.{i}.self_attn.v_proj.weight` | `(n_kv_heads·head_dim, d)` | V projection |
| `model.layers.{i}.self_attn.o_proj.weight` | `(d, d)` | output projection |
| `model.layers.{i}.mlp.gate_proj.weight` | `(ffn_dim, d)` | SwiGLU gate |
| `model.layers.{i}.mlp.up_proj.weight` | `(ffn_dim, d)` | SwiGLU up |
| `model.layers.{i}.mlp.down_proj.weight` | `(d, ffn_dim)` | SwiGLU down |

---

## Read More

- *Mistral 7B* (Jiang et al. 2023): <https://arxiv.org/abs/2310.06825>
- HF `MistralConfig` docs: <https://huggingface.co/docs/transformers/model_doc/mistral>
- Task 303 (`llama_model`) — the Llama forward this is based on.

---

## How to Test

```bash
# Grade the operator + whole-model hermetic fixture (always-on):
uv run grade 305

# Real-weights test (optional — downloads ~2 MB from HF):
bash 305_sliding_window_attention/download.sh
uv run grade 305
```

The real-weights test is automatically skipped unless `mistral_tiny.npz` is present
(populated by `download.sh`).
