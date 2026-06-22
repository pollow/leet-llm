# 307 — Long-Context RoPE Scaling: Llama-3.1 Whole-Model Forward

## Description

Llama-3.1 is the baseline Llama (303) with exactly one localized delta: the RoPE
**inverse frequencies are rescaled** so the same pretrained weights generalise from
the 8K pretraining window out to 128K tokens. The attention math, GQA, SwiGLU,
RMSNorm and rotate-half RoPE are all unchanged — only the *frequencies* that go into
the rotation are bent. You implement the frequency schedule and the apply step, then
assemble a runnable `llama31_forward → logits`.

Two new operators plus the whole-model assembly:

1. **`rope_scaled_freqs(head_dim, base, scaling)`** — compute the per-pair inverse
   frequencies `inv_freq` for `default` (or `None`) and Llama-3.1's native
   `rope_type="llama3"`. With `scaling=None` this is plain RoPE; `llama3` bends low
   frequencies so far-apart positions *interpolate* rather than extrapolate.
2. **`rope_from_freqs(x, positions, inv_freq)`** — rotate-half RoPE applied with a
   **precomputed** `inv_freq`. Identical rotation to 213's `rope_half`; the only
   difference is that the (rescaled) frequencies are supplied rather than derived from
   `base`.
3. **`Llama31Config` / `Llama31Params` / `load_llama31` / `llama31_forward`** — the
   Llama-3.1 decoder-only model.

**GIVEN — the Llama-3.1 wrinkle** (architecture-as-spec):

- **RoPE frequency scaling** — a `rope_scaling` config dict selects how `inv_freq` is
  rescaled (see *The Math*). Llama-3.1 ships `rope_type="llama3"`.
- **Everything else is 303's Llama** — bias-free QKV/O projections, GQA, SwiGLU FFN
  (`gate`/`up`/`down`), pre-norm RMSNorm, rotate-half RoPE, full causal masking,
  attention scale `head_dim ** -0.5` with `head_dim = hidden_size // n_heads`.

> **→ L4:** this task implements the forward-pass arithmetic only. The KV-cache side
> of long-context decoding — chunked prefill and position bookkeeping past the
> original window — is an inference-systems concern, deferred to Level 4.

---

## The Math

Write `d = head_dim` and the default (unscaled) frequencies as

```
inv_freq[j] = 1 / base ** (2j / d)      for j = 0 .. d/2 - 1
wavelen[j]  = 2π / inv_freq[j]
```

### `rope_scaled_freqs`

`scaling["rope_type"]` picks the schedule:

- **`default` / `None`** — `inv_freq` unchanged.
- **`llama3`** — keep high frequencies, divide low frequencies by `factor`, and
  smoothly interpolate the band in between. With `low_freq_wavelen = O / low_freq_factor`,
  `high_freq_wavelen = O / high_freq_factor`, and `O = original_max_position_embeddings`:

  ```
  if wavelen > low_freq_wavelen:           inv_freq[j] / factor        # low freq → scaled
  if wavelen < high_freq_wavelen:          inv_freq[j]                 # high freq → kept
  otherwise (medium band):                                            # smooth blend
      s          = (O / wavelen − low_freq_factor) / (high_freq_factor − low_freq_factor)
      inv_freq[j] = (1 − s) · inv_freq[j] / factor + s · inv_freq[j]
  ```

### `rope_from_freqs`

The apply step is the rotate-half rotation, unchanged from 213:

```
angle = positions[:, None] * inv_freq            # (L, d/2)
angle = concat([angle, angle], axis=-1)          # (L, d)
out   = x · cos(angle) + rotate_half(x) · sin(angle)
rotate_half(x) = concat([-x[..., d/2:], x[..., :d/2]], axis=-1)
```

### Whole-model `llama31_forward`

```
inv_freq = rope_scaled_freqs(head_dim, rope_base, cfg.rope_scaling)   # once
h = embedding(input_ids, tok_embed)

for layer in layers:
    a = rms_norm(h, input_layernorm)
    q = (a @ q_proj.T) → (B, H,   L, head_dim) ; rope_from_freqs(q, positions, inv_freq)
    k = (a @ k_proj.T) → (B, KVH, L, head_dim) ; rope_from_freqs(k, positions, inv_freq)
    v = (a @ v_proj.T) → (B, KVH, L, head_dim)
    k, v   = repeat_kv(k, v, H // KVH)                       # GQA
    a = sdpa(q, k, v, triangular_mask(L)) → merge heads → @ o_proj.T
    h = h + a                                                # residual 1

    f = rms_norm(h, post_attention_layernorm)
    f = swiglu_ffn(f, {gate, up, down})
    h = h + f                                                # residual 2

h = rms_norm(h, final_norm)
logits = h @ lm_head.T
```

**Note:** the attention math composes from granular primitives (rotate-half RoPE,
`sdpa`), *not* from `llama_decoder_block` (216, which is interleaved-RoPE only).

---

## GIVEN — HF weight map (`load_llama31`)

Llama-3.1 uses rotate-half RoPE, so weights map **as-is** (no un-permute), and the
attention projections are **bias-free**:

```
model.embed_tokens.weight                       (V, d)
model.norm.weight                               (d,)
lm_head.weight                                  (V, d)   # absent → tie to embed

model.layers.{i}.input_layernorm.weight             (d,)
model.layers.{i}.post_attention_layernorm.weight    (d,)
model.layers.{i}.self_attn.q_proj.weight            (n_heads    * head_dim, d)
model.layers.{i}.self_attn.k_proj.weight            (n_kv_heads * head_dim, d)
model.layers.{i}.self_attn.v_proj.weight            (n_kv_heads * head_dim, d)
model.layers.{i}.self_attn.o_proj.weight            (d, n_heads * head_dim)
model.layers.{i}.mlp.gate_proj.weight               (intermediate, d)
model.layers.{i}.mlp.up_proj.weight                 (intermediate, d)
model.layers.{i}.mlp.down_proj.weight               (d, intermediate)
```

---

## Function Signatures

```python
def rope_scaled_freqs(
    head_dim: int,
    base: float,
    scaling: dict | None = None,   # rope_scaling config; None → default RoPE
) -> np.ndarray:                   # (head_dim / 2,) inverse frequencies

def rope_from_freqs(
    x: np.ndarray,                 # (..., head_dim)
    positions: np.ndarray,         # (L,)
    inv_freq: np.ndarray,          # (head_dim / 2,)
) -> np.ndarray:                   # same shape as x

def load_llama31(weights: dict, cfg: Llama31Config) -> Llama31Params: ...

def llama31_forward(
    input_ids: np.ndarray,         # (B, L)
    params: Llama31Params,
    cfg: Llama31Config,
    start_pos: int = 0,
) -> np.ndarray:                   # (B, L, vocab_size)
```

---

## Read More

- Llama-3.1 RoPE scaling (the `llama3` schedule): `transformers.modeling_rope_utils`
  (`_compute_llama3_parameters`) — the exact arithmetic reference.
- `303_llama_model/` — the Llama baseline this re-skins
- Task 213 (`rope_half` / `rope_interleaved`), 205 (`sdpa`), 214 (`swiglu_ffn`), 009 (`triangular_mask`)

**Real-weights layer (B):** `download.sh` fetches `llamafactory/tiny-random-Llama-3`
(config + safetensors only), maps the HF names → `Llama31Params`, and commits
`tests/fixtures/real_ref.npz` from a genuine `LlamaForCausalLM`. This ungated tiny
checkpoint ships an **active** `rope_type=llama3` schedule — exactly the delta 307
implements — so nothing is forced (contrast 309, which forced default RoPE). The
weights are random (no demo): no small *trained* Llama-3.1 is ungated (the real 1B+
models are license-gated), so there is no Tier-C end-to-end demo. This layer exists
as the grade-time genuine-HF cross-check + loader coverage (Tier B).

---

## How to Test

```bash
uv run grade 307                  # grade your implementation
uv run grade 307 -v               # verbose output

# optional: real-weights parity (downloads a tiny checkpoint, ~MBs)
bash 307_llama31_model/download.sh
```

The test suite checks:

- `test_rope_default_matches_base_freqs` / `test_rope_none_is_default` /
  `test_rope_scaled_matches_hf_golden` / `test_rope_llama3_bends_low_frequencies`:
  `rope_scaled_freqs` for `default` and `llama3` vs frozen genuine-HF `inv_freq`
  goldens and scaling invariants.
- `test_rope_from_freqs_equals_rope_half_on_default` /
  `test_rope_from_freqs_zero_position_is_identity`: the apply step matches 213's
  `rope_half` and is the identity at position 0.
- `test_llama31_logits_match_oracle`: whole-model logit parity vs the committed float64
  oracle at `rtol=1e-9`.
- `test_llama31_logits_shape` / `test_llama31_causal`: output shape and causal masking.
- `test_llama31_scaling_changes_logits`: the `rope_scaling` schedule is actually wired
  into the forward (llama3 ≠ default).
- `test_llama31_real_weights_logits` *(skipped until `download.sh` runs)*: parity vs a
  genuine `LlamaForCausalLM` on real weights with the native llama3 schedule.
