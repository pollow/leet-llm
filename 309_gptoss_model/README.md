# 309 — Attention Sinks + GPT-OSS MoE: GPT-OSS Whole-Model Forward

## Description

GPT-OSS is OpenAI's open-weight Llama-style decoder. It keeps GQA + rotate-half
RoPE but adds two localized deltas on top of the baseline (303): **attention
sinks** and a **GPT-OSS-flavoured sparse mixture-of-experts FFN**. You implement
each delta and then assemble a runnable `gptoss_forward → logits`.

Two new operators plus the whole-model assembly:

1. **`attention_with_sinks(scores, sink_logits, mask=None)`** — softmax with one
   extra learned **sink** logit per head in the denominator. The sink column is
   appended, the softmax is taken over the widened axis, and the sink column is
   then dropped — so each attention row sums to **less than 1** (mass leaks into
   the discarded sink).
2. **`gptoss_moe_ffn(...)`** — GPT-OSS's sparse MoE. It is **not** Mixtral's
   `moe_ffn` (308); see the contrast table below.
3. **`GptOssConfig` / `GptOssParams` / `load_gptoss` / `gptoss_forward`** — the
   GPT-OSS decoder-only model.

**GIVEN — the GPT-OSS wrinkles** (each is architecture-as-spec):

- **Attention sinks** — each head has one learned scalar `sink_logit`. It enters
  the softmax denominator as an extra (key-less) column; after softmax it is
  discarded, so the attention weights no longer sum to 1.
- **QKV/O biases** — unlike Llama/Mistral, GPT-OSS attention projections all carry
  biases (`config.attention_bias = True`).
- **Attention scale** — `head_dim ** -0.5` (and `n_heads * head_dim` need not equal
  `hidden_size`; `head_dim` is set explicitly).
- **Alternating SWA/full layers** — even-indexed layers (0, 2, …) use a
  sliding-window causal mask (`sliding_window_mask`, reused from 305); odd-indexed
  layers use full causal.
- **GPT-OSS MoE** — top-k routing with a **biased** router, a softmax taken over the
  **selected** logits, interleaved gate/up, and a clamped GLU activation (below).
- **RoPE** — GPT-OSS's real schedule is **YaRN** long-context scaling:
  `inv_freq = rope_scaled_freqs(head_dim, rope_base, cfg.rope_scaling)` and the attention
  temperature `af = rope_attention_scale(cfg.rope_scaling)`, applied as
  `rope_from_freqs(.., inv_freq) * af` (rotate-half). `cfg.rope_scaling=None` → plain
  rotate-half RoPE.

### GPT-OSS MoE vs Mixtral `moe_ffn` (308)

| | Mixtral `moe_ffn` (308) | GPT-OSS `gptoss_moe_ffn` (309) |
|---|---|---|
| router bias | none | **yes** (`mlp.router.bias`) |
| gate softmax | over **all** experts, then top-k, then renormalise | over the **selected top-k** logits only |
| expert bias | none | **yes** (`gate_up_proj_bias`, `down_proj_bias`) |
| gate/up split | halves `[ :Fd ]` / `[ Fd: ]` | **interleaved** `[::2]` / `[1::2]` |
| activation | `SiLU(gate) * up` | clamped `(up+1) · gate · σ(α·gate)` |
| weight layout | `x @ W.T` | `x @ W` (input dim first, **no transpose**) |

Because every routing/expert detail differs, GPT-OSS gets its **own** MoE operator
rather than reusing 308's.

> **→ L4:** this task implements the forward-pass arithmetic only. The
> **streaming sink + sliding-window KV-cache eviction** during decode (keeping the
> per-head sink while dropping out-of-window keys) is an inference-systems concern,
> deferred to Level 4.

---

## The Math

### `attention_with_sinks`

Given per-head scores `S ∈ R^(B,H,L,L)` (already `= Q Kᵀ · scale`), an additive
`mask`, and per-head sinks `z ∈ R^H`:

```
S'        = S + mask                              # (B, H, L, L)
combined  = concat([S', broadcast(z)], axis=-1)   # (B, H, L, L+1)
P         = softmax(combined, axis=-1)            # over the L+1 columns
weights   = P[..., :-1]                            # drop the sink column
```

Each row of `weights` sums to `1 − P[..., -1]` (the sink mass). Setting
`z = -inf` zeroes the sink column and recovers the plain softmax (rows sum to 1).

### `gptoss_moe_ffn`

Let `x ∈ R^(T, d)`, router `(W_r, b_r)`, and `E` experts with packed weights.

```
router_logits = x @ W_rᵀ + b_r                       # (T, E)
vals, idx     = top_k(router_logits, k)              # (T, k) — k largest, descending
scores        = softmax(vals)                         # softmax over the SELECTED k
```

For each token `t` and each selected expert `e = idx[t, j]`:

```
gate_up = x_t @ gate_up_proj[e] + gate_up_bias[e]    # (2F,)
gate    = gate_up[::2]                                 # even columns
up      = gate_up[1::2]                                # odd  columns
gate    = min(gate, limit)                            # clamp max only (limit = 7.0)
up      = clip(up, -limit, limit)
glu     = gate · sigmoid(alpha · gate)               # alpha = 1.702
out_e   = (up + 1) · glu @ down_proj[e] + down_bias[e]   # (d,)

moe(x_t) = Σ_j  scores[t, j] · out_{idx[t,j]}
```

### Whole-model `gptoss_forward`

```
inv_freq = rope_scaled_freqs(head_dim, rope_base, cfg.rope_scaling)   # YaRN
af       = rope_attention_scale(cfg.rope_scaling)                     # ≈1.35 for YaRN
h = embedding(input_ids, tok_embed)

for i, layer in enumerate(layers):
    a = rms_norm(h, input_layernorm)
    q = (a @ q_proj.T + q_bias) → (B, H,   L, head_dim) ; rope_from_freqs(q, positions, inv_freq) * af
    k = (a @ k_proj.T + k_bias) → (B, KVH, L, head_dim) ; rope_from_freqs(k, positions, inv_freq) * af
    v = (a @ v_proj.T + v_bias) → (B, KVH, L, head_dim)
    k, v   = repeat_kv(k, v, H // KVH)                       # GQA
    scores = (q @ k.T) * (head_dim ** -0.5)
    mask   = sliding_window_mask(L, window) if i even else full causal
    probs  = attention_with_sinks(scores, sinks, mask)
    a = probs @ v → merge heads → @ o_proj.T + o_bias
    h = h + a                                                # residual 1

    f = rms_norm(h, post_attention_layernorm)
    f = gptoss_moe_ffn(f, router, experts, num_experts_per_tok)
    h = h + f                                                # residual 2

h = rms_norm(h, final_norm)
logits = h @ lm_head.T
```

**Note:** the attention math composes from granular primitives (rotate-half RoPE,
explicit scores → sink softmax), *not* from `llama_decoder_block` (216, which is
interleaved-RoPE only).

---

## GIVEN — HF weight map (`load_gptoss`)

GPT-OSS uses rotate-half RoPE, so weights map **as-is** (no un-permute):

```
model.embed_tokens.weight                       (V, d)
model.norm.weight                               (d,)
lm_head.weight                                  (V, d)   # absent → tie to embed

model.layers.{i}.input_layernorm.weight             (d,)
model.layers.{i}.post_attention_layernorm.weight    (d,)
model.layers.{i}.self_attn.q_proj.weight            (n_heads    * head_dim, d)
model.layers.{i}.self_attn.q_proj.bias              (n_heads    * head_dim,)
model.layers.{i}.self_attn.k_proj.weight            (n_kv_heads * head_dim, d)
model.layers.{i}.self_attn.k_proj.bias              (n_kv_heads * head_dim,)
model.layers.{i}.self_attn.v_proj.weight            (n_kv_heads * head_dim, d)
model.layers.{i}.self_attn.v_proj.bias              (n_kv_heads * head_dim,)
model.layers.{i}.self_attn.o_proj.weight            (d, n_heads * head_dim)
model.layers.{i}.self_attn.o_proj.bias              (d,)
model.layers.{i}.self_attn.sinks                    (n_heads,)
model.layers.{i}.mlp.router.weight                  (num_experts, d)
model.layers.{i}.mlp.router.bias                    (num_experts,)
model.layers.{i}.mlp.experts.gate_up_proj           (num_experts, d, 2 * intermediate)
model.layers.{i}.mlp.experts.gate_up_proj_bias      (num_experts, 2 * intermediate)
model.layers.{i}.mlp.experts.down_proj              (num_experts, intermediate, d)
model.layers.{i}.mlp.experts.down_proj_bias         (num_experts, d)
```

---

## Function Signatures

```python
def attention_with_sinks(
    scores: np.ndarray,        # (B, H, L, L) = Q Kᵀ * scale (no mask applied yet)
    sink_logits: np.ndarray,   # (H,)
    mask: np.ndarray | None = None,   # additive, broadcastable to scores
) -> np.ndarray:               # (B, H, L, L), rows sum to < 1

def gptoss_moe_ffn(
    x: np.ndarray,             # (T, d)
    router_weight: np.ndarray, # (E, d)
    router_bias: np.ndarray,   # (E,)
    gate_up_proj: np.ndarray,  # (E, d, 2*F)
    gate_up_bias: np.ndarray,  # (E, 2*F)
    down_proj: np.ndarray,     # (E, F, d)
    down_bias: np.ndarray,     # (E, d)
    top_k: int,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> np.ndarray:               # (T, d)

def load_gptoss(weights: dict, cfg: GptOssConfig) -> GptOssParams: ...

def gptoss_forward(
    input_ids: np.ndarray,     # (B, L)
    params: GptOssParams,
    cfg: GptOssConfig,
    start_pos: int = 0,
) -> np.ndarray:               # (B, L, vocab_size)
```

---

## Read More

- GPT-OSS model card: <https://huggingface.co/openai/gpt-oss-20b>
- `modeling_gpt_oss.py` in the transformers package (exact arithmetic reference)
- Attention-sink intuition (StreamingLLM): <https://arxiv.org/abs/2309.17453>
- `303_llama_model/` — the Llama baseline this re-skins
- `305_sliding_window_attention/` — the band mask reused for the even (sliding) layers
- `307_llama31_model/` — Llama-3.1 `llama3` RoPE scaling (contrast reference)
- `308_mixtral_model/` — Mixtral's MoE, contrasted in the table above
- Task 213 (`rope_half`), 005 (`softmax`), 007 (`top_k`)

**Real-weights layer (B):** `download.sh` fetches
`hf-internal-testing/tiny-random-GptOssForCausalLM` (config + safetensors only),
maps the HF names → `GptOssParams`, and commits `tests/fixtures/real_ref.npz` from a
genuine `GptOssForCausalLM` with its **native YaRN** RoPE active. The checkpoint's
weights are random (no demo) — it exists only as the grade-time genuine-HF cross-check
+ loader coverage (Tier B). The only forced setting is **eager attention** (the
explicit softmax-with-sink path our forward mirrors).

> **→ L4:** GPT-OSS's real RoPE is YaRN long-context scaling. This task includes the
> YaRN `inv_freq` + attention-temperature wiring in forward; long-context decode
> (KV-cache/eviction behavior) is an inference-systems concern deferred to L4.

---

## How to Test

```bash
uv run grade 309                  # grade your implementation
uv run grade 309 -v               # verbose output

# optional: real-weights parity (downloads a tiny checkpoint, ~MBs)
bash 309_gptoss_model/download.sh
```

The test suite checks:

- `test_sinks_matches_oracle` / `test_sinks_rows_sum_below_one` /
  `test_sinks_neg_inf_recovers_plain_softmax` / `test_sinks_respects_mask`:
  `attention_with_sinks` vs the float64 oracle and its sink invariants.
- `test_moe_matches_oracle` / `test_moe_routing_depends_only_on_top_k` /
  `test_moe_gate_scores_sum_to_one` / `test_moe_clamps_preactivations`:
  `gptoss_moe_ffn` vs the float64 oracle, routing sparsity, score normalisation,
  and pre-activation clamping.
- `test_gptoss_logits_match_oracle`: whole-model logit parity vs the committed float64
  oracle at `rtol=1e-9`.
- `test_gptoss_logits_shape` / `test_gptoss_causal`: output shape and causal masking.
- `test_gptoss_alternating_sliding_full_masks`: even layers consult the sliding window.
- `test_gptoss_yarn_rope_is_wired`: the YaRN schedule is applied (yarn ≠ default RoPE).
- `test_gptoss_real_weights_logits` *(skipped until `download.sh` runs)*: parity vs a
  genuine `GptOssForCausalLM` on real weights.
