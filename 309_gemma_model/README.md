# 309 — GeGLU + Soft-cap + Sandwich Norm: Gemma-2 Whole-Model Forward

## Description

Gemma-2 is the heaviest re-skin of the Llama baseline (303): it keeps the same
attention machinery (GQA + rotate-half RoPE) but changes the **norm**, the
**activation**, the **scale**, and adds **soft-capping** and **alternating masks**.
No new attention mechanism — just a bundle of localized deltas, each of which you
implement and then assemble into a runnable `gemma_forward → logits`.

Two new operators plus the whole-model assembly:

1. **`softcap(x, cap)`** — `cap * tanh(x / cap)`. Applied twice: to the attention
   logits before softmax (`attn_logit_softcapping`) and to the final lm_head logits
   (`final_logit_softcapping`).
2. **`geglu_ffn(x, params)`** — GeGLU MLP with the **GELU-tanh** activation
   (`gelu_pytorch_tanh`), *not* SiLU.
3. **`GemmaConfig` / `GemmaParams` / `load_gemma` / `gemma_forward`** — the Gemma-2
   decoder-only model.

**GIVEN — the Gemma-2 wrinkles** (each is architecture-as-spec):

- **√d embedding scale** — after `embedding`, multiply the hidden state by
  `sqrt(hidden_size)`.
- **`(1+w)` RMSNorm** — Gemma normalizes with gain `(1 + weight)`, i.e.
  `(1 + w) * (x / rms(x))`, *not* the plain `w * (x / rms(x))` of Llama.
- **Sandwich norm** — **four** RMSNorms per layer: `input_layernorm` before
  attention, `post_attention_layernorm` on the attention output (before the
  residual add), `pre_feedforward_layernorm` before the FFN, and
  `post_feedforward_layernorm` on the FFN output (before its residual add).
- **GeGLU FFN** — GELU-tanh gate, not SiLU.
- **Attention logit soft-cap** — applied to the pre-softmax scores.
- **Final logit soft-cap** — applied to the lm_head output.
- **`query_pre_attn_scalar`** — the attention scale is `query_pre_attn_scalar ** -0.5`,
  which may differ from `head_dim ** -0.5`.
- **Alternating SWA/full layers** — even-indexed layers (0, 2, …) use a
  sliding-window causal mask (`sliding_window_mask`, reused from 305); odd-indexed
  layers use full causal.
- **Tied embeddings** — `lm_head.weight` is absent; reuse `model.embed_tokens.weight`
  as the output projection.

Reuse: `embedding` (201), `rope_half` (213), `sliding_window_mask` (305),
`softmax` (005).

> **→ L4:** this task implements the forward-pass arithmetic only. The
> **sliding-window KV-cache eviction** for the SWA layers (incrementally dropping
> keys that fall outside each sliding layer's window during decode) is an
> inference-systems concern, deferred to Level 4.

---

## The Math

### `softcap`

```
softcap(x, c) = c * tanh(x / c)
```

For `|x| ≪ c` this is ≈ `x` (near-linear); as `|x| → ∞` it saturates toward `±c`.

### GeGLU FFN

```
GELU_tanh(z) = 0.5 * z * (1 + tanh( sqrt(2/π) * (z + 0.044715 * z³) ))

GeGLU-FFN(x) = down_proj( GELU_tanh(x @ gate_proj.T) ⊙ (x @ up_proj.T) )
```

### `(1+w)` RMSNorm

```
rms(x)            = sqrt( mean(x², axis=-1) + eps )
gemma_rms_norm(x) = (1 + w) * (x / rms(x))
```

### Whole-model `gemma_forward`

```
h = embedding(input_ids, tok_embed) * sqrt(dim)

for i, layer in enumerate(layers):
    # attention sub-block (sandwich)
    a = gemma_rms_norm(h, input_layernorm)
    q = (a @ q_proj.T) → (B, H,   L, head_dim) ; rope_half(q, positions)
    k = (a @ k_proj.T) → (B, KVH, L, head_dim) ; rope_half(k, positions)
    v = (a @ v_proj.T) → (B, KVH, L, head_dim)
    k, v = repeat_kv(k, v, H // KVH)                      # GQA
    scores = (q @ k.T) * (query_pre_attn_scalar ** -0.5)
    scores = softcap(scores, attn_logit_softcapping)
    scores += (sliding_window_mask if i even else full causal)
    a = softmax(scores) @ v  → merge heads → @ o_proj.T
    a = gemma_rms_norm(a, post_attention_layernorm)
    h = h + a                                              # residual 1

    # FFN sub-block (sandwich)
    f = gemma_rms_norm(h, pre_feedforward_layernorm)
    f = geglu_ffn(f, {gate_proj, up_proj, down_proj})
    f = gemma_rms_norm(f, post_feedforward_layernorm)
    h = h + f                                              # residual 2

h = gemma_rms_norm(h, final_norm)
logits = h @ tok_embed.T                                  # tied lm_head
logits = softcap(logits, final_logit_softcapping)
```

**Note:** the attention math composes from granular primitives (rotate-half RoPE,
explicit scores → softcap → mask → softmax), *not* from `llama_decoder_block` (216,
which is interleaved-RoPE only).

---

## GIVEN — HF weight map (`load_gemma`)

Gemma-2 uses rotate-half RoPE, so weights map **as-is** (no un-permute):

```
model.embed_tokens.weight                       (V, d)   # also the tied lm_head
model.norm.weight                               (d,)
                                                         # lm_head.weight ABSENT (tied)

model.layers.{i}.input_layernorm.weight             (d,)
model.layers.{i}.post_attention_layernorm.weight    (d,)
model.layers.{i}.pre_feedforward_layernorm.weight   (d,)
model.layers.{i}.post_feedforward_layernorm.weight  (d,)
model.layers.{i}.self_attn.q_proj.weight            (n_heads    * head_dim, d)
model.layers.{i}.self_attn.k_proj.weight            (n_kv_heads * head_dim, d)
model.layers.{i}.self_attn.v_proj.weight            (n_kv_heads * head_dim, d)
model.layers.{i}.self_attn.o_proj.weight            (d, n_heads * head_dim)
model.layers.{i}.mlp.gate_proj.weight               (intermediate_size, d)
model.layers.{i}.mlp.up_proj.weight                 (intermediate_size, d)
model.layers.{i}.mlp.down_proj.weight               (d, intermediate_size)
```

---

## Function Signatures

```python
def softcap(x: np.ndarray, cap: float) -> np.ndarray: ...

def geglu_ffn(x: np.ndarray, params: GeGLUParams) -> np.ndarray: ...   # (B, L, d) -> (B, L, d)

def load_gemma(weights: dict, cfg: GemmaConfig) -> GemmaParams: ...

def gemma_forward(
    input_ids: np.ndarray,   # (B, L)
    params: GemmaParams,
    cfg: GemmaConfig,
    start_pos: int = 0,
) -> np.ndarray:             # (B, L, vocab_size)
```

---

## Read More

- Gemma-2 technical report: https://arxiv.org/abs/2408.00118
- `modeling_gemma2.py` in the transformers package (exact arithmetic reference)
- `303_llama_model/` — the Llama baseline this re-skins
- `305_sliding_window_attention/` — the band mask reused for the even (sliding) layers
- `214_swiglu/` — the SwiGLU FFN that GeGLU contrasts with (GELU-tanh vs SiLU gate)

**Real-weights layer (B):** `download.sh` fetches
`hf-internal-testing/tiny-random-Gemma2ForCausalLM` (config + safetensors only),
maps the HF names → `GemmaParams`, verifies `gemma_forward` against a genuine
`Gemma2ForCausalLM` (eager attention), and commits `tests/fixtures/real_ref.npz`.
The checkpoint's `sliding_window=4096` is larger than the test sequence, so the
band reduces to full causal there — the alternating sliding/full behavior is
exercised by the always-on hermetic 2-layer fixture (one sliding + one full layer)
and by the mask-isolation test.

---

## How to Test

```bash
uv run grade 309                  # grade your implementation
uv run grade 309 -v               # verbose output

# optional: real-weights parity (downloads a tiny checkpoint, ~MBs)
bash 309_gemma_model/download.sh
```

The test suite checks:

- `test_softcap_matches_oracle` / `test_softcap_saturates` /
  `test_softcap_near_linear_small`: `softcap` vs the closed form, saturation, and
  near-linear regime.
- `test_geglu_matches_oracle` / `test_geglu_uses_gelu_not_silu`: GeGLU vs a float64
  oracle and that the gate is GELU-tanh (not SiLU).
- `test_gemma_logits_match_oracle`: whole-model logit parity vs the committed float64
  oracle at `rtol=1e-9`.
- `test_gemma_logits_shape` / `test_gemma_causal`: output shape and causal masking.
- `test_gemma_final_logits_softcapped`: logits bounded by the final soft-cap.
- `test_one_plus_w_rmsnorm`: the norm uses the `(1+w)` gain (not plain `w*`).
- `test_sqrt_d_embedding_scale`: the `sqrt(dim)` embedding scale is applied.
- `test_alternating_sliding_full_masks`: even layers consult the sliding window.
- `test_attention_softcap_active`: the attention soft-cap affects the output.
- `test_gemma_real_weights_logits` *(skipped until `download.sh` runs)*: parity vs a
  genuine `Gemma2ForCausalLM` on real weights.
