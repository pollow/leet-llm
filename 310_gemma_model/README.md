# 310 — GeGLU + Soft-cap + Sandwich Norm: Gemma-2 Whole-Model Forward

## Description

### 1) Orientation

Task 310 starts from the Llama-style decoder baseline (303): GQA attention, rotate-half RoPE, causal decoder wiring. Gemma-2 keeps that backbone, then adds localized deltas in norm, FFN activation, scaling, masking policy, and soft-capping. Your implementation target is two new operators (`softcap`, `geglu_ffn`) plus full-model wiring (`load_gemma`, `gemma_forward`).

Out of scope: serving-time KV-cache eviction policy for sliding-window layers (Level 4 systems topic).

### 2) Baseline and delta map

| Component | Baseline behavior | Task 310 delta | Where wired |
|---|---|---|---|
| Embedding output | `h = embedding(ids, tok_embed)` | multiply by `sqrt(dim)` | `gemma_forward` prologue |
| RMSNorm gain | Llama-style `w * (x / rms)` | Gemma-style `(1 + w) * (x / rms)` | every norm site in `gemma_forward` |
| Layer norm layout | pre-attn + pre-ffn | sandwich norm (4 norms/layer) | attn and ffn sub-blocks |
| FFN nonlinearity | SwiGLU / SiLU gate | GeGLU + GELU-tanh gate | `geglu_ffn` |
| Attention score scaling | often `head_dim**-0.5` | `query_pre_attn_scalar**-0.5` | score computation |
| Attention logits | raw scores -> mask -> softmax | apply `softcap` before mask/softmax | per-layer attention |
| Mask policy | all full-causal or all sliding | even layers sliding window, odd layers full causal | per-layer mask branch |
| Output projection | separate `lm_head.weight` in many models | tied embedding projection (`tok_embed.T`) | final logits |
| Final logits | unbounded | `softcap(logits, final_logit_softcapping)` | output tail |

Non-obvious design knobs (why + tradeoff):

- `query_pre_attn_scalar` decouples attention temperature from raw `head_dim`, so checkpoint behavior is preserved even when scale conventions differ. Tradeoff: one more config-dependent constant that is easy to wire incorrectly.
- alternating sliding/full masks reduce average attention cost while keeping periodic global mixing from full-causal layers. Tradeoff: even (sliding) layers cannot see all far-past tokens.
- sandwich norm stabilizes both sub-block input and sub-block output scales before each residual merge. Tradeoff: more norm calls and stricter ordering constraints than simpler pre-norm layouts.

### 3) Prerequisite checklist

You should already have these operators correct before implementing 310:

- `embedding` (201): token ids `(B, L)` -> hidden `(B, L, d)`.
- `softmax` (005): numerically stable, along last axis for attention probabilities.
- `rope_half` (213): rotate-half RoPE for q/k tensors.
- GQA wiring from earlier tasks (especially 215): split heads, repeat kv heads to query-head count.
- `sliding_window_mask` (305): boolean mask for local causal attention (`True` = masked out).

If one prerequisite is shaky, fix it first; 310 composes all of them.

### 4) Step-by-step implementation path

#### Step A — `softcap`

- **Why add this?** Gemma constrains extreme logits without hard clipping. This stabilizes score/logit magnitude while keeping gradients smooth (`tanh`). Tradeoff: slight compression of very confident logits.
- **Purpose:** bound attention logits and final logits with one reusable operator.
- **What:** input/output same shape; scalar `cap` controls saturation range.
- **How:** `softcap(x, cap) = cap * tanh(x / cap)`.
- **Check:** large `|x|` saturates near `±cap`; small `|x|` is near-linear.

#### Step B — `geglu_ffn`

- **Why add this?** Gemma uses GeGLU with GELU-tanh gate instead of SiLU-based SwiGLU. This changes FFN behavior and must match checkpoint arithmetic.
- **Purpose:** implement Gemma FFN block with correct gate function and projection order.
- **What:** `x: (B, L, d)` -> output `(B, L, d)` using `gate/up/down` weights.
- **How:** `g = GELU_tanh(x @ gate.T)`, `u = x @ up.T`, output `(g * u) @ down.T`.
- **Check:** matches oracle formula; output differs from a SiLU-gate implementation.

#### Step C — `load_gemma`

- **Why add this?** correct whole-model math still fails if HF tensor names map to wrong slots.
- **Purpose:** deterministic mapping from HF state dict to `GemmaParams`.
- **What:** collect global tensors (`embed`, `final_norm`) plus per-layer norms/attn/mlp weights.
- **How:** map keys directly (no RoPE un-permute); tie lm head to `model.embed_tokens.weight`.
- **Note:** do not load a separate `lm_head.weight`; final projection uses `model.embed_tokens.weight.T`.
- **Check:** all expected keys present; per-layer tensor shapes align with config.

#### Step D — `gemma_forward`

- **Why add this?** local operators passing unit tests is not enough; parity depends on exact order of norm, residual, mask policy, and scaling constants.
- **Purpose:** assemble Gemma-2 forward pass from reusable primitives in a deterministic order.
- **What:** `input_ids (B, L)` -> `logits (B, L, vocab_size)`.
- **How:** follow the integration order in section 6 exactly.
- **Check:** tiny fixture parity (`rtol=1e-9`), causal invariants, soft-cap and alternating-mask effect tests.

### 5) Cross-task dependency contract

| Primitive | Implemented in | Consumed in 310 | Compatibility constraints |
|---|---|---|---|
| `embedding` | 201 | forward input stage | ids are integer token indices |
| `rope_half` | 213 | q/k after projection | rotate-half layout; positions are `arange(0, L)` |
| `softmax` | 005 | attention probabilities | apply on masked score axis (last dim) |
| `sliding_window_mask` | 305 | even layers only | boolean mask with `True` meaning forbidden attention |
| GQA head repeat | 215 path | attention core | `n_heads % n_kv_heads == 0` |

Task 310 owns: `softcap`, `geglu_ffn`, `load_gemma`, and the whole forward assembly.

### 6) Integration assembly path (deterministic)

Use this exact wiring order:

1. `h = embedding(input_ids, tok_embed) * sqrt(dim)`.
2. For each layer `i`:
   - `a_in = gemma_rms_norm(h, input_layernorm)` using `(1+w)`.
   - q/k/v projections from `a_in`, then head reshape.
   - build positions as `positions = arange(0, L)` for RoPE.
   - apply `rope_half` to q and k.
   - repeat kv heads for GQA.
   - compute scores with `query_pre_attn_scalar**-0.5`.
   - apply attention `softcap`.
   - apply boolean mask: even `i` -> sliding-window causal, odd `i` -> full causal.
   - set masked score entries to `-inf` before `softmax`.
   - `softmax(scores)` then weighted sum with v, merge heads, `o_proj`.
   - post-attn norm, then first residual add.
   - pre-ffn norm, `geglu_ffn`, post-ffn norm, second residual add.
3. Final norm with `(1+w)`.
4. Logits via tied projection: `h @ tok_embed.T`.
5. Final logits `softcap`.

Do not collapse this into `llama_decoder_block` (216); 310 requires explicit Gemma-specific arithmetic.

### 7) Out-of-scope boundary

Not required in 310:

- KV-cache eviction mechanics for sliding-window decode.
- serving/runtime optimizations (paged attention, tensor parallelism, etc.).
- training-time concerns (backprop, optimizer behavior).

---

## The Math

### `softcap`

```
softcap(x, c) = c * tanh(x / c)
```

- `|x| << c`: approximately linear.
- `|x| -> large`: saturates toward `±c`.

### GeGLU FFN

```
GELU_tanh(z) = 0.5 * z * (1 + tanh( sqrt(2/pi) * (z + 0.044715 * z^3) ))

GeGLU(x) = down_proj( GELU_tanh(x @ gate_proj.T) * (x @ up_proj.T) )
```

### Gemma `(1+w)` RMSNorm

```
rms(x) = sqrt(mean(x^2, axis=-1) + eps)
gemma_rms_norm(x, w) = (1 + w) * (x / rms(x))
```

### Attention core with Gemma deltas

```
scores = (q @ k^T) * (query_pre_attn_scalar ** -0.5)
scores = softcap(scores, attn_logit_softcapping)
scores = where(mask, -inf, scores)   # bool mask: True means masked
attn   = softmax(scores)
out    = attn @ v
```

Mask ordering note: apply mask after softcap. If you softcap already-masked large negatives, forbidden positions can be unintentionally de-magnified.

---

## Function Signature

```python
def softcap(x: np.ndarray, cap: float) -> np.ndarray: ...

def geglu_ffn(x: np.ndarray, params: GeGLUParams) -> np.ndarray: ...   # (B, L, d) -> (B, L, d)

def load_gemma(weights: dict, cfg: GemmaConfig) -> GemmaParams: ...

def gemma_forward(
    input_ids: np.ndarray,   # (B, L)
    params: GemmaParams,
    cfg: GemmaConfig,
) -> np.ndarray:             # (B, L, vocab_size)
```

`gemma_rms_norm` in this README is an internal helper name (or inlined math), not an additional required public API.

---

## Read More

- Gemma-2 technical report: https://arxiv.org/abs/2408.00118
- `transformers` reference: `modeling_gemma2.py`
- `303_llama_model/` (baseline mental model)
- `305_sliding_window_attention/` (mask primitive reused in even layers)
- `214_swiglu/` (contrast with GeGLU gate choice)

Optional real-weights layer:

- `bash 310_gemma_model/download.sh` fetches `hf-internal-testing/tiny-random-Gemma2ForCausalLM`.
- `convert.py` maps HF weights and saves `tests/fixtures/real_ref.npz`.
- `test_gemma_real_weights_logits` then compares your forward vs genuine HF logits.

---

## How to Test

```bash
uv run grade 310
uv run grade 310 -v
uv run pytest 310_gemma_model/tests/test_gemma.py -k "softcap or geglu"  # fast local loop

# optional real-weights parity
bash 310_gemma_model/download.sh
```

### Verification ladder (run in order)

1. **Operator checks**
   - `test_softcap_matches_oracle`
   - `test_softcap_saturates`
   - `test_softcap_near_linear_small`
   - `test_geglu_matches_oracle`
   - `test_geglu_uses_gelu_not_silu`
2. **Cross-task wiring checks**
   - `test_alternating_sliding_full_masks`
   - `test_attention_softcap_active`
3. **Whole-model parity and invariants**
   - `test_gemma_logits_match_oracle`
   - `test_gemma_logits_shape`
   - `test_gemma_causal`
   - `test_one_plus_w_rmsnorm`
   - `test_gemma_final_logits_softcapped`
4. **Optional real-weight parity**
   - `test_gemma_real_weights_logits` (after `download.sh`)

### Debug playbook (symptom -> likely cause -> first check)

- `softcap` tests fail -> formula wrong or cap misplaced -> compare output against `cap * tanh(x/c)` on a tiny array.
- GeGLU close-but-wrong -> accidentally used SiLU or wrong projection order -> verify `GELU_tanh(gate)` then multiply by `up`.
- parity fails, shape is correct -> forward order mismatch -> re-check section 6 order, especially post-attn/post-ffn norm placement.
- causal test fails -> mask broadcast/triangular logic broken -> inspect mask shape and future-token entries.
- alternating-mask test fails -> same mask path used for all layers -> branch on layer index parity.
- real-weight test fails only -> loader mapping issue -> audit `load_gemma` key-to-slot mapping and tied embedding behavior.
