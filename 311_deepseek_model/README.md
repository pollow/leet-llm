# 311 — Multi-head Latent Attention (MLA) + DeepSeek-V3 Whole-Model Forward

## Description

This task starts from the Llama-style decoder assembly you already built and
adds the two DeepSeek-V3 deltas that materially change forward behavior:
MLA (low-rank KV + decoupled RoPE slice) and DeepSeek MoE (sigmoid routing,
selection bias, group-limited top-k, always-on shared experts).

This task is **forward-only arithmetic**. You are not building cache management,
kernel fusion, or serving optimizations here.

### Baseline and delta map

| Component | Baseline behavior | Task delta | Where wired |
|---|---|---|---|
| Attention projection | Full per-head K/V projections | MLA: `kv_a_proj_with_mqa` latent compression + `kv_b_proj` reconstruction | `mla_project` |
| Positional encoding in attention | RoPE applied on whole Q/K heads | Decoupled RoPE: apply `rope_half` only to rope slices (`q_rope`, shared `k_rope`) | `mla_project` |
| FFN per layer | Same FFN type for all layers | Hybrid FFN: dense SwiGLU for early layers, DeepSeek MoE for later layers | `deepseek_forward` |
| MoE routing | Softmax top-k (Mixtral style) | Sigmoid scores + additive selection bias + group-limited top-k | `deepseek_forward` (via `moe_ffn`) |
| Shared experts | Optional/absent in baseline tracks | Always-on shared SwiGLU branch added to routed MoE output | `deepseek_forward` |

### Prerequisite checklist

Before implementing 311, ensure these operators already work in your stack:

- `rms_norm(x, weight, eps)` returns shape-preserving normalized activations.
- `rope_half(x, positions, base)` uses rotate-half convention on last dim.
- `sdpa(q, k, v, mask, scale)` supports causal boolean mask + scaling.
- `swiglu_ffn(x, gate_proj, up_proj, down_proj)` is numerically stable.
- `moe_ffn(...)` supports token-to-expert dispatch with top-k routing.
- `embedding`, `add_residual`, and `triangular_mask` follow prior tasks.

### Out of scope

- Latent-KV cache layout and paged cache updates (L4 topic).
- `rope_interleave=True` / extended YaRN implementations.
- Training/backprop behavior.

---

## The Math

### Step-by-step implementation path

### Step 1 — Implement `mla_project`

**Why add this?**
- Full K/V heads are memory-heavy at long context; MLA reduces KV footprint with
  latent compression while preserving per-head attention behavior after reconstruction.
- Decoupled RoPE lets DeepSeek keep positional signal in a dedicated rope slice.
- Tradeoff: more projection bookkeeping (split/concat/broadcast) and stricter shape checks.

**Purpose**
- Produce one layer's attention output `y` with shape `(B, L, d)` using DeepSeek MLA.

**What (shape contract)**
- Input `x`: `(B, L, d)`.
- Output `y`: `(B, L, d)`.
- Internal invariants:
  - `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim`
  - `kv_a_proj_with_mqa` outputs `(kv_lora_rank + qk_rope_head_dim)` per token.
  - `k_rope` is shared MQA rope key: `(B, 1, L, qk_rope_head_dim)` before broadcast.

**How**

1. Build `q`:
   - direct path: `q = x @ q_proj.T`
   - low-rank path: `qa = x @ q_a_proj.T` -> `rms_norm` -> `q = qa @ q_b_proj.T`
2. Reshape `q -> (B, H, L, qk_head_dim)` and split into `q_nope | q_rope`.
3. Build latent KV:
   - `compressed = x @ kv_a_proj_with_mqa.T`
   - split `compressed -> c_kv | k_rope`
   - `c_kv` -> `rms_norm` -> `kv_b_proj` -> reshape -> split `k_nope | v`
4. Apply `rope_half` **only** to `q_rope` and `k_rope`.
5. Broadcast `k_rope` from `(B, 1, L, qk_rope_head_dim)` to `(B, H, L, qk_rope_head_dim)`.
6. Concatenate:
   - `q_full = concat(q_nope, q_rope)`
   - `k_full = concat(k_nope, k_rope)`
7. Run causal SDPA with `scale = qk_head_dim**(-0.5)` (plus mscale adjustment for yarn configs).
8. Merge heads and apply `o_proj`.

**Check**
- `mla_project` output shape is exactly `(B, L, d)`.
- Changing non-uniform position spacing (e.g. `[0,1,2,3]` vs `[0,2,4,6]`) changes output.
- Perturbing `kv_b_proj` changes output (proves latent path is active).

### Step 2 — Implement DeepSeek MoE branch in `deepseek_forward`

**Why add this?**
- DeepSeek routing uses sigmoid confidence and group-limited candidate search for stable,
  sparse expert usage.
- Shared experts provide an always-on dense capacity path to reduce routing brittleness.
- Tradeoff: routing is more complex than plain top-k and easier to get subtly wrong.

**Purpose**
- For MoE layers (`layer_index >= first_k_dense_replace`), produce FFN output that matches
  DeepSeek's routing semantics.

**What (shape contract)**
- Flatten hidden states to `(T, d)` where `T = B * L`.
- Router scores shape `(T, n_routed_experts)`.
- Selected experts per token shape `(T, num_experts_per_tok)`.

**How**

1. Compute sigmoid routing scores:
   - `scores = sigmoid(x @ W_gate.T)`
2. Compute biased scores for selection only:
   - `scores_biased = scores + e_score_correction_bias`
3. Group-limited candidate selection:
   - reshape biased scores to `(T, n_group, experts_per_group)`
   - per group: sum top-2 -> `group_scores`
   - select top `topk_group` groups
   - mask out other groups with `-inf`
   - run expert top-k on masked biased scores
4. Gather **unbiased** selected token weights from `scores`, then:
   - normalize if `norm_topk_prob`
   - multiply by `routed_scaling_factor`
5. Dispatch tokens through routed experts with these weights.
6. Compute shared experts dense SwiGLU output.
7. Return `routed_output + shared_output`.

**Check**
- Zeroing shared expert weights changes model output.
- Zeroing a non-selected routed expert can be a no-op on that fixture.

### Step 3 — Assemble full forward deterministically

**Why add this?**
- Most parity bugs come from correct operators wired in the wrong order.
- Deterministic assembly removes ambiguity and makes failures localizable.

**Purpose**
- Produce logits `(B, L, vocab_size)` with causal behavior.

**What**
- `start_pos` offsets position ids.
- Layer behavior switches from dense to MoE at `first_k_dense_replace`.

**How (integration assembly path)**

1. `h = embedding(input_ids, tok_embed)`
2. `positions = arange(start_pos, start_pos + L)`
3. For each layer `i` in order:
   - `h_attn_in = rms_norm(h, input_layernorm)`
   - `attn_out = mla_project(h_attn_in, layer, cfg, positions)`
   - `h = add_residual(h, attn_out)`
   - `h_ffn_in = rms_norm(h, post_attention_layernorm)`
   - if `i < first_k_dense_replace`: dense `swiglu_ffn`
   - else: DeepSeek MoE branch
   - `h = add_residual(h, ffn_out)`
4. `h = rms_norm(h, final_norm)`
5. `logits = h @ lm_head.T`

**Check**
- Causal invariant: changing the last token must not change earlier logits.

### Cross-task dependency contract

- **Primitive math responsibility**
  - `213`: `rope_half` math.
  - `212`: `rms_norm`.
  - `205`: SDPA behavior.
  - `214`: dense SwiGLU FFN.
  - `308`: routed MoE dispatch core.
- **311 responsibility**
  - DeepSeek-specific MLA split/recombine wiring.
  - DeepSeek-specific routing policy (sigmoid + bias + group top-k + shared experts).
  - Whole-model residual/norm ordering and dense-vs-MoE switch.
- **Compatibility constraints**
  - `qk_head_dim == qk_nope_head_dim + qk_rope_head_dim`
  - `n_routed_experts % n_group == 0`
  - `topk_group <= n_group`
  - `num_experts_per_tok <= n_routed_experts`

---

## Function Signature

```python
def mla_project(
    x: np.ndarray,          # (B, L, d)
    layer: dict,            # per-layer MLA weights from load_deepseek
    cfg: DeepseekConfig,
    positions: np.ndarray,  # (L,) integer position indices
) -> np.ndarray:            # (B, L, d)
```

```python
def deepseek_forward(
    input_ids: np.ndarray,  # (B, L)
    params: DeepseekParams,
    cfg: DeepseekConfig,
    start_pos: int = 0,
) -> np.ndarray:            # (B, L, vocab_size)
```

---

## Read More

- DeepSeek-V3 technical report: https://arxiv.org/abs/2412.19437
- `transformers` reference: `modeling_deepseek_v3.py`
- `215_gqa/`: baseline grouped-query attention wiring
- `308_mixtral_model/`: MoE dispatch baseline for comparison

Tier-B real-weights parity is currently unavailable in this task:
public tiny checkpoints rely on yarn + `rope_interleave=True`, while 311
targets default rotate-half RoPE. See `download.sh` and `convert.py`.

---

## How to Test

```bash
uv run grade 311
uv run grade 311 -v
```

### Verification ladder

1. **Operator-level check**
   - `test_mla_project_matches_oracle`
2. **Cross-task wiring checks**
   - `test_mla_kv_lora_rank_is_compressed`
   - `test_mla_rope_slice_carries_position`
   - `test_moe_shared_experts_always_contribute`
   - `test_moe_non_selected_expert_noop`
3. **Whole-model parity/invariants**
   - `test_deepseek_logits_match_oracle`
   - `test_deepseek_logits_shape`
   - `test_deepseek_causal`
4. **Optional real-weight parity**
   - not enabled until a suitable tiny default-RoPE checkpoint exists

### Debug playbook

| Symptom | Likely cause | First check |
|---|---|---|
| `mla_project` shape mismatch | wrong head reshape/split sizes | verify `qk_head_dim = qk_nope + qk_rope` and transpose order |
| Position test fails (no sensitivity) | RoPE applied to wrong slice or not broadcasted | confirm only `q_rope`/`k_rope` go through `rope_half`; `k_rope` is broadcast to all heads |
| MoE output unstable/NaN | masked scores not using `-inf`, or wrong normalization | inspect group mask and `norm_topk_prob` denominator (`+1e-20`) |
| Shared-expert test no effect | shared branch omitted | confirm final MoE output adds routed + shared outputs |
| Causal test fails | missing/incorrect causal mask in attention | verify SDPA receives a boolean causal mask (`True` means masked) for every layer |
