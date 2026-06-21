# 311 — Multi-head Latent Attention (MLA) + DeepSeek-V3 Whole-Model Forward

## Description

DeepSeek-V3 introduces two architectural innovations over the Llama baseline:

1. **Multi-head Latent Attention (MLA)** — low-rank KV compression with a
   *decoupled RoPE* slice. Instead of storing full K/V tensors per head, MLA
   projects to a compact latent vector and reconstructs K and V on the fly.
   A small shared `k_rope` slice carries position (applied with rotate-half RoPE);
   the `nope` (no-positional-encoding) slice carries content.

2. **DeepSeek MoE** — sigmoid gating + per-expert selection bias +
   *group-limited top-k* routing + always-on shared experts.  The first
   `first_k_dense_replace` layers use a plain dense SwiGLU MLP; all later
   layers use this MoE.

Reuse: `moe_ffn` (308), `rope_half` (213), `rms_norm` (212), `softmax` (005),
`swiglu_ffn` (214), `embedding` (201), `triangular_mask` (009), `add_residual` (208).

---

## The Math

### MLA — Multi-head Latent Attention

**Q projection** (two variants):
- *Direct* (when `q_lora_rank` is `None`):
  `q = x @ W_q.T` → reshape to `(B, H, L, qk_head_dim)`
- *Low-rank* (when `q_lora_rank` is set):
  `qa = x @ W_qa.T`, `qa = rms_norm(qa, W_qa_norm)`, `q = qa @ W_qb.T`

Split: `q = [q_nope (qk_nope_head_dim) | q_rope (qk_rope_head_dim)]`

**KV down-projection** (`W_kv_a` is `kv_a_proj_with_mqa`):
```
compressed = x @ W_kv_a.T          # (B, L, kv_lora_rank + qk_rope_head_dim)
c_kv   = compressed[..., :kv_lora_rank]        # latent
k_rope = compressed[..., kv_lora_rank:]        # shared (MQA), shape (B, L, qk_rope_head_dim)
```

**KV up-projection**:
```
c_kv_norm = rms_norm(c_kv, W_kv_a_norm)
kv = c_kv_norm @ W_kv_b.T                     # (B, L, H*(qk_nope_head_dim + v_head_dim))
k_nope, v = split(kv, [qk_nope_head_dim, v_head_dim], axis=-1)
```

**Decoupled RoPE** — apply `rope_half` ONLY to the rope slices:
```
k_rope → (B, 1, L, qk_rope_head_dim)          # MQA: 1 shared rope key
q_rope = rope_half(q_rope, positions, rope_base)
k_rope = rope_half(k_rope, positions, rope_base)
k_rope = broadcast(k_rope, (B, H, L, qk_rope_head_dim))   # expand to all heads
```

**Full Q, K**:
```
q = concat([q_nope, q_rope], axis=-1)          # (B, H, L, qk_head_dim)
k = concat([k_nope, k_rope], axis=-1)          # (B, H, L, qk_head_dim)
```

**Attention**:
```
scale = qk_head_dim ** (-0.5)
# For Yarn RoPE: scale *= mscale^2, where mscale = 0.1*mscale_all_dim*log(factor) + 1.0
scores = (q @ k.T) * scale + causal_mask       # (B, H, L, L)
attn = softmax(scores) @ v                     # (B, H, L, v_head_dim)
output = reshape(attn, (B, L, H*v_head_dim)) @ W_o.T
```

**Latent-KV cache** is out of scope (deferred to L4). Forward arithmetic only.

---

### DeepSeek MoE

**Given (GIVENs you must match exactly)**:

1. **Sigmoid gating**: `scores = sigmoid(x @ W_gate.T)`

2. **Additive selection bias**: `scores_biased = scores + e_score_correction_bias`
   (used for routing decisions only, NOT for the final token weights)

3. **Group top-k selection**:
   - Reshape biased scores to `(T, n_group, experts_per_group)`
   - Take top-2 per group, sum → group_scores `(T, n_group)`
   - Select `topk_group` groups → group mask
   - Expand mask to per-expert and mask out unselected groups with `-inf`
   - Select top-k experts from the masked biased scores

4. **Token weights**: gather the UNBIASED sigmoid scores for selected experts:
   ```
   topk_weights = scores[tokens, topk_indices]   # unbiased sigmoid
   if norm_topk_prob:
       topk_weights /= sum(topk_weights) + 1e-20
   topk_weights *= routed_scaling_factor
   ```

5. **Shared experts** (always on, dense SwiGLU):
   ```
   output = routed_expert_output + shared_experts_swiglu(x)
   ```
   `shared_experts` has `intermediate_size = n_shared_experts * moe_intermediate_size`.

6. **Dense layers** (`layer_index < first_k_dense_replace`): use a plain SwiGLU MLP
   (`swiglu_ffn`) with `intermediate_size` as the FFN dimension.

---

## Function Signature

```python
def mla_project(
    x: np.ndarray,          # (B, L, d)
    layer: dict,            # per-layer weight dict from load_deepseek
    cfg: DeepseekConfig,
    positions: np.ndarray,  # (L,) integer position indices
) -> np.ndarray:            # (B, L, d)
```

```python
def deepseek_forward(
    input_ids: np.ndarray,   # (B, L)
    params: DeepseekParams,
    cfg: DeepseekConfig,
    start_pos: int = 0,
) -> np.ndarray:             # (B, L, vocab_size)
```

---

## Read More

- DeepSeek-V3 technical report: https://arxiv.org/abs/2412.19437
- `modeling_deepseek_v3.py` in the transformers package (exact arithmetic reference)
- `215_gqa/` — GQA (DeepSeek MLA is the low-rank generalization)
- `308_mixtral_model/` — Mixtral MoE (DeepSeek MoE is the sigmoid+group-topk variant)

**Real-weights layer (B)**: Both publicly available tiny DeepSeek-V3 checkpoints
(`bzantium/tiny-deepseek-v3` and `hf-internal-testing/tiny-random-DeepseekV3ForCausalLM`)
use Yarn RoPE with `rope_interleave=True`, which requires an extended implementation
beyond the scope of this task (the standard `rope_half` only covers default half-rotate
RoPE). No suitable public tiny checkpoint with default RoPE is available, so the
real-weights parity test is skipped. `download.sh` documents this limitation.

---

## How to Test

```bash
uv run grade 311                  # grade your implementation
uv run grade 311 -v               # verbose output
```

The test suite checks:
- `test_deepseek_logits_match_oracle`: whole-model logit parity vs the committed
  float64 oracle at `rtol=1e-9`
- `test_deepseek_logits_shape`: output shape `(B, L, vocab_size)`
- `test_deepseek_causal`: causal masking invariant
- `test_mla_project_matches_oracle`: `mla_project` unit test vs float64 oracle
- `test_mla_kv_lora_rank_is_compressed`: verifies the low-rank KV path is active
- `test_mla_rope_slice_carries_position`: verifies RoPE sensitivity to position spacings
- `test_moe_shared_experts_always_contribute`: shared experts always affect output
- `test_moe_non_selected_expert_noop`: non-selected expert weights are no-ops
