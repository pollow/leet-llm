# 311 — Multi-head Latent Attention (MLA) + DeepSeek-V3 Whole-Model Forward

## Description

This README is designed as a short lecture for students who completed `310` and
all prerequisites before it. The goal is not to copy formulas, but to understand
which modeling decisions DeepSeek-V3 changes, and then implement them cleanly.

### Learning goals

By the end of this task, you should be able to:

1. Explain why MLA replaces full per-head KV storage with latent KV compression.
2. Explain what "decoupled RoPE slices" means in tensor terms.
3. Implement DeepSeek MoE routing where:
   - selection uses biased scores,
   - weighting uses unbiased scores,
   - shared experts are always active.
4. Wire the full forward pass in a deterministic, testable order.

### From 310 to 311: what changes?

| Component | In 310-style baseline | In 311 (DeepSeek-V3) | Where you code it |
|---|---|---|---|
| Attention projection | Full per-head `K` and `V` directly from hidden states | MLA: latent compression (`kv_a`) then reconstruction (`kv_b`) | `mla_project` |
| Positional channels | RoPE applied to full Q/K head channels | RoPE only on rope slices; nope slices stay content-only | `mla_project` |
| FFN policy | Single FFN style per layer region | Early layers dense SwiGLU, later layers DeepSeek MoE | `deepseek_forward` |
| MoE routing | Standard top-k pattern | Sigmoid + selection bias + group-limited top-k + shared experts | `deepseek_moe_ffn` |

### Prerequisite checklist (must already work)

- `rms_norm` (`212`)
- `sigmoid` (`202`)
- `rope_half` (`213`)
- `sdpa` (`205`) with **boolean causal mask** (`True = blocked`, `False = visible`)
- `swiglu_ffn` (`214`)
- `embedding`, `triangular_mask`, `add_residual` (`201`, `009`, `208`)

### Terminology primer

- **Full KV heads**: each token produces full K and V channels for every head.
  If `H` heads and `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim`, then full
  K width per token is `H * qk_head_dim`; V width is `H * v_head_dim`.
- **Latent KV**: a smaller vector `c_kv` that is later expanded to per-head K/V.
- **RoPE slice** (`qk_rope_head_dim`): channels that carry positional phase (RoPE applied).
- **NOPE slice** (`qk_nope_head_dim`): channels that keep non-positional content features.
- **Decoupled RoPE**: apply RoPE to rope slices only; do not rotate NOPE channels.

### Out of scope

- KV-cache systems and paged attention runtime mechanics (L4 scope).
- `rope_interleave=True` and full YaRN/interleaved variants.
- Training/backward pass.

---

## The Math

We structure this as five teaching units:

- Unit A: Why DeepSeek-V3 changes the baseline
- Unit B: Low-rank Q projection (separate from MLA core)
- Unit C: MLA implementation
- Unit D: DeepSeek MoE routing
- Unit E: Full-model assembly

### Unit A — Why these two deltas exist

**Problem 1: Full KV is expensive.**  
Long-context inference is often bottlenecked by KV memory traffic. MLA keeps a
small latent representation and reconstructs K/V on demand.

**Problem 2: Position and content are entangled.**  
Applying RoPE to all channels mixes positional phase into every feature channel.
DeepSeek separates channels into:
- rope slice (positional), and
- nope slice (content).

This separation gives cleaner control over what is position-sensitive.

---

### Unit B — Low-rank Q projection (prelude to MLA)

DeepSeek-V3/V3.1 mainstream checkpoints use low-rank Q projection:

1. `qa = x @ q_a_proj.T`
2. `qa_norm = rms_norm(qa, q_a_layernorm)`
3. `q = qa_norm @ q_b_proj.T`

What you should implement first:
- Load the low-rank Q weights: `q_a_proj`, `q_a_layernorm`, and `q_b_proj`.
- Implement `_project_q_low_rank`: compute `qa`, normalize it, then project it
  back to full Q width and reshape to `(B, H, L, qk_head_dim)`.
- Treat this as the way 311 builds Q before the MLA split into `q_nope | q_rope`.

The reason to learn this before MLA: it is a small, local projection block. Once
`q` has shape `(B, H, L, qk_head_dim)`, the rest of MLA can treat it like any
other attention query tensor.

Direct path supplement (for comparison only):
- Some reference implementations expose `q = x @ q_proj.T` as an optional fallback.
- It is **not** the primary path for this task's teaching flow.

---

### Unit C — MLA (`mla_project`)

#### Why add MLA?

- Reduces KV storage footprint.
- Keeps per-head attention behavior by reconstructing head-wise tensors.
- Adds explicit control over positional channels via decoupled RoPE.

#### Purpose

Given `x: (B, L, d)`, return attention output `y: (B, L, d)`.

#### What (contracts)

- `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim`
- `kv_a_proj_with_mqa` is one linear projection from the hidden state `x`.
  For each token, it produces one combined vector:
  `compressed = x @ kv_a_proj_with_mqa.T`.
- The last dimension of `compressed` has width `kv_lora_rank + qk_rope_head_dim`.
  Split that last dimension into two slices:
  - `c_kv = compressed[..., :kv_lora_rank]`
  - `k_rope = compressed[..., kv_lora_rank:]`
- `c_kv` is the compressed content latent. It is not yet per-head K/V; you must
  normalize it and pass it through `kv_b_proj` to reconstruct per-head `k_nope`
  and `v`.
- `k_rope` is the shared positional key slice. It is already the right RoPE
  width, but it is shared across heads, so later you reshape it to
  `(B, 1, L, qk_rope_head_dim)` and broadcast across `H` heads.
- `k_rope` is shared across heads before broadcast: `(B, 1, L, qk_rope_head_dim)`

#### Mini numeric example

Suppose:
- `H=4`, `qk_nope_head_dim=8`, `qk_rope_head_dim=4`, `v_head_dim=8`, `kv_lora_rank=8`

Then:
- baseline full K width per token: `4 * (8 + 4) = 48`
- baseline full V width per token: `4 * 8 = 32`
- MLA latent storage before reconstruction: `c_kv(8) + k_rope(4) = 12`

This is the core compression intuition.

#### How (ordered implementation steps)

1. Build Q branch:
   - `q = _project_q_low_rank(x, layer, cfg)`
2. Split `q` into `q_nope | q_rope`.
3. Build KV latent branch:
   - `compressed = x @ kv_a_proj_with_mqa.T`
   - split `compressed -> c_kv | k_rope`
   - `c_kv -> rms_norm -> kv_b_proj` to get per-head `[k_nope, v]`
4. Apply `rope_half` only to `q_rope` and `k_rope`.
5. Broadcast `k_rope` from head-shared shape to all heads.
6. Reconstruct full tensors:
   - `q_full = concat(q_nope, q_rope)`
   - `k_full = concat(k_nope, k_rope)`
7. Run SDPA with:
   - `sdpa(q_full, k_full, v, causal_bool_mask)`.
   - `sdpa` applies `1 / sqrt(qk_head_dim)` internally.
   - The mask is boolean (`True = blocked`).
8. Merge heads and project with `o_proj`.

#### Check (before moving on)

- `mla_project` output shape is `(B, L, d)`.
- Non-uniform position spacing changes output.
- Perturbing `kv_b_proj` changes output (confirms latent up-projection path is active).

---

### Unit D — DeepSeek MoE (`deepseek_moe_ffn`)

#### Why this routing design?

- Sigmoid routing gives independent confidence per expert.
- Selection bias helps choose experts without directly distorting final token weights.
- Group-limited selection constrains candidate search.
- Shared experts provide a stable dense path for every token.

#### Purpose

For layers `i >= first_k_dense_replace`, replace dense FFN with DeepSeek MoE behavior.
Implement this as a separate `deepseek_moe_ffn` operator first, then call it from
`deepseek_forward`.

#### What (contracts)

- Flatten hidden states to `(T, d)` where `T = B * L`.
- Router scores shape: `(T, n_routed_experts)`.
- Selected experts per token: `(T, num_experts_per_tok)`.

#### Mini routing example

Assume:
- `n_routed_experts=8`, `n_group=2`, `topk_group=1`, `num_experts_per_tok=2`

For each token:
1. Compute `scores = sigmoid(x @ W_gate.T)`.
2. Compute `scores_biased = scores + e_score_correction_bias`.
3. Use `scores_biased` for group filtering and top-k selection.
4. Gather token weights from **unbiased** `scores` at selected indices.
5. Normalize and apply `routed_scaling_factor`.
6. Add shared expert dense output.

Critical invariant: **biased scores choose experts; unbiased scores weight experts.**

This is not the same operator as `308`'s `moe_ffn`: Mixtral uses softmax routing
over all experts, while DeepSeek uses sigmoid scores, selection bias, group-limited
top-k, and shared experts.

#### How (ordered implementation steps)

1. Compute scores with `sigmoid(x @ gate.T)`.
2. Add selection bias only for routing decisions.
3. Group-limited selection:
   - reshape by groups,
   - use `top_k` to score groups,
   - use `top_k` again to keep top groups,
   - mask other groups out,
   - use `top_k` on the masked scores to select top experts.
4. Gather unbiased selected weights.
5. Normalize and scale weights.
6. Dispatch to routed experts.
7. Compute shared experts branch and add it.

#### Check

- Zeroing shared expert weights changes output.
- Zeroing a non-selected routed expert can be no-op on the tiny fixture.
- Changing selection bias should mainly change selected indices, not the weighting formula.

---

### Unit E — Whole-model assembly (`deepseek_forward`)

#### Why emphasize wiring order?

Most parity bugs in L3 whole-model tasks are wiring mistakes, not operator math mistakes.
Use one deterministic order and do not improvise.

#### Deterministic integration path

1. `h = embedding(input_ids, tok_embed)`
2. `positions = arange(0, L)`
3. `causal_bool_mask = triu(ones((L, L), bool), k=1)`
4. For each layer `i`:
   - `h_attn_in = rms_norm(h, input_layernorm)`
   - `attn_out = mla_project(h_attn_in, layer, cfg, positions, causal_bool_mask)`
   - `h = add_residual(h, attn_out)`
   - `h_ffn_in = rms_norm(h, post_attention_layernorm)`
   - if `i < first_k_dense_replace`: dense `swiglu_ffn`
   - else: `deepseek_moe_ffn(h_ffn_in, layer, cfg)`
   - `h = add_residual(h, ffn_out)`
5. `h = rms_norm(h, final_norm)`
6. `logits = h @ lm_head.T`

#### Cross-task dependency contract

| Primitive | Implemented in | How 311 uses it |
|---|---|---|
| `rms_norm` | `212` | pre-attn, post-attn, final norm |
| `rope_half` | `213` | only rope slices (`q_rope`, `k_rope`) |
| SDPA + mask policy | `205` + `009` | scaled attention with boolean causal mask |
| `swiglu_ffn` | `214` | dense branch for early layers and shared experts |

#### Compatibility constraints

- `qk_head_dim == qk_nope_head_dim + qk_rope_head_dim`
- `n_routed_experts % n_group == 0`
- `topk_group <= n_group`
- `num_experts_per_tok <= n_routed_experts`

---

## Function Signature

```python
def _project_q_low_rank(
    x: np.ndarray,      # (B, L, d)
    layer: dict,        # contains q_a_proj, q_a_layernorm, q_b_proj
    cfg: DeepseekConfig,
) -> np.ndarray:        # (B, H, L, qk_head_dim)
```

This is a private helper for the task, not a registry export. Implement it first,
then call it from `mla_project`.

```python
def load_deepseek(
    weights: dict,
    cfg: DeepseekConfig,
) -> DeepseekParams:
```

`load_deepseek` maps HF-style weight names into the internal per-layer dicts.
For Q projection, 311 expects the low-rank path only:

- `model.layers.{i}.self_attn.q_a_proj.weight`
- `model.layers.{i}.self_attn.q_a_layernorm.weight`
- `model.layers.{i}.self_attn.q_b_proj.weight`

```python
def mla_project(
    x: np.ndarray,          # (B, L, d)
    layer: dict,            # per-layer MLA weights from load_deepseek
    cfg: DeepseekConfig,
    positions: np.ndarray,  # (L,) integer positions
    mask: np.ndarray,       # (L, L) boolean mask; True = blocked
) -> np.ndarray:            # (B, L, d)
```

```python
def deepseek_moe_ffn(
    x: np.ndarray,          # (B, L, d)
    layer: dict,            # per-layer MoE weights from load_deepseek
    cfg: DeepseekConfig,
) -> np.ndarray:            # (B, L, d)
```

```python
def deepseek_forward(
    input_ids: np.ndarray,  # (B, L)
    params: DeepseekParams,
    cfg: DeepseekConfig,
) -> np.ndarray:            # (B, L, vocab_size)
```

---

## Read More

- DeepSeek-V3 report: https://arxiv.org/abs/2412.19437
  - Focus on MLA and MoE routing design motivations.
- `transformers` reference: `modeling_deepseek_v3.py`
  - Focus on tensor shapes and routing path details.
- `215_gqa/`
  - Use as a baseline attention wiring comparison.
- `308_mixtral_model/`
  - Compare MoE dispatch core; then add DeepSeek-specific routing logic.

Tier-B real-weights parity is currently unavailable for this task:
available tiny checkpoints rely on yarn + `rope_interleave=True`, while 311
teaches default rotate-half RoPE only. See `download.sh` and `convert.py`.

---

## How to Test

```bash
uv run grade 311
uv run grade 311 -v
```

### Verification ladder (run in this order)

1. **Operator parity**
   - `test_mla_project_matches_oracle`
   - `test_deepseek_moe_ffn_matches_oracle`
2. **Cross-task wiring checks**
   - `test_mla_kv_lora_rank_is_compressed`
   - `test_mla_rope_slice_carries_position`
   - `test_moe_shared_experts_always_contribute`
   - `test_moe_non_selected_expert_noop`
3. **Whole-model checks**
   - `test_deepseek_logits_match_oracle`
   - `test_deepseek_logits_shape`
   - `test_deepseek_causal`
4. **Optional real-weight parity**
   - enabled only when a suitable tiny default-RoPE checkpoint exists

### Debug playbook

| Symptom | Likely cause | First check |
|---|---|---|
| `mla_project` shape mismatch | wrong split/reshape order | verify `qk_head_dim`, transpose order, and concat axes |
| position-sensitive tests fail | RoPE applied to wrong channels | confirm only rope slices are rotated |
| MoE behavior unstable | group mask or normalization bug | inspect selected indices and normalized weights per token |
| shared-expert test no effect | shared branch not added | verify final FFN output is `routed + shared` |
| causal test fails | mask policy mismatch | confirm boolean causal mask (`True=blocked`) reaches every layer |
