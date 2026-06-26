## 309 — From Llama baseline to GPT-OSS forward

### Learning objective

Starting from a Llama-style decoder baseline, implement GPT-OSS forward by adding five localized deltas:

1. attention sinks,
2. GPT-OSS MoE,
3. YaRN RoPE wiring,
4. alternating sliding/full attention masks,
5. GPT-OSS loader mapping.

---

### 0) Start from the right baseline

Use this as your mental baseline:

- Attention backbone: Llama/Mistral-style GQA with rotate-half RoPE.
- Residual structure: pre-norm RMSNorm, then attention residual, then FFN residual.
- Decoder-only causal forward: embed -> N blocks -> final norm -> lm head.

Do not use `216_llama_decoder_block` as the implementation template for 309 details.  
Task 309 requires explicit GPT-OSS attention and MoE behavior that is not represented there.

---

### 1) Delta map (what changed vs baseline)

| Component | Baseline | GPT-OSS delta in 309 |
|---|---|---|
| Attention normalization | softmax over keys | softmax over keys plus one sink column, then drop sink |
| Attention projections | often bias-free in Llama/Mistral | q/k/v/o all include bias |
| Attention mask policy | usually full causal or all sliding | even layers sliding-window, odd layers full causal |
| FFN | dense SwiGLU or Mixtral MoE | GPT-OSS-specific MoE (different from 308 in routing + expert math) |
| RoPE schedule | default/llama3 schedule | YaRN schedule + attention scale wiring |
| Head scaling | `1/sqrt(head_dim)` | same formula, but `head_dim` is an explicit config field |

### 2) Step A: `attention_with_sinks`

#### Why add this?

In plain causal softmax, each query must distribute 100% probability over visible tokens, even when none is a strong match.  
At long context, that constraint can force weak token-to-token couplings simply because the probability mass must be assigned somewhere.  
GPT-OSS adds a learned sink channel so the model can represent "none of the above" without inventing a token alignment.

#### Purpose

Add a learned sink channel so each attention row can route part of its mass outside visible keys when that is preferable.

#### What

- Input: `scores` shape `(B, H, L, L)`, optional additive mask, `sink_logits` shape `(H,)`.
- Output: `(B, H, L, L)`; row sums are `< 1` when sink logits are finite.

#### How

1. Apply additive mask to `scores` if present.
2. Broadcast sink logits to `(B, H, L, 1)`.
3. Concatenate as an extra last-axis column -> `(B, H, L, L+1)`.
4. Run softmax over the last axis.
5. Drop the sink column and return the remaining `L` columns.

#### Check before continuing

- finite sink -> row sums `< 1`
- sink `=-inf` -> recovers plain softmax
- masked future positions remain zero

---

### 3) Step B: `gptoss_moe_ffn`

#### Why add this?

GPT-OSS does not reuse Mixtral routing unchanged; it defines a different routing contract.

In Mixtral-style routing, scores are normalized over all experts, then truncated and renormalized.  
In GPT-OSS, routing weights are computed only from selected experts: top-k logits first, softmax over selected logits second.

GPT-OSS also includes router/expert biases and clamps key activation paths in the expert computation.  
For this task, treat those choices as required reference behavior, not optional tuning.

#### Purpose

Match GPT-OSS sparse routing behavior (router bias, selected-topk softmax, interleaved gate/up split, clamped activation path) so whole-model parity matches the reference.

#### What

Token-wise sparse expert routing with GPT-OSS-specific routing and expert math.

#### How

1. Router logits with bias:
   - `router_logits = x @ router_weight.T + router_bias`
2. Select top-k logits and indices.
3. Softmax only over selected top-k logits (not over all experts).
4. For each selected expert:
   - `gate_up = x_t @ gate_up_proj[e] + gate_up_bias[e]` (no transpose)
   - split interleaved:
     - `gate = gate_up[::2]`
     - `up = gate_up[1::2]`
   - clamp:
     - `gate = min(gate, limit)`
     - `up = clip(up, -limit, limit)`
   - activation:
     - `glu = gate * sigmoid(alpha * gate)`
     - `gated = (up + 1) * glu`
   - down projection:
     - `out_e = gated @ down_proj[e] + down_bias[e]`
5. Compute the weighted sum of selected experts using top-k softmax scores.

#### Check before continuing

- output shape equals input shape
- non-selected experts can be zeroed without changing output
- selected gate scores sum to 1
- clamp bounds extreme pre-activation effects

---

### 4) Step C: YaRN explained and wired (core theory section)

- Implement reusable YaRN primitives in `213` (`rope_scaled_freqs`, `rope_attention_scale`).
- Return to `309` and wire those primitives through the `213/215` interfaces.
- Do not duplicate RoPE/YaRN directly inside `309` forward. Keep RoPE logic reusable at the primitive layer.

#### Why add this?

Default RoPE frequencies are tuned for pretraining context ranges.  
When context is extended far beyond that range, direct extrapolation can degrade phase behavior at long distances.

GPT-OSS uses YaRN to reshape the frequency schedule for long-context behavior while preserving useful short-range structure.  
YaRN also introduces attention-scale correction so q/k magnitudes remain calibrated after frequency reshaping.

#### Purpose

Reshape RoPE frequencies for long context while preserving calibrated attention temperature through YaRN scale.

#### 4.1 RoPE recap

For pair index `i` in head dimension:

- `inv_freq[i] = base^(-2i/d)`
- angle at position `p`: `theta(p, i) = p * inv_freq[i]`

RoPE rotates q/k pairs by `theta`, converting absolute position into relative phase interaction.

#### 4.2 Why long context needs scaling

When context length extends far beyond pretraining range, the default frequency progression can produce poor long-distance phase behavior.  
A scaled schedule is needed to preserve useful positional behavior at both short and long ranges.

#### 4.3 YaRN frequency schedule intuition

YaRN blends two regimes:

- **extrapolation-like branch** (original frequencies),
- **interpolation-like branch** (frequencies divided by scaling factor).

A ramp between low/high rotation bands determines each pair's blend between these branches.  
In code, this is the `rope_scaled_freqs(..., scaling={"rope_type": "yarn", ...})` path.

#### 4.4 Why YaRN also has attention scale

GPT-OSS YaRN includes an additional scalar:

- `af = rope_attention_scale(scaling)`

In this repo's GPT-OSS wiring, both q and k are multiplied by `af` after RoPE.  
That scaling adjusts effective attention temperature under the YaRN frequency regime and is part of the reference behavior expected by tests.

#### 4.5 Exact implementation surface across tasks

- Task 213 owns reusable primitives:
  - `rope_scaled_freqs`
  - `rope_attention_scale`
  - `rope_from_freqs`
- Task 215 provides the RoPE hook path in GQA call sites.
- Task 309 consumes them in whole-model forward:
  - `inv_freq = rope_scaled_freqs(head_dim, rope_base, cfg.rope_scaling)`
  - `af = rope_attention_scale(cfg.rope_scaling)`
  - `q = rope_from_freqs(q, positions, inv_freq, pair_type="half") * af`
  - `k = rope_from_freqs(k, positions, inv_freq, pair_type="half") * af`

#### Check before continuing

- yarn config output differs from default RoPE output
- switching `rope_scaling=None` changes logits
- 309 cross-task tests for 213/215 YaRN wiring pass

---

### 5) Step D: assemble `gptoss_forward` in one deterministic order

#### Why add this?

By this point, local operators may be correct in isolation, but GPT-OSS parity still depends on exact wiring order.

#### Purpose

Ensure final logits reflect GPT-OSS architecture semantics, not just individually correct subfunctions.

#### Delta note: attention projection biases (`q/k/v/o`)

- Why add this? GPT-OSS checkpoints include affine attention projections.
- Purpose: preserve parity by applying affine projections (with bias) before RoPE on q/k.

#### Delta note: alternating sliding/full mask

- Why add this? GPT-OSS alternates local and global attention layers.
- Purpose: apply sliding-window masks on even layers and full causal masks on odd layers.

#### Delta note: explicit `head_dim`

- Why add this? GPT-OSS config exposes `head_dim` explicitly.
- Purpose: derive head split and score scaling from config instead of implicit size assumptions.

Per layer:

1. `a = rms_norm(h, input_layernorm)`
2. q/k/v affine projection with bias
3. head split
4. YaRN RoPE on q/k (rotate-half path)
5. repeat kv to query-head count
6. scores with `head_dim**-0.5`
7. mask selection:
   - even layer index -> sliding-window mask
   - odd layer index -> full causal mask
8. `attention_with_sinks`
9. weighted value aggregation + o projection with bias
10. first residual add
11. post-attn norm
12. `gptoss_moe_ffn`
13. second residual add

After final layer:

- final RMSNorm
- logits via `lm_head.T`

---

### 6) Step E: `load_gptoss`

#### Why add this?

Whole-model parity can fail even when operator math is correct if HF tensors are mapped to the wrong internal slots.

#### Purpose

Build a deterministic bridge from HF state-dict names to internal parameters so `gptoss_forward` runs correctly on both tiny fixtures and real weights.

Map HF names directly to internal parameter slots, including GPT-OSS-specific fields:

- q/k/v/o bias tensors
- per-head sinks
- router bias
- expert bias tensors
- `lm_head` fallback to tied embedding if absent

No un-permute path is required for rotate-half layout.

---

### 7) Verification ladder (run in this order)

1. **Operator parity**
   - sinks oracle tests
   - GPT-OSS MoE oracle tests
2. **Cross-task YaRN checks**
   - `rope_scaled_freqs` + `rope_attention_scale`
   - GQA RoPE-hook effect
3. **Whole-model parity**
   - tiny fixture logits parity
   - shape + causal invariants
   - alternating mask effect
   - YaRN-vs-default behavioral difference
4. **Optional real-weight parity**
   - after running `download.sh`

---

### 8) Debug playbook (symptom -> likely cause)

- Row sums are exactly 1 in sinks tests  
  -> sink column was not appended, or it was not removed correctly after softmax.

- MoE outputs are close but off from oracle  
  -> Mixtral-style logic was used accidentally (softmax over all experts, wrong split, or wrong transpose).

- YaRN tests fail but 213 base RoPE tests pass  
  -> `rope_scaling` path is not wired, or `af` is not applied.

- Alternating-mask test fails  
  -> one mask policy is being reused for all layers; branch by layer index.

- Real-weight parity fails while tiny parity passes  
  -> loader key mapping or dtype/shape assumptions are incorrect.

---

## Delta coverage checklist for 309

This tutorial covers all deltas required for a student with completed prerequisites to implement 309:

- [x] attention sinks operator and invariants
- [x] GPT-OSS MoE routing/expert arithmetic deltas vs Mixtral
- [x] q/k/v/o biases in attention path
- [x] explicit `head_dim` scaling
- [x] alternating sliding/full mask policy
- [x] YaRN theory, schedule intent, and wiring surface
- [x] whole-forward assembly order
- [x] HF loader mapping specifics
- [x] verification and debugging workflow

Deferred to later levels (explicitly out of scope):

- streaming sink KV-cache behavior and eviction details
- serving/runtime system optimizations

