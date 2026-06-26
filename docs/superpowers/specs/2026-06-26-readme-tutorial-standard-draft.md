# README Tutorial Standard (Draft v0)

## Why this draft exists

Current advanced task READMEs (especially `309_gptoss_model`) are accurate as specs, but often not executable as tutorials. Students can understand facts yet still not know:

- what to implement first,
- what to ignore,
- where the likely mistakes are,
- and how to prove each step before moving on.

This draft defines a new standard for L3+ READMEs, then provides a full 309 tutorial-style example (assuming students have completed all prerequisites before 309).

---

## Scope and assumptions

- This is an authoring standard for task READMEs.
- Student-facing README shape still stays:
  - Description
  - The Math
  - Function Signature
  - Read More
  - How to Test
- We improve execution guidance without leaking copy-paste solutions.
- Assumption for the 309 walkthrough: students already completed tasks before 309.

---

## Definition of "pain-free" README

A README is "pain-free" when a prepared student can pass from blank stub to passing grade by following it linearly, without guessing architecture intent.

Minimum acceptance bar:

1. The README starts from a known baseline and names only the real deltas.
2. Each delta has a concrete "implement + self-check" step.
3. There is a verification ladder (operator -> integration -> full model).
4. Common failure modes are listed with quick diagnosis.
5. Deferred concerns (for later levels) are explicitly called out.

---

## Standard template for advanced README rewrites

Use this section order for L3+ tasks with multiple deltas.

### 1) Orientation (short)

- One paragraph: "what model family, what baseline, what changed."
- One sentence: what is out of scope.

### 2) Baseline and delta map (mandatory)

Add a table:

| Component | Baseline behavior | Task delta | Where wired |
|---|---|---|---|

Goal: students can instantly see what is unchanged.

### 3) Prerequisite checklist (mandatory)

Before coding, list exact prerequisite operators and expected behavior (not just task numbers).

### 4) Step-by-step implementation path (mandatory)

For each step, always include:

- **Why add this?** what gap in the baseline this delta solves.
  - Must explain architecture intent, not just "this test needs it".
  - Include: what is weak in the baseline, and why this delta is the chosen tradeoff.
- **Purpose** what behavior we want after adding it.
- **What** to build (shape-level contracts).
- **How** (formula or pseudocode, no full code).
- **Check** (a fast test or invariant before moving on).

### 5) Integration assembly path (mandatory)

One explicit execution order for `forward` wiring, with the exact layer order and mask/position logic.

### 6) Verification ladder (mandatory)

Strict order:

1. Unit tests for new operators
2. Wiring checks for cross-task dependencies
3. Whole-model parity
4. Optional real-weight parity

### 7) Debug playbook (mandatory)

Map common symptoms -> most likely causes -> first check to run.

---

## 309 tutorial draft (lecture-style)

## 309 — From Llama baseline to GPT-OSS forward

### Learning objective

Starting from Llama-style decoder intuition, implement GPT-OSS forward by adding localized deltas:

1. attention sinks,
2. GPT-OSS MoE,
3. YaRN RoPE wiring,
4. alternating sliding/full attention masks,
5. GPT-OSS loader mapping.

---

### 0) Start from the right baseline

Use this mental baseline:

- Attention backbone: Llama/Mistral-style GQA with rotate-half RoPE.
- Residual structure: pre-norm RMSNorm, attention residual, FFN residual.
- Decoder-only causal forward: embed -> N blocks -> final norm -> lm head.

Do not start from `216_llama_decoder_block` for 309 implementation details, because 309 needs explicit GPT-OSS-specific attention and MoE behavior.

---

### 1) Delta map (what changed vs baseline)

| Component | Baseline | GPT-OSS delta in 309 |
|---|---|---|
| Attention normalization | softmax over keys | softmax over keys plus one sink column, then drop sink |
| Attention projections | often bias-free in Llama/Mistral | q/k/v/o all include bias |
| Attention mask policy | usually full causal or all sliding | even layers sliding-window, odd layers full causal |
| FFN | dense SwiGLU or Mixtral MoE | GPT-OSS-specific MoE (different from 308 in routing + expert math) |
| RoPE schedule | default/llama3 schedule | YaRN schedule + attention scale wiring |
| Head scaling | `1/sqrt(head_dim)` | same formula, but `head_dim` is explicit config field |

### 2) Step A: `attention_with_sinks`

#### Why add this?

In plain causal softmax, every query must distribute 100% probability over visible tokens, even when none is a good match.  
At long context, this can create forced, low-quality token-to-token coupling: attention mass still has to go somewhere.  
GPT-OSS adds a learned sink channel so the model can represent "none of the above" without inventing spurious token alignments.

#### Purpose

Add a learned sink channel so each row can leak part of the mass outside visible keys when beneficial.

#### What

- Input: `scores` shape `(B, H, L, L)`, optional additive mask, `sink_logits` shape `(H,)`.
- Output: `(B, H, L, L)`, rows sum to `< 1` when sink is finite.

#### How

1. Apply additive mask to `scores` if present.
2. Broadcast sink logits to `(B, H, L, 1)`.
3. Concatenate as extra last-axis column -> `(B, H, L, L+1)`.
4. Softmax over last axis.
5. Drop the sink column and return remaining `L` columns.

#### Check before continuing

- finite sink -> row sums `< 1`
- sink `=-inf` -> recovers plain softmax
- masked future positions remain zero

---

### 3) Step B: `gptoss_moe_ffn`

#### Why add this?

GPT-OSS is not trying to reuse Mixtral routing unchanged; it changes the routing contract itself.

In Mixtral-style routing, scores are normalized over all experts first, then truncated and renormalized.  
With many experts, this makes selected weights indirectly depend on tail experts that were not selected, which can make dispatch less local and less stable under expert-count/scale shifts.

GPT-OSS instead makes dispatch depend only on selected experts (top-k logits -> softmax over selected), adds router/expert biases for better per-expert calibration, and clamps the gate/up path to bound extreme activations.  
So this is not "Mixtral can't be used"; it is "GPT-OSS chooses a different MoE operating point: top-k-local routing + calibrated affine terms + bounded expert dynamics."

#### Purpose

Match GPT-OSS-native sparse routing behavior (router bias, selected-topk softmax, interleaved gate/up, clamped activation) so whole-model parity matches reference.

#### What

Token-wise sparse expert routing with GPT-OSS-specific rules.

#### How

1. Router logits with bias:
   - `router_logits = x @ router_weight.T + router_bias`
2. Top-k select logits and indices.
3. Softmax only over selected top-k logits (not all experts).
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
5. Weighted sum of selected experts by top-k softmax scores.

#### Check before continuing

- output shape equals input shape
- non-selected experts can be zeroed without affecting output
- selected gate scores sum to 1
- clamp prevents blow-up on huge pre-activations

---

### 4) Step C: YaRN

This section must exist explicitly in 309-level tutorials.

Course flow contract for this repo's design:

- `309` teaches the YaRN requirement and target behavior first.
- Students then implement reusable YaRN primitives in `213` (`rope_scaled_freqs`, `rope_attention_scale`).
- Then return to `309` and wire them through the existing `213/215` interfaces.

Do not duplicate YaRN math directly inside `309` forward; keep RoPE logic reusable at the primitive layer.

#### Why add this?

Default RoPE frequencies are tuned for pretraining context ranges; direct extrapolation to much longer windows can distort phase behavior and harm long-range attention quality.

GPT-OSS adopts YaRN to change the frequency schedule so long-context behavior is better conditioned, while preserving useful short-range structure.
YaRN also introduces attention-scale correction so the q/k magnitude regime stays calibrated after frequency reshaping.

#### Purpose

Reshape frequencies for long-context behavior while keeping attention temperature calibrated through YaRN scale.

#### 4.1 RoPE recap

For pair index `i` in head dimension:

- `inv_freq[i] = base^(-2i/d)`
- angle at position `p`: `theta(p, i) = p * inv_freq[i]`

RoPE rotates q/k pairs by `theta`, turning absolute position into relative phase interaction.

#### 4.2 Why long context needs scaling

When context length is extended far beyond pretraining range, default frequency progression can produce poor phase behavior at long distances. We need a schedule that stretches usable positional behavior while preserving useful short-range structure.

#### 4.3 YaRN frequency schedule intuition

YaRN blends two regimes:

- **extrapolation-like branch** (original frequencies),
- **interpolation-like branch** (frequencies divided by scaling factor).

A ramp between low/high rotation bands decides how much each pair uses each branch. In code this is the `rope_scaled_freqs(..., scaling={"rope_type": "yarn", ...})` path.

#### 4.4 Why YaRN also has attention scale

GPT-OSS YaRN applies an additional scalar:

- `af = rope_attention_scale(scaling)`

In this repo's GPT-OSS wiring, q and k are both multiplied by `af` after RoPE application. This adjusts effective attention temperature under the scaled frequency regime, and is part of the reference behavior expected by tests.

#### 4.5 Exact implementation surface across tasks

- Task 213 owns reusable primitives:
  - `rope_scaled_freqs`
  - `rope_attention_scale`
  - `rope_from_freqs`
- Task 215 provides RoPE hook path in GQA call sites.
- Task 309 consumes them in whole-model forward:
  - `inv_freq = rope_scaled_freqs(head_dim, rope_base, cfg.rope_scaling)`
  - `af = rope_attention_scale(cfg.rope_scaling)`
  - `q = rope_from_freqs(q, positions, inv_freq, pair_type="half") * af`
  - `k = rope_from_freqs(k, positions, inv_freq, pair_type="half") * af`

#### Check before continuing

- yarn config output differs from default rope output
- switching `rope_scaling=None` changes logits
- 309 cross-task tests for 213/215 YaRN wiring pass

---

### 5) Step D: assemble `gptoss_forward` in one deterministic order

#### Why add this?

The previous steps produce correct local operators, but GPT-OSS parity depends on exact wiring order.

#### Purpose

Guarantee the final logits reflect GPT-OSS architecture semantics, not just individually correct subfunctions.

#### Delta note: attention projection biases (`q/k/v/o`)

- Why add this? Bias-free projections are elegant, but they remove an affine degree of freedom that can help attention calibration under distribution shifts and heterogeneous token statistics.
- Purpose: keep affine flexibility in attention projections while preserving RoPE compatibility by ordering as affine(with bias) first, then RoPE on q/k.

#### Delta note: alternating sliding/full mask

- Why add this? Full attention everywhere is expensive, but sliding-only everywhere can under-propagate global information across layers.
- Purpose: combine local-efficiency layers (even) with global-context layers (odd).

#### Delta note: explicit `head_dim`

- Why add this? Not all model families keep strict `hidden_size = n_heads * head_dim` assumptions in config surfaces.
- Purpose: drive head split and score scaling from explicit config, avoiding hidden coupling bugs.

Per layer:

1. `a = rms_norm(h, input_layernorm)`
2. q/k/v affine with bias
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

Whole-model parity can fail even with correct math if HF tensors are mapped into wrong slots.

#### Purpose

Create a deterministic bridge from HF state-dict names to internal params so `gptoss_forward` runs on both tiny fixtures and real weights.

Map HF names directly to internal parameter slots, preserving GPT-OSS-specific fields:

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
   - GQA rope hook effect
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
  -> sink column not appended or not excluded correctly.

- MoE outputs close but off from oracle  
  -> accidentally using Mixtral logic (softmax over all experts, wrong split, or wrong transpose).

- YaRN tests fail but 213 base RoPE tests pass  
  -> `rope_scaling` path not wired, or `af` not applied.

- Alternating mask test fails  
  -> same mask used for all layers; parity logic should branch on layer index.

- Real-weight parity fails while tiny parity passes  
  -> loader key mapping or dtype/shape assumptions wrong.

---

## Delta coverage checklist for 309

This tutorial covers all deltas needed for a completed-prereq student to implement 309:

- [x] attention sinks operator and invariants
- [x] GPT-OSS MoE routing/expert arithmetic deltas vs Mixtral
- [x] q/k/v/o biases in attention path
- [x] explicit `head_dim` scaling
- [x] alternating sliding/full mask policy
- [x] YaRN theory, schedule math intent, and wiring surface
- [x] whole-forward assembly order
- [x] HF loader mapping specifics
- [x] verification and debugging workflow

Deferred to later levels (explicitly out of scope):

- streaming sink KV-cache behavior and eviction system details
- serving/runtime system optimizations

---

## Rollout checklist for README iteration across tasks

When rewriting old READMEs to this standard:

1. Add baseline-vs-delta map table.
2. Add ordered implementation steps with per-step checks.
3. Add cross-task dependency section (if any).
4. Add verification ladder and debug playbook.
5. Keep fixed student-facing section shape and avoid solution leakage.

