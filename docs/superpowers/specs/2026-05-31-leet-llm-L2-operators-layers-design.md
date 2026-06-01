# leet-llm — Level 2: Operators & Layers (Design)

> The computation layer. Learners build the transformer's operators and blocks by hand in
> NumPy — first the classic *Attention Is All You Need* stack (encoder / decoder / the GPT
> block), then the Llama-3 upgrades (RMSNorm, RoPE, SwiGLU, GQA) as **contrasting deltas**.
> Every unit is a pure function the learner writes from the math, graded against a frozen
> **PyTorch golden fixture**.

- **Status:** design approved 2026-05-31. Extends the locked ladder in
  `2026-05-31-leet-llm-curriculum-design.md` (§2) and follows its refined authoring rules
  (§1 "Show the math, not the method"; "Solutions policy"; the L2-onward testing refinement).
- **Level goal:** turn the batched tensors from L1 into transformer outputs — understand
  every projection, every normalization, every attention score by hand — and understand
  *why* modern decoder-only LLMs made the choices they did.

---

## 1. Anchor: what the L3 capstone actually uses

The L3 capstone (`llama3.np`, sibling workspace) runs Karpathy's **stories15M** — a tiny
**Llama-2-architecture, decoder-only** model: `dim=288, n_layers=6, n_heads=6, n_kv_heads=6`.
Its repeating unit is the **Llama decoder block** built in task **216**:

- **pre-norm RMSNorm** (not post-norm LayerNorm),
- **RoPE** rotary positions applied to q/k inside attention (not additive sinusoidal PE),
- **SwiGLU** gated FFN (not a GELU MLP),
- attention with `n_kv_heads == n_heads`, so **GQA reduces to MHA** here. Real GQA
  (`n_kv_heads < n_heads`, Llama-3 / 34B-70B) is exercised by the task but only *used* with
  smaller `n_kv_heads` in an L3 OSS-zoo delta.

**Capstone-path tasks:** 201–203, 205–208, 212–216 (the Llama block and everything it
needs). **Foundation/contrast tasks** that the capstone does *not* run — `sinusoidal_pe`
(204), `encoder_block` (209), the full cross-attention `decoder_block` (210), and the
classic `gpt_block` (211) — exist to teach the classic architecture and to make
"decoder-only" *mean* something. `gpt_block` is the conceptual parent the Llama block (216)
is a delta against, even though llama3.np runs only 216. They are kept deliberately; not
dead weight, but not on the generation critical path.

---

## 2. The settled API: pure functions + a params dataclass

**Decision (locked):** every graded unit is a **pure function** taking its weights as
explicit arguments. Composite layers (attention, FFN, blocks) take a small **frozen
dataclass** of parameters so signatures stay readable.

```python
def layer_norm(x, gamma, beta, eps=1e-5): ...          # leaf op: explicit weights
def rms_norm(x, weight, eps=1e-5): ...
def sdpa(q, k, v, mask=None): ...                       # mask: bool, True ⇒ −∞ (reuses L0 009)

@dataclass(frozen=True)
class AttnParams:    Wq; Wk; Wv; Wo                     # composite: a params bag
def mha(x_q, params, n_heads, x_kv=None, mask=None): ...
```

**Why functional (vs llama3.np's mini-classes):** a llama3.np class like `RMSNorm` is just
`{weight} + __call__`, and that `__call__` body *is* our `rms_norm(x, weight)`. So the
functional core loses nothing — at L3 assembly each llama3.np class is recoverable as a
3-line wrapper over the function the learner already wrote. Pure functions are also trivial
to test (no construction/state), keep every parameter visible (nothing hidden in `self`),
and match L0/L1 and the `leet_llm` facade (reuse-by-name).

**Weight initialization** is *not* a learner concern at L2 — operators receive weights;
tests supply them from fixtures. No init helpers, no dtype/precision handling (deferred to
L3/L4 per the curriculum).

---

## 3. The task list (16 tasks, ids 201–216)

Level theme allows full attention vocabulary now (heads, tokens, attention, cross-attention).
Reused L0/L1 primitives are imported via the facade — never rebuilt.

### Classic components (201–208)

| Id | Slug | Builds (primary signature) | Reuses | Designer note (NOT in README) |
|----|------|----------------------------|--------|-------------------------------|
| **201** | `embedding` | `embedding(ids, table)` — gather token vectors from a `(V, d)` table | L0 008 `gather_rows` | token embedding lookup |
| **202** | `activations` | `gelu(x)` (exact erf form), `silu(x)` | — | GELU → classic FFN; SiLU → SwiGLU (214) |
| **203** | `layer_norm` | `layer_norm(x, gamma, beta, eps)` | L0 002 `standardize` | contrasted by RMSNorm (212) |
| **204** | `positional_encoding` | `sinusoidal_pe(seq_len, dim)`; learned-PE noted as a position `embedding` | 201 | contrasted by RoPE (213); not capstone-path |
| **205** | `scaled_dot_product_attention` | `sdpa(q, k, v, mask=None)` = `softmax(QKᵀ/√dₖ + mask)·V` | L0 005 `softmax`, 009 `masked_fill`/`triangular_mask`, 004 matmul | the attention core |
| **206** | `multi_head_attention` | `mha(x_q, params, n_heads, x_kv=None, mask=None)` — split heads, per-head SDPA, merge, output proj. **`x_kv` ⇒ cross-attention** | L0 001 `group_last_axis`, 205, 003 `affine` | self- and cross-attention in one fn |
| **207** | `feed_forward` | `ffn(x, params)` — `linear → GELU → linear` (classic MLP) | L0 003 `affine`, 202 | contrasted by SwiGLU (214) |
| **208** | `residual_norm` | `add_residual(x, sublayer_out)`; **pre-norm vs post-norm** placement | 203 | the field moved post→pre for training stability |

### Classic blocks (209–211)

| Id | Slug | Builds | Norm placement | Designer note |
|----|------|--------|----------------|---------------|
| **209** | `encoder_block` | bidirectional self-attn + FFN (BERT-style) | post-norm (Vaswani) | no causal mask; not capstone-path |
| **210** | `decoder_block` | masked self-attn + **cross-attn** + FFN (original seq2seq decoder) | post-norm | cross-attn = `mha(x_q=dec, x_kv=enc_out)`; not capstone-path |
| **211** | `gpt_block` | masked self-attn + FFN (**the GPT block**, decoder-only) | pre-norm (GPT-2) | = 210 minus cross-attention; the modern LLM unit |

### Llama deltas (212–216) — each contrasts a baseline task

| Id | Slug | Builds | Contrasts | Designer note |
|----|------|--------|-----------|---------------|
| **212** | `rms_norm` | `rms_norm(x, weight, eps)` | 203 LayerNorm | no mean-subtraction, no bias; capstone norm |
| **213** | `rope` | `rope_interleaved` + `rope_half` + `rope_qk_dot` (q/k rotation in **both** conventions + relative-position checker) | 204 sinusoidal | interleaved (Meta/`llama3.np`, capstone) **and** rotate-half (HF); reuses L0 011 |
| **214** | `swiglu` | `swiglu_ffn(x, params)` = `(SiLU(xW₁) ⊙ xW₃)W₂`, bias-free | 207 FFN | reuses 202 `silu`; capstone FFN |
| **215** | `gqa` | `gqa(x, params, n_heads, n_kv_heads, mask=None)` — `repeat_kv` then attend | 206 MHA | `n_kv_heads==n_heads` ⇒ MHA (invariant test) |
| **216** | `llama_decoder_block` | pre-norm RMSNorm → RoPE-GQA → RMSNorm → SwiGLU | 211 gpt_block | **the capstone block**; RoPE wired here, not inside 215 |

**Folded, not separate tasks:** *Linear* = reuse L0 003 `affine` (Llama linears pass `b=0`,
bias-free); *cross-attention* = `mha` with a distinct `x_kv` (206); *RoPE application* is
wired in the Llama block (216), keeping `gqa` (215) a clean grouping-only contrast to MHA.
**Out of L2 scope:** final norm, LM head (→ vocab logits), weight tying — all L3 whole-model
assembly. KV-cache → L4. Precision/dtype → L3 (weight load) / L4.

---

## 4. Testing strategy: PyTorch golden fixtures (primary) + invariants (complement)

Nearly every L2 operator has a trusted PyTorch reference. So the gold standard is a
**frozen golden fixture from an external oracle** — the same principle as L1's
tiktoken/clone parity, with **torch as the oracle**.

**The loop:**
1. An **authoring-time generator script** per task seeds random weights + input
   (`np.random.default_rng(seed)`), runs the **torch reference**, and saves
   `(input, weights…, expected_output)` to a small `.npz` (a few KB each).
2. The runtime test loads the `.npz`, runs the **learner's** function, and asserts
   `np.allclose(out, expected, rtol, atol)`.

**Why this satisfies "don't replicate the solution in the test":** the test file contains
**no parallel NumPy implementation** — only frozen arrays from torch. The solution is never
written twice, and nothing in the committed test reveals the method.

**Four rules that make it robust:**

1. **Runtime stays torch-free.** `torch` is an **authoring/dev-only** dependency (an extra
   `[dependency-groups] gen` group); the committed `.npz` fixtures carry the goldens, so
   `uv run grade` never imports torch. (This pulls torch in earlier than the planned L5 —
   but only as a fixture generator, never at grade time.)
2. **Generate the reference in float64** (`.double()`). NumPy defaults to float64, so a
   double-precision torch oracle matches to ~1e-12 → tight tolerances (`rtol=1e-9` ballpark),
   deterministic, no fp32 fuzz.
3. **The fixture pins the convention.** RoPE ships **both** conventions — `rope_interleaved`
   (Meta/`llama3.np`, golden = official torch complex `view_as_complex`/`polar`; this is the
   capstone form 216/L3 use) and `rope_half` (golden = HF `transformers.rotate_half`); plus
   `rope_qk_dot` whose tests check the relative-position property. GELU (**exact erf**, not
   tanh approx), norm `eps` placement, attention scale `1/√dₖ`, mask polarity (bool True ⇒
   −∞, from L0 009). The fixture *is* the spec.
4. **Many small cases + invariants/edges.** Parametrize the generator over shapes & seeds for
   lots of KB-sized fixtures, and keep cheap, self-documenting checks alongside:
   - **Invariants:** softmax rows sum to 1; LayerNorm out mean≈0/var≈1 on last axis; RMSNorm
     scale-equivariant; RoPE preserves norm and makes `⟨q_m,k_n⟩` depend only on `m−n`;
     Linear is linear; `n_kv_heads==n_heads` ⇒ `gqa ≡ mha`; `heads==1` ⇒ `mha ≡ sdpa`.
   - **Edges:** causal mask leaves token *t* unchanged when tokens `>t` are perturbed; all-pad
     row; single token; `seq_len==1`.

**Building faithful torch references:** compose torch *primitives* to match our exact weight
layout — e.g. explicit q/k/v `nn.Linear`/`F.linear` projections + `F.scaled_dot_product_attention`,
**not** `nn.MultiheadAttention` (its fused-qkv packing won't line up with our signatures).
`F.layer_norm`, `nn.RMSNorm` (or manual), `F.silu`/`F.gelu` cover the leaves; RoPE uses
official torch complex rotation (interleaved) and HF `rotate_half` (rotate-half); GQA uses
`repeat_kv` (+ `enable_gqa` where available).

**Fixture layout (per task):**
```
2NN_slug/
├── tests/
│   ├── test_2NN.py            # loads fixtures, runs learner fn, allclose + invariants
│   ├── gen_fixtures.py        # AUTHORING ONLY: torch ref → .npz (needs the `gen` group)
│   └── fixtures/
│       ├── case_0001.npz      # {x, W…, expected}; a few KB
│       └── …
```

The eventual end-to-end oracles still stand downstream: L3 reproduces `llama3.np`'s
generated text, and L5's keystone asserts a torch Llama block matches the learner's L2 block.

---

## 5. Reuse registrations (`leet_llm/_registry.py`)

Add as each task lands (single source of truth = the task stub). Tentative public names:
`embedding` (201); `gelu`,`silu` (202); `layer_norm` (203); `sinusoidal_pe` (204); `sdpa`
(205); `mha` (206); `ffn` (207); `add_residual` (208); `encoder_block` (209); `decoder_block`
(210); `gpt_block` (211); `rms_norm` (212); `rope_interleaved`/`rope_half`/`rope_qk_dot` (213); `swiglu_ffn` (214); `gqa` (215);
`llama_decoder_block` (216). Params dataclasses (`AttnParams`, `FFNParams`, `SwiGLUParams`,
`BlockParams`) are exported alongside their owning task. (Exact names finalized per-task at
scaffold time.)

---

## 6. README rules for L2 (per curriculum §1, refined 2026-05-31)

Each task README: **goal** (concept-level what & why) · **the math it needs** (the canonical
*formula* — the problem spec, e.g. `softmax(QKᵀ/√dₖ)·V`, the RMSNorm definition) · **Read
More** (good docs + the famous paper: Attention Is All You Need; RoFormer = RoPE;
GLU-Variants = SwiGLU; the LayerNorm / RMSNorm papers; BERT & GPT-2 for the encoder/decoder
asides) · **function signature** · **how to test**. **No** step-by-step list of NumPy
operations — that is the solution. Ship **README + stub + tests only**; `solution.py` is left
as `NotImplementedError` for the learner.

---

## 7. Out of scope (L2)

- Backward passes / autograd (PyTorch phase, L5+).
- Weight initialization schemes, dtype/precision, mixed precision (L3 weight-load, L4).
- KV-cache and any inference-time caching (L4 inference systems).
- The final model head, logits, weight tying, full generation loop (L3 assembly).
- MoE, MLA, sliding-window, RoPE-scaling and other architectural variants (L3 OSS-zoo deltas).
- Real GQA models (`n_kv_heads < n_heads` checkpoints) as a *capstone* — built here, used in L3.
