# L3 Track C — OSS-Zoo Whole-Model Forwards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal (redesigned 2026-06-20):** Author the L3 Track C tasks as a **runnable whole-model forward for each OSS family**, not isolated operator deltas. The end goal is: load each family's (tiny) weights and reproduce its logits with a hand-written numpy forward — exactly what 303 does for Llama/stories15M, repeated across the modern zoo. Every family's model is **303's assembly (`embed → N× decoder block → final norm → lm_head`) + one localized delta + that family's weight loader.** All fundamental operators already live in the 2xx series (`rms_norm` 212, `rope_interleaved`/`rope_half` 213, `swiglu_ffn` 214, `gqa` 215, `sdpa` 205, `embedding` 201, `add_residual` 208 …); architecture is *assembly*.

> **Why the redesign:** the original plan made each task an isolated component delta graded by a forward hook. The user's goal is a full runnable model per family. So each task now bundles **(a) its delta operator(s)** (registered for reuse) **+ (b) `<Family>Config` / `load_<family>` / `<family>_forward` → logits.** 305 (Mistral band mask) and 306 (Qwen3 qk-norm) were already built operator-only and are **retrofitted** to add the whole-model forward.

> **No C0 prerequisite.** 213 already ships **both** `rope_interleaved` (Meta) and `rope_half` (HF rotate-half), the latter registered and tested at `rtol=1e-9` against `transformers`' genuine `apply_rotary_pos_emb`. Families that use rotate-half (`Gemma2`, `DeepSeek`, every HF class) just `from leet_llm import rope_half`. The Llama capstone (303) and stories15M use interleaved.

**Tech Stack:** Python 3.11+, NumPy 2.x (runtime), `uv`. Authoring-only `gen` group: `torch` + `transformers` (used by `gen_fixtures.py` and `convert.py`; never at grade time). Installed `transformers` is **5.9.0**, which ships every oracle class: `MistralForCausalLM`, `Qwen3ForCausalLM`, `MixtralForCausalLM`, `DeepseekV3ForCausalLM`, `Gemma2ForCausalLM`, `GptOssForCausalLM`. Raise the `gen`-group floor to `transformers>=5.0`. Network is reachable for `download.sh`.

---

## Family map (one whole-model per task)

| Task | Family | new delta op(s) | reuses | real tiny checkpoint | effort |
|---|---|---|---|---|---|
| 305 | Mistral | `sliding_window_mask` ✓done | rope_half, sdpa, rms_norm, swiglu_ffn, band mask | `hf-internal-testing/tiny-random-MistralForCausalLM` (2M, loads) | retrofit (small) |
| 306 | Qwen3 | `qk_norm` ✓done | rope_half, sdpa, rms_norm, swiglu_ffn, qk_norm | `Qwen/Qwen3-0.6B` (~1.2 GB, download.sh) | retrofit (small) |
| 307 | Mixtral | `moe_ffn` | 303 assembly, swiglu_ffn, top_k, softmax | tiny-random-Mixtral (probe) | medium |
| 308 | DeepSeek-V3 | `mla_project` + reuse `moe_ffn` | 307, rope_half | tiny-random-DeepseekV3 (probe) | large |
| 309 | Gemma-2 | `softcap`, `geglu_ffn` + reuse 305 band, rope_half | 303 assembly | tiny-random / gemma-2-2b (probe) | large |
| 310 | GPT-OSS | `attention_with_sinks` + reuse `moe_ffn` | 307, rope (+yarn 311) | tiny-random-GptOss (probe) | medium |
| 311 | Llama-3.1 | `rope_scaled_freqs` (+ rope variant accepting `inv_freq`) | 303 assembly | Llama-3.2-1B (download.sh) | small–medium |

`moe_ffn` lives in 307 (Mixtral, canonical MoE) and is imported by 308/310. `softcap` lives in 309 (Gemma); `attention_with_sinks` in 310; `mla_project` in 308. **Slugs:** 305/306 keep their existing operator-named folders (already committed) and gain model functions; new tasks 307–311 use family-named folders (`307_mixtral_model`, `308_deepseek_model`, `309_gemma_model`, `310_gptoss_model`, `311_llama31_model`). (Optional cosmetic rename of 305/306 → `*_model` deferred to finalization.)

---

## Cross-cutting design decisions (read before scaffolding any task)

1. **Whole-model grading with two parity layers (the 303/304 pattern).** Each task ships:
   - **(A) Always-on hermetic fixture** — a **composed float64 torch oracle** in `gen_fixtures.py` (genuine torch primitives `F.linear`/`F.rms_norm`/`F.silu`/`F.scaled_dot_product_attention` + the family delta + RoPE, exactly as 303's gen_fixtures, NOT the genuine `*ForCausalLM` — see decision 2), at a tiny **seeded** config (`hidden_size≈32, num_hidden_layers=2, heads=4, kv=2, vocab=64`, plus family fields). Freeze the random HF-named weights **+** the float64 oracle logits into a committed `tiny_<family>.npz`. The test runs `<family>_forward` on those weights and asserts **whole-model logit parity at `rtol=1e-9`/`atol=1e-9`** (303's tolerance).
   - **(B) Skippable real-weights capstone** — a `download.sh` (+`convert.py`) fetches the real checkpoint (only the needed files; map HF names → params, **no un-permute** — the zoo uses rotate-half, see decision 2.5), the test compares `<family>_forward` against a committed **`real_ref.npz`** golden (the *reference output* — logits and/or greedy ids — not the weights), gated by `@pytest.mark.skipif(not weights.exists(), reason="run 3NN_*/download.sh")`, at the looser real tolerance 303/304 use (`rtol≈1e-5`/`atol≈1e-4`, or token-equality). Where no real checkpoint exists/loads, omit layer (B) and say so in the README.
   - Plus **operator invariant tests** for the new delta (kept from the component work, e.g. 305's band-mask invariants, 306's qk-norm invariants).

2. **Composed-oracle, not the genuine class, for the hermetic fixture (303 precedent + 306 anchor).** The hermetic oracle is a **composed float64 torch** forward (decision 1A) because the genuine `*ForCausalLM` casts to float32 internally (RMSNorm, softmax upcast), which would forbid `rtol=1e-9`. The float64 composition matches the student's numpy math exactly → clean `1e-9`. **To prove the composition faithful to the genuine class (not self-circular):** in `gen_fixtures.py` also instantiate the genuine `<Family>ForCausalLM` **on the same random weights** and assert the composed oracle matches its logits at `rtol≈1e-3`/`atol≈1e-3` (whole-model float32 accumulation). This anchor is authoring-only. (For single-op operator fixtures like 306, the same anchor applies at `rtol≈1e-4`.)

2.5. **RoPE convention: the zoo is rotate-half (`rope_half`), HF layout as-is.** Unlike 303 (interleaved + un-permuted, because stories15M is interleaved), the zoo reproduces the **genuine HF families, which all use rotate-half**, and several deltas (qk-norm over head_dim, MLA) interact with head-dim ordering — so un-permuting would also require permuting norm weights. Therefore each `<family>_forward` composes from granular primitives — `embedding` (201), `rms_norm` (212), q/k/v `affine` (003) + head split (001), **`rope_half`** (213) on q/k, `sdpa` (205) with the family mask, `add_residual` (208), `swiglu_ffn`/`geglu_ffn`/`moe_ffn`, lm_head — **not** `llama_decoder_block` (216, which is interleaved-only). `convert.py` maps HF names → params with **no un-permute**.

3. **Ship UNSOLVED (deliverable = tests + fixtures).** Both the learner stub `<file>.py` and `solution.py` are **byte-identical** and raise `NotImplementedError`; the learner writes `solution.py`. Validate by **temporarily** drafting a real `solution.py`, running `uv run grade -s 3NN` (the L2 deps ARE solved — `grade -s 303` passes), then **reverting** `solution.py` before commit; confirm `uv run grade 3NN` (stub) fails **cleanly** (only `NotImplementedError`, no collection/import/KeyError).

4. **Lean stubs, math in the README.** Stub docstrings stay 215-style (name the building blocks + one-line orientation); the closed-form math lives only in `README.md`'s "The Math" section. READMEs never mention `grade -s`. README shape: Description · The Math · Function Signatures · Read More · How to Test, naming each family wrinkle as architecture-as-spec with **GIVEN HF facts** (config fields, weight names, layout). The `→ L4` line stays where a cache/serving facet is deferred (305 windowed eviction, 308/310 latent/streaming cache, 311 long-context decode).

---

## Task 305 (retrofit): `305_sliding_window_attention` → add the **Mistral** whole-model

**Status:** band-mask operator `sliding_window_mask` already shipped & reviewed (commit 8f8ee53). This task ADDS the Mistral whole-model forward in the same folder.

**Delta:** Mistral = the rotate-half Llama forward with the **band mask** swapped in for the causal mask. **rope_half** RoPE, GQA, SwiGLU FFN, RMSNorm. (Mistral has no QKV bias; tied/untied lm_head per config — loader notes.)

- [ ] **Step 1: Stub** — add `MistralConfig` (adds `sliding_window`), `MistralParams`, `load_mistral(weights, cfg)`, `mistral_forward(input_ids, params, cfg) -> logits` to a new `mistral_model.py` in the folder (keep `sliding_window_attention.py` as the operator stub). Both new file + its `solution.py` byte-identical `NotImplementedError`. `mistral_forward` composes the forward from granular primitives per decision 2.5 (`embedding`, `rms_norm`, q/k/v `affine`+head-split, **`rope_half`** on q/k, `sdpa` with `sliding_window_mask`, `add_residual`, `swiglu_ffn`, lm_head) — **not** 216 (interleaved). `load_mistral` maps the Llama-style HF names (same as `load_llama`, no un-permute).
- [ ] **Step 2: Hermetic fixture** — composed float64 torch oracle (303 gen_fixtures style, but **rope_half** + band mask) at tiny seeded config with small `sliding_window` so the band activates at short L; freeze random weights + float64 oracle logits → `tiny_mistral.npz`; in gen_fixtures also assert the oracle matches a genuine `MistralForCausalLM` built on the same weights at `rtol≈1e-3` (decision 2).
- [ ] **Step 3: Real-weights** — `download.sh` + `convert.py` for `hf-internal-testing/tiny-random-MistralForCausalLM`; commit `real_ref.npz`; skippable parity test. (Note: that checkpoint's `sliding_window=4096` won't activate the band at small L — the band is exercised by the operator tests + the hermetic fixture's small window.)
- [ ] **Step 4: Tests** — whole-model logit parity (A) + skippable real (B) + keep the existing band-mask invariants. **Step 5:** registry `MistralConfig`/`load_mistral`/`mistral_forward`. **Step 6:** extend README with the Mistral assembly + GIVEN weight map. **Steps 7–8:** validate (draft→`grade -s 305`→revert), clean unsolved, commit.

---

## Task 306 (retrofit): `306_qk_norm` → add the **Qwen3** whole-model

**Status:** `qk_norm` operator shipped & reviewed (commit dff3d52). ADD the Qwen3 whole-model.

**Delta:** Qwen3 = Llama with **per-head qk-norm before RoPE** (rotate-half RoPE; `rope_half`). GQA, SwiGLU, RMSNorm otherwise.

- [ ] **Step 1: Stub** — `Qwen3Config`, `Qwen3Params`, `load_qwen3`, `qwen3_forward` in `qwen3_model.py` (+byte-identical solution). Reuse 303 assembly + `qk_norm` + `rope_half`.
- [ ] **Step 2: Hermetic fixture** — tiny seeded `Qwen3ForCausalLM`, float64, weights + float64-oracle logits → `tiny_qwen3.npz`; HF-anchor assert at `rtol≈1e-3`.
- [ ] **Step 3: Real-weights** — `download.sh`+`convert.py` for `Qwen/Qwen3-0.6B` (~1.2 GB, needed files only); `real_ref.npz`; skippable test.
- [ ] **Step 4: Tests** — whole-model parity (A) + skippable real (B) + existing qk-norm invariants. **Step 5:** registry. **Step 6:** README extend. **Steps 7–8:** validate, clean unsolved, commit.

---

## Task 307: `307_mixtral_model` — **Mixtral** (introduces `moe_ffn`)

**Delta:** Mixtral = Llama with a **sparse MoE FFN** replacing SwiGLU: router → top-k → softmax-over-selected-k gate → `Σ_k gate_k · SwiGLU_{e_k}(x)`. RoPE rotate-half.

- [ ] **Step 1: Stubs** — operator `moe_ffn(x, router_weight, experts, top_k)` (`experts: list[SwiGLUParams]`, reuse 005 softmax, 007 top_k, 214 swiglu) **and** `MixtralConfig`/`MixtralParams`/`load_mixtral`/`mixtral_forward`. Byte-identical solution.
- [ ] **Step 2: Hermetic fixture** — tiny seeded `MixtralForCausalLM(num_local_experts=4, num_experts_per_tok=2)`, float64, weights (router `block_sparse_moe.gate`, experts `w1/w2/w3`) + float64-oracle logits → `tiny_mixtral.npz`; HF-anchor.
- [ ] **Step 3: Real-weights** — probe `hf-internal-testing/tiny-random-MixtralForCausalLM`; if it loads, `download.sh`+`convert.py`+`real_ref.npz`+skippable test; else README notes none.
- [ ] **Step 4: Tests** — whole-model parity + MoE invariants (routing depends only on top-k; zeroing a non-selected expert is a no-op; selected gate weights sum to 1) + skippable real. **Step 5:** registry `moe_ffn`,`MixtralConfig`,`load_mixtral`,`mixtral_forward`. **Step 6:** README. **Steps 7–8:** validate, clean unsolved, commit.

---

## Task 308: `308_deepseek_model` — **DeepSeek-V3** (MLA + MoE)

**Delta:** DeepSeek = Llama with **Multi-head Latent Attention** (low-rank KV via a latent `c_kv`, up-projected per head; decoupled-RoPE slice carries position via `rope_half`) **and** MoE FFN (reuse `moe_ffn` from 307, plus DeepSeek's shared-expert + sigmoid-gate specifics as GIVEN). Forward-pass arithmetic only; the latent-KV **cache** is **→ L4**.

- [ ] **Step 1: Stubs** — operator `mla_project(x, w_dkv, w_uk, w_uv, …) -> (K, V)` (+ decoupled-RoPE slice via `rope_half`) **and** `DeepseekConfig`/`DeepseekParams`/`load_deepseek`/`deepseek_forward` (reuse `moe_ffn`). Byte-identical solution.
- [ ] **Step 2: Hermetic fixture** — tiny seeded `DeepseekV3ForCausalLM`, float64, weights (`kv_a_proj_with_mqa`, `kv_b_proj`, `q_*`, experts) + float64-oracle logits → `tiny_deepseek.npz`; HF-anchor.
- [ ] **Step 3: Real-weights** — probe a tiny-random DeepseekV3; download.sh/real_ref if available, else README notes none.
- [ ] **Step 4: Tests** — whole-model parity + MLA invariants (latent rank `<` `n_kv_heads·head_dim`; reconstructed K/V parity; decoupled slice carries position, latent slice does not) + MoE reuse. **Step 5:** registry `mla_project`,`DeepseekConfig`,`load_deepseek`,`deepseek_forward`. **Step 6:** README (GIVEN `kv_lora_rank`,`q_lora_rank`,`qk_rope_head_dim`; → L4 latent cache). **Steps 7–8:** validate, clean unsolved, commit.

---

## Task 309: `309_gemma_model` — **Gemma-2** (the heaviest re-skin)

**Delta bundle:** GeGLU FFN (`geglu_ffn`, GELU gate vs SwiGLU's SiLU); **`(1+w)` RMSNorm**; **√d embedding scale**; **sandwich norm** (`input` + `post_attention` + `pre_feedforward` + `post_feedforward`); **attention logit soft-cap** + **final logit soft-cap** (`softcap`); **rotate-half RoPE** (`rope_half`); **alternating sliding/full layers** (reuse 305 `sliding_window_mask` per layer index).

- [ ] **Step 1: Stubs** — operators `softcap(x, cap)`, `geglu_ffn(...)` **and** `GemmaConfig` (`query_pre_attn_scalar`, `final_logit_softcapping`, `attn_logit_softcapping`, `sliding_window`), `GemmaParams`, `load_gemma`, `gemma_forward`. Byte-identical solution.
- [ ] **Step 2: Hermetic fixture** — tiny seeded `Gemma2ForCausalLM` (2 layers → one sliding + one full), float64, weights + float64-oracle logits → `tiny_gemma.npz`; HF-anchor at `rtol≈1e-3`.
- [ ] **Step 3: Real-weights** — probe tiny-random-Gemma2 / `gemma-2-2b` (large, license-gated — may be omitted); README notes.
- [ ] **Step 4: Tests** — whole-model parity + ablation isolations for `(1+w)` norm and √d scale + per-layer sliding/full mask + softcap saturation. **Step 5:** registry `softcap`,`geglu_ffn`,`GemmaConfig`,`load_gemma`,`gemma_forward`. **Step 6:** README naming each wrinkle. **Steps 7–8:** validate, clean unsolved, commit.

> The natural home for "a clean GPT, re-skinned": swapping norm/activation/scale/RoPE-convention turns Llama into Gemma with no new attention machinery.

---

## Task 310: `310_gptoss_model` — **GPT-OSS** (attention sinks + MoE)

**Delta:** GPT-OSS = Llama with **attention sinks** (an extra learned per-head logit in the softmax denominator: `softmax([scores ; sink])`, drop the sink column → rows sum `< 1`) **and** MoE FFN (reuse `moe_ffn`). YaRN rope (reuse 311 `rope_scaled_freqs` if ordering allows, else GIVEN). Sink streaming eviction is **→ L4**.

- [ ] **Step 1: Stubs** — operator `attention_with_sinks(scores, sink_logits, mask=None)` **and** `GptOssConfig`/`GptOssParams`/`load_gptoss`/`gptoss_forward` (reuse `moe_ffn`). Byte-identical solution.
- [ ] **Step 2: Hermetic fixture** — tiny seeded `GptOssForCausalLM`, float64, weights + float64-oracle logits → `tiny_gptoss.npz`; HF-anchor.
- [ ] **Step 3: Real-weights** — probe tiny-random-GptOss; download.sh/real_ref if available.
- [ ] **Step 4: Tests** — whole-model parity + sink invariants (rows sum to `1 − sink_mass`; `sink_logits=-inf` recovers plain softmax) + MoE reuse. **Step 5:** registry `attention_with_sinks`,`GptOssConfig`,`load_gptoss`,`gptoss_forward`. **Step 6:** README (→ L4 streaming eviction). **Steps 7–8:** validate, clean unsolved, commit.

---

## Task 311: `311_llama31_model` — **Llama-3.1** (long-context RoPE scaling)

**Delta:** Llama-3.1 = 303's Llama with **`inv_freq` rescaled** per a `rope_scaling` config (linear / dynamic-NTK / llama3 / yarn). Needs a rope variant that accepts a precomputed `inv_freq` (213's rope functions take only `base`) — add `rope_from_freqs` (or extend) so the scaled frequencies feed rotation. Long-context decode is **→ L4**.

- [ ] **Step 1: Stubs** — operator `rope_scaled_freqs(head_dim, base, scaling: dict) -> inv_freq` (`rope_type ∈ {default,linear,dynamic,llama3,yarn}`) + a rope-from-`inv_freq` apply helper **and** `Llama31Config`/`llama31_forward` (mostly 303 + scaled freqs). Byte-identical solution.
- [ ] **Step 2: Hermetic fixture** — drive HF `transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS` at a tiny config to freeze `inv_freq` per `rope_type`, plus a tiny seeded `LlamaForCausalLM(rope_scaling=…)` whole-model logit fixture → `tiny_llama31.npz`.
- [ ] **Step 3: Real-weights** — `download.sh`+`convert.py` for `meta-llama/Llama-3.2-1B` (gated; or an ungated mirror) — skippable; else reuse 303's stories15M path note.
- [ ] **Step 4: Tests** — per-`rope_type` `inv_freq` parity vs HF + `default` == 213 base freqs + linear factor halves effective positions + whole-model logit parity. **Step 5:** registry `rope_scaled_freqs`,`Llama31Config`,`llama31_forward`. **Step 6:** README (GIVEN `rope_scaling` schema, Llama-3/YaRN bound fields; → L4 long-context). **Steps 7–8:** validate, clean unsolved, commit.

---

## Finalization

- [ ] Update root `README.md` L3 row → Track C = whole-model forwards per OSS family (305 Mistral, 306 Qwen3, 307 Mixtral, 308 DeepSeek, 309 Gemma-2, 310 GPT-OSS, 311 Llama-3.1).
- [ ] Pin `transformers>=5.0` in the `gen` group.
- [ ] Confirm `_registry.py` resolves all new names; each unsolved `uv run grade 3NN` fails cleanly; each `uv run grade -s 3NN` (with a drafted solution during dev) passes.
- [ ] (Optional) cosmetic rename 305/306 folders → `*_model`.
- [ ] Commit `docs: L3 Track C reframed as whole-model forwards per OSS family`.

## Self-Review

**End goal:** each task yields a runnable `<family>_forward` matching the genuine HF model's logits — on a committed tiny hermetic fixture (always-on, float64 oracle, HF-anchored) and on real downloaded weights (skippable, `download.sh` + `real_ref.npz`). ✓ maps to the user's "forward pass for each OSS model on its tiny weights."

**Reuse compounds:** operators registered once (`moe_ffn` 307 → 308/310; `softcap` 309; `rope_half` 213 → 306/308/309), assembly from 303. New per-family surface = Config/load/forward + the one delta.

**Numerics:** float32-cast anchoring (decision 2) keeps tight student grading while proving the oracle faithful to the genuine class. Real-weights goldens use documented tolerances.

**Ship discipline:** unsolved (byte-identical stub/solution), lean stubs, math in README, validate via temporary draft + `grade -s` + revert.

**Open probes (resolve at scaffold time):** real tiny-checkpoint availability for Mixtral/DeepSeek/GptOss/Gemma2 (`hf-internal-testing/tiny-random-*` — confirmed for Mistral, absent for Qwen3); if a family has none that loads, omit parity layer (B) and note it.
