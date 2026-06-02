# leet-llm — Level 3: Whole-Model & Inference (Design)

> The assembly level. Learners stop building *layers* and start building *models*: they
> wire the L2 operators into two complete, runnable transformers and generate from real
> pretrained weights. First the **classic encoder-decoder** (Vaswani 2017) realized on
> Hugging Face's **`Helsinki-NLP/opus-mt-en-zh`** translation model, then the
> **decoder-only Llama** realized on Karpathy's **`stories15M`** — i.e. a from-scratch
> rebuild of **`llama3.np`**. Finally, the model is morphed into the modern open-source zoo
> (MoE, MLA, sliding-window, RoPE-scaling, QK-norm…) as a catalog of **deltas**.

- **Status:** design approved 2026-06-01. Extends the locked ladder in
  `2026-05-31-leet-llm-curriculum-design.md` (§2) and builds directly on
  `2026-05-31-leet-llm-L2-operators-layers-design.md`. Follows the same authoring rules
  ("Show the math, not the method"; "ship README + stub + tests, no reference `solution.py`";
  PyTorch/HF-oracle fixtures).
- **Level goal:** turn the L2 operators into whole working models. Understand model
  *assembly* (weight layout, block stacking, the final head), *inference* (causal masks,
  the autoregressive loop, sampling), and *why the modern zoo is just deltas* over one
  decoder-only baseline. The end-to-end oracle the whole course was pointing at —
  reproducing `llama3.np`'s output — lands here.

---

## 1. Anchor: what L3 assembles, and from what

Every operator L3 needs already exists as a learner-written pure function in L2, imported
by name through the `leet_llm` facade. **L3 adds no new operators on the capstone path** —
it adds *assembly*, *weight loading*, *masks*, *sampling*, and the *generation loop*.

| Capstone | Real model | Repeating unit (already built) | Reuses (facade) |
|----------|-----------|-------------------------------|-----------------|
| **A. Classic encoder-decoder** | `Helsinki-NLP/opus-mt-en-zh` (Marian) | L2 `encoder_block` (209) + `decoder_block` (210, cross-attn) | 201 embedding, 204 sinusoidal_pe, 202/207 ffn, 203 layer_norm, 205 sdpa, 206 mha, 208 add_residual |
| **B. Decoder-only Llama** | `stories15M` (= `llama3.np`) | L2 `llama_decoder_block` (216) | 201, 212 rms_norm, 213 rope_interleaved, 214 swiglu_ffn, 215 gqa; L0 005/007/009/010 |

This is why the L2 "classic path" tasks (204, 207, 209, 210) that were *not* on any
generation critical path now have a real home: **opus-mt is the classic encoder-decoder
made real.** Historical arc preserved: 2017 seq2seq encoder-decoder → its decoder-only
descendant (Llama) → the variant explosion (zoo).

`stories15M` config (`llama3.np/config.py`): `dim=288, n_layers=6, n_heads=6,
n_kv_heads=6, vocab=32000, max_seq_len=256, norm_eps=1e-6`. Its tokenizer is the **Llama-2
SentencePiece 32k** already built in L1 (the score-greedy encoder in `106`/`107`).

---

## 2. Locked design decisions

1. **Pure functional, consistent with L2.** Whole models are pure functions over an explicit
   params object: `transformer_logits(...)`, `translate(...)`, `llama_forward(...)`,
   `generate(...)`. A thin runnable `__main__` / class wrapper exists only for the CLI demo
   — the "3-line wrapper over the function you already wrote" the L2 memo predicted.
2. **Stateless recompute; KV-cache is L4.** Each decode step re-runs the full prefix (the
   encoder runs once; the decoder recomputes its prefix every step). **L3 = architecture +
   numerical correctness; L4 = systems speed** (the KV-cache is a pure speed delta over the
   *same* model). This matches the locked KV-cache→L4 split.
3. **Greedy decoding.** opus-mt translation is greedy (`num_beams=1`, deterministic — a clean
   token-sequence oracle). Beam search is **out of L3** (defer to L4 inference systems if ever).
   Llama story generation adds temperature / top-k / top-p sampling (its own task), but the
   *graded* token-sequence parity uses greedy/argmax for determinism.
4. **Two tasks per capstone**, split on a real competency boundary — *build the model*
   (parametric forward → logits, graded by per-layer activation parity) vs *run the model*
   (the autoregressive loop + sampling + the real-weight capstone, graded by token-sequence
   parity). L3 is the *assembly* altitude: the unit of work is a whole model, not a layer.
5. **No shipped weights.** Real checkpoints are 60–300 MB and license-encumbered
   (`opus-mt` is CC-BY-4.0; `stories15M` is MIT and already sits at
   `../llama3.np/stories15M.model.npz`). A `download.sh` + `convert.py` fetches and converts
   them; only tiny derived **fixtures** are committed, with attribution.

---

## 3. Testing model: tiny genuine-HF fixtures (graded) + a real-weight capstone (gated)

L3 inherits L2's principle — *grade against the genuine PyTorch/Hugging Face implementation,
captured as a frozen fixture* — and applies it at whole-model scale. One authoring script
per task, `gen_fixtures.py`, is the single fixture tool; it produces **two** kinds of
fixture, both committed:

**Tier 1 — hermetic wiring fixtures (the graded oracle).**
`gen_fixtures.py` instantiates the *genuine HF class at a tiny config* with random init,
runs it once, and commits `{input_ids, weights, per-layer hidden states, final logits,
greedy token sequence}`:

```python
# authoring-only (gen group): the real target impl, shrunk
cfg = MarianConfig(d_model=32, encoder_layers=2, decoder_layers=2,
                   encoder_attention_heads=4, decoder_attention_heads=4,
                   decoder_ffn_dim=64, encoder_ffn_dim=64, vocab_size=64, ...)
model = MarianMTModel(cfg).double().eval()          # float64 → tight tolerances
# ... fixed input, dump activations/logits/tokens → case_*.npz
```

Because the config is tiny, the **weights fit inside the fixture** — the learner's forward
receives its weights from the fixture, so `uv run grade 3xx` is fully hermetic (no
download, no network). The tiny config is chosen to exercise the wrinkles: GQA with
`kv_heads < heads`, encoder↔decoder cross-attention, tied embeddings + `final_logits_bias`,
`n_layers ≥ 2`. **Per-layer hidden-state fixtures localize bugs inside a single coarse
task** — you learn *which* layer is wrong without splitting assembly into micro-tasks.

**Tier 2 — real-weight capstone (generated once, gated at grade time).**
The author runs `download.sh` once, loads the real `opus-mt` / `stories15M` weights **one
time**, runs the real HF model on the fixed prompt, and commits only the small **expected
outputs** (the en→zh / story token sequence; optionally final logits). The big weights are
never committed. The capstone test then:
- **does** load real weights to run the learner's inference (inference genuinely needs them),
  so it is **`@pytest.mark.skipif(weights absent)`** — green for the author / CI-with-weights,
  skipped otherwise;
- compares the learner's greedy run to the committed reference token sequence.

So **routine grading never loads real weights** (Tier 1 carries correctness); real weights
appear only (a) once at authoring to bake the reference, and (b) in the optional capstone /
demo run where a learner watches *their own* code actually translate / tell a story.

**Tolerances:** Tier-1 fixtures generate in float64 (`.double()`) → `rtol≈1e-9` like L2.
The real-weight reference is float32 → token-sequence equality (exact) + looser logit
`atol` if asserted.

**Fixture layout (per task)** — identical to L2:
```
3NN_slug/
├── tests/
│   ├── test_3NN.py            # loads fixtures, runs learner fn, allclose + token-seq + invariants
│   ├── gen_fixtures.py        # AUTHORING ONLY: tiny-HF-config + one real-weight pass → .npz
│   └── fixtures/
│       ├── tiny_case_0001.npz # {input, weights, activations, logits, tokens}; a few KB
│       └── real_ref.npz       # {prompt_ids, expected_tokens}; tiny, from the real model
├── download.sh                # (capstone tasks) fetch real checkpoint from HF / Karpathy
└── convert.py                 # (capstone tasks) HF checkpoint → our .npz naming/layout
```

---

## 4. The task list

Capstones first (the user-locked priority), then the zoo. Ids are `3xx`. Each task ships
**README + stub + tests only**; `solution.py` is left `NotImplementedError` for the learner.

### Track A — Classic encoder-decoder Transformer → `opus-mt-en-zh` (build first)

| Id | Slug | Builds (primary signature) | Reuses | Designer note (NOT in README) |
|----|------|----------------------------|--------|-------------------------------|
| **301** | `transformer_model` | `transformer_logits(src_ids, tgt_ids, params, cfg)` → logits. Weight load (HF→`MarianParams`) + scaled token-embed (×√d_model) + sinusoidal PE + N×`encoder_block` → memory + N×`decoder_block` (causal self-mask + cross padding-mask) + tied-embedding head + `final_logits_bias`. | 201, 204, 209, 210; L0 009 masks | the whole parametric forward; Marian wrinkles = embed-scale, tie, bias, post-norm. Graded by per-layer hidden + logit parity vs tiny `MarianMTModel`. |
| **302** | `translate` | `translate(src_ids, params, cfg, max_len)` → tgt_ids. Encode source once; greedy autoregressive decode from `decoder_start_token_id`, stop on EOS (stateless recompute). + real en→zh capstone (`download.sh`/`convert.py`). | 301, L0 007 argmax | encode-once/decode-many; the en→zh payoff. Graded by greedy token-seq parity (tiny) + gated real run. |

### Track B — Decoder-only Llama → `stories15M` / `llama3.np`

| Id | Slug | Builds (primary signature) | Reuses | Designer note (NOT in README) |
|----|------|----------------------------|--------|-------------------------------|
| **303** | `llama_model` | `llama_forward(input_ids, params, cfg, start_pos=0)` → logits. Weight load (stories15M HF names → `LlamaParams` = per-layer `LlamaBlockParams` + embed + final `rms_norm` + lm_head, with the `.T` layout) + embed → N×`llama_decoder_block` → final `rms_norm` → lm_head; prefill causal mask + positions. | 201, 212, 216; L0 009 | the decoder-only forward; recovers `llama3.np`'s `Llama.__call__`. Graded vs tiny `LlamaForCausalLM`. |
| **304** | `generate` | `sample(logits, temperature, top_k, top_p, rng)` (greedy/temp/top-k/**top-p**=new) + `generate(input_ids, params, cfg, max_new_tokens, sampler)`: prefill + autoregressive decode (stateless), stop on EOS. + the stories15M / `llama3.np` capstone. | 303; L0 005/007/010 | sampling + the loop; top-p (nucleus) is the one new op. Graded by argmax token-seq parity vs `llama3.np` (local npz). |

**Folded deliberately:** weight loading lives *inside* each build task (301/303), not as a
separate task — it is half of "build the model," and the per-layer fixtures still localize a
bad name-map. Config is a small dataclass introduced in 301 and reused by 303 (or one shared
`leet_llm` config helper). Causal/padding mask construction reuses L0 009.

### Track C — OSS-zoo deltas (breadth catalog; "all major OSS variants")

Each zoo task is a **delta over the Llama baseline (B)**, framed as "what this family
changes," graded by a tiny-config fixture from its genuine HF class
(`MixtralForCausalLM`, `MistralForCausalLM`, `Gemma2ForCausalLM`, `Qwen2MoeForCausalLM`,
`Qwen3ForCausalLM`, DeepSeek where available…). Detailed per-task at scaffold time; defined
now grouped by the component they perturb. Provisional ids `305+`.

- **Attention deltas** — MLA / low-rank latent KV (DeepSeek-V2/V3); sliding-window mask
  (Mistral / Gemma-2 / GPT-OSS); attention sinks (StreamingLLM / GPT-OSS); QK-norm
  (Qwen3 / OLMo-2); attention logit soft-cap (Gemma-2); QKV bias (Qwen2 / GPT-2).
- **FFN deltas** — MoE router + top-k gating + expert combine + shared experts
  (Mixtral / Qwen-MoE / DeepSeek-MoE / OLMoE); GeGLU vs SwiGLU (Gemma).
- **Norm / embedding deltas** — Gemma `(1+w)` RMSNorm + √d embed-scale + sandwich (pre+post)
  norm; final-logit soft-cap (Gemma-2); tied vs untied embeddings.
- **Position deltas** — RoPE scaling: linear / NTK-aware / YaRN / Llama-3 `rope_scaling`
  (delta over 213); ALiBi (optional).

Per the locked priority, the zoo is breadth-defined now and detailed after both capstones
land. YAGNI applies — prune at scaffold time.

---

## 5. Reuse registrations (`leet_llm/_registry.py`)

Add as each task lands (single source of truth = the task stub). Tentative public names:
`transformer_logits`, `MarianParams` (301); `translate` (302); `llama_forward`,
`LlamaParams` (303); `sample`, `generate` (304). Zoo names finalized per-task at scaffold
time. Weight-loader functions (`load_marian`, `load_llama`) are exported from their build
task. The capstone wrappers (a runnable `Llama` / `Marian` class) are demo-only, not graded
units.

---

## 6. README rules for L3 (per curriculum §1)

Each task README: **goal** (concept-level what & why) · **the architecture it assembles**
(the block diagram / data flow as the *problem spec* — "embed → N×block → norm → head", the
mask shapes, the Marian/Llama wrinkles named but not coded) · **Read More** (the famous
papers — Attention Is All You Need, the Llama / Marian model cards & HF docs) · **function
signature** · **how to test** (incl. how to run `download.sh` for the real capstone). **No**
step-by-step list of array operations — the architecture is the problem statement; the glue
code is the solution. Ship **README + stub + tests only**.

**HF-specific facts are *given*, not derived.** A new line at L3: the framework-plumbing
details — the **config field meanings** (`d_model`, `decoder_start_token_id`, `norm_eps`,
`scale_embedding`, `rope_theta`…), the **weight name map** (`model.layers.N.self_attn.q_proj.weight`
→ which params slot), the **layout/transpose conventions** (HF stores Linear weight as
`(out, in)`; whether a `.T` is needed for our `affine`), **special-token ids**, the
**embedding ×√d_model scale**, **tied embeddings + `final_logits_bias`** — are *not*
derivable from the math and are *not* the learner's puzzle. The README **states them
explicitly** (a small "HF config & weight layout" table). They are framework trivia, not
architecture; withholding them would just make the task a guessing game about HuggingFace
internals. The learner derives the **assembly/wiring**; the HF facts are reference data
handed over up front.

---

## 7. Out of scope (L3) → deferred

- **KV-cache and any inference-time caching** → L4 (a pure speed delta over L3's stateless model).
- **Beam search, speculative decoding, batched/continuous batching** → L4.
- **FlashAttention / PagedAttention / scheduler / tensor & pipeline parallelism** → L4.
- **Backward passes / autograd / training / LoRA / RL** → L5–L6.
- **Quantization** beyond the float32 weight load → L4+.
- **Shipping model weights** — always `download.sh` + `convert.py`; only tiny fixtures committed.

---

## 8. Open items (resolve at scaffold time)

- Confirm `opus-mt-en-zh` specifics against its actual `config.json`: activation
  (`swish`/`gelu`), `normalize_before` (post-norm expected), `scale_embedding`, decoder
  start token, presence of `final_logits_bias`, tie pattern.
- Decide whether config is one shared `leet_llm` helper or a per-capstone dataclass.
- Pin the exact graded prompts and `max_new_tokens` for both capstone references.
- Zoo: assign concrete `3xx` ids + one-line signatures once capstones are green.
