# leet-llm — Level 4: Inference Systems & Serving (Design)

> The systems level. Learners stop building *models* and start building the *machinery
> that serves them fast*. L3 produced correct-but-slow stateless forwards; L4 keeps the
> arithmetic identical and makes it serveable, in three escalating tracks on **two models**:
> **(1)** a KV-cached, continuously-batched, paged single-node engine on `Qwen3-0.6B`;
> **(2)** that same dense model **sharded across hosts** (collectives from scratch + tensor /
> pipeline parallelism), ending in a real multi-process capstone; **(3)** the frontier deltas
> on `DeepSeek-V3` — **MLA KV-compression** and **expert parallelism**.

- **Status:** design approved 2026-06-20; reconciled with the shipped L3 code and
  restructured (3 tracks, 2 models, no bonus) 2026-06-28 (§4a). Extends the locked ladder in
  `2026-05-31-leet-llm-curriculum-design.md` (§2, L4 row) and builds directly on
  `2026-06-01-leet-llm-L3-whole-model-inference-design.md` and the L3 Track-C zoo
  (`2026-06-20-leet-llm-L3-trackC-oss-zoo.md`). Still the **NumPy phase** — zero
  framework autograd; PoC/simulation, not GPU kernels or real clusters.
- **Level goal:** turn the L3 stateless forwards into *serving systems*. Understand the
  stateless→stateful shift (the **KV-cache**), KV memory management (**paged + prefix/radix**),
  request scheduling (**continuous batching**), scaling a model past one device
  (**collectives, tensor / pipeline / expert parallelism**), and **KV compression as
  architecture** (MLA). The satisfying payoff: the learner's own L3 model generating the
  *same tokens*, much faster, behind a mini-vLLM engine — and then running across real OS
  processes.

---

## 1. The altitude shift: from operator-filling to system-building

L0–L3 are **operator-filling**: implement one function, graded by output parity against a
genuine-HF oracle. If L4 were just *more, bigger* functions to fill, it would still read as
function-filling. **L4 is system-building.** We hand the learner a **contract** — a thin
registered API plus a behavioral spec — and the learner **designs the system** behind it:
the cache layout, the scheduler policy, the block allocator, the radix tree, the ring
collective algorithm, the multi-process orchestration. The README states *what the system
must guarantee*, never *how to structure it inside*. There is no sub-function checklist.

What makes this gradeable — and **logic-perfect** — is the boundary, not the internals
(see §3): every L4 system's API **returns logits or token ids**, which the harness compares
*exactly* against the learner's own L3 stateless output. Free design inside; exact oracle at
the edge.

**L4 adds no new model arithmetic on the main path.** Every optimization is a pure
*speed / memory* delta over math the learner already wrote in L3. So the reference is L3
itself — no new external oracle is needed for any L4 task.

---

## 2. Anchors: two models, three tracks (no bonus)

L4 does **not** re-cover the whole OSS zoo. It anchors on **two** models the learner already
built in L3, escalating one axis per track, and lists the rest as one-line extension notes.
The hardest model (DeepSeek) is sealed in the final track so the first two tracks are fully
authorable and shippable without it.

| Track | Anchor | Lessons | Why this model | L3 reuse |
|-------|--------|---------|----------------|----------|
| **1 — Single-node serving** | **Qwen3-0.6B** (306) | KV-cache · continuous batching · paged + radix | Most popular community OSS model with a small, **ungated** (Apache-2.0) ~0.6 B checkpoint. Dense GQA + qk-norm keeps the serving systems clean and model-agnostic, and gives a **real fast-generation demo** (Tier C). | `qwen3_forward`, `load_qwen3`, `Qwen3Config` (306); `generate`/`sample` (304); `sdpa` (205) |
| **2 — Multi-host (dense)** | **Qwen3-0.6B** (306) | collectives from scratch · tensor + pipeline parallel · real-process serving | Keep the *same* familiar dense model so the learner meets distribution **before** any exotic architecture. TP/PP/collectives are model-agnostic — Qwen3 teaches them with the least incidental complexity. | `qwen3_forward` (306); `embedding`/`rms_norm`/`affine`/`sdpa` primitives |
| **3 — Frontier deltas** | **DeepSeek-V3** (311) | **MLA KV-compression** (+ GQA-vs-MLA size comparison) · **expert parallelism** | The frontier MLA + MoE architecture (latent-KV attention, 256 routed + 1 shared expert). MLA *is* a KV-cache compression — the natural capstone of the KV arc; its MoE is *the* reason to do expert parallelism. Both reuse 311 directly. | `mla_project`, `deepseek_moe_ffn`, `deepseek_forward` (311); `rope_half` (213) |

**Why DeepSeek for both frontier deltas (not Mixtral):** DeepSeek-V3 already carries MLA
*and* a MoE, so one model teaches both compression and expert parallelism — no extra anchor.
Its MoE is richer than Mixtral's, but that complexity lands in the **frontier track where it
belongs**, after the learner already knows collectives/TP/PP from Track 2. Expert parallelism
generalizes to any MoE (e.g. Mixtral 308) — noted as an extension, not a separate task.

**Why MLA is single-node, not sharding:** MLA exists to *shrink the KV cache* (one latent per
token vs full per-head K/V). That is a **serving/memory** lesson — the capstone of Track 1's
KV arc, contrasted against the paged GQA cache — not a sharding lesson. Coupling it to
sharding would be arbitrary; coupling it to KV memory is exactly its reason for existing.

**Deferred frontier (not in scope):** DeepSeek-V4's Compressed Sparse Attention (Lightning
Indexer + top-`k` compressed blocks) is a genuine L4-flavored topic, but the golden
`DeepseekV4` module is heavy and **does not reuse L3's MLA** (V4 dropped `kv_lora_rank`). It
is recorded in §8 as a future stretch, not a v1 task.

---

## 3. The grading model: contract, not recipe (and logic-perfect)

L0–L3 grade a *prescribed function* by activation parity. L4 grades a *student-designed
system* by a **conformance harness** bound only to a thin registered API. Three mechanisms,
**no new external oracle anywhere** — L3 is the oracle for every L4 task.

### 3.1 The logits/token-id output contract (the spine)

Every L4 system exposes an output the harness can compare **exactly**:

- return **logits** → compared to the learner's L3 stateless logits at **`rtol≈1e-9`**
  (float64 equivalence, the L3 tolerance), **or**
- return **token ids** → compared to the learner's L3 greedy/sampled token sequence at
  **exact equality**.

The internals are the learner's design, but the API's *return type* is **pinned** to logits
or token ids. This is what reconciles "the learner owns the architecture" with "graded
logic-perfectly." Concretely: an `Engine.step()` returns each active request's next **token
id** (checked against `generate`, 304); a `sharded_forward(...)` returns **logits** (checked
against `qwen3_forward`, 306); the `mla_kv_cache` decode returns **logits** (checked against
`deepseek_forward`, 311).

### 3.2 The equivalence oracle = L3 itself

Because every L4 path is a pure speed/memory delta over L3's arithmetic, **the reference is
the learner's own L3 forward.** A KV-cached / paged / sharded run must reproduce the *same*
logits/tokens as the stateless L3 run on the same input. This is the same trick L3 used
internally (its locked decision: "KV-cache is a pure speed delta"). It is free, exact, and
self-contained — the learner's earlier code is the oracle, resolved through the `leet_llm`
facade exactly as in L3. (Authoring-time anchor: the fixture's stateless logits are themselves
frozen from the L3 float64 oracle / golden HF class, as in L3.) **L4 never *wraps* this
forward** to obtain the fast path — it re-implements the loop/kernel and compares against it.
L3 is L4's **oracle + primitive library, not a building block** (§4 decision 8, §4a).

### 3.3 Invariants + behavioral assertions

The defining properties of each *system* — not just its output:

- **memory**: paged cache uses O(blocks) not O(L²); the block allocator never double-allocates
  and returns freed blocks; the MLA cache stores **one latent per token** (not per-head K/V),
  so its bytes/token/layer `(kv_lora_rank + rope_dim)` ≪ GQA's `2·n_kv_heads·head_dim`.
- **scheduler**: every active sequence advances exactly one token per `step`; finished
  sequences are retired; a prefix cache **hit does not recompute** the shared prefix.
- **collectives**: ring all-reduce == naive `np.sum`; every rank ends with identical
  reduced state; `dispatch ∘ combine` (all-to-all) is the identity (the expert-parallel core).
- **reconstruction**: paged gather reconstructs the contiguous KV exactly.

### 3.4 Simulated hardware (pure NumPy)

Per curriculum §8 ("PoC/simulation only"), L4 stays **pure NumPy**. Single-node tasks are
single-process. Multi-host is **hybrid** (locked decision, §4): the *graded* parallel tasks
are **in-process and deterministic** (ranks = a list of per-rank arrays; collectives =
ring algorithms over that list), and **one skippable capstone** runs a real `multiprocessing`
process group — each worker holds only its shard and exchanges tensors over an IPC backend
the learner writes — so the learner genuinely confronts "a rank can't see the other shards;
you must communicate." No torch, no MPI, no GPU.

---

## 4. Locked design decisions

1. **System-building, contract-graded.** Each task registers a thin public API; grading is a
   conformance harness (§3), not a prescribed function tree. The README gives guarantees,
   not structure.
2. **Logic-perfect via the logits/token-id boundary.** Every API returns logits (`rtol≈1e-9`)
   or token ids (exact); compared against the learner's own L3 output (§3.1–3.2).
3. **No new arithmetic on the main path.** L4 = speed/memory deltas over L3. The equivalence
   oracle is L3 itself; no new external oracle for any task.
4. **Two models, three tracks, no bonus** (§2): Qwen3-0.6B carries Track 1 (single-node
   serving) and Track 2 (multi-host dense); DeepSeek-V3 carries Track 3 (MLA compression +
   expert parallelism). Mixtral is dropped; the DeepSeek-V4 CSA frontier is deferred (§8).
   The rest of the zoo = extension notes.
5. **Hybrid multi-host** (§3.4): graded tasks in-process/deterministic; one real
   `multiprocessing` capstone, **skippable** like L3's Tier-C real-weights capstone.
6. **Pure NumPy, PoC/simulation.** No GPU kernels, no real distributed runtime,
   no autograd.
7. **Stub UNSOLVED, `solution.py` REAL** (corrected 2026-06-30 — supersedes the earlier
   "byte-identical NotImplementedError solution"). As actually shipped across L2/L3 (301–311)
   and mandated by CLAUDE.md ("`solution.py` is the only place the full implementation lives"):
   the learner stub `<file>.py` raises `NotImplementedError`; `solution.py` is the **real,
   working reference** (NOT byte-identical, NOT reverted). This is what makes `grade -s`'s
   all-solutions stack resolve across tasks (402 imports 401's solution, etc.). Validate in the
   **shipped state**: `uv run grade -s 4NN` GREEN and `uv run grade 4NN` fails **cleanly** (only
   `NotImplementedError`). The stub must never leak logic.
8. **L3 is oracle + primitives, not machinery** (§4a). Every L4 system *re-authors* its
   forward loop; the L3 forward and `generate` are reused only (a) as the equivalence oracle
   (§3.2) and (b) as the imported primitive library (`embedding`, `rms_norm`, `qk_norm`,
   `rope_half`, `sdpa`, q/k/v/o `affine`, `swiglu_ffn`, `mla_project`, `deepseek_moe_ffn`).
   No task wraps an L3 forward to obtain the fast path — the K/V seam, the batched loop, and
   mid-layer collectives all live *outside* the stateless box, by design.

---

## 4a. Reconciliation with the shipped L3 code (2026-06-28)

A review of the L3 solutions against this design surfaced four plug points; all are resolved
in favor of "re-author in L4, simplify L3":

- **Forwards aren't cache-pluggable (was G1).** `qwen3_forward` builds `positions`/`mask`
  internally and discards K/V; `deepseek_forward` is monolithic. Resolution: re-authoring the
  cache-aware (Track 1) and shard-aware (Track 2/3) forward loop *is* the task. L3 forwards
  stay stateless and serve as oracles + a primitive library — they are never wrapped for the
  fast path (decision 8).
- **`start_pos` is speculative and inconsistent (was G2).** Six L3 forwards (303/307/308/309/
  310/311) carry an unused, untested `start_pos` half-hook ("only used by L4 KV-cache"), and
  it doesn't actually enable caching (square mask, K/V recomputed); 305/306 lack it. Since L4
  re-authors the loop, `start_pos` is dead. Resolution: **delete it across L3** (separate
  commit) so every forward is uniformly `positions = arange(0, L)`.
- **`generate` is Llama-only (was G3).** `generate` (304) hardwires `llama_forward`.
  Resolution: do **not** generalize it; the token-id oracle is frozen in `gen_fixtures.py`
  (greedy over the anchor's stateless logits at authoring time). *(If a live Qwen3 baseline is
  ever wanted for the real-weights demo, revisit a `forward_fn` param then — grading does not
  need it.)*
- **No offset/non-square causal mask primitive (was G4).** Building the `(Lq×kv_len)` decode
  mask is part of the L4 student rewrite; L3 needs no new mask primitive.

**Cache-design corollary (Tracks 1–2 are MLA-free).** Tasks 401–406 never reference MLA, so
their KV cache should be **GQA-specific** (fixed per-head K/V layout) — do *not* pre-generalize
it to cover MLA's latent layout. MLA enters only in Track 3 (407), which builds its own latent
cache from scratch and does the GQA-vs-MLA comparison as a **backward** reference to the paged
GQA cache from 403. Nothing before 407 depends on DeepSeek.

---

## 5. The task list (8 tasks · 3 tracks)

Ids are `4xx`, ordered so "finish in order" holds (no task depends on a later one). Each
ships **README + stub + tests only**. Signatures below are the **registered contract
boundary** (what the grader binds to); the per-task scaffold finalizes exact field names.
Everything *inside* the contract is the learner's design.

### Track 1 — Qwen3 deployment & optimization (single-node, real demo)

| Id | Task | Registered contract (returns logits / token ids) | The system to build | Reuses | Oracle |
|----|------|--------------------------------------------------|---------------------|--------|--------|
| **401** | `kv_cache` | a per-layer KV cache + `prefill(prompt_ids)` / `decode_step(token) -> logits`; greedy loop → **token ids** | stateless→stateful: prefill once, then single-seq incremental decode reusing cached K/V (offset positions, `(1×kv_len)` mask) | 306, 205, 304 (oracle) | generated **token ids** == L3 `generate` (single seq) |
| **402** | `continuous_batching` | `Engine(params, cfg)`; `add_request(prompt_ids) -> req_id`; `step() -> list[(req_id, next_token_id)]` | ragged multi-request scheduler over the cache: admit/retire, exactly one token/step | 401 | each request's **token ids** == its standalone `generate`; every active seq advances 1/step |
| **403** | `paged_kv` | a KV-manager the engine plugs into: `allocate / append / gather` over fixed-size blocks + `match_prefix / insert` (radix) | paged KV (block table) **and** prefix/radix sharing across common prompts | 402 | paged/prefix-cached **logits** == contiguous-cache logits (`rtol≈1e-9`); shared prefix not recomputed; allocator invariants |

*Extension notes (not tasks):* online-softmax / FlashAttention (tile attention, never
materialize the L×L scores — a kernel idea whose memory-hierarchy payoff needs a GPU, so it
stays a note, not a NumPy task); 305 → ring-buffer eviction (sliding window); 309 →
evict-but-keep-sinks (StreamingLLM); speculative decoding (draft-and-verify); quantized KV.

### Track 2 — Qwen3 multi-host simulation (dense, no expert parallelism)

| Id | Task | Registered contract (returns logits / token ids) | The system to build | Reuses | Oracle |
|----|------|--------------------------------------------------|---------------------|--------|--------|
| **404** | `collectives` | `all_reduce / all_gather / reduce_scatter / all_to_all(shards)` over a list of per-rank arrays | ring collective algorithms from scratch, **in-process** | — | ring all-reduce == `np.sum`; every rank identical; `dispatch ∘ combine` == identity |
| **405** | `sharded_forward` | `sharded_forward(input_ids, sharded_params, cfg, world) -> logits` | **tensor + pipeline** parallelism on Qwen3 dense GQA, wired with 404's collectives over a rank list | 404, 306 | sharded **logits** == `qwen3_forward` (306) at `rtol≈1e-9` |
| **406** | `distributed_serve` *(capstone, skippable)* | a real `multiprocessing` harness: N workers, each holds one shard, collectives over the learner's IPC backend → `serve(input_ids) -> token ids` | the same sharding, but over **real OS processes** with genuine IPC | 405, 401 | distributed **token ids** == single-process Qwen3; `@pytest.mark.skipif` when the process group can't start |

*Extension notes (not tasks):* disaggregated prefill/decode (separate workers + KV handoff);
collective-comm overlap; MLA-aware sharding (shard the latent-KV built in 407).

### Track 3 — DeepSeek frontier: compression + expert parallelism (deferred for difficulty)

| Id | Task | Registered contract (returns logits / token ids) | The system to build | Reuses | Oracle |
|----|------|--------------------------------------------------|---------------------|--------|--------|
| **407** | `mla_kv_cache` | an MLA latent-KV cache + cached `decode_step(token) -> logits`; plus `kv_bytes_per_token(cfg)` for the comparison | MLA compressed KV-cache (store the latent, not per-head K/V) + a **GQA-vs-MLA bytes/token comparison** against the 403 cache | 311 `mla_project`, 403 | MLA cached-decode **logits** == stateless `deepseek_forward` (311); **assert `(kv_lora_rank + rope_dim)` ≪ `2·n_kv_heads·head_dim`** |
| **408** | `expert_parallel` | `expert_parallel(tokens, router, local_experts, world) -> out` + the sharded MoE forward | **expert parallelism**: route over all experts, all-to-all dispatch tokens to the rank owning each expert, compute locally, all-to-all combine | 404, 311 `deepseek_moe_ffn` | sharded MoE **logits** == `deepseek_forward` (311) at `rtol≈1e-9`; `dispatch ∘ combine` == identity |

*Extension notes (not tasks):* expert parallelism generalizes to any MoE (e.g. Mixtral 308);
MLA-aware tensor sharding of the latent-KV; DeepSeek-V4 Compressed Sparse Attention (§8).

---

## 6. Testing & fixtures

L4 inherits L3's fixture machinery (`gen_fixtures.py` per task, committed `.npz`, the gen
group's `torch`/`transformers` used only at authoring), specialized to the contract model:

- **Equivalence fixtures (the graded oracle).** `gen_fixtures.py` freezes a tiny seeded
  config's **stateless L3 output** — logits and the greedy/sampled token sequence — from the
  same float64 oracle L3 uses (the composed-oracle or golden HF class, anchored at authoring
  time, exactly as the L3 zoo). The L4 harness then drives the learner's *system* and asserts
  its returned logits/token-ids match that frozen stateless reference (§3.1–3.2). No real
  weights are needed for any graded path — the tiny config fits in the fixture (hermetic, like
  L3 Tier 1).
- **Invariant/behavioral tests** (§3.3) accompany every task — memory bounds, scheduler
  advance/retire, prefix-hit-skips-recompute, ring-collective identities, paged gather
  reconstruction, all-to-all `dispatch∘combine` identity.
- **The MLA KV-size comparison (407)** is asserted **analytically** from the two configs
  (`kv_lora_rank + qk_rope_head_dim` vs `2·n_kv_heads·head_dim`), not from a fragile fixture —
  it's a closed-form memory claim the student computes and the test checks.
- **The real-weights demo (Track 1).** `401`–`403` ship a `download.sh` for `Qwen/Qwen3-0.6B`
  (reusing 306's converter): the learner watches their *own* engine generate real text fast,
  with output token-equal to the stateless 304/306 run — the Tier-C payoff. Skippable.
- **The multi-process capstone (Track 2, `406`).** Skippable, gated on the process group
  starting; grades distributed token-ids == single-process Qwen3.

**Tolerances:** logit equivalence at `rtol≈1e-9`/`atol≈1e-9` (float64, L3's bar); token-id
equivalence is exact.

---

## 7. README rules for L4 (per curriculum §1)

Each task README keeps the fixed shape (**Description · The Math/The Contract · Function
Signatures · Read More · How to Test**) but at system altitude:

- **State the contract, not the recipe.** Give the **registered API** the harness drives, the
  **guarantees** it must meet (the equivalence oracle + invariants), and the **GIVEN systems
  facts** (block size, world size, the IPC primitive available, the MLA latent/rope dims).
  Do **not** enumerate internal sub-functions or a step-by-step build — the architecture is
  the learner's to design.
- **Name the technique and cite the paper** (PagedAttention/vLLM; RadixAttention/SGLang;
  Megatron-LM tensor parallelism; GPipe/1F1B pipeline parallelism; GShard/Switch expert
  parallelism; DeepSeek-V3 MLA; FlashAttention for the online-softmax extension note) — the
  *what/why*, never the *how*.
- READMEs never mention `grade -s`. The `→ L4` lines L3 left behind (305 windowed eviction,
  309 streaming-sink eviction, 311 latent-KV cache) are realized here as the engine (311
  latent-KV → 407) or as extension notes — cross-reference them.

---

## 8. Out of scope (L4) → deferred

- **Real GPU kernels / Triton / CUDA** — L4 is a NumPy PoC of the *algorithms* (block tables,
  ring collectives, latent-KV), not performant kernels.
- **FlashAttention as a graded task** — online softmax is an operator-level rewrite whose real
  payoff is the SRAM/HBM memory hierarchy, which a pure-NumPy sim cannot exhibit. Kept as an
  extension note (§5 Track 1), not a task.
- **A real distributed runtime** (NCCL/MPI/torch.distributed) — collectives are hand-written;
  the capstone uses stdlib `multiprocessing` only.
- **Backward passes / autograd / training / LoRA / RL** → L5–L6.
- **Quantization** beyond a KV-cache quantization note → out of scope for v1.
- **Beam search** (L3 already deferred it) — speculative decoding appears only as an
  extension note.
- **DeepSeek-V4 Compressed Sparse Attention** — a genuine frontier inference topic, but the
  golden `DeepseekV4` module is heavy (CSA + HCA caches, two compressors, the indexer,
  hyper-connection, hash router, `sqrtsoftplus` MoE, MTP) and does **not** reuse L3's MLA. A
  future stretch, not a v1 task.
- **Shipping model weights** — always `download.sh` + `convert.py`; only tiny fixtures
  committed.

---

## 9. Open items (resolve at scaffold time)

- **Track 1 (401–403): RESOLVED — see §10** (2026-06-30 scaffold-decisions pass).
- Confirm the `multiprocessing` IPC primitive for `406` (Pipe vs shared-memory vs a tiny
  socket ring) and whether the capstone grades on macOS spawn-start reliably. *(Track 2.)*
- Pin the world sizes for the Track-2 fixtures. *(Track 2.)*
- Confirm 407's MLA latent-KV cache and the analytic KV-size comparison exercise real
  sparsity at the tiny fixture config (latent dim < full per-head K/V). *(Track 3.)*

---

## 10. Track-1 scaffold decisions (2026-06-30) — 401–403 pinned

This pass authors **Track 1 only** (401 `kv_cache` → 402 `continuous_batching` → 403
`paged_kv`) on Qwen3-0.6B. Track 2/3 stay as designed above. Every decision below is final
for scaffolding; anything not listed defers to §§1–7.

### 10.1 The KVCache-as-seam decision (the spine of the track)

**Decision.** 401 registers **two separable surfaces**, not one fused engine:

1. a **`KVCache` storage object** — the K/V seam, with a fixed interface, and
2. **`prefill` / `decode_step`** — the re-authored Qwen3 decode forward, written against
   *any* object satisfying that interface.

402's scheduler holds one `KVCache` per request; 403's `PagedKVCache` **implements the same
interface**, so the 401 `prefill`/`decode_step` forward runs over it **unchanged**. The
paged/radix machinery is a *drop-in substitution behind the seam*, never a rewrite of the
learner's earlier code.

**Why (lecturer's rationale).** The one property that makes this a *systems* track rather
than three disconnected function-fills is that the fast-path variations (batched, paged,
prefix-shared) are all **the same forward over a different memory manager**. If the seam is
discovered late (bundle-then-refactor, §Q3 option 2) the learner rewrites 401 at 403 and
never feels that invariance. Pinning the seam at 401 is what makes "free design inside, exact
oracle at the edge" (§3) concrete: the *edge* is the `KVCache` interface. This also inoculates
against the §9 "operator-filling smell at the seams" — the seam is an interface contract, not
a sub-function checklist. **This does not leak the internal design:** the README states the
interface the harness binds to (what `append`/`get`/`length` must guarantee) and the
equivalence oracle, never how to lay out blocks, offsets, or the decode loop.

### 10.2 Registered contract (final field names → `_registry.py`)

Everything *inside* is the learner's design; only these names are pinned and graded.

**401 `kv_cache.py`**
- `class KVCache` — per-layer preallocated K/V storage. `KVCache(cfg)`; `.append(layer, k, v)`;
  `.get(layer) -> (K, V)` (contiguous, length-`self.length`); `.length` (tokens cached so far).
  Per-layer store shape `(n_kv_heads, max_seq_len, head_dim)` — **GQA-specific** (decision §4a
  cache corollary; no MLA generalization).
- `prefill(prompt_ids, params, cfg, cache) -> logits` — full-prompt forward at
  `positions = arange(0, len)`, fills `cache`, returns **last-position** logits `(1, V)`.
- `decode_step(token_id, params, cfg, cache) -> logits` — single token at offset
  `cache.length` with a `(1 × kv_len)` causal mask, appends its K/V, returns logits `(1, V)`.
- `kv_generate(prompt_ids, params, cfg, n_new) -> list[int]` — greedy demo driver
  (`prefill` then `n_new` × `decode_step`), the real-weights Tier-C entry point.

**402 `continuous_batching.py`**
- `class Engine` — `Engine(params, cfg)`; `.add_request(prompt_ids) -> req_id`;
  `.step() -> list[tuple[req_id, int]]` (each active request's next token id, exactly one per
  step); `.is_finished(req_id) -> bool`. Holds one 401 `KVCache` per live request.

**403 `paged_kv.py`**
- `class PagedKVCache` — satisfies the **401 `KVCache` interface** (`append`/`get`/`length`)
  over fixed-size blocks; adds `.allocate()`, `.free()`, and exposes `.block_table`.
  Constructed with an explicit `block_size`.
- `class RadixCache` — prefix sharing: `.match_prefix(ids) -> (node, matched_len)`;
  `.insert(ids, ...)`. A shared prefix **must not recompute** its K/V (§3.3).

### 10.3 Grading (per task) — teacher-forced logits + free-run tokens

**401.** Two checks (the §Q2 decision):
- **Test A (primary, loud):** freeze the composed-oracle per-position logits over the full
  greedy sequence; `prefill(prompt)` then `decode_step` through the *known* frozen tokens and
  assert each step's logits `≈` the oracle at that position at **`rtol≈1e-9`**. A cache/offset/
  mask bug fails at the exact step it occurs.
- **Test B (secondary, exact):** `kv_generate` free-runs from the prompt and its token ids
  must equal the frozen greedy sequence exactly.
- **Invariants:** `cache.length` advances by exactly 1 per `decode_step`; `get(layer)` returns
  the contiguous prefix; prefill of length `p` leaves `length == p`.

**402.** Each request's emitted token ids == its standalone `kv_generate`; every active
sequence advances exactly one token per `step`; finished sequences are retired and stop
appearing in `step()` output.

**403.** Paged/prefix-cached **logits** == the 401 contiguous-cache logits at `rtol≈1e-9`
(same oracle); a radix prefix **hit does not recompute** the shared prefix (assert via a
recompute counter / call spy); allocator invariants — no block double-allocated, freed blocks
return to the pool, `get` reconstructs the contiguous K/V exactly from the block table.

### 10.4 Fixtures (reuse 306's tiny Qwen3, extend for decode)

- **Config:** the **exact 306 tiny config** — `V=64, dim=16, n_layers=2, n_heads=4,
  n_kv_heads=2, head_dim=4, ffn=32, base=1e4, eps=1e-6` — so 401–403 inherit 306's
  composed-oracle machinery verbatim. `max_seq_len=64` for cache preallocation.
- **Sequence:** prompt = 306's seeded `input_ids` (len 5); `n_new = 8` greedy tokens.
  `gen_fixtures.py` greedy-decodes the composed float64 oracle for 8 steps and freezes **both**
  the full token sequence (len 13) *and* the full-sequence per-position logits `(1, 13, V)` —
  Test A reads the logits, Test B reads the tokens. HF-anchor at authoring time (`rtol≈1e-3`)
  exactly as 306.
- **Block size (403):** `block_size = 4` for the fixture → a 13-token sequence spans 4 blocks
  (genuine multi-block paging) and a block-aligned shared prefix (len 8 = 2 blocks) exercises
  radix cleanly. README notes production vLLM uses 16; the small value is a fixture choice.

### 10.5 Real-weights demo (Tier C, skippable)

401–403 ship a `download.sh` for **`Qwen/Qwen3-0.6B`** reusing **306's `convert.py`**; the
learner watches their own `kv_generate` / `Engine` produce real text, token-equal to the
stateless 304/306 run, much faster. `@pytest.mark.skipif` when weights are absent — the graded
path stays hermetic on the tiny fixture (no download needed to grade).

### 10.7 Prod-fidelity & performance — "prove the mechanism, not just the output"

L4 is where **correctness stops being the only bar**. Each Track-1 task mirrors a specific
mechanism from a *named production serving framework*, and every task carries at least one
assertion that **fails on a correct-but-naive implementation** — an implementation that
returns the right logits the slow/wasteful way. That single rule is what makes this a systems
course rather than three more forward passes. Because L4 is a pure-NumPy sim (no kernels, no
wall-clock), "performance" is asserted **structurally/analytically** — compute-tensor shapes,
recompute counts, live-block counts — the way one proves an algorithm's complexity, never by
timing. READMEs state these as *guarantees the system must meet* (the contract), never as a
recipe.

**401 — the prefill/decode split & O(1) incremental decode.** Mirrors the universal
engine split every framework makes: a **prefill** phase (whole prompt once, one `L×L`
attention, *compute-bound*) and a **decode** phase (one token, `(1×kv_len)` attention,
*memory-bandwidth-bound* — reloading KV + weights). The cache is a **contiguous preallocated**
per-layer K/V store (HF `StaticCache`, the physical thing vLLM later pages); positions offset
by `cache.length`; the decode mask is `(1×kv_len)`, not square.
*Performance property (tested):* decode does **not recompute the prefix** — assert the score
tensor built per decode step is `(n_heads, 1, kv_len)` (a single query row), so decode is
`O(kv_len)`/token and generation `O(L²)` not the stateless `O(L³)`. Test B's token-parity with
stateless `generate` then proves *same output at 1/L the FLOPs*.

**402 — continuous (iteration-level) batching (Orca → vLLM scheduler).** Mirrors **Orca's
iteration-level scheduling** (= vLLM continuous batching): a *ragged* running set advanced one
token per iteration; an EOS request is **retired mid-batch** and its slot immediately reused by
a waiting request — no head-of-line blocking, unlike static/request-level batching that pads to
the longest and stalls throughput.
*Performance property (tested):* **no wasted compute on finished sequences** — a retired
request never appears in a later `step()` and is not advanced; **slot reuse** — a queued
request begins the very next step after a short one finishes, not after the whole batch drains;
**iteration-level** — every live request advances exactly one token per `step`. A naive
batch-until-all-done scheduler fails the slot-reuse assertion.

**403 — PagedAttention (vLLM) + RadixAttention (SGLang).** Mirrors **PagedAttention** (KV in
fixed-size **blocks** via a per-request **block table**, physically non-contiguous → no external
fragmentation, no "reserve max_seq_len/request" waste) and **RadixAttention** (a prefix/radix
tree so requests sharing a prompt prefix **physically share** its KV blocks, computed once).
*Performance property (tested):* **memory is O(used blocks)** not `O(max_seq_len × requests)`
— assert `⌈tokens/block⌉` live blocks/request, internal fragmentation ≤ one block; **prefix hit
skips recompute** — a call-spy/recompute counter shows a shared prefix's K/V is computed only
for the novel suffix; **shared blocks are physically shared** — N requests with a common prefix
hold `< N×` the blocks; **allocator invariants** — freed blocks return to the pool, no physical
block double-mapped (except intentional read-shared prefix blocks), and paged `get`
reconstructs the exact contiguous K/V 401 would (`rtol≈1e-9`). A 403 that *copies* the shared
prefix produces identical logits but fails the no-recompute and shared-block assertions.

### 10.6 Build order

Prototype **401 end-to-end first** (README + unsolved stub + REAL `solution.py` + tests +
fixtures; `uv run grade -s 401` green and `uv run grade 401` clean-fail in the shipped state per
decision 7) to validate the `KVCache` seam and the teacher-forced harness against working code
**before** 402/403 lean on it. Then 402, then 403.

---

## 11. Track-1 learnings → carry into Track 2/3 (2026-06-30)

Track 1 (401–403) shipped and passed a blind student-sim ("best-designed L4 track"). The
patterns below are what *made* it cohere; reuse them for Tracks 2–3 (they are not re-litigated
per task).

- **Interface-seam is the structural spine (generalize the KVCache seam).** The win in Track 1
  was registering a *thin interface* (the `KVCache` storage seam) separate from the driver
  (`prefill`/`decode_step`), so 402 (one cache/request) and 403 (`PagedKVCache`) substituted
  **behind the seam with the driver unchanged**. Do the same in Track 2: the **collectives API
  (404) over a per-rank array list is the seam**, and `sharded_forward` (405) / the
  multiprocessing capstone (406) plug into it unchanged — 406 swaps the in-process ring for a
  real-IPC ring *behind the same collective interface*, exactly as 403 swapped the cache. In
  Track 3, the **MLA latent cache (407)** reuses the 401→403 cache seam. Register the interface;
  let the internals be the learner's.

- **"Prove the mechanism" applies to every track (§10.7), not just Track 1.** Each Track-2/3
  task needs ≥1 assertion that fails on a correct-but-naive impl, observed through the API:
  *collectives (404)* — ring all-reduce == `np.sum` **and** every rank ends identical **and**
  `dispatch∘combine` == identity (a single-buffer shortcut that ignores the ring must fail);
  *TP/PP (405)* — assert work is *actually sharded* (each rank holds/computes only its slice;
  a one-rank fallback that ignores `world` would still match logits and must fail a
  shard-shape / where-computed check); *expert-parallel (408)* — only local experts compute +
  all-to-all identity; *MLA (407)* — the **analytic** `kv_bytes_per_token` inequality
  (`kv_lora_rank + rope_dim` ≪ `2·n_kv_heads·head_dim`), asserted from the config, not a fixture.

- **One hermetic tiny fixture per track, frozen once.** Track 1 reused 306's exact tiny Qwen3
  config + composed float64 oracle across 401/402/403 (frozen logits/tokens, HF-anchored at
  `rtol≈1e-3`, graded at `rtol=1e-9/atol=0`). Track 2 reuses the *same* Qwen3 oracle
  (sharded/distributed logits == `qwen3_forward`); Track 3 reuses 311's DeepSeek oracle. Freeze
  the reference once in `gen_fixtures.py`; never call L3 live at grade time.

- **Ship convention (decision 7, corrected).** Stub UNSOLVED + `solution.py` REAL. This is what
  makes `grade -s`'s all-solutions stack resolve across a track's dependency chain (405 imports
  404's solution, etc.) — do **not** ship blank/byte-identical solutions.

- **Authoring process that worked.** Subagent-driven, one task at a time, controller verifies
  **inline** (byte-identical? no — now: stub raises + no leaked logic; `grade -s NNN` green;
  `grade NNN` clean `NotImplementedError`-only fail; read any shared-infra change), one whole-
  branch review at the end, then the **blind student-sim gate** (§7 / AGENTS.md) as the real
  acceptance test. Pass Global Constraints to each implementer separately — the task-brief
  extractor only pulls one task's section.
