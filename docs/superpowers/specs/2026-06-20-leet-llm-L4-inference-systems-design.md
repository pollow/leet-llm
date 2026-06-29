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
7. **Ship UNSOLVED.** As in L2/L3: learner stub `<file>.py` and `solution.py` are
   **byte-identical** and raise `NotImplementedError`; validate by temporarily drafting a
   real `solution.py`, running `uv run grade -s 4NN` (L3 deps ARE solved), then reverting
   before commit; confirm the stub fails **cleanly** (only `NotImplementedError`).
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

- Finalize the exact registered API field names per task (the `Engine` / KV-manager /
  collective / MLA-cache signatures) and add them to `leet_llm/_registry.py` as each lands.
- Confirm the `multiprocessing` IPC primitive for `406` (Pipe vs shared-memory vs a tiny
  socket ring) and whether the capstone grades on macOS spawn-start reliably.
- Pin the graded prompts, `max_new_tokens`, world sizes, and block size for each fixture.
- Confirm the `401 → 402 → 403` split granularity holds once the contract is prototyped
  (cache vs scheduler vs paging — guard against the operator-filling smell at the seams).
- Confirm 407's MLA latent-KV cache and the analytic KV-size comparison exercise real
  sparsity at the tiny fixture config (latent dim < full per-head K/V).
