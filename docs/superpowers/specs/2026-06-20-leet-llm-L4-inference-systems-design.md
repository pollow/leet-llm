# leet-llm — Level 4: Inference Systems & Serving (Design)

> The systems level. Learners stop building *models* and start building the *machinery
> that serves them fast*. L3 produced correct-but-slow stateless forwards; L4 keeps the
> arithmetic identical and makes it serveable: a KV-cached, continuously-batched,
> paged, prefix-sharing **single-node engine** on `Qwen3-0.6B`, then **shards a model
> across hosts** (tensor / expert / pipeline parallel + collectives from scratch) on
> `DeepSeek-V3`. A bonus track touches the frontier: DeepSeek-V4's Compressed Sparse
> Attention.

- **Status:** design approved 2026-06-20. Extends the locked ladder in
  `2026-05-31-leet-llm-curriculum-design.md` (§2, L4 row) and builds directly on
  `2026-06-01-leet-llm-L3-whole-model-inference-design.md` and the L3 Track-C zoo
  (`2026-06-20-leet-llm-L3-trackC-oss-zoo.md`). Still the **NumPy phase** — zero
  framework autograd; PoC/simulation, not GPU kernels or real clusters.
- **Level goal:** turn the L3 stateless forwards into *serving systems*. Understand the
  stateless→stateful shift (the **KV-cache**), online/tiled attention (**FlashAttention**),
  KV memory management (**paged + prefix/radix**), request scheduling (**continuous
  batching**), and scaling a model past one device (**collectives, tensor / expert /
  pipeline parallelism, disaggregated serving**). The satisfying payoff: the learner's
  own L3 model generating the *same tokens*, much faster, behind a mini-vLLM engine — and
  then running across real OS processes.

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
itself — no new external oracle is needed for the single-node and multi-host tracks (the
V4 bonus is the one exception; it grades against the golden HF class).

---

## 2. Anchors (two focus models + one bonus)

L4 does **not** re-cover the whole OSS zoo. It anchors on two models — one per track — and
lists the rest as one-line extension notes. The anchors were chosen so the systems lessons
land on the best-fit architecture while **reusing L3 heavily**.

| Track | Anchor | Why this model | L3 reuse |
|-------|--------|----------------|----------|
| **Single-node serving** | **Qwen3-0.6B** (task 306) | The most popular community OSS model with a small, **ungated** (Apache-2.0) real checkpoint (~0.6 B). Dense GQA + qk-norm → the serving systems stay model-agnostic and clean, and the learner gets a **real fast-generation demo** (Tier C). | `qwen3_forward`, `load_qwen3`, `Qwen3Config` (306); `generate`/`sample` (304); `sdpa` (205) |
| **Multi-host** | **DeepSeek-V3** (task 311) | The frontier MLA + MoE architecture (latent-KV attention, 256 routed + 1 shared expert). MoE → **expert parallelism** is *the* reason modern models go multi-host; MLA → **MLA-aware sharding** is the cutting-edge bonus. Already built in L3, so it **reuses the most**. | `deepseek_forward`, `mla_project`, `load_deepseek` (311); `moe_ffn` (308); `rope_half` (213) |
| **Bonus** | **DeepSeek-V4** | The frontier *inference* architecture: a Lightning Indexer drives **Compressed Sparse Attention** (compress the KV sequence, attend to only top-`k` compressed blocks per query) — i.e. the architecture *is* a sparse/compressed KV optimization, squarely an L4 topic. transformers 5.9 ships a **golden built-in `DeepseekV4ForCausalLM`** → no remote code; random-init a shrunk config for the oracle. | (stretch — not required to reuse L3) |

**Why not Mixtral / Llama for multi-host:** DeepSeek-V3's MoE gives the same expert-parallel
lesson Mixtral would, *and* reuses an L3 task we already built (Mixtral 308's forward would
also work, but V3 adds MLA-aware sharding — the frontier delta — for free). Plain dense
Llama would miss expert parallelism entirely.

**Why V4 is bonus, not an anchor:** the golden `DeepseekV4` module is large (CSA + HCA
caches, two compressors, the indexer, hyper-connection, hash router, `sqrtsoftplus` MoE,
MTP) and **does not reuse L3's MLA** (V4 dropped `kv_lora_rank` for compressed-sparse
attention). A faithful whole-V4 build is heavier than any L3 task, so we isolate only the
inference-relevant core (CSA + Indexer) as an optional stretch.

---

## 3. The grading model: contract, not recipe (and logic-perfect)

L0–L3 grade a *prescribed function* by activation parity. L4 grades a *student-designed
system* by a **conformance harness** bound only to a thin registered API. Three mechanisms,
no new external oracle (except the V4 bonus):

### 3.1 The logits/token-id output contract (the spine)

Every L4 system exposes an output the harness can compare **exactly**:

- return **logits** → compared to the learner's L3 stateless logits at **`rtol≈1e-9`**
  (float64 equivalence, the L3 tolerance), **or**
- return **token ids** → compared to the learner's L3 greedy/sampled token sequence at
  **exact equality**.

The internals are the learner's design, but the API's *return type* is **pinned** to logits
or token ids. This is what reconciles "the learner owns the architecture" with "graded
logic-perfectly." Concretely: an `Engine.step()` returns each active request's next **token
id** (full generated sequence checked against `generate`, 304); a `sharded_forward(...)`
returns **logits** (checked against `deepseek_forward`, 311); `flash_attention(...)` returns
the attention output tensor (checked against `sdpa`, 205).

### 3.2 The equivalence oracle = L3 itself

Because every L4 path is a pure speed/memory delta over L3's arithmetic, **the reference is
the learner's own L3 forward.** A KV-cached / paged / flash / sharded run must reproduce the
*same* logits/tokens as the stateless L3 run on the same input. This is the same trick L3
used internally (its locked decision: "KV-cache is a pure speed delta"). It is free, exact,
and self-contained — the learner's earlier code is the oracle, resolved through the
`leet_llm` facade exactly as in L3. (Authoring-time anchor: the fixture's stateless logits
are themselves frozen from the L3 float64 oracle / golden HF class, as in L3.)

### 3.3 Invariants + behavioral assertions

The defining properties of each *system* — not just its output:

- **memory**: paged cache uses O(blocks) not O(L²); FlashAttention never materializes the
  L×L score matrix; the block allocator never double-allocates and returns freed blocks.
- **scheduler**: every active sequence advances exactly one token per `step`; finished
  sequences are retired; a prefix cache **hit does not recompute** the shared prefix.
- **collectives**: ring all-reduce == naive `np.sum`; every rank ends with identical
  reduced state; `dispatch ∘ combine` (all-to-all) is the identity.
- **reconstruction**: paged gather reconstructs the contiguous KV exactly; the online
  softmax running stats after all tiles equal the one-shot softmax stats.

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
   oracle is L3 itself; no new external oracle for the single-node/multi-host tracks.
4. **Two anchors + one bonus** (§2): Qwen3-0.6B (single-node), DeepSeek-V3 (multi-host),
   DeepSeek-V4 CSA (bonus). The rest of the zoo = extension notes.
5. **Hybrid multi-host** (§3.4): graded tasks in-process/deterministic; one real
   `multiprocessing` capstone, **skippable** like L3's Tier-C real-weights capstone.
6. **Pure NumPy, PoC/simulation.** No GPU kernels, no real distributed runtime,
   no autograd.
7. **Ship UNSOLVED.** As in L2/L3: learner stub `<file>.py` and `solution.py` are
   **byte-identical** and raise `NotImplementedError`; validate by temporarily drafting a
   real `solution.py`, running `uv run grade -s 4NN` (L3 deps ARE solved), then reverting
   before commit; confirm the stub fails **cleanly** (only `NotImplementedError`).

---

## 5. The task list (≈5 + bonus)

Ids are `4xx`, ordered so "finish in order" holds (no task depends on a later one). Each
ships **README + stub + tests only**. Signatures below are the **registered contract
boundary** (what the grader binds to); the per-task scaffold finalizes exact field names.
Everything *inside* the contract is the learner's design.

### Track 1 — Single-node serving · anchor **Qwen3-0.6B** (real demo)

| Id | Task | Registered contract (returns logits / token ids) | The system to build | Reuses | Oracle |
|----|------|--------------------------------------------------|---------------------|--------|--------|
| **401** | `inference_engine` | `Engine(params, cfg)`; `add_request(prompt_ids) -> req_id`; `step() -> list[(req_id, next_token_id)]` | KV-cache + continuous-batching engine: prefill, then incremental decode reusing cached K/V, ragged multi-sequence batch, admit/retire | 306, 304, 205 | each request's generated **token ids** == L3 stateless `generate` (304); every active seq advances exactly 1 token/step |
| **402** | `kv_memory` | a KV-manager the engine plugs into: `allocate / append / gather` over fixed-size blocks + `match_prefix / insert` (radix) | paged KV (block table) **and** prefix/radix sharing across common prompts | 401 | paged/prefix-cached **logits** == contiguous-cache logits (`rtol≈1e-9`); shared prefix not recomputed; allocator invariants |
| **403** | `flash_attention` | `flash_attention(q, k, v, mask, tile) -> out` | tiled online-softmax attention; never materialize the L×L score matrix | 005/006, 205 | output == one-shot `sdpa` (205) at `rtol≈1e-9`; running stats == global softmax; O(tile) memory |

*Extension notes (not tasks):* 305 → ring-buffer eviction (sliding window); 307 →
long-context decode under the scaled `inv_freq`; 309 → evict-but-keep-sinks (StreamingLLM);
speculative decoding (draft-and-verify) and quantized KV cache.

### Track 2 — Multi-host · anchor **DeepSeek-V3** · hybrid process model

| Id | Task | Registered contract (returns logits / token ids) | The system to build | Reuses | Oracle |
|----|------|--------------------------------------------------|---------------------|--------|--------|
| **404** | `sharded_forward` | `all_reduce / all_gather / reduce_scatter / all_to_all(shards)` + `sharded_forward(input_ids, sharded_params, cfg, world) -> logits` | collectives from scratch (ring) + tensor / expert / pipeline parallelism (incl. **MLA-aware** sharding) over a list of ranks, **in-process** | 311, 308 MoE | sharded **logits** == `deepseek_forward` (311) at `rtol≈1e-9`; ring all-reduce == `np.sum`; `dispatch ∘ combine` == identity |
| **405** | `distributed_serve` *(capstone, skippable)* | a real `multiprocessing` harness: N workers, each holds one shard, ring all-reduce / all-to-all over the learner's IPC backend → `serve(input_ids) -> token ids` | the same sharding, but over **real OS processes** with genuine IPC | 404, 401 | distributed **token ids** == single-process 311; `@pytest.mark.skipif` when the process group can't start |

*Extension notes (not tasks):* disaggregated prefill/decode (separate prefill/decode
workers + KV handoff); collective-comm overlap.

### Bonus — frontier inference architecture

| Id | Task | Registered contract | The system to build | Oracle |
|----|------|---------------------|---------------------|--------|
| **490** | `compressed_sparse_attention` *(bonus)* | `compressed_sparse_attention(...) -> out` | Lightning Indexer + Compressed Sparse Attention: compress the KV sequence, select top-`k` compressed blocks per query, attend only to those | == golden tiny-random `DeepseekV4` CSA block (transformers 5.9, **no remote code**); `rtol` per the float64/float32 split (§6) |

---

## 6. Testing & fixtures

L4 inherits L3's fixture machinery (`gen_fixtures.py` per task, committed `.npz`, the gen
group's `torch`/`transformers` used only at authoring), specialized to the contract model:

- **Equivalence fixtures (the graded oracle).** `gen_fixtures.py` freezes a tiny seeded
  config's **stateless L3 output** — logits and the greedy/sampled token sequence — from the
  same float64 oracle L3 uses (the composed-oracle or golden HF class, anchored at authoring
  time, exactly as the L3 zoo). The L4 harness then drives the learner's *system* and asserts
  its returned logits/token-ids match that frozen stateless reference (§3.1–3.2). No real
  weights are needed for the single-node/multi-host graded path — the tiny config fits in the
  fixture (hermetic, like L3 Tier 1).
- **Invariant/behavioral tests** (§3.3) accompany every task — memory bounds, scheduler
  advance/retire, prefix-hit-skips-recompute, ring-collective identities, paged gather
  reconstruction, online-softmax running stats.
- **The real-weights demo (Track 1).** `401`/`402` ship a `download.sh` for `Qwen/Qwen3-0.6B`
  (reusing 306's converter): the learner watches their *own* engine generate real text fast,
  with output token-equal to the stateless 304/306 run — the Tier-C payoff. Skippable.
- **The multi-process capstone (Track 2, `405`).** Skippable, gated on the process group
  starting; grades distributed token-ids == single-process 311.
- **The V4 bonus** is the one task that grades against an **external** oracle: a golden
  tiny-random `DeepseekV4ForCausalLM` (shrunk `DeepseekV4Config`, random init, float64),
  frozen in `gen_fixtures.py`. No remote code, no real weights.

**Tolerances:** logit equivalence at `rtol≈1e-9`/`atol≈1e-9` (float64, L3's bar); token-id
equivalence is exact; the V4 golden anchor follows L3's float32 accumulation looseness where
the genuine class upcasts.

---

## 7. README rules for L4 (per curriculum §1)

Each task README keeps the fixed shape (**Description · The Math/The Contract · Function
Signatures · Read More · How to Test**) but at system altitude:

- **State the contract, not the recipe.** Give the **registered API** the harness drives, the
  **guarantees** it must meet (the equivalence oracle + invariants), and the **GIVEN systems
  facts** (block size, world size, the IPC primitive available, the tile size convention).
  Do **not** enumerate internal sub-functions or a step-by-step build — the architecture is
  the learner's to design.
- **Name the technique and cite the paper** (FlashAttention; PagedAttention/vLLM; RadixAttention/
  SGLang; Megatron-LM tensor parallelism; GShard/Switch expert parallelism; GPipe/1F1B
  pipeline parallelism; DeepSeek-V3 MLA; DeepSeek sparse attention for the V4 bonus) — the
  *what/why*, never the *how*.
- READMEs never mention `grade -s`. The `→ L4` lines L3 left behind (305 windowed eviction,
  307 long-context decode, 309 streaming-sink eviction, 311 latent-KV cache) are now realized
  as the engine/extension notes here — cross-reference them.

---

## 8. Out of scope (L4) → deferred

- **Real GPU kernels / Triton / CUDA** — L4 is a NumPy PoC of the *algorithms* (online
  softmax, block tables, ring collectives), not performant kernels.
- **A real distributed runtime** (NCCL/MPI/torch.distributed) — collectives are hand-written;
  the capstone uses stdlib `multiprocessing` only.
- **Backward passes / autograd / training / LoRA / RL** → L5–L6.
- **Quantization** beyond a KV-cache quantization note → out of scope for v1.
- **Beam search** (L3 already deferred it) — speculative decoding appears only as an
  extension note.
- **A faithful whole DeepSeek-V4 forward** (hyper-connection, hash router, MTP) — only the
  CSA + Indexer core is in scope, as the bonus.
- **Shipping model weights** — always `download.sh` + `convert.py`; only tiny fixtures
  committed.

---

## 9. Open items (resolve at scaffold time)

- Finalize the exact registered API field names per task (the `Engine` / KV-manager /
  collective signatures) and add them to `leet_llm/_registry.py` as each lands.
- Confirm the `multiprocessing` IPC primitive for `405` (Pipe vs shared-memory vs a tiny
  socket ring) and whether the capstone grades on macOS spawn-start reliably.
- Pin the graded prompts, `max_new_tokens`, world sizes, block size, and tile size for each
  fixture.
- Verify the shrunk `DeepseekV4Config` for the bonus instantiates and runs the genuine class
  at a tiny size (CSA needs `seq_len > compress_rate · index_topk` to exercise sparsity);
  confirm which submodule boundary to grade (CSA block vs whole tiny model).
- Decide whether `401` and `402` are two tasks or one larger "serving engine" task once the
  contract is prototyped (granularity check against the operator-filling smell).
