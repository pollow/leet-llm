# L4 Track 1 — Qwen3 Serving (`kv_cache` · `continuous_batching` · `paged_kv`) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development`,
> run with the **lean loop** the user standardized for this curriculum (L3 Track C directive):
> implementer subagent → controller verifies **inline** (byte-identical stub==solution, leak
> grep, `uv run grade -s 4NN` green + `uv run grade 4NN` fails cleanly with only
> `NotImplementedError`, read any risky file) → fix inline or one focused fix dispatch → one
> ledger line. **No per-task reviewer subagent.** One whole-branch review at the very end, then
> student-simulation agents.

**Goal.** Author L4 Track 1: turn the L3 stateless Qwen3 forward (306) into a *serving system*
in three tasks — **401 `kv_cache`** (stateful prefill/decode), **402 `continuous_batching`**
(iteration-level scheduler), **403 `paged_kv`** (PagedAttention block table + RadixAttention
prefix sharing). Anchor model: Qwen3-0.6B. Each ships **README + stub + tests + committed
fixtures**, shipped **UNSOLVED**. This is the **NumPy phase** — no autograd, no kernels; a
PoC/simulation of the *algorithms*.

**Design source of truth:** `docs/superpowers/specs/2026-06-20-leet-llm-L4-inference-systems-design.md`,
especially **§10** (Track-1 scaffold decisions) and **§10.7** (prod-fidelity & performance).
Read §10 before implementing any task — it holds the rationale the briefs below compress.

**Tech stack:** Python 3.11+, NumPy 2.x (runtime), `uv`. Authoring-only `gen` group
(`torch` + `transformers` 5.9) is used *only* by `gen_fixtures.py` — never at grade time.
The reuse facade is `from leet_llm import <name>`; add L4 names to `leet_llm/_registry.py`.

---

## Global Constraints (bind every task — copy into each implementer + final-review dispatch)

1. **Stub UNSOLVED, `solution.py` REAL (repo convention — overrides design decision 7).**
   Per CLAUDE.md ("`solution.py` is the only place the full implementation lives") and every
   committed task (301–311), ship the **student stub `<slug>.py` raising `NotImplementedError`**
   and a **REAL, working `solution.py`** — NOT byte-identical, NOT reverted. This is what makes
   `grade -s`'s all-solutions stack work (402 imports 401's solution, etc.). Validate in the
   **shipped state**: `uv run grade -s 4NN` is GREEN (real solution) and `uv run grade 4NN`
   FAILS CLEANLY (student stub, *only* `NotImplementedError`, no import/collection error). Commit
   README + tests + fixtures + the UNSOLVED stub + the REAL solution + registry entry. The stub
   must never leak logic (CLAUDE.md Stub-vs-solution principle).
2. **Reuse is mandatory (decision 8).** The re-authored forward imports L2/L3 **primitives**
   from the facade — `embedding`, `rms_norm`, `qk_norm`, `rope_half`, `sdpa`, `affine`,
   `group_last_axis`, `ungroup_last_axis`, `swiglu_ffn`, `add_residual`, `masked_fill`,
   `triangular_mask`, `softmax`, `argmax`/`sample` — and **never re-inlines** them. But it
   **must NOT call `qwen3_forward` / `qwen3_decoder_block`** (those are stateless and discard
   K/V); the per-layer decode loop with the K/V seam is re-authored. L3 is *oracle + primitive
   library, not machinery*. `qwen3_decoder_block`/`_rope_gqa_qk_norm` in 306 are unregistered
   and off-limits.
3. **The grading oracle is a frozen fixture, hermetic.** Per L3 Tier A, `gen_fixtures.py`
   freezes the **composed float64 torch oracle** output (the exact `_composed_oracle` machinery
   from `306_qk_norm/tests/gen_fixtures.py`) at authoring time; the committed `.npz` golden is
   what grade-time compares against — **no live L3 call at grade time**, no real weights needed
   to grade. HF-anchor at authoring: assert the composed oracle matches genuine
   `Qwen3ForCausalLM` at `rtol≈1e-3` (proves the oracle faithful, non-circular), exactly as 306.
4. **Fixture config (all three tasks share it).** The exact 306 tiny Qwen3 config:
   `V=64, dim=16, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=4, ffn(intermediate)=32,
   rope_base=1e4, norm_eps=1e-6, qk_norm_eps=1e-6`, and `max_seq_len=64` for cache
   preallocation. Graded sequence: **prompt = 306's seeded `input_ids` (len 5)**, `n_new = 8`
   greedy tokens (total len 13). `gen_fixtures.py` greedy-decodes the composed oracle for 8
   steps and freezes **both** the full token id sequence (len 13) **and** the full-sequence
   per-position logits `(1, 13, V)`. **403** additionally uses `block_size = 4`.
5. **Tolerances.** Logit equivalence `rtol=1e-9, atol=0` (or `1e-12`), never `atol=1e-9`;
   token-id equivalence is **exact**.
6. **Masks are boolean** (project contract): `True` = masked/hidden. Additive `-inf` form is at
   most an internal detail of an operator, never the API. The decode mask is `(1 × kv_len)`.
7. **Prove the mechanism, not just the output (§10.7).** Every task carries ≥1 assertion that
   **fails on a correct-but-naive implementation** (recompute/shape/live-block checks), plus the
   correctness parity. Tests **encode invariants**, so a partial/wrong stub fails loudly — and
   any numpy-only invariant test must observe the property **through the registered API** (not
   recompute it in the test), so it can't pass against a `NotImplementedError` stub vacuously.
8. **README = tutorial at system altitude (§7 + README Tutorial Standard).** Fixed shape
   (Description · The Contract/The Math · Function Signatures · Read More · How to Test). State
   the **registered API + guarantees + GIVEN systems facts**, cite the **paper/framework**
   (PagedAttention/vLLM, RadixAttention/SGLang, Orca continuous batching, HF StaticCache), teach
   *what/why/cost* — **never** the internal recipe or a sub-function checklist, **never** mention
   `grade -s`. Cross-reference the `→ L4` breadcrumbs L3 left (305 windowed / 309 sink eviction
   as extension notes).
9. **Folder + registry.** Slugs: `401_kv_cache/`, `402_continuous_batching/`, `403_paged_kv/`.
   Each folder: `README.md`, `<slug>.py` (stub==solution), `solution.py`, `tests/` (with
   `gen_fixtures.py`, committed `fixtures/*.npz`, `test_*.py`), and for 401 a `download.sh`
   (reuse `306_qk_norm/convert.py`) for the skippable real-weights demo. Add the registered
   names (§ each task) to `leet_llm/_registry.py`.
10. **Grader.** `uv run grade 4NN` (student), `uv run grade -s 4NN` (solution, authoring only).
    Mirror the `tests/` + `conftest.py` layout of an existing L3 task (e.g. `306_qk_norm`).

---

## Task 401: `kv_cache` — stateful prefill/decode (the K/V seam)

**Where it fits:** first L4 task; converts the stateless Qwen3 forward (306) into a stateful
engine. Establishes the `KVCache` interface that 402 (one cache/request) and 403 (paged cache)
reuse *unchanged*. Mirrors the universal prefill/decode split + HF `StaticCache` (§10.7).

**Read first:** design §10.1–10.4, §10.7 (401 paragraph); `306_qk_norm/solution.py`
(`qwen3_forward`, `qwen3_decoder_block`, `_rope_gqa_qk_norm` — the math to re-author with a
cache seam) and `306_qk_norm/tests/gen_fixtures.py` (`_composed_oracle` to reuse for the golden).

**Registered contract (add to `_registry.py`):**
- `class KVCache` — per-layer preallocated K/V storage. `KVCache(cfg)`; `append(layer, k, v)`
  (append this step's keys/values for one layer); `get(layer) -> (K, V)` returning the
  contiguous cached K/V of length `self.length`; `length` (int, tokens cached). Per-layer store
  shape `(n_kv_heads, max_seq_len, head_dim)`. **GQA-specific** (§4a corollary — do NOT
  generalize to MLA).
- `prefill(prompt_ids, params, cfg, cache) -> logits` — re-authored full-prompt Qwen3 forward
  at `positions = arange(0, len)`, fills `cache` for every layer, returns **last-position**
  logits `(1, V)`.
- `decode_step(token_id, params, cfg, cache) -> logits` — single-token forward at
  `positions = [cache.length]` with a `(1 × kv_len)` causal mask (all visible — the new query
  attends to all cached keys), appends its per-layer K/V, returns logits `(1, V)`.
- `kv_generate(prompt_ids, params, cfg, n_new) -> list[int]` — greedy driver: `prefill` then
  `n_new` × (`argmax` → `decode_step`); returns prompt+generated ids. The Tier-C demo entry.

**Fixtures (`tests/gen_fixtures.py`, authoring-only):** reuse 306's `_composed_oracle`; config +
sequence per Global Constraint 4. Greedy-decode the oracle 8 steps; save to
`fixtures/kv_cache.npz`: `input_ids` (1,5), `token_ids` (13,), `logits` (1,13,V), all HF-named
weights, and the config scalars (as 306 does). HF-anchor assert at `rtol≈1e-3`.

**Tests (`tests/test_kv_cache.py`):**
- **A — teacher-forced logits (primary, loud):** `prefill(prompt)` then `decode_step` through
  the *known frozen* `token_ids[5:]`; assert each step's logits ≈ `logits[0, pos]` at
  `rtol=1e-9, atol=0`. Localizes any offset/mask/append bug to its step.
- **B — free-run tokens (exact):** `kv_generate(prompt, n_new=8)` == `token_ids` exactly.
- **Perf/mechanism (§10.7, fails-on-naive):** assert `decode_step` builds a `(n_heads, 1,
  kv_len)` score row — i.e. the prefix is **not** recomputed. Observe via the registered path
  (e.g. spy on `sdpa`/`get`, or expose the per-step score shape through the cache) so a stateless
  re-forward would fail it.
- **Invariants:** `cache.length` advances by exactly 1 per `decode_step`; after `prefill` of
  length `p`, `cache.length == p`; `get(layer)` returns exactly the contiguous cached prefix.

**README.md:** teach the stateless→stateful shift, prefill (compute-bound) vs decode
(memory-bandwidth-bound), why KV-cache is the highest-leverage inference optimization, the
`(1×kv_len)` decode mask and offset positions. Cite HF `StaticCache`. State the `KVCache`
interface as the contract 402/403 build on. `download.sh` + note on the skippable real demo.

**Definition of done:** Global Constraints 1 (ship-unsolved validated), 3 (hermetic golden +
HF-anchor), 7 (mechanism assertion present). Ledger line.

---

## Task 402: `continuous_batching` — iteration-level scheduler (Orca / vLLM)

**Where it fits:** wraps 401's `KVCache` + `decode_step` into a multi-request engine. Mirrors
**Orca iteration-level scheduling** = vLLM continuous batching (§10.7).

**Read first:** design §10.2 (402 row), §10.7 (402 paragraph); Task 401's shipped `KVCache` /
`prefill` / `decode_step` / `kv_generate` interface (it is the primitive 402 drives).

**Registered contract (add to `_registry.py`):**
- `class Engine` — `Engine(params, cfg)`; `add_request(prompt_ids) -> req_id`;
  `step() -> list[tuple[req_id, int]]` (each **currently-running** request's next token id,
  exactly one per step; admits waiting requests onto free slots, prefilling on admission);
  `is_finished(req_id) -> bool`. Holds one 401 `KVCache` per live request; retires a request on
  EOS (or when it hits its length budget).

**Fixtures (`tests/gen_fixtures.py`):** reuse 401's frozen per-request oracle — freeze the greedy
`token_ids` (via the composed oracle) for **each** of a few prompts of **different lengths** so
raggedness/retirement are exercised. Save `fixtures/continuous_batching.npz` with the prompts and
their expected token sequences (+ weights/config as 401).

**Tests (`tests/test_continuous_batching.py`):**
- **Correctness:** each request's emitted ids (concatenated across `step`s) == its standalone
  `kv_generate` (401) exactly, regardless of interleaving with other requests.
- **Perf/mechanism (§10.7, fails-on-naive):** **slot reuse** — after a short request retires, a
  waiting request begins on the **next** `step` (not after the batch drains); **no wasted
  compute** — a retired request never appears in a later `step()` output; **iteration-level** —
  every running request advances exactly one token per `step`. A batch-until-all-done scheduler
  fails slot-reuse.
- **Invariants:** admitting more requests than free slots queues the overflow; `is_finished`
  flips exactly when a request emits EOS/hits budget.

**README.md:** contrast static/request-level batching (pad-to-longest, head-of-line blocking,
wasted throughput) vs continuous/iteration-level (Orca paper, vLLM scheduler); the running-set +
waiting-queue model; admit-on-free-slot, retire-on-EOS. State the `Engine` contract only — the
scheduler policy internals are the learner's.

**Definition of done:** Global Constraints as 401. Ledger line.

---

## Task 403: `paged_kv` — PagedAttention block table + RadixAttention prefix sharing

**Where it fits:** replaces 401's contiguous `KVCache` with a **paged** manager the engine plugs
into *unchanged* (same interface), and adds cross-request prefix sharing. Mirrors
**PagedAttention (vLLM)** + **RadixAttention (SGLang)** (§10.7). Capstone of the KV arc; the
GQA-vs-MLA comparison is a *forward* reference realized later in 407.

**Read first:** design §10.2 (403 row), §10.7 (403 paragraph); Task 401's `KVCache` interface
(403's `PagedKVCache` must satisfy it so `prefill`/`decode_step` run over it unchanged).

**Registered contract (add to `_registry.py`):**
- `class PagedKVCache` — satisfies the **401 `KVCache` interface** (`append(layer, k, v)` /
  `get(layer) -> (K,V)` / `length`) but over fixed-size **blocks**; constructed
  `PagedKVCache(cfg, block_size)`. Adds `allocate()` (grab a free block), `free()` (return a
  request's blocks to the pool), and exposes `block_table` (logical position → physical block).
- `class RadixCache` — prefix sharing: `match_prefix(ids) -> (node, matched_len)`;
  `insert(ids, ...)`. A shared prefix's blocks are **reference-shared**, not copied, and a hit
  **must not recompute** the prefix's K/V.

**Fixtures (`tests/gen_fixtures.py`):** reuse 401's frozen logits/token oracle (same config,
`block_size = 4`). Add a **shared-prefix** scenario: two prompts sharing a block-aligned prefix
(len 8 = 2 blocks). Save `fixtures/paged_kv.npz`.

**Tests (`tests/test_paged_kv.py`):**
- **Correctness:** running 401's `prefill`/`decode_step` over `PagedKVCache` reproduces the 401
  contiguous-cache logits at `rtol=1e-9` (paged `get` reconstructs contiguous K/V exactly).
- **Perf/mechanism (§10.7, fails-on-naive):** **O(used blocks)** — a request of `t` tokens holds
  exactly `⌈t/block_size⌉` live blocks (internal frag ≤ one block), not `max_seq_len`; **prefix
  hit skips recompute** — a call-spy/recompute counter shows the shared prefix's K/V computed
  only for the novel suffix; **physical sharing** — two requests with a common prefix hold
  `< 2×` the blocks (shared blocks are the same physical ids). A copy-the-prefix impl passes
  correctness but fails these.
- **Allocator invariants:** no physical block double-allocated (except intentional read-shared
  prefix blocks); `free()` returns blocks to the pool; re-allocation reuses freed blocks.

**README.md:** PagedAttention (block table, non-contiguous KV, no external fragmentation, kills
the reserve-max_seq_len waste) + RadixAttention (prefix/radix tree, automatic KV reuse). State
`block_size` as a GIVEN systems fact (fixture uses 4; note production vLLM uses 16). State the
two class contracts + the guarantees; the block-allocator and tree internals are the learner's.
Cross-reference 305 windowed / 309 streaming-sink eviction as extension notes; 407 will contrast
this GQA cache against MLA.

**Definition of done:** Global Constraints as 401. Ledger line.

---

## After all tasks — validation

1. **One whole-branch review** (final code-reviewer, most capable model) over
   `merge-base..HEAD`, pointed at the Global Constraints + the Minor roll-up.
2. **Student-simulation agents (fresh context).** Dispatch, per task, an agent that is given
   *only* the student-visible material (README + blank stub + tests, **not** the solution or the
   design doc) and asked to implement from scratch, then report: was it *solvable without
   guessing hidden design intent* (informative enough) and *non-trivial* (challenging enough)?
   Surface any README gap or leak. This is the real acceptance test of the course design.
