# 403 — Paged KV Cache: PagedAttention + RadixAttention

**Level 4 · Track 1 — Inference Systems & Serving**

## Description

Task 401 gave every sequence a **contiguous** `KVCache`: one preallocated buffer of
shape `(n_kv_heads, max_seq_len, head_dim)` per layer. That is exactly what HF's
`StaticCache` does, and it has a fatal problem at serving scale — you must **reserve
`max_seq_len` per request up front**, even though most requests generate far fewer
tokens. With hundreds of concurrent requests that reservation dominates GPU memory and
most of it sits empty. Worse, the buffers are contiguous, so freeing a short request
leaves a hole too small for a long one: **external fragmentation**.

Production serving fixes both problems the same way an operating system manages RAM —
with **paging**.

**PagedAttention (vLLM).** Chop the KV store into fixed-size **blocks** of `block_size`
tokens, drawn from a shared **pool**. Each request keeps a **block table** mapping its
logical token positions to physical block ids. The blocks a request holds need not be
contiguous in memory, so:

- there is **no external fragmentation** — any free block fits any request;
- there is **no reserve-`max_seq_len` waste** — a request of `t` tokens holds exactly
  `ceil(t / block_size)` blocks, growing one block at a time as it decodes. Internal
  fragmentation is at most one partly-filled block.

Attention itself is unchanged: `get(layer)` gathers a request's logical blocks back
into the same contiguous `(n_kv_heads, length, head_dim)` K/V that 401 produced, so
**401's `prefill` / `decode_step` run over the paged cache unmodified** and the logits
are bit-for-bit the same.

**RadixAttention (SGLang).** Once KV lives in shareable blocks, requests that begin
with the **same prompt prefix** (a shared system prompt, a few-shot preamble, a chat
history) can **physically share** the blocks holding that prefix's KV — computed
**once**, referenced many times. A **radix tree** keyed by token ids remembers which
blocks cache which prefixes; a new request looks up its longest cached prefix,
**reference-shares** those blocks (a refcount bump — never a copy), and only computes
the KV for its **novel suffix**. A block is returned to the pool only when its **last**
holder frees it.

**What changes vs 401.** Nothing about the Qwen3 math, and nothing about the
`prefill` / `decode_step` forward. The only change is *where the K/V lives* and *who
may share it*: a `BlockPool` + a paged cache with the same `append` / `get` / `length`
seam, plus a radix tree for prefix reuse.

## The Contract

`PagedKVCache` is a **drop-in replacement** for 401's `KVCache` — it satisfies the
same interface, so treat `append` / `get` / `length` as fixed (401's forward calls
them). The paging and the tree internals are yours to design.

### `BlockPool` — the shared physical store

- `BlockPool(cfg, block_size)` — a pool of fixed-size physical blocks. Each block stores
  **two** arrays, each of shape `(n_layers, n_kv_heads, block_size, head_dim)` — one for
  post-RoPE keys and one for raw values — for `block_size` tokens across all layers.
  `block_size` is also a **public attribute** (`pool.block_size`) readable by downstream
  classes (e.g. `RadixCache`).
- `allocate() -> int` — hand out a free block id (reusing a freed one when available),
  and mark it referenced once.
- `incref(bid)` — add a reference (a second request adopting a shared block).
- `free(bid)` — release **one** reference; the block returns to the free list only when
  its refcount reaches zero.
- `capacity` — physical blocks materialised so far. `n_free` — blocks currently free.

### `PagedKVCache` — one sequence, paged (satisfies the 401 `KVCache` interface)

- `PagedKVCache(cfg, block_size, pool=None)` — build over a shared `BlockPool`
  (a fresh private pool when `None`). `block_size` is a **GIVEN** systems fact.
- `append(layer, k, v)` — write this step's `(n_kv_heads, t, head_dim)` keys/values for
  one layer, **allocating any new blocks the write crosses into**.
- `get(layer) -> (K, V)` — gather the paged K/V into the **exact contiguous** prefix
  401 returns, each shaped `(n_kv_heads, length, head_dim)`.
- `length` — tokens cached so far (advances by exactly 1 per `decode_step`).
- `allocate() -> int` — grab a free block and append it to `block_table`.
- `reuse_prefix(block_ids, length)` — adopt a reference-shared, **block-aligned**
  prefix (a radix hit): point `block_table` at the blocks that already hold the
  prefix's K/V, bump their refcount, and set `length` — so the prefix is **not
  recomputed**.
- `free()` — return this request's blocks to the pool.
- `block_table` — logical block index → physical block id.

### `RadixCache` — prefix sharing (RadixAttention)

- `RadixCache(pool)` — a radix tree over the same pool the caches use. `block_size` is
  read from `pool.block_size`.
- `insert(ids, block_ids)` — record that a **block-aligned** prefix `ids` is cached by
  `block_ids` (`len(ids) == len(block_ids) * block_size`).
- `match_prefix(ids) -> (node, matched_len)` — the longest cached, **block-aligned**
  prefix of `ids`; `matched_len` is a whole number of blocks. Returns `(None, 0)` on a
  miss. **`node.block_ids` is the cumulative list of physical block ids from the root to
  that node, in order** — not just the node's own edge, but every block needed to
  reconstruct the full matched prefix. You can pass it directly to
  `reuse_prefix(node.block_ids, matched_len)`. **Reuse those blocks; do not recompute
  them.**

**GIVEN systems facts.**
- `block_size` is fixed for the pool (the fixture uses **4**; production vLLM uses
  **16**). It trades internal fragmentation (smaller = less waste) against block-table
  and kernel overhead (larger = fewer blocks to track).
- Prefix sharing is **block-granular**: only whole shared blocks can be reused, so
  `match_prefix` rounds a match down to a multiple of `block_size`.
- The cache remains **GQA-specific** (one entry per KV head), exactly as 401. MLA
  (task **407**) caches a compressed latent instead — a *different* paged object; this
  task will be the GQA side of that comparison.

## The Math

There is no new tensor math — the guarantee here is about **memory and reuse**, and it
is expressed as invariants over the block table and the pool.

Let `p = block_size`. A sequence of `t` cached tokens occupies logical blocks
`0 … ceil(t/p) - 1`; token at absolute position `pos` lives at

```
logical_block = pos // p          physical = block_table[logical_block]
slot          = pos %  p          store[physical][layer, :, slot, :] = k/v
```

so `len(block_table) == ceil(length / p)` at all times — **O(used blocks)**, never
`max_seq_len`. `get(layer)` walks `block_table`, slices `store[physical][layer]` for
each block (trimming the last to `length`), and concatenates → the identical contiguous
K/V 401 built.

For two sequences with a common prefix of `s` tokens (`s` a multiple of `p`), a correct
`reuse_prefix` gives them the **same physical block ids** for logical blocks `0 …
s/p - 1`. The prefix's K/V is computed once (during the first request's `prefill`); the
second request only runs the model on its `t - s` suffix tokens. A naive implementation
that *copies* the prefix produces identical logits but holds **distinct** blocks and
**recomputes** the KV — which the mechanism tests catch.

## Function Signatures

```python
class BlockPool:
    block_size: int                           # public — read by RadixCache etc.
    def __init__(self, cfg: Qwen3Config, block_size: int) -> None: ...
    def allocate(self) -> int: ...
    def incref(self, bid: int) -> None: ...
    def free(self, bid: int) -> None: ...
    @property
    def capacity(self) -> int: ...
    @property
    def n_free(self) -> int: ...

class PagedKVCache:                       # satisfies 401's KVCache interface
    def __init__(self, cfg: Qwen3Config, block_size: int, pool: BlockPool | None = None) -> None: ...
    @property
    def length(self) -> int: ...
    def append(self, layer: int, k: np.ndarray, v: np.ndarray) -> None: ...
    def get(self, layer: int) -> tuple[np.ndarray, np.ndarray]: ...
    def allocate(self) -> int: ...
    def reuse_prefix(self, block_ids: list[int], length: int) -> None: ...
    def free(self) -> None: ...
    block_table: list[int]                # logical block idx -> physical block id

class RadixCache:
    def __init__(self, pool: BlockPool) -> None: ...
    def insert(self, ids, block_ids: list[int]): ...
    def match_prefix(self, ids): ...      # -> (node, matched_len); node.block_ids set
```

Reuse your earlier work — `prefill` / `decode_step` (401), `Qwen3Config` (306) — from
`leet_llm`. **Do not touch the forward:** 403 only changes where KV is stored and who
may share it. If your `get` reconstructs the contiguous prefix correctly, 401's forward
does the rest.

## Read More

- vLLM — *Efficient Memory Management for LLM Serving with PagedAttention* (Kwon et
  al., 2023): <https://arxiv.org/abs/2309.06180> (the block table + pool you build here)
- SGLang — *Efficient Execution of Structured Language Model Programs* (Zheng et al.,
  2024), the RadixAttention paper: <https://arxiv.org/abs/2312.07104>
- vLLM's automatic prefix caching (the production form of block-level prefix reuse):
  <https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html>

**Extension notes (not graded here).** Real engines also *evict* blocks under pressure:
**305**'s sliding-window attention bounds the live KV to the last `w` tokens, and
**309**'s attention sink keeps the first few tokens pinned while streaming the rest —
both are block-eviction policies layered on exactly this pool. **407** contrasts this
GQA block cache against **MLA**, which pages a compressed latent instead of per-head
K/V.

## How to Test

```bash
# Grade the hermetic fixture (tiny Qwen3, frozen float64 oracle — no download):
uv run grade 403
```

The graded checks: **correctness** (401's `prefill` / `decode_step` over
`PagedKVCache` reproduce the frozen oracle logits at every position, `rtol=1e-9` — paged
`get` reconstructs contiguous K/V exactly; a reused prefix yields correct suffix
logits), the **mechanism guarantees** (memory is `O(used blocks)` not `max_seq_len`;
a prefix hit skips recompute — only the novel suffix reaches attention; a shared prefix
is *physically* shared, same block ids, `< 2×` the blocks), and the **allocator
invariants** (independent requests get disjoint blocks; `free` returns blocks to the
pool; re-allocation reuses freed blocks; a shared block survives until its last holder
frees; `block_table` ids stay in-bounds).
