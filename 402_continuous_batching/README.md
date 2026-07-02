# 402 — Continuous Batching: Iteration-Level Scheduling

**Level 4 · Track 1 — Inference Systems & Serving**

## Description

Task 401 gave you a *single-sequence* engine: one `KVCache`, one `prefill`, then
`decode_step` in a loop. A real server does not get one request at a time — it gets a
*stream* of them, arriving and finishing at different moments, each a different length.
This task builds the **scheduler** that runs many requests at once on top of 401's
per-sequence cache.

The naive way to batch is **static / request-level batching**: collect `B` requests,
pad them to the longest, run the whole batch step-by-step, and only start the *next*
batch once **every** sequence in the current one has finished. This wastes throughput
two ways:

- **Head-of-line blocking.** A batch of a 5-token reply and a 500-token reply runs for
  500 steps; the short request finished at step 5 but its slot sits occupied — burning
  compute on padding — for 495 more steps.
- **No mid-batch admission.** New requests wait for the *entire* batch to drain before
  they can even start, so latency spikes under load.

**Continuous batching** (a.k.a. **iteration-level scheduling**, from the **Orca** paper,
and the core of the **vLLM** scheduler) fixes both. The unit of scheduling is a *single
decoding iteration*, not a whole request:

- A **running set** of at most `MAX_CONCURRENT_REQUESTS` requests is advanced **one
  token per iteration**.
- The instant a request emits its end-of-sequence token (or hits its length budget) it
  is **retired mid-batch** and its slot is freed.
- A **waiting queue** holds requests that could not fit; the moment a slot frees, the
  next waiting request is **admitted** — prefilled and folded into the running set on
  the very next iteration, *without* waiting for the rest of the batch to finish.

The arithmetic is *identical* to running each request alone (this task changes *when*
tokens are computed, never *what* they are) — but the GPU stays saturated with useful
work instead of padding, which is why continuous batching is the single biggest
throughput win in modern LLM serving.

## The Contract — `Engine` wraps one 401 `KVCache` per request

`Engine` is a scheduler, not a model. It owns the model weights and, for each live
request, **one independent 401 `KVCache`** (its own K/V timeline). It never re-authors
any forward math — it *drives* 401's `prefill` / `decode_step`.

- `Engine(params, cfg)` — build a scheduler for one model. (An optional
  `eos_token_id` makes the engine retire a request the moment it emits that token;
  omitted, requests retire only at the length budget.)
- `add_request(prompt_ids) -> req_id` — register a request and return an id you use to
  track it. The request does **not** run yet; it joins the waiting queue.
- `step() -> list[(req_id, token_id)]` — run **one scheduler iteration**. Returns the
  next token for **every currently-running request**, exactly one entry per running
  request. Within a step the engine first **admits** waiting requests onto any free
  slots (prefilling them, which emits their first token) and then advances the
  already-running requests by one `decode_step`. **A newly admitted request's prefill
  IS that step's first token emission** — it appears in this `step()`'s return list,
  not the next one. A request admitted and a request already running both contribute
  exactly one `(req_id, token_id)` entry to the same step's output.
- `is_finished(req_id) -> bool` — whether that request has retired (emitted eos or hit
  the length budget).

**Guarantees the engine must meet** (these are what is graded — the *how* is yours):

1. **Same tokens as standalone.** Each request's emitted ids, concatenated across the
   steps in which it appears, equal exactly what 401's `kv_generate` would produce for
   that prompt in isolation. Interleaving with other requests changes nothing.
2. **Iteration-level.** Every running request advances by **exactly one** token per
   `step`, and no step advances more than `MAX_CONCURRENT_REQUESTS` requests.
3. **Retire mid-batch, reuse the slot immediately.** A finished request never appears
   in a later `step` (no wasted compute on it), and a waiting request begins the step
   right after a slot frees — **not** after the whole batch drains.

**GIVEN systems facts.**
- `MAX_CONCURRENT_REQUESTS = 2` — the number of KV-cache slots (max requests running
  concurrently). Kept deliberately small so a handful of requests exercises the waiting
  queue and slot reuse; a production engine sizes this from available KV memory.
- **Length budget.** A request retires when its total sequence length reaches
  `cfg.max_seq_len` (its context budget) — the hard cap that guarantees termination even
  without an eos. In this task's fixture the budget is small (`12`), and prompts of
  different lengths therefore generate different numbers of tokens, so requests finish
  at *different* steps — exactly the raggedness a scheduler must handle.

## The Model

Nothing about the Qwen3 forward changes — you reuse 401 wholesale. The engine is pure
bookkeeping around it. One iteration, conceptually:

```
step():
    # admit: fill free slots from the waiting queue
    for each free slot while the queue is non-empty:
        give the admitted request its own KVCache and prefill its prompt
        (this emits its first token)
    # advance: one decode for every request already running
    for each already-running request:
        decode_step its last token over its own cache  ->  next token
    # retire: any request that emitted eos or reached the length budget leaves the
    #         running set; its slot is available next step
    return (req_id, token) for every request that emitted this step
```

The three registered methods and the two GIVENs above are the entire contract. **How**
you represent the running set, the waiting queue, admission order, and retirement is
your design — there is no required helper breakdown.

## Function Signatures

```python
MAX_CONCURRENT_REQUESTS: int   # = 2, the number of concurrent KV-cache slots

class Engine:
    def __init__(self, params: Qwen3Params, cfg: Qwen3Config,
                 eos_token_id: int | None = None) -> None: ...
    def add_request(self, prompt_ids) -> int: ...            # -> req_id (queued)
    def step(self) -> list[tuple[int, int]]: ...             # [(req_id, token_id), ...]
    def is_finished(self, req_id: int) -> bool: ...
```

Reuse your earlier work — `KVCache`, `prefill`, `decode_step` from 401 (via
`from leet_llm import ...`), and `Qwen3Config` / `Qwen3Params` / `load_qwen3` from 306.
Use `np.argmax` to pick the greedy next token so your stream is byte-identical to
`kv_generate`. Do **not** re-author the forward pass — the engine only orchestrates
401's primitives.

## Read More

- **Orca: A Distributed Serving System for Transformer-Based Generative Models** (Yu et
  al., OSDI 2022) — the paper that introduced *iteration-level scheduling*:
  <https://www.usenix.org/conference/osdi22/presentation/yu>
- **vLLM** — *Efficient Memory Management for LLM Serving with PagedAttention* (Kwon et
  al., SOSP 2023): <https://arxiv.org/abs/2309.06180>. vLLM's scheduler is continuous
  batching; 403 will page the per-request cache this engine holds.
- vLLM docs on continuous batching / the scheduler:
  <https://docs.vllm.ai/en/latest/>

## How to Test

```bash
uv run grade 402
```

The graded checks: **correctness** (each request's tokens match its standalone
`kv_generate` under arbitrary interleaving), the **mechanism guarantees** (slot reuse —
a queued request starts while a longer one is still running; no wasted compute on
retired requests; exactly one token per running request per step), and the
**invariants** (overflow requests are queued, and `is_finished` flips exactly when a
request hits the length budget or emits eos).
