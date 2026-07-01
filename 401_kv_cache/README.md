# 401 — KV Cache: Stateful Prefill / Decode

**Level 4 · Track 1 — Inference Systems & Serving**

## Description

In L3 you built a **stateless** forward: give it `input_ids` of length `L`, it
returns logits `(B, L, V)`. To *generate* text you called it once per new token on
the whole growing sequence — `generate` re-ran the entire prompt-plus-so-far through
every layer at every step. That is correct, and catastrophically wasteful: producing
token `t` recomputes the keys and values of tokens `0…t-1` that never change. Total
cost is `O(L³)` over a sequence of length `L`.

Every production inference engine (HF `transformers`, vLLM, TensorRT-LLM, SGLang)
fixes this by splitting generation into **two phases** and keeping a **KV cache**
between them:

- **Prefill** — run the whole prompt **once**. One big `L×L` causal attention. This
  phase is **compute-bound** (a dense matmul over the prompt). It populates the cache
  with every layer's keys and values for the prompt tokens, and emits the logits for
  the *last* position (the first token to sample).
- **Decode** — run **one token at a time**. The single new query attends to **all
  cached keys** (`1×kv_len` attention), appends its own key/value to the cache, and
  emits the next-token logits. Each step is **memory-bandwidth-bound**: almost no
  math, just streaming the cached KV and the layer weights through the ALU. This is
  why decode, not prefill, dominates latency — and why the KV cache is the single
  highest-leverage inference optimization.

With the cache, decode is `O(kv_len)` per token and a full generation is `O(L²)`
instead of `O(L³)` — the *same output* at a fraction of the FLOPs.

**What changes vs 306.** Nothing about the Qwen3 math. Same GQA + qk-norm +
rotate-half RoPE + SwiGLU blocks. The only new things are:

1. a **cache object** that stores each layer's keys/values as tokens arrive, and
2. two entry points, `prefill` and `decode_step`, that thread that cache through the
   forward instead of recomputing keys/values every call.

Two facts make the decode step correct:

- **Offset positions.** The new token is the `cache.length`-th token, so its RoPE
  position is `cache.length` (not `0`). Keys are cached **after** RoPE, so every
  cached key already carries its own absolute-position rotation.
- **The decode mask is `(1 × kv_len)`, not square.** The single new query is causally
  *allowed to see every key already in the cache* (all of them precede it), so the
  decode mask is all-visible: a boolean row of `False`. (Project mask contract:
  `True` = hidden, `False` = visible.) Contrast prefill, whose mask is the familiar
  square lower-triangular `(L × L)`.

## The Contract — `KVCache` is the seam

`KVCache` is the interface the rest of this track is built on. **402**
(`continuous_batching`) holds one `KVCache` per in-flight request; **403**
(`paged_kv`) replaces it with a block-paged store that satisfies the *same*
interface, so `prefill` / `decode_step` run over it **unchanged**. Treat these three
methods as a fixed API — that invariance is the whole point of the track.

- `KVCache(cfg)` — preallocate per-layer key and value storage sized for the config.
  Each layer's store is **GQA-shaped `(n_kv_heads, max_seq_len, head_dim)`**: one
  slot per *key/value* head (not per query head), for up to `max_seq_len` tokens.
  This is exactly HF's `StaticCache` — a fixed, contiguous buffer, the physical thing
  vLLM later chops into pages.
- `append(layer, k, v)` — write this step's keys and values for one layer at the
  current write position. `k`, `v` have shape `(n_kv_heads, t, head_dim)` (`t = L` on
  prefill, `t = 1` on decode).
- `get(layer) -> (K, V)` — return the **contiguous** cached keys and values for that
  layer, each of length `self.length`.
- `length` — number of tokens cached so far. It is `0` at construction, equals the
  prompt length after `prefill`, and **advances by exactly 1 per `decode_step`**.

**GIVEN systems facts.**
- The cache stores **post-RoPE keys** and **raw values** (values are never rotated).
- Storage is **GQA-specific** — one entry per KV head. (MLA, task 407, caches a
  latent instead; that is a *different* cache and out of scope here.)
- `max_seq_len` is the preallocation budget (the tiny fixture uses `64`); a real
  engine caps generation length to it.

## The Math

Let the cache already hold `p = cache.length` tokens. For a new token `x` at
position `p`, one layer computes (all L2/L3 primitives you already own):

```
q = qk_norm(rope_half(project_q(x), pos=p))     # shape (n_heads,    1, head_dim)
k = qk_norm(rope_half(project_k(x), pos=p))     # shape (n_kv_heads, 1, head_dim)
v =          project_v(x)                        # shape (n_kv_heads, 1, head_dim)

cache.append(layer, k, v)                        # cache grows to p+1 tokens
K, V = cache.get(layer)                          # (n_kv_heads, p+1, head_dim)

o = sdpa(q, K, V, mask)      # mask is (1 × (p+1)) all-visible → score row (…, 1, p+1)
```

The score tensor is a **single query row** `(n_heads, 1, p+1)`: decode never rebuilds
the `(p+1)×(p+1)` block prefill built. That shape *is* the performance guarantee — a
naive "re-run the whole sequence" decode would produce a `(p+1)×(p+1)` score and the
same logits, but at `L×` the cost. `prefill` is this same computation with `t = L`
tokens at `positions = arange(L)` and the square causal mask.

## Function Signatures

```python
class KVCache:
    def __init__(self, cfg: Qwen3Config) -> None: ...
    @property
    def length(self) -> int: ...                       # tokens cached so far
    def append(self, layer: int, k: np.ndarray, v: np.ndarray) -> None: ...
    def get(self, layer: int) -> tuple[np.ndarray, np.ndarray]: ...

def prefill(prompt_ids, params: Qwen3Params, cfg: Qwen3Config, cache: KVCache) -> np.ndarray:
    """Full-prompt forward at positions arange(len); fills `cache` for every layer;
    returns LAST-position logits, shape (1, V)."""

def decode_step(token_id: int, params: Qwen3Params, cfg: Qwen3Config, cache: KVCache) -> np.ndarray:
    """Single token at positions [cache.length] with a (1 × kv_len) causal mask;
    appends its per-layer K/V; returns logits, shape (1, V)."""

def kv_generate(prompt_ids, params: Qwen3Params, cfg: Qwen3Config, n_new: int) -> list[int]:
    """Greedy driver: prefill, then n_new × (argmax → decode_step).
    Returns prompt + generated token ids."""
```

Reuse your earlier work — `load_qwen3` / `Qwen3Config` / `Qwen3Params` (306),
`embedding`, `rms_norm`, `qk_norm`, `rope_half`, `sdpa`, `affine`, `group_last_axis` /
`ungroup_last_axis`, `swiglu_ffn`, `add_residual`, `triangular_mask` — from
`leet_llm`. The block math is 306's; do **not** call `qwen3_forward` (it is stateless
and throws the keys/values away). Re-author the per-layer loop so the cache is filled
and read at the attention seam.

## Read More

- HF `transformers` **`StaticCache`** — the preallocated contiguous KV cache this task
  mirrors: <https://huggingface.co/docs/transformers/en/kv_cache>
- vLLM — *Efficient Memory Management for LLM Serving with PagedAttention* (Kwon et
  al., 2023): <https://arxiv.org/abs/2309.06180> (the cache you build here is what 403
  will page)
- The prefill/decode split and its compute- vs memory-bound characters:
  <https://arxiv.org/abs/2308.16369> (Sarathi)

## How to Test

```bash
# Grade the hermetic fixture (tiny Qwen3, frozen float64 oracle — no download):
uv run grade 401

# Optional real-weights demo (~1.2 GB, Qwen/Qwen3-0.6B — reuses 306's converter):
bash 401_kv_cache/download.sh
uv run grade 401
```

The graded checks: **teacher-forced logits** (prefill then decode through the frozen
tokens match the oracle at every position), **free-run tokens** (`kv_generate`
reproduces the greedy sequence exactly), the **mechanism guarantee** (decode builds a
single `(…, 1, kv_len)` query row — the prefix is not recomputed), and the **cache
invariants** (`length` advances by exactly one per decode; `get` returns the
contiguous prefix).
