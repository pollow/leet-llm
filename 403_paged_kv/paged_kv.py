"""403 — Paged KV cache + prefix sharing (Qwen3, GQA).

Replaces 401's contiguous ``KVCache`` with a **paged** manager that satisfies the
*same* ``append`` / ``get`` / ``length`` interface, so 401's ``prefill`` /
``decode_step`` run over it **unchanged**. Two ideas from production serving:

- **PagedAttention (vLLM).** KV lives in fixed-size **blocks** (``block_size`` tokens
  each) drawn from a shared pool; a per-request **block table** maps logical token
  positions to physical blocks. Blocks need not be contiguous, so there is no external
  fragmentation and no "reserve ``max_seq_len`` per request" waste — a request of
  ``t`` tokens holds exactly ``ceil(t / block_size)`` blocks.
- **RadixAttention (SGLang).** A radix tree keyed by token ids remembers which blocks
  already hold the KV of previously-seen prefixes. When a new request shares a
  block-aligned prompt prefix, it **reference-shares** those physical blocks instead
  of recomputing them — the prefix's K/V is computed exactly once.

Registered surfaces (see README.md):

- ``BlockPool``      — shared pool of fixed-size physical KV blocks with reference
                       counting. ``allocate`` / ``incref`` / ``free``; ``capacity`` /
                       ``n_free``.
- ``PagedKVCache``   — one sequence's block-paged KV; satisfies 401's ``KVCache``
                       interface (``append`` / ``get`` / ``length``) plus ``allocate``,
                       ``free``, ``reuse_prefix``, and exposes ``block_table``.
- ``RadixCache``     — prefix sharing: ``match_prefix`` / ``insert``.

``get(layer)`` must reconstruct the EXACT contiguous ``(n_kv_heads, length, head_dim)``
array 401's ``KVCache.get`` returns, so decode logits match 401 at ``rtol=1e-9``.

Run ``uv run grade 403`` to check your work.

Reuse: 401's ``prefill`` / ``decode_step`` drive this cache unchanged; ``Qwen3Config``
(306) for shapes. Do NOT re-author the forward — 403 only changes where KV is stored.
"""

from __future__ import annotations

import numpy as np

from leet_llm import Qwen3Config

__all__ = ["PagedKVCache", "RadixCache", "BlockPool"]


class BlockPool:
    """A shared pool of fixed-size physical KV blocks with reference counting.

    Each physical block stores post-RoPE keys and raw values for ``block_size`` tokens
    across *all* layers — as **two** separate arrays, each of shape
    ``(n_layers, n_kv_heads, block_size, head_dim)``: one for keys (post-RoPE) and one
    for values (raw). ``block_size`` is a **public attribute** (``pool.block_size``) so
    downstream classes such as ``RadixCache`` can read it directly.

    ``free`` decrements a refcount and only returns a block to the free list when its
    last holder releases it, so a reference-shared prefix block survives until every
    request that adopted it frees.
    """

    def __init__(self, cfg: Qwen3Config, block_size: int) -> None:
        raise NotImplementedError

    def allocate(self) -> int:
        """Grab a free physical block (reusing a freed one if available)."""
        raise NotImplementedError

    def incref(self, bid: int) -> None:
        """Add a reference (a second request adopting a shared block)."""
        raise NotImplementedError

    def free(self, bid: int) -> None:
        """Release one reference; return the block to the free list at refcount 0."""
        raise NotImplementedError

    @property
    def capacity(self) -> int:
        """Total physical blocks ever materialised."""
        raise NotImplementedError

    @property
    def n_free(self) -> int:
        """Physical blocks currently on the free list."""
        raise NotImplementedError


class PagedKVCache:
    """Block-paged KV store for one sequence — a drop-in for 401's ``KVCache``.

    Satisfies the 401 interface (``append`` / ``get`` / ``length``) so 401's
    ``prefill`` / ``decode_step`` run over it unchanged, but stores KV in fixed-size
    blocks addressed by ``block_table`` (logical block index -> physical block id).
    Blocks are allocated lazily as tokens arrive: a ``t``-token sequence holds exactly
    ``ceil(t / block_size)`` blocks — never ``max_seq_len``.

    - ``PagedKVCache(cfg, block_size, pool=None)`` — build over an optional shared
      ``BlockPool`` (a fresh private pool when ``None``).
    - ``append(layer, k, v)`` / ``get(layer) -> (K, V)`` / ``length`` — the 401 seam.
    - ``allocate()`` — grab a free block and append it to ``block_table``.
    - ``reuse_prefix(block_ids, length)`` — adopt a reference-shared, block-aligned
      prefix (a RadixAttention hit) so it is not recomputed.
    - ``free()`` — return this request's blocks to the pool.
    - ``block_table`` — logical block index -> physical block id.
    """

    def __init__(
        self, cfg: Qwen3Config, block_size: int, pool: BlockPool | None = None
    ) -> None:
        raise NotImplementedError

    @property
    def length(self) -> int:
        """Tokens cached (layer 0's timeline, like HF ``seen_tokens``)."""
        raise NotImplementedError

    def allocate(self) -> int:
        """Grab a free block from the pool and append it to this request's table."""
        raise NotImplementedError

    def append(self, layer: int, k: np.ndarray, v: np.ndarray) -> None:
        """Write this step's keys/values (``(n_kv_heads, t, head_dim)``) for one layer,
        allocating any new blocks the write crosses into."""
        raise NotImplementedError

    def get(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """Gather the paged K/V into the exact contiguous prefix 401 returns, each
        shaped ``(n_kv_heads, length, head_dim)``."""
        raise NotImplementedError

    def reuse_prefix(self, block_ids: list[int], length: int) -> None:
        """Adopt a reference-shared, block-aligned prefix (RadixAttention hit): point
        ``block_table`` at the blocks that already hold the prefix's K/V, bump their
        refcount, and set ``length`` — so the prefix is not recomputed."""
        raise NotImplementedError

    def free(self) -> None:
        """Return this request's blocks to the pool (decref each physical block)."""
        raise NotImplementedError


class RadixCache:
    """A radix tree of cached prefixes for automatic KV reuse (RadixAttention).

    Keyed by token ids; each edge remembers the physical blocks holding that segment's
    KV. Sharing is by reference — blocks are never copied — so a hit reuses
    already-computed K/V.

    - ``RadixCache(pool)`` — build over the same ``BlockPool`` the caches use.
      ``block_size`` is read from ``pool.block_size``.
    - ``insert(ids, block_ids)`` — record that a block-aligned prefix ``ids`` is cached
      by ``block_ids`` (``len(ids) == len(block_ids) * block_size``).
    - ``match_prefix(ids) -> (node, matched_len)`` — longest cached, block-aligned
      prefix; ``matched_len`` is a whole number of blocks. **``node.block_ids`` is the
      cumulative list of physical block ids from the root to that node, in order** — not
      just the blocks at this node's own edge, but every block needed to reconstruct the
      full matched prefix. Pass it directly to ``reuse_prefix(node.block_ids,
      matched_len)``. Returns ``(None, 0)`` on a miss.
    """

    def __init__(self, pool: BlockPool) -> None:
        raise NotImplementedError

    def insert(self, ids, block_ids: list[int]):
        """Record that ``ids`` (a block-aligned prefix) are cached by ``block_ids``."""
        raise NotImplementedError

    def match_prefix(self, ids):
        """Return ``(node, matched_len)`` for the longest cached block-aligned prefix
        (``(None, 0)`` on a miss)."""
        raise NotImplementedError
