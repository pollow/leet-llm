"""403 — Paged KV cache + prefix sharing (Qwen3, GQA) — REFERENCE SOLUTION.

Replaces 401's contiguous ``KVCache`` with a **paged** manager that satisfies the
*same* ``append`` / ``get`` / ``length`` interface, so 401's ``prefill`` /
``decode_step`` run over it **unchanged**. Two ideas from production serving:

- **PagedAttention (vLLM).** KV lives in fixed-size **blocks** (``block_size`` tokens
  each) drawn from a shared pool; a per-request **block table** maps logical token
  positions to physical blocks. The blocks a request holds need not be contiguous, so
  there is no external fragmentation and no "reserve ``max_seq_len`` per request"
  waste — a request of ``t`` tokens holds exactly ``ceil(t / block_size)`` blocks.
- **RadixAttention (SGLang).** A radix tree keyed by token ids remembers which blocks
  already hold the KV of previously-seen prefixes. When a new request shares a
  block-aligned prompt prefix, it **reference-shares** those physical blocks (refcount
  bump) instead of recomputing them — the prefix's K/V is computed exactly once.

``get(layer)`` gathers the logical blocks back into the exact contiguous ``(n_kv_heads,
length, head_dim)`` array 401 would have built, so decode logits match 401 at
``rtol=1e-9``.

Reuse (do NOT re-inline): 401's ``prefill`` / ``decode_step`` drive this cache
unchanged; ``Qwen3Config`` (306) for shapes.
"""

from __future__ import annotations

import os
from math import ceil

import numpy as np

os.environ.setdefault("LEET_LLM_TARGET", "solution")

from leet_llm import Qwen3Config

__all__ = ["PagedKVCache", "RadixCache", "BlockPool"]


class BlockPool:
    """A shared pool of fixed-size physical KV blocks with reference counting.

    Each physical block stores the post-RoPE keys and raw values for ``block_size``
    tokens across *all* layers: shape ``(n_layers, n_kv_heads, block_size, head_dim)``.
    Blocks are handed out by integer id; ``free`` decrements a refcount and only
    returns a block to the free list when its last holder releases it (so a
    reference-shared prefix block survives until every request that adopted it frees).
    """

    def __init__(self, cfg: Qwen3Config, block_size: int) -> None:
        self.cfg = cfg
        self.block_size = block_size
        self._block_shape = (cfg.n_layers, cfg.n_kv_heads, block_size, cfg.head_dim)
        self._k: list[np.ndarray] = []       # physical id -> (NL, n_kv, bs, hd)
        self._v: list[np.ndarray] = []
        self._refcount: list[int] = []
        self._free: list[int] = []           # ids currently free (reusable)

    def _grow(self) -> int:
        bid = len(self._k)
        self._k.append(np.zeros(self._block_shape, dtype=np.float64))
        self._v.append(np.zeros(self._block_shape, dtype=np.float64))
        self._refcount.append(0)
        return bid

    def allocate(self) -> int:
        """Grab a free physical block (reusing a freed one if available), refcount 1."""
        bid = self._free.pop() if self._free else self._grow()
        self._refcount[bid] = 1
        return bid

    def incref(self, bid: int) -> None:
        """Add a reference (a second request adopting a shared block)."""
        self._refcount[bid] += 1

    def free(self, bid: int) -> None:
        """Release one reference; return the block to the free list at refcount 0."""
        self._refcount[bid] -= 1
        if self._refcount[bid] == 0:
            self._free.append(bid)

    @property
    def capacity(self) -> int:
        """Total physical blocks ever materialised."""
        return len(self._k)

    @property
    def n_free(self) -> int:
        """Physical blocks currently on the free list."""
        return len(self._free)


class PagedKVCache:
    """Block-paged KV store for one sequence — a drop-in for 401's ``KVCache``.

    Satisfies the 401 interface (``append`` / ``get`` / ``length``) so 401's
    ``prefill`` / ``decode_step`` run over it unchanged, but stores KV in fixed-size
    blocks addressed by ``block_table`` (logical block index -> physical block id).
    Blocks are allocated lazily as tokens arrive: a ``t``-token sequence holds exactly
    ``ceil(t / block_size)`` blocks — never ``max_seq_len``.
    """

    def __init__(
        self, cfg: Qwen3Config, block_size: int, pool: BlockPool | None = None
    ) -> None:
        self.cfg = cfg
        self.block_size = block_size
        self.pool = pool if pool is not None else BlockPool(cfg, block_size)
        self.block_table: list[int] = []          # logical block idx -> physical id
        self._sizes = [0 for _ in range(cfg.n_layers)]

    @property
    def length(self) -> int:
        """Tokens cached (layer 0's timeline, like HF ``seen_tokens``)."""
        return self._sizes[0]

    def allocate(self) -> int:
        """Grab a free block from the pool and append it to this request's table."""
        bid = self.pool.allocate()
        self.block_table.append(bid)
        return bid

    def _ensure_blocks(self, upto_tokens: int) -> None:
        """Make sure the block table can address ``upto_tokens`` token positions."""
        need = ceil(upto_tokens / self.block_size)
        while len(self.block_table) < need:
            self.allocate()

    def append(self, layer: int, k: np.ndarray, v: np.ndarray) -> None:
        """Write this step's keys/values (``(n_kv_heads, t, head_dim)``) for one layer.

        Allocates any new blocks the write crosses into (only the first layer of a step
        grows the shared table; later layers write into the same freshly-allocated
        blocks), then scatters each token into its ``(block, slot)`` position.
        """
        t = k.shape[1]
        off = self._sizes[layer]
        self._ensure_blocks(off + t)
        for j in range(t):
            pos = off + j
            blk = self.block_table[pos // self.block_size]
            slot = pos % self.block_size
            self.pool._k[blk][layer, :, slot, :] = k[:, j, :]
            self.pool._v[blk][layer, :, slot, :] = v[:, j, :]
        self._sizes[layer] = off + t

    def get(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """Gather the paged K/V back into the exact contiguous prefix 401 returns.

        Returns ``(K, V)`` each shaped ``(n_kv_heads, length, head_dim)``.
        """
        n = self._sizes[layer]
        ks, vs = [], []
        pos = 0
        for blk in self.block_table:
            if pos >= n:
                break
            take = min(self.block_size, n - pos)
            ks.append(self.pool._k[blk][layer, :, :take, :])
            vs.append(self.pool._v[blk][layer, :, :take, :])
            pos += take
        if not ks:  # empty cache
            empty = np.zeros((self.cfg.n_kv_heads, 0, self.cfg.head_dim), dtype=np.float64)
            return empty, empty.copy()
        return np.concatenate(ks, axis=1), np.concatenate(vs, axis=1)

    def reuse_prefix(self, block_ids: list[int], length: int) -> None:
        """Adopt a reference-shared, block-aligned prefix (RadixAttention hit).

        Points this request's table at the physical blocks that already hold the
        prefix's K/V (bumping their refcount) and sets ``length`` to the shared token
        count — so the prefix is **not recomputed**. Must be called on a fresh cache
        and ``length`` must be a whole number of blocks.
        """
        if self.block_table:
            raise ValueError("reuse_prefix must be called on a fresh cache")
        if length % self.block_size != 0 or length != len(block_ids) * self.block_size:
            raise ValueError("shared prefix must be block-aligned")
        self.block_table = list(block_ids)
        for bid in self.block_table:
            self.pool.incref(bid)
        self._sizes = [length for _ in range(self.cfg.n_layers)]

    def free(self) -> None:
        """Return this request's blocks to the pool (decref each physical block)."""
        for bid in self.block_table:
            self.pool.free(bid)
        self.block_table = []
        self._sizes = [0 for _ in range(self.cfg.n_layers)]


class _RadixNode:
    """A radix-tree node owning one token segment and the blocks that cache it."""

    __slots__ = ("segment", "block_ids", "children")

    def __init__(self, segment: tuple[int, ...], block_ids: list[int]) -> None:
        self.segment = segment                 # token ids on the edge into this node
        self.block_ids = block_ids             # physical blocks for THIS segment
        self.children: dict[int, _RadixNode] = {}  # first-token -> child


class RadixCache:
    """A radix tree of cached prefixes for automatic KV reuse (RadixAttention).

    Keyed by token ids; each edge stores the physical blocks holding that segment's
    KV. ``match_prefix`` walks the longest matching, **block-aligned** prefix of a
    query and returns the physical blocks covering it; ``insert`` records a request's
    prefix so later requests can reference-share it. Sharing is by reference — blocks
    are never copied — so a hit reuses already-computed K/V.
    """

    def __init__(self, pool: BlockPool) -> None:
        self.pool = pool
        self.block_size = pool.block_size
        self._root = _RadixNode((), [])

    def insert(self, ids, block_ids: list[int]) -> _RadixNode:
        """Record that ``ids`` (a block-aligned prefix) are cached by ``block_ids``.

        ``ids`` length must be ``len(block_ids) * block_size``. Returns the leaf node.
        """
        ids = tuple(int(x) for x in np.asarray(ids).reshape(-1))
        if len(ids) != len(block_ids) * self.block_size:
            raise ValueError("insert expects a block-aligned prefix")
        node = self._root
        pos = 0
        while pos < len(ids):
            first = ids[pos]
            child = node.children.get(first)
            if child is None:
                seg = ids[pos:]
                child = _RadixNode(seg, list(block_ids[pos // self.block_size :]))
                node.children[first] = child
                return child
            # follow the shared portion of this edge
            seg = child.segment
            match = 0
            while match < len(seg) and pos + match < len(ids) and seg[match] == ids[pos + match]:
                match += 1
            if match == len(seg):
                node = child
                pos += match
            else:  # partial edge overlap — good enough for block-aligned prefixes
                return child
        return node

    def match_prefix(self, ids) -> tuple[_RadixNode | None, int]:
        """Return ``(node, matched_len)`` for the longest cached block-aligned prefix.

        ``matched_len`` is a whole number of blocks; ``node.block_ids`` (via the walked
        path) covers the matched tokens. Reuse those blocks — do NOT recompute them.
        """
        ids = tuple(int(x) for x in np.asarray(ids).reshape(-1))
        node = self._root
        pos = 0
        blocks: list[int] = []
        last_node: _RadixNode | None = None
        while pos < len(ids):
            child = node.children.get(ids[pos])
            if child is None:
                break
            seg = child.segment
            match = 0
            while match < len(seg) and pos + match < len(ids) and seg[match] == ids[pos + match]:
                match += 1
            full = match // self.block_size  # only whole shared blocks are reusable
            blocks.extend(child.block_ids[:full])
            pos += full * self.block_size
            if match == len(seg) and full * self.block_size == len(seg):
                node = child
                last_node = child
                continue
            break
        matched_len = len(blocks) * self.block_size
        if matched_len == 0:
            return None, 0
        # Return a lightweight result node carrying the flat matched-block list, so the
        # caller reads ``node.block_ids`` without us mutating the stored tree.
        _ = last_node
        return _RadixNode(ids[:matched_len], blocks), matched_len
