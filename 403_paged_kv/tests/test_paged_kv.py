"""403 — tests for the paged KV cache + RadixAttention prefix sharing.

Categories:
  A. Correctness — 401's ``prefill`` / ``decode_step`` run **unchanged** over a
     ``PagedKVCache(cfg, block_size)`` and reproduce the frozen contiguous-cache
     logits at ``rtol=1e-9, atol=0`` at every position. This proves paged ``get``
     reconstructs the exact contiguous K/V 401 would build. Reused prefix blocks
     yield identical suffix logits too.
  B. Mechanism (§10.7, fails-on-naive) — observed through the registered API:
       (a) O(used blocks)     — a ``t``-token request holds exactly
           ``ceil(t/block_size)`` live blocks (a reserve-``max_seq_len`` impl holds
           ``max_seq_len/block_size``).
       (b) prefix hit skips recompute — with a radix hit, the model computes K/V only
           for the novel suffix (a call-spy on 401's ``sdpa`` sees only suffix tokens;
           a no-sharing impl re-runs the whole prefix).
       (c) physical sharing   — two requests with a common prefix hold the *same
           physical block ids* for it and fewer than ``2x`` the blocks (a copy impl
           holds distinct ids).
  C. Allocator invariants — independent requests get disjoint physical blocks (no
     double-allocation); ``free`` returns blocks to the pool; re-allocation reuses a
     freed id; reference-shared prefix blocks survive until their last holder frees;
     ``block_table`` positions are in-bounds.

Every check inspects the registered API (``PagedKVCache`` / ``RadixCache``) driven by
401's ``prefill`` / ``decode_step`` — so a NotImplemented stub fails loudly.
"""

from __future__ import annotations

import pathlib
from math import ceil

import numpy as np

from leet_llm import Qwen3Config, decode_step, load_qwen3, prefill
from leet_llm._loader import load_task
from leet_llm.grader import load

_m = load(__file__)
PagedKVCache = _m.PagedKVCache
RadixCache = _m.RadixCache
BlockPool = _m.BlockPool

# 401's module (same target), for spying on the attention call it makes.
_m401 = load_task("401_kv_cache")

FIX = pathlib.Path(__file__).parent / "fixtures"
_F = np.load(FIX / "paged_kv.npz")

PROMPT_LEN = 5
BLOCK_SIZE = int(_F["block_size"])
SHARED_LEN = int(_F["shared_len"])


def _cfg():
    return Qwen3Config(
        dim=int(_F["dim"]),
        n_layers=int(_F["n_layers"]),
        n_heads=int(_F["n_heads"]),
        n_kv_heads=int(_F["n_kv_heads"]),
        head_dim=int(_F["head_dim"]),
        vocab_size=int(_F["vocab_size"]),
        max_seq_len=int(_F["max_seq_len"]),
        norm_eps=float(_F["norm_eps"]),
        qk_norm_eps=float(_F["qk_norm_eps"]),
        rope_base=float(_F["rope_base"]),
    )


def _params(cfg):
    return load_qwen3({k: _F[k] for k in _F.files}, cfg)


# ---------------------------------------------------------------------------
# A. Correctness — paged cache reproduces the contiguous-cache logits exactly
# ---------------------------------------------------------------------------


def test_prefill_last_logits_match_oracle_over_paged_cache():
    cfg = _cfg()
    params = _params(cfg)
    tokens = _F["token_ids"]
    cache = PagedKVCache(cfg, BLOCK_SIZE)
    out = prefill(tokens[:PROMPT_LEN][None, :], params, cfg, cache)
    assert out.shape == (1, int(_F["vocab_size"]))
    np.testing.assert_allclose(
        out[0], _F["logits"][0, PROMPT_LEN - 1], rtol=1e-9, atol=0,
        err_msg="paged prefill last-position logits diverge from the oracle",
    )


def test_teacher_forced_decode_matches_oracle_each_step_over_paged_cache():
    """401's prefill+decode over PagedKVCache reproduce the oracle at every position —
    proving paged ``get`` reconstructs the exact contiguous K/V (rtol=1e-9, atol=0)."""
    cfg = _cfg()
    params = _params(cfg)
    tokens = _F["token_ids"]
    oracle = _F["logits"][0]

    cache = PagedKVCache(cfg, BLOCK_SIZE)
    prefill(tokens[:PROMPT_LEN][None, :], params, cfg, cache)
    for pos in range(PROMPT_LEN, len(tokens)):
        out = decode_step(int(tokens[pos]), params, cfg, cache)
        np.testing.assert_allclose(
            out[0], oracle[pos], rtol=1e-9, atol=0,
            err_msg=f"paged decode_step logits diverge from oracle at position {pos}",
        )


def test_get_reconstructs_contiguous_prefix_shape():
    cfg = _cfg()
    params = _params(cfg)
    tokens = _F["token_ids"]
    cache = PagedKVCache(cfg, BLOCK_SIZE)
    prefill(tokens[:PROMPT_LEN][None, :], params, cfg, cache)
    decode_step(int(tokens[PROMPT_LEN]), params, cfg, cache)
    expected = PROMPT_LEN + 1
    assert cache.length == expected
    for layer in range(cfg.n_layers):
        k, v = cache.get(layer)
        assert k.shape == (cfg.n_kv_heads, expected, cfg.head_dim)
        assert v.shape == (cfg.n_kv_heads, expected, cfg.head_dim)


def test_reused_prefix_yields_correct_suffix_logits():
    """A radix hit reuses A's prefix blocks; decoding B's novel suffix over them
    reproduces B's own teacher-forced logits — the shared KV is physically correct."""
    cfg = _cfg()
    params = _params(cfg)
    seq_a, seq_b = _F["seq_a"], _F["seq_b"]
    logits_b = _F["logits_b"][0]

    pool = BlockPool(cfg, BLOCK_SIZE)
    cache_a = PagedKVCache(cfg, BLOCK_SIZE, pool)
    prefill(seq_a[None, :], params, cfg, cache_a)

    radix = RadixCache(pool)
    radix.insert(seq_a[:SHARED_LEN], cache_a.block_table[: SHARED_LEN // BLOCK_SIZE])

    node, matched = radix.match_prefix(seq_b)
    assert matched == SHARED_LEN

    cache_b = PagedKVCache(cfg, BLOCK_SIZE, pool)
    cache_b.reuse_prefix(node.block_ids, matched)
    assert cache_b.length == SHARED_LEN
    for pos in range(SHARED_LEN, len(seq_b)):
        out = decode_step(int(seq_b[pos]), params, cfg, cache_b)
        np.testing.assert_allclose(
            out[0], logits_b[pos], rtol=1e-9, atol=0,
            err_msg=f"reused-prefix suffix logits diverge from oracle at position {pos}",
        )


# ---------------------------------------------------------------------------
# B. Mechanism (§10.7, fails-on-naive)
# ---------------------------------------------------------------------------


def test_memory_is_O_used_blocks_not_max_seq_len():
    """A t-token request holds exactly ceil(t/block_size) blocks — internal
    fragmentation is at most one block. A reserve-max_seq_len impl would hold
    ceil(max_seq_len/block_size) blocks (here 16, not 2..4)."""
    cfg = _cfg()
    params = _params(cfg)
    tokens = _F["token_ids"]

    cache = PagedKVCache(cfg, BLOCK_SIZE)
    prefill(tokens[:PROMPT_LEN][None, :], params, cfg, cache)
    assert len(cache.block_table) == ceil(PROMPT_LEN / BLOCK_SIZE)
    assert len(cache.block_table) < ceil(cfg.max_seq_len / BLOCK_SIZE)

    for pos in range(PROMPT_LEN, len(tokens)):
        decode_step(int(tokens[pos]), params, cfg, cache)
        assert len(cache.block_table) == ceil(cache.length / BLOCK_SIZE), (
            "block count must track ceil(length/block_size), not reserve max_seq_len"
        )


def test_prefix_hit_skips_recompute(monkeypatch):
    """With a radix hit, the model computes K/V only for the novel suffix.

    A spy on 401's ``sdpa`` records the query-length of every attention call. In the
    WARM path (reuse the shared prefix blocks, decode only the suffix) the prefix
    tokens never reach ``sdpa`` — total attended query rows == suffix length per layer.
    A no-sharing impl (re-prefill the whole prompt B) would attend all tokens; asserted
    by contrast against a COLD prefill of the same prompt."""
    cfg = _cfg()
    params = _params(cfg)
    seq_a, seq_b = _F["seq_a"], _F["seq_b"]
    suffix_len = len(seq_b) - SHARED_LEN

    real_sdpa = _m401.sdpa
    qlens: list[int] = []

    def spy(q, k, v, mask=None, sink_logits=None):
        qlens.append(q.shape[-2])
        return real_sdpa(q, k, v, mask, sink_logits)

    monkeypatch.setattr(_m401, "sdpa", spy)

    pool = BlockPool(cfg, BLOCK_SIZE)
    cache_a = PagedKVCache(cfg, BLOCK_SIZE, pool)
    prefill(seq_a[None, :], params, cfg, cache_a)
    radix = RadixCache(pool)
    radix.insert(seq_a[:SHARED_LEN], cache_a.block_table[: SHARED_LEN // BLOCK_SIZE])

    # WARM: reuse prefix → decode only the suffix.
    qlens.clear()
    node, matched = radix.match_prefix(seq_b)
    cache_b = PagedKVCache(cfg, BLOCK_SIZE, pool)
    cache_b.reuse_prefix(node.block_ids, matched)
    for pos in range(SHARED_LEN, len(seq_b)):
        decode_step(int(seq_b[pos]), params, cfg, cache_b)
    warm_rows = sum(qlens)
    assert all(q == 1 for q in qlens), "decode must build single query rows"
    assert warm_rows == suffix_len * cfg.n_layers, (
        "reused prefix must NOT be recomputed — only the suffix reaches sdpa"
    )

    # COLD: no reuse → re-prefill the whole prompt B recomputes the prefix.
    qlens.clear()
    cache_cold = PagedKVCache(cfg, BLOCK_SIZE, pool)
    prefill(seq_b[None, :], params, cfg, cache_cold)
    assert sum(qlens) > warm_rows, "cold prefill must recompute more than the suffix"


def test_common_prefix_is_physically_shared():
    """Two requests with a common prefix hold the SAME physical block ids for it and
    strictly fewer than 2x the blocks. A copy-the-prefix impl holds distinct ids."""
    cfg = _cfg()
    params = _params(cfg)
    seq_a, seq_b = _F["seq_a"], _F["seq_b"]
    n_shared_blocks = SHARED_LEN // BLOCK_SIZE

    pool = BlockPool(cfg, BLOCK_SIZE)
    cache_a = PagedKVCache(cfg, BLOCK_SIZE, pool)
    prefill(seq_a[None, :], params, cfg, cache_a)
    radix = RadixCache(pool)
    radix.insert(seq_a[:SHARED_LEN], cache_a.block_table[:n_shared_blocks])

    node, matched = radix.match_prefix(seq_b)
    cache_b = PagedKVCache(cfg, BLOCK_SIZE, pool)
    cache_b.reuse_prefix(node.block_ids, matched)
    for pos in range(SHARED_LEN, len(seq_b)):
        decode_step(int(seq_b[pos]), params, cfg, cache_b)

    a_pre = cache_a.block_table[:n_shared_blocks]
    b_pre = cache_b.block_table[:n_shared_blocks]
    assert a_pre == b_pre, "shared prefix must map to the SAME physical blocks"

    distinct = set(cache_a.block_table) | set(cache_b.block_table)
    assert len(distinct) < len(cache_a.block_table) + len(cache_b.block_table), (
        "shared blocks must not be duplicated (copy-the-prefix would)"
    )


# ---------------------------------------------------------------------------
# C. Allocator invariants
# ---------------------------------------------------------------------------


def test_independent_requests_get_disjoint_blocks():
    cfg = _cfg()
    params = _params(cfg)
    tokens = _F["token_ids"]
    pool = BlockPool(cfg, BLOCK_SIZE)
    c1 = PagedKVCache(cfg, BLOCK_SIZE, pool)
    c2 = PagedKVCache(cfg, BLOCK_SIZE, pool)
    prefill(tokens[:PROMPT_LEN][None, :], params, cfg, c1)
    prefill(tokens[:PROMPT_LEN][None, :], params, cfg, c2)
    assert set(c1.block_table).isdisjoint(c2.block_table), (
        "unrelated requests must not share a physical block"
    )
    for bid in c1.block_table + c2.block_table:
        assert 0 <= bid < pool.capacity, "block_table id out of bounds"


def test_free_returns_blocks_and_reallocation_reuses_them():
    cfg = _cfg()
    params = _params(cfg)
    tokens = _F["token_ids"]
    pool = BlockPool(cfg, BLOCK_SIZE)
    cache = PagedKVCache(cfg, BLOCK_SIZE, pool)
    prefill(tokens[:PROMPT_LEN][None, :], params, cfg, cache)
    held = list(cache.block_table)
    assert pool.n_free == 0
    cache.free()
    assert pool.n_free == len(held), "free must return every block to the pool"
    assert cache.block_table == []
    reallocated = [pool.allocate() for _ in held]
    assert set(reallocated) == set(held), "re-allocation must reuse freed blocks"


def test_shared_block_survives_until_last_holder_frees():
    """A reference-shared prefix block is not returned to the pool while another
    request still holds it — freeing the sharer decrefs but does not release it."""
    cfg = _cfg()
    params = _params(cfg)
    seq_a, seq_b = _F["seq_a"], _F["seq_b"]
    n_shared = SHARED_LEN // BLOCK_SIZE

    pool = BlockPool(cfg, BLOCK_SIZE)
    cache_a = PagedKVCache(cfg, BLOCK_SIZE, pool)
    prefill(seq_a[None, :], params, cfg, cache_a)  # seq_a → capacity blocks, none free
    assert pool.n_free == 0
    total = pool.capacity

    radix = RadixCache(pool)
    radix.insert(seq_a[:SHARED_LEN], cache_a.block_table[:n_shared])
    node, matched = radix.match_prefix(seq_b)
    cache_b = PagedKVCache(cfg, BLOCK_SIZE, pool)
    cache_b.reuse_prefix(node.block_ids, matched)  # B references A's shared blocks

    cache_b.free()  # B releases its reference — shared blocks still held by A
    assert pool.n_free == 0, "shared prefix blocks must stay live while A holds them"
    cache_a.free()  # last holder releases → everything returns to the pool
    assert pool.n_free == total, (
        "shared blocks must return to the pool once the last holder frees"
    )
