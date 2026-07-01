"""401 — tests for stateful prefill / decode over a KV cache.

Four categories:
  A. Teacher-forced logits (primary, loud) — prefill then decode_step through the
     *frozen* token ids; each step's logits ≈ the composed-oracle logits at that
     position (rtol=1e-9, atol=0). Localises any offset/mask/append bug to its step.
  B. Free-run tokens (exact) — kv_generate reproduces the frozen greedy token ids.
  C. Mechanism (§10.7, fails-on-naive) — decode_step builds a single query row over
     the whole cached prefix: q has query-length 1, k has kv_len keys. A stateless
     re-forward (query-length = full seq) fails this. Observed through the registered
     path by spying on the module's ``sdpa``.
  D. Cache invariants — length advances by exactly 1 per decode_step; prefill of
     length p leaves length == p; get(layer) returns the contiguous cached prefix.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from leet_llm import load_qwen3
from leet_llm.grader import load

_m = load(__file__)
KVCache = _m.KVCache
prefill = _m.prefill
decode_step = _m.decode_step
kv_generate = _m.kv_generate

FIX = pathlib.Path(__file__).parent / "fixtures"
_F = np.load(FIX / "kv_cache.npz")

PROMPT_LEN = 5


def _cfg():
    from leet_llm import Qwen3Config

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
# A. Teacher-forced logits (primary, loud)
# ---------------------------------------------------------------------------


def test_prefill_last_logits_match_oracle():
    """prefill returns the last-prompt-position logits (1, V) == oracle[0, 4]."""
    cfg = _cfg()
    params = _params(cfg)
    prompt = _F["token_ids"][:PROMPT_LEN]
    cache = KVCache(cfg)
    out = prefill(prompt[None, :], params, cfg, cache)
    assert out.shape == (1, int(_F["vocab_size"]))
    np.testing.assert_allclose(
        out[0], _F["logits"][0, PROMPT_LEN - 1], rtol=1e-9, atol=0,
        err_msg="prefill last-position logits diverge from the oracle",
    )


def test_teacher_forced_decode_matches_oracle_each_step():
    """Feed the frozen tokens one at a time; every step's logits match the oracle at
    that position. A cache/offset/mask bug fails at the exact step it occurs."""
    cfg = _cfg()
    params = _params(cfg)
    tokens = _F["token_ids"]
    oracle = _F["logits"][0]  # (13, V)

    cache = KVCache(cfg)
    prefill(tokens[:PROMPT_LEN][None, :], params, cfg, cache)

    for pos in range(PROMPT_LEN, len(tokens)):
        out = decode_step(int(tokens[pos]), params, cfg, cache)
        assert out.shape == (1, int(_F["vocab_size"]))
        np.testing.assert_allclose(
            out[0], oracle[pos], rtol=1e-9, atol=0,
            err_msg=f"decode_step logits diverge from oracle at position {pos}",
        )


# ---------------------------------------------------------------------------
# B. Free-run tokens (exact)
# ---------------------------------------------------------------------------


def test_kv_generate_reproduces_frozen_tokens():
    """Greedy kv_generate from the prompt must reproduce the frozen token ids exactly."""
    cfg = _cfg()
    params = _params(cfg)
    prompt = _F["token_ids"][:PROMPT_LEN]
    out = kv_generate(prompt[None, :], params, cfg, n_new=len(_F["token_ids"]) - PROMPT_LEN)
    assert list(out) == _F["token_ids"].tolist()


# ---------------------------------------------------------------------------
# C. Mechanism — decode does NOT recompute the prefix (§10.7, fails-on-naive)
# ---------------------------------------------------------------------------


def test_decode_step_is_single_query_row(monkeypatch):
    """Each decode_step's attention builds a (…, 1, kv_len) score row — one query
    over the whole cached prefix — proving the prefix is not recomputed.

    A correct-but-naive stateless re-forward would pass a full-length query
    (query-length = current sequence length) and fail the ``q_len == 1`` assertion.
    Observed through the registered path by spying on the module's ``sdpa``.
    """
    cfg = _cfg()
    params = _params(cfg)
    tokens = _F["token_ids"]

    real_sdpa = _m.sdpa
    calls: list[tuple[int, int]] = []

    def spy(q, k, v, mask=None, sink_logits=None):
        # q: (1, KV, n_g, q_len, hd)   k: (1, KV, 1, kv_len, hd)
        calls.append((q.shape[-2], k.shape[-2]))
        return real_sdpa(q, k, v, mask, sink_logits)

    monkeypatch.setattr(_m, "sdpa", spy)

    cache = KVCache(cfg)
    prefill(tokens[:PROMPT_LEN][None, :], params, cfg, cache)

    # Prefill attends L×L (compute-bound); every prefill sdpa call is a full square.
    assert calls, "prefill never called sdpa"
    for q_len, kv_len in calls:
        assert q_len == PROMPT_LEN and kv_len == PROMPT_LEN, (
            f"prefill should attend {PROMPT_LEN}x{PROMPT_LEN}, saw {q_len}x{kv_len}"
        )

    # Decode one token: a single query row over the grown cache — NOT a re-forward.
    calls.clear()
    decode_step(int(tokens[PROMPT_LEN]), params, cfg, cache)
    assert len(calls) == cfg.n_layers, "decode_step should call sdpa once per layer"
    for q_len, kv_len in calls:
        assert q_len == 1, (
            f"decode must build a single query row, saw query-length {q_len} "
            "(a stateless re-forward recomputes the whole prefix)"
        )
        assert kv_len == PROMPT_LEN + 1, (
            f"decode query must attend all {PROMPT_LEN + 1} cached keys, saw {kv_len}"
        )


# ---------------------------------------------------------------------------
# D. Cache invariants (observed through the registered API)
# ---------------------------------------------------------------------------


def test_prefill_sets_length_to_prompt_len():
    cfg = _cfg()
    params = _params(cfg)
    prompt = _F["token_ids"][:PROMPT_LEN]
    cache = KVCache(cfg)
    assert cache.length == 0
    prefill(prompt[None, :], params, cfg, cache)
    assert cache.length == PROMPT_LEN


def test_decode_advances_length_by_exactly_one():
    cfg = _cfg()
    params = _params(cfg)
    tokens = _F["token_ids"]
    cache = KVCache(cfg)
    prefill(tokens[:PROMPT_LEN][None, :], params, cfg, cache)
    prev = cache.length
    for pos in range(PROMPT_LEN, len(tokens)):
        decode_step(int(tokens[pos]), params, cfg, cache)
        assert cache.length == prev + 1, "each decode_step must cache exactly one token"
        prev = cache.length


def test_get_returns_contiguous_prefix_shape():
    """get(layer) returns GQA-shaped K/V of length == cache.length for every layer."""
    cfg = _cfg()
    params = _params(cfg)
    tokens = _F["token_ids"]
    cache = KVCache(cfg)
    prefill(tokens[:PROMPT_LEN][None, :], params, cfg, cache)
    decode_step(int(tokens[PROMPT_LEN]), params, cfg, cache)
    expected = PROMPT_LEN + 1
    for layer in range(cfg.n_layers):
        k, v = cache.get(layer)
        assert k.shape == (cfg.n_kv_heads, expected, cfg.head_dim)
        assert v.shape == (cfg.n_kv_heads, expected, cfg.head_dim)


def test_get_prefix_is_stable_under_decode():
    """The cached prefix's K/V is not disturbed by a later decode_step (append-only)."""
    cfg = _cfg()
    params = _params(cfg)
    tokens = _F["token_ids"]
    cache = KVCache(cfg)
    prefill(tokens[:PROMPT_LEN][None, :], params, cfg, cache)
    k0_before, v0_before = cache.get(0)
    k0_before = k0_before.copy()
    v0_before = v0_before.copy()
    decode_step(int(tokens[PROMPT_LEN]), params, cfg, cache)
    k0_after, v0_after = cache.get(0)
    np.testing.assert_array_equal(k0_after[:, :PROMPT_LEN], k0_before)
    np.testing.assert_array_equal(v0_after[:, :PROMPT_LEN], v0_before)
