"""309 — tests for ``attention_with_sinks`` / ``gptoss_moe_ffn`` and the GPT-OSS whole-model.

Categories:
  1. Operator unit tests:
     - ``attention_with_sinks`` vs a float64 oracle; sink invariants
       (rows sum to ``1 - sink_mass < 1``; ``sink_logits=-inf`` recovers plain softmax;
      pre-masked scores are respected).
     - ``gptoss_moe_ffn`` vs a float64 oracle; routing invariants (depends only on the
       selected top-k; zeroing a non-selected expert is a no-op; the selected gate
       scores sum to 1; the gate/up pre-activations are clamped).
  2. Cross-task YaRN checks — validate the new 213/215 behavior required by 309.
  3. Whole-model parity (A) — ``gptoss_forward`` vs the composed float64 oracle in
     ``tiny_gptoss.npz`` at ``rtol=1e-9``.
  4. Wrinkle isolations observed THROUGH ``gptoss_forward`` (sinks lower the row mass;
     alternating sliding/full masks; causal; the YaRN RoPE schedule is wired in).
  5. Real-weights parity (B, skippable) — ``gptoss_forward`` vs ``real_ref.npz`` logits
     from a genuine ``GptOssForCausalLM`` (native YaRN). Run ``309_gptoss_model/download.sh`` first.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from leet_llm import AttnParams, RopeParams, gqa, rope_attention_scale, rope_scaled_freqs
from leet_llm.grader import load

_m = load(__file__)
attention_with_sinks = _m.attention_with_sinks
gptoss_moe_ffn = _m.gptoss_moe_ffn
GptOssConfig = _m.GptOssConfig
load_gptoss = _m.load_gptoss
gptoss_forward = _m.gptoss_forward

FIX = pathlib.Path(__file__).parent / "fixtures"
_TINY = np.load(FIX / "tiny_gptoss.npz")

ALPHA = 1.702
LIMIT = 7.0


# ---------------------------------------------------------------------------
# float64 oracles
# ---------------------------------------------------------------------------

def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - np.where(np.isfinite(m), m, 0.0))
    return e / e.sum(axis=axis, keepdims=True)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _yarn_inv_freq_reference(head_dim: int, base: float, sc: dict) -> np.ndarray:
    """Reference YaRN inverse-frequency schedule used for cross-task wiring checks."""
    factor = float(sc["factor"])
    old = float(sc["original_max_position_embeddings"])
    beta_fast = float(sc.get("beta_fast", 32.0))
    beta_slow = float(sc.get("beta_slow", 1.0))
    truncate = bool(sc.get("truncate", True))

    pos_freqs = base ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    extrap = 1.0 / pos_freqs
    interp = 1.0 / (factor * pos_freqs)

    low = (head_dim * np.log(old / (beta_fast * 2 * np.pi))) / (2 * np.log(base))
    high = (head_dim * np.log(old / (beta_slow * 2 * np.pi))) / (2 * np.log(base))
    if truncate:
        low, high = np.floor(low), np.ceil(high)
    low = max(low, 0.0)
    high = min(high, float(head_dim - 1))
    if low == high:
        high += 1e-3
    ramp = np.clip(
        (np.arange(head_dim // 2, dtype=np.float64) - low) / (high - low),
        0.0,
        1.0,
    )
    extrap_factor = 1.0 - ramp
    return interp * (1.0 - extrap_factor) + extrap * extrap_factor


def _sinks_oracle(scores, sink_logits):
    s = scores.astype(np.float64)
    B, Hh, L, _ = s.shape
    sink = np.asarray(sink_logits, np.float64).reshape(1, Hh, 1, 1)
    sink = np.broadcast_to(sink, (B, Hh, L, 1))
    combined = np.concatenate([s, sink], axis=-1)
    return _softmax(combined, axis=-1)[..., :-1]


def _moe_oracle(x, rw, rb, gup, gub, dp, db, top_k):
    x_shape = x.shape
    d_model = x_shape[-1]
    x = x.astype(np.float64).reshape(-1, d_model)
    T = x.shape[0]
    logits = x @ rw.T + rb
    idx = np.argsort(logits, axis=-1)[:, ::-1][:, :top_k]
    top_val = np.take_along_axis(logits, idx, axis=-1)
    scores = _softmax(top_val, axis=-1)
    out = np.zeros_like(x)
    for t in range(T):
        for j in range(top_k):
            e = idx[t, j]
            gate_up = x[t] @ gup[e] + gub[e]
            gate = np.minimum(gate_up[::2], LIMIT)
            up = np.clip(gate_up[1::2], -LIMIT, LIMIT)
            glu = gate * _sigmoid(gate * ALPHA)
            gated = (up + 1.0) * glu
            out[t] += scores[t, j] * (gated @ dp[e] + db[e])
    return out.reshape(x_shape)


# ---------------------------------------------------------------------------
# 1a. attention_with_sinks
# ---------------------------------------------------------------------------

def test_sinks_matches_oracle():
    rng = np.random.default_rng(1)
    B, Hh, L = 2, 3, 5
    scores = rng.standard_normal((B, Hh, L, L))
    sink = rng.standard_normal((Hh,))
    out = attention_with_sinks(scores, sink)
    ref = _sinks_oracle(scores, sink)
    assert out.shape == (B, Hh, L, L)
    np.testing.assert_allclose(out, ref, rtol=1e-9, atol=0)


def test_sinks_rows_sum_below_one():
    """A finite sink steals probability mass: each row sums to < 1."""
    rng = np.random.default_rng(2)
    scores = rng.standard_normal((1, 2, 4, 4))
    sink = np.array([0.0, 1.0])          # finite → strictly positive sink mass
    out = attention_with_sinks(scores, sink)
    row_sums = out.sum(axis=-1)
    assert np.all(row_sums < 1.0), "rows must sum to < 1 with a finite sink"
    assert np.all(row_sums > 0.0)


def test_sinks_neg_inf_recovers_plain_softmax():
    """sink_logits = -inf zeros the sink column → plain softmax (rows sum to 1)."""
    rng = np.random.default_rng(3)
    scores = rng.standard_normal((1, 2, 4, 4))
    sink = np.array([-np.inf, -np.inf])
    out = attention_with_sinks(scores, sink)
    plain = _softmax(scores, axis=-1)
    np.testing.assert_allclose(out, plain, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(out.sum(axis=-1), 1.0, atol=1e-9)


def test_sinks_respects_mask():
    """A pre-applied -inf mask zeroes those attention entries."""
    rng = np.random.default_rng(4)
    L = 4
    scores = rng.standard_normal((1, 2, L, L))
    rows = np.arange(L)[:, None]
    cols = np.arange(L)[None, :]
    mask = np.where(rows >= cols, 0.0, -np.inf)   # causal
    masked_scores = scores + mask
    out = attention_with_sinks(masked_scores, np.zeros(2))
    ref = _sinks_oracle(masked_scores, np.zeros(2))
    np.testing.assert_allclose(out, ref, rtol=1e-9, atol=0)
    # upper triangle (future) must be exactly zero
    fut = np.triu(np.ones((L, L), bool), k=1)
    assert np.all(out[:, :, fut] == 0.0)


# ---------------------------------------------------------------------------
# 1b. gptoss_moe_ffn
# ---------------------------------------------------------------------------

def _rand_moe(rng, B, L, d, F, E):
    return dict(
        x=rng.standard_normal((B, L, d)),
        rw=rng.standard_normal((E, d)),
        rb=rng.standard_normal((E,)),
        gup=rng.standard_normal((E, d, 2 * F)),
        gub=rng.standard_normal((E, 2 * F)),
        dp=rng.standard_normal((E, F, d)),
        db=rng.standard_normal((E, d)),
    )


def test_moe_matches_oracle():
    rng = np.random.default_rng(5)
    p = _rand_moe(rng, B=2, L=3, d=8, F=16, E=4)
    out = gptoss_moe_ffn(p["x"], p["rw"], p["rb"], p["gup"], p["gub"], p["dp"], p["db"], 2)
    ref = _moe_oracle(p["x"], p["rw"], p["rb"], p["gup"], p["gub"], p["dp"], p["db"], 2)
    assert out.shape == p["x"].shape
    np.testing.assert_allclose(out, ref, rtol=1e-9, atol=0)


def test_moe_bld_and_td_equivalent():
    """(B, L, d) and flattened (T, d) inputs should produce identical token outputs."""
    rng = np.random.default_rng(51)
    B, L, d, F, E, K = 2, 4, 8, 16, 4, 2
    p = _rand_moe(rng, B, L, d, F, E)
    x_td = p["x"].reshape(-1, d).copy()

    out_bld = gptoss_moe_ffn(p["x"], p["rw"], p["rb"], p["gup"], p["gub"], p["dp"], p["db"], K)
    out_td = gptoss_moe_ffn(x_td, p["rw"], p["rb"], p["gup"], p["gub"], p["dp"], p["db"], K)

    assert out_bld.shape == p["x"].shape
    assert out_td.shape == x_td.shape
    np.testing.assert_allclose(out_bld.reshape(-1, d), out_td, rtol=1e-9, atol=0)


def test_moe_routing_depends_only_on_top_k():
    """Zeroing every non-selected expert's weights must not change the output."""
    rng = np.random.default_rng(0)
    B, L, d, F, E, K = 1, 4, 8, 16, 6, 2   # E>(B*L)*K so some expert is reliably non-selected
    p = _rand_moe(rng, B, L, d, F, E)
    out = gptoss_moe_ffn(p["x"], p["rw"], p["rb"], p["gup"], p["gub"], p["dp"], p["db"], K)

    logits = p["x"].reshape(-1, d) @ p["rw"].T + p["rb"]
    idx = np.argsort(logits, axis=-1)[:, ::-1][:, :K]
    selected = set(idx.flatten().tolist())
    non_selected = set(range(E)) - selected
    if not non_selected:
        pytest.skip("all experts selected this seed")

    gup2, gub2, dp2, db2 = (p["gup"].copy(), p["gub"].copy(), p["dp"].copy(), p["db"].copy())
    for e in non_selected:
        gup2[e] = 0.0
        gub2[e] = 0.0
        dp2[e] = 0.0
        db2[e] = 0.0
    out2 = gptoss_moe_ffn(p["x"], p["rw"], p["rb"], gup2, gub2, dp2, db2, K)
    np.testing.assert_allclose(out2, out, rtol=1e-9, atol=0)


def test_moe_gate_scores_sum_to_one():
    """With all experts identical, the routed output collapses to a single expert.

    GPT-OSS softmaxes the selected top-k router logits, so the gate scores sum to 1.
    If every expert shares one weight set, ``Σ_k score_k · expert(x)`` = ``expert(x)``
    exactly — observed THROUGH gptoss_moe_ffn (not recomputed against itself).
    """
    rng = np.random.default_rng(7)
    B, L, d, F, E, K = 1, 7, 8, 16, 4, 2
    x = rng.standard_normal((B, L, d))
    x_flat = x.reshape(-1, d)
    rw = rng.standard_normal((E, d))
    rb = rng.standard_normal((E,))
    gup_one = rng.standard_normal((d, 2 * F))
    gub_one = rng.standard_normal((2 * F,))
    dp_one = rng.standard_normal((F, d))
    db_one = rng.standard_normal((d,))
    gup = np.broadcast_to(gup_one, (E, d, 2 * F)).copy()
    gub = np.broadcast_to(gub_one, (E, 2 * F)).copy()
    dp = np.broadcast_to(dp_one, (E, F, d)).copy()
    db = np.broadcast_to(db_one, (E, d)).copy()

    out = gptoss_moe_ffn(x, rw, rb, gup, gub, dp, db, K)

    gate_up = x_flat @ gup_one + gub_one
    gate = np.minimum(gate_up[:, ::2], LIMIT)
    up = np.clip(gate_up[:, 1::2], -LIMIT, LIMIT)
    glu = gate * _sigmoid(gate * ALPHA)
    ref = (((up + 1.0) * glu) @ dp_one + db_one).reshape(x.shape)
    np.testing.assert_allclose(out, ref, rtol=1e-9, atol=0,
                               err_msg="selected gate scores don't sum to 1")


def test_moe_clamps_preactivations():
    """The gate/up pre-activations are clamped at ±limit (7.0).

    Drive the gate far past the limit; the clamped reference (and any correct
    implementation) saturates, while an unclamped GLU would blow up. We compare
    against the clamped oracle for a single-expert (top_k=1) routed token.
    """
    B, L, d, F, E = 1, 1, 4, 3, 1
    x = np.ones((B, L, d))
    rw = np.zeros((E, d))                              # one expert, trivial router
    rb = np.zeros((E,))
    gup = np.full((E, d, 2 * F), 50.0)                 # x@gup ≈ 200 ≫ limit
    gub = np.zeros((E, 2 * F))
    dp = np.ones((E, F, d))
    db = np.zeros((E, d))
    out = gptoss_moe_ffn(x, rw, rb, gup, gub, dp, db, 1)
    ref = _moe_oracle(x, rw, rb, gup, gub, dp, db, 1)
    np.testing.assert_allclose(out, ref, rtol=1e-9, atol=0)
    # clamped gate≈7 → glu≈7; up clamped 7 → (7+1)*7=56 per F → @ones(F)=168
    assert np.all(np.abs(out) < 1e3), "pre-activations not clamped (output exploded)"


# ---------------------------------------------------------------------------
# 2. Cross-task YaRN wiring checks (asserted here, not in 213/215)
# ---------------------------------------------------------------------------


def test_213_yarn_freqs_and_scale_match_reference():
    cfg = _tiny_cfg()
    sc = cfg.rope_scaling
    got_freqs = rope_scaled_freqs(cfg.head_dim, cfg.rope_base, sc)
    ref_freqs = _yarn_inv_freq_reference(cfg.head_dim, cfg.rope_base, sc)
    np.testing.assert_allclose(got_freqs, ref_freqs, rtol=1e-12, atol=0)

    got_af = rope_attention_scale(sc)
    ref_af = 0.1 * np.log(float(sc["factor"])) + 1.0
    np.testing.assert_allclose(got_af, ref_af, rtol=1e-12, atol=0)


def test_215_gqa_rope_hook_applies_yarn_scale():
    cfg = _tiny_cfg()
    sc = cfg.rope_scaling
    rng = np.random.default_rng(9)

    B, L, d_model = 1, 5, 16
    n_heads, n_kv_heads = 4, 2
    head_dim = d_model // n_heads
    x = rng.standard_normal((B, L, d_model))
    params = AttnParams(
        Wq=rng.standard_normal((d_model, d_model)),
        Wk=rng.standard_normal((n_kv_heads * head_dim, d_model)),
        Wv=rng.standard_normal((n_kv_heads * head_dim, d_model)),
        Wo=rng.standard_normal((d_model, d_model)),
        bq=rng.standard_normal((d_model,)),
        bk=rng.standard_normal((n_kv_heads * head_dim,)),
        bv=rng.standard_normal((n_kv_heads * head_dim,)),
        bo=rng.standard_normal((d_model,)),
    )
    positions = np.arange(L, dtype=np.int64)

    out_default = gqa(
        x,
        params,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        positions=positions,
        rope_params=RopeParams(base=cfg.rope_base, pair_type="half", scaling=None),
    )
    out_yarn = gqa(
        x,
        params,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        positions=positions,
        rope_params=RopeParams(base=cfg.rope_base, pair_type="half", scaling=sc),
    )
    assert out_default.shape == out_yarn.shape == (B, L, d_model)
    assert not np.allclose(out_yarn, out_default, atol=1e-6), (
        "gqa rope_params hook had no effect — YaRN schedule/scale not applied"
    )


# ---------------------------------------------------------------------------
# 3. Whole-model parity — tiny hermetic fixture (always-on)
# ---------------------------------------------------------------------------

def _tiny_cfg(rope_scaling="__keep__"):
    return GptOssConfig(
        dim=int(_TINY["dim"]),
        n_layers=int(_TINY["n_layers"]),
        n_heads=int(_TINY["n_heads"]),
        n_kv_heads=int(_TINY["n_kv_heads"]),
        head_dim=int(_TINY["head_dim"]),
        vocab_size=int(_TINY["vocab_size"]),
        intermediate_size=int(_TINY["intermediate_size"]),
        num_local_experts=int(_TINY["num_local_experts"]),
        num_experts_per_tok=int(_TINY["num_experts_per_tok"]),
        sliding_window=int(_TINY["sliding_window"]),
        norm_eps=float(_TINY["norm_eps"]),
        rope_base=float(_TINY["rope_base"]),
        max_seq_len=int(_TINY["max_seq_len"]),
        rope_scaling=json.loads(str(_TINY["rope_scaling"])) if rope_scaling == "__keep__" else rope_scaling,
    )


def _weights():
    return {k: _TINY[k] for k in _TINY.files}


def _tiny_params():
    return load_gptoss(_weights(), _tiny_cfg())


def test_gptoss_logits_match_oracle():
    """gptoss_forward must reproduce the composed float64 oracle logits at rtol=1e-9."""
    out = gptoss_forward(_TINY["input_ids"], _tiny_params(), _tiny_cfg())
    np.testing.assert_allclose(out, _TINY["logits"], rtol=1e-9, atol=1e-9)


def test_gptoss_logits_shape():
    out = gptoss_forward(_TINY["input_ids"], _tiny_params(), _tiny_cfg())
    B, L = _TINY["input_ids"].shape
    assert out.shape == (B, L, int(_TINY["vocab_size"]))


def test_gptoss_causal():
    """Changing the last token must NOT affect earlier logits (causal masking)."""
    p, cfg = _tiny_params(), _tiny_cfg()
    base = gptoss_forward(_TINY["input_ids"], p, cfg)
    ids2 = _TINY["input_ids"].copy()
    ids2[0, -1] = (ids2[0, -1] + 1) % int(_TINY["vocab_size"])
    pert = gptoss_forward(ids2, p, cfg)
    np.testing.assert_allclose(base[0, :-1], pert[0, :-1], atol=1e-9)


def test_gptoss_alternating_sliding_full_masks():
    """Even-indexed layers use sliding-window attention; shrinking the window changes
    the logits (a full-causal-everywhere model would be window-invariant)."""
    cfg = _tiny_cfg()
    L = int(_TINY["input_ids"].shape[1])
    big = GptOssConfig(**{**cfg.__dict__, "sliding_window": L})
    small = GptOssConfig(**{**cfg.__dict__, "sliding_window": 1})
    out_big = gptoss_forward(_TINY["input_ids"], _tiny_params(), big)
    out_small = gptoss_forward(_TINY["input_ids"], _tiny_params(), small)
    assert not np.allclose(out_big, out_small, atol=1e-6), (
        "sliding window had no effect — alternating sliding/full masks not applied"
    )


def test_gptoss_yarn_rope_is_wired():
    """GPT-OSS's YaRN schedule is actually applied: yarn ≠ default rotate-half RoPE."""
    p = _tiny_params()
    out_yarn = gptoss_forward(_TINY["input_ids"], p, _tiny_cfg())
    out_default = gptoss_forward(_TINY["input_ids"], p, _tiny_cfg(rope_scaling=None))
    assert not np.allclose(out_yarn, out_default, atol=1e-6), (
        "rope_scaling had no effect — YaRN inv_freq / attention scale not used"
    )


# ---------------------------------------------------------------------------
# B. Real-weights parity — skippable (run download.sh first)
# ---------------------------------------------------------------------------

_WEIGHTS_PATH = pathlib.Path(__file__).resolve().parents[1] / "gptoss_tiny.npz"
_REAL_REF = FIX / "real_ref.npz"


@pytest.mark.skipif(
    not (_WEIGHTS_PATH.exists() and _REAL_REF.exists()),
    reason="run 309_gptoss_model/download.sh to fetch real weights",
)
def test_gptoss_real_weights_logits():
    """gptoss_forward on the real tiny-random-GptOss weights must match real_ref.npz.

    real_ref.npz logits were produced by a genuine GptOssForCausalLM (eager, float32)
    with its native YaRN long-context schedule active (reused from 307).
    Our forward runs in float64 on the same weights: tolerance rtol=1e-2/atol=1e-2.
    """
    ref = np.load(_REAL_REF)
    weights = dict(np.load(str(_WEIGHTS_PATH)))
    cfg = GptOssConfig(
        dim=int(ref["dim"]),
        n_layers=int(ref["n_layers"]),
        n_heads=int(ref["n_heads"]),
        n_kv_heads=int(ref["n_kv_heads"]),
        head_dim=int(ref["head_dim"]),
        vocab_size=int(ref["vocab_size"]),
        intermediate_size=int(ref["intermediate_size"]),
        num_local_experts=int(ref["num_local_experts"]),
        num_experts_per_tok=int(ref["num_experts_per_tok"]),
        sliding_window=int(ref["sliding_window"]),
        norm_eps=float(ref["norm_eps"]),
        rope_base=float(ref["rope_base"]),
        max_seq_len=int(ref["max_seq_len"]),
        rope_scaling=json.loads(str(ref["rope_scaling"])),
    )
    params = load_gptoss(weights, cfg)
    out = gptoss_forward(ref["input_ids"], params, cfg)
    np.testing.assert_allclose(out, ref["logits"], rtol=1e-2, atol=1e-2)
