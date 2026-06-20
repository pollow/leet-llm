"""306 — tests for ``qk_norm`` and the Qwen3 whole-model.

Three categories:
  1. qk_norm operator invariants (original) — fixture parity, shape/dtype,
     identity-weight, per-head independence, Q/K independence.
  2. Whole-model parity (A) — ``qwen3_forward`` vs the composed float64 oracle in
     ``tiny_qwen3.npz`` at ``rtol=1e-9``.
  3. Real-weights parity (B, skippable) — ``qwen3_forward`` vs ``real_ref.npz`` logits
     produced by a genuine ``Qwen3ForCausalLM`` (float64) on the downloaded
     ``Qwen/Qwen3-0.6B`` weights.  Run ``306_qk_norm/download.sh`` to populate the
     weights.  Test B is a genuine parity check at ``rtol=1e-5``, NOT self-circular.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from leet_llm import rms_norm
from leet_llm.grader import load

_m = load(__file__)
qk_norm = _m.qk_norm
Qwen3Config = _m.Qwen3Config
load_qwen3 = _m.load_qwen3
qwen3_forward = _m.qwen3_forward

FIX = pathlib.Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# 1. Real-fixture parity — qk_norm operator
# ---------------------------------------------------------------------------


def test_matches_qwen3_fixture():
    """``qk_norm`` must match the float64 reference from a Qwen3 forward pass."""
    d = np.load(FIX / "qknorm.npz")
    q_pre = d["q_pre"]    # (n_q_heads, L, head_dim)
    k_pre = d["k_pre"]    # (n_kv_heads, L, head_dim)
    q_post = d["q_post"]  # expected post-norm Q
    k_post = d["k_post"]  # expected post-norm K
    q_weight = d["q_weight"]
    k_weight = d["k_weight"]
    eps = float(d["eps"])

    q_out, k_out = qk_norm(q_pre, k_pre, q_weight, k_weight, eps=eps)

    assert q_out.shape == q_pre.shape, f"Q shape: expected {q_pre.shape}, got {q_out.shape}"
    assert k_out.shape == k_pre.shape, f"K shape: expected {k_pre.shape}, got {k_out.shape}"
    np.testing.assert_allclose(q_out, q_post, rtol=1e-9, atol=1e-12,
                               err_msg="Q parity failed vs Qwen3 fixture")
    np.testing.assert_allclose(k_out, k_post, rtol=1e-9, atol=1e-12,
                               err_msg="K parity failed vs Qwen3 fixture")


# ---------------------------------------------------------------------------
# 2. Shape preserved
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_q,n_kv,L,d", [
    (4, 2, 5, 4),
    (8, 4, 10, 16),
    (1, 1, 3, 8),
    (6, 3, 7, 32),
])
def test_shape_preserved(n_q, n_kv, L, d):
    """Output shapes must match input shapes exactly."""
    rng = np.random.default_rng(0)
    q = rng.standard_normal((n_q, L, d))
    k = rng.standard_normal((n_kv, L, d))
    q_w = rng.standard_normal(d)
    k_w = rng.standard_normal(d)

    q_out, k_out = qk_norm(q, k, q_w, k_w)

    assert q_out.shape == q.shape, f"Q shape mismatch: {q_out.shape} != {q.shape}"
    assert k_out.shape == k.shape, f"K shape mismatch: {k_out.shape} != {k.shape}"


def test_dtype_preserved():
    """Output dtype must be float64 when input is float64."""
    rng = np.random.default_rng(1)
    q = rng.standard_normal((4, 5, 4))
    k = rng.standard_normal((2, 5, 4))
    q_w = rng.standard_normal(4)
    k_w = rng.standard_normal(4)

    q_out, k_out = qk_norm(q, k, q_w, k_w)

    assert q_out.dtype == np.float64
    assert k_out.dtype == np.float64


# ---------------------------------------------------------------------------
# 3. Identity weight reduces to plain RMSNorm
# ---------------------------------------------------------------------------


def test_identity_weight_matches_rms_norm_q():
    """With q_weight = ones, qk_norm(q) == rms_norm(q, ones, eps) from (212)."""
    rng = np.random.default_rng(2)
    q = rng.standard_normal((4, 7, 8))
    k = rng.standard_normal((2, 7, 8))
    head_dim = q.shape[-1]
    ones = np.ones(head_dim)
    eps = 1e-6

    q_out, _ = qk_norm(q, k, q_weight=ones, k_weight=ones, eps=eps)
    q_ref = rms_norm(q, ones, eps=eps)

    np.testing.assert_allclose(
        q_out, q_ref, rtol=1e-9, atol=1e-9,
        err_msg="identity-weight Q should equal rms_norm(q, ones)",
    )


def test_identity_weight_matches_rms_norm_k():
    """With k_weight = ones, qk_norm(k) == rms_norm(k, ones, eps) from (212)."""
    rng = np.random.default_rng(3)
    q = rng.standard_normal((4, 7, 8))
    k = rng.standard_normal((2, 7, 8))
    head_dim = k.shape[-1]
    ones = np.ones(head_dim)
    eps = 1e-6

    _, k_out = qk_norm(q, k, q_weight=ones, k_weight=ones, eps=eps)
    k_ref = rms_norm(k, ones, eps=eps)

    np.testing.assert_allclose(
        k_out, k_ref, rtol=1e-9, atol=1e-9,
        err_msg="identity-weight K should equal rms_norm(k, ones)",
    )


def test_weight_scaling_applied_correctly():
    """A scalar weight ``w`` applied to each head_dim element scales the output by ``w``."""
    rng = np.random.default_rng(4)
    q = rng.standard_normal((3, 6, 8))
    k = rng.standard_normal((3, 6, 8))
    ones = np.ones(8)
    scale_factor = 3.14
    w = np.full(8, scale_factor)
    eps = 1e-6

    q_out_w, k_out_w = qk_norm(q, k, q_weight=w, k_weight=w, eps=eps)
    q_out_1, k_out_1 = qk_norm(q, k, q_weight=ones, k_weight=ones, eps=eps)

    np.testing.assert_allclose(
        q_out_w, scale_factor * q_out_1, rtol=1e-9, atol=1e-9,
        err_msg="uniform weight w should scale the unit-weight output by w",
    )
    np.testing.assert_allclose(
        k_out_w, scale_factor * k_out_1, rtol=1e-9, atol=1e-9,
    )


# ---------------------------------------------------------------------------
# 4. Per-head independence — norm acts over head_dim only
# ---------------------------------------------------------------------------


def test_per_head_independence_q():
    """Changing one Q head does not affect the normalised output of other heads."""
    rng = np.random.default_rng(5)
    n_q, L, d = 4, 6, 8
    q = rng.standard_normal((n_q, L, d))
    k = rng.standard_normal((2, L, d))
    q_w = rng.standard_normal(d)
    k_w = rng.standard_normal(d)

    q_out_orig, _ = qk_norm(q, k, q_w, k_w)

    q_perturbed = q.copy()
    q_perturbed[2] = rng.standard_normal((L, d)) * 100.0

    q_out_pert, _ = qk_norm(q_perturbed, k, q_w, k_w)

    for h in range(n_q):
        if h != 2:
            np.testing.assert_allclose(
                q_out_pert[h], q_out_orig[h], rtol=1e-12, atol=1e-12,
                err_msg=f"head {h} changed when only head 2 was perturbed",
            )


def test_per_position_independence_q():
    """Changing one token position in a head does not affect other positions."""
    rng = np.random.default_rng(6)
    n_q, L, d = 4, 6, 8
    q = rng.standard_normal((n_q, L, d))
    k = rng.standard_normal((2, L, d))
    q_w = rng.standard_normal(d)
    k_w = rng.standard_normal(d)

    q_out_orig, _ = qk_norm(q, k, q_w, k_w)

    q_perturbed = q.copy()
    q_perturbed[1, 3] = rng.standard_normal(d) * 100.0

    q_out_pert, _ = qk_norm(q_perturbed, k, q_w, k_w)

    for pos in range(L):
        if pos != 3:
            np.testing.assert_allclose(
                q_out_pert[1, pos], q_out_orig[1, pos], rtol=1e-12, atol=1e-12,
                err_msg=f"head 1, position {pos} changed when only position 3 was perturbed",
            )


# ---------------------------------------------------------------------------
# 5. Q and K are normalised independently
# ---------------------------------------------------------------------------


def test_q_k_independent():
    """Perturbing K does not change Q output, and vice versa."""
    rng = np.random.default_rng(7)
    q = rng.standard_normal((4, 5, 8))
    k = rng.standard_normal((2, 5, 8))
    q_w = rng.standard_normal(8)
    k_w = rng.standard_normal(8)

    q_out, k_out = qk_norm(q, k, q_w, k_w)

    k2 = rng.standard_normal((2, 5, 8)) * 1000.0
    q_out2, _ = qk_norm(q, k2, q_w, k_w)
    np.testing.assert_allclose(
        q_out2, q_out, rtol=1e-12, atol=1e-12,
        err_msg="Q output changed when only K was perturbed",
    )

    q3 = rng.standard_normal((4, 5, 8)) * 1000.0
    _, k_out3 = qk_norm(q3, k, q_w, k_w)
    np.testing.assert_allclose(
        k_out3, k_out, rtol=1e-12, atol=1e-12,
        err_msg="K output changed when only Q was perturbed",
    )


# ---------------------------------------------------------------------------
# 6. Batch dimension passthrough
# ---------------------------------------------------------------------------


def test_batch_dim_passthrough():
    """qk_norm supports a leading batch dimension (..., n_heads, L, head_dim)."""
    rng = np.random.default_rng(8)
    B, n_q, L, d = 2, 4, 5, 8
    q = rng.standard_normal((B, n_q, L, d))
    k = rng.standard_normal((B, 2, L, d))
    q_w = rng.standard_normal(d)
    k_w = rng.standard_normal(d)

    q_out, k_out = qk_norm(q, k, q_w, k_w)

    assert q_out.shape == (B, n_q, L, d)
    assert k_out.shape == (B, 2, L, d)

    for b in range(B):
        q_single, k_single = qk_norm(q[b], k[b], q_w, k_w)
        np.testing.assert_allclose(q_out[b], q_single, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(k_out[b], k_single, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# A. Whole-model parity — tiny hermetic fixture (always-on)
# ---------------------------------------------------------------------------

_TINY = np.load(FIX / "tiny_qwen3.npz")


def _tiny_cfg():
    return Qwen3Config(
        dim=int(_TINY["dim"]),
        n_layers=int(_TINY["n_layers"]),
        n_heads=int(_TINY["n_heads"]),
        n_kv_heads=int(_TINY["n_kv_heads"]),
        head_dim=int(_TINY["head_dim"]),
        vocab_size=int(_TINY["vocab_size"]),
        max_seq_len=int(_TINY["max_seq_len"]),
        norm_eps=float(_TINY["norm_eps"]),
        qk_norm_eps=float(_TINY["qk_norm_eps"]),
        rope_base=float(_TINY["rope_base"]),
    )


def _tiny_params():
    return load_qwen3({k: _TINY[k] for k in _TINY.files}, _tiny_cfg())


def test_qwen3_logits_match_oracle():
    """qwen3_forward must reproduce the composed float64 oracle logits at rtol=1e-9."""
    out = qwen3_forward(_TINY["input_ids"], _tiny_params(), _tiny_cfg())
    np.testing.assert_allclose(out, _TINY["logits"], rtol=1e-9, atol=1e-9)


def test_qwen3_logits_shape():
    out = qwen3_forward(_TINY["input_ids"], _tiny_params(), _tiny_cfg())
    B, L = _TINY["input_ids"].shape
    assert out.shape == (B, L, int(_TINY["vocab_size"]))


def test_qwen3_causal():
    """Changing the last token must NOT affect earlier logits (causal masking)."""
    p, cfg = _tiny_params(), _tiny_cfg()
    base = qwen3_forward(_TINY["input_ids"], p, cfg)
    ids2 = _TINY["input_ids"].copy()
    ids2[0, -1] = (ids2[0, -1] + 1) % int(_TINY["vocab_size"])
    pert = qwen3_forward(ids2, p, cfg)
    np.testing.assert_allclose(base[0, :-1], pert[0, :-1], atol=1e-9)


# ---------------------------------------------------------------------------
# B. Real-weights parity — skippable (run download.sh first)
# ---------------------------------------------------------------------------

_WEIGHTS_PATH = pathlib.Path(__file__).resolve().parents[1] / "qwen3_0_6b.npz"
_REAL_REF = FIX / "real_ref.npz"


@pytest.mark.skipif(
    not _WEIGHTS_PATH.exists(),
    reason="run 306_qk_norm/download.sh to fetch real weights",
)
def test_qwen3_real_weights_logits():
    """qwen3_forward on the real Qwen/Qwen3-0.6B weights must match the committed
    real_ref.npz logits.

    real_ref.npz logits were produced by a genuine Qwen3ForCausalLM (float64) on the
    downloaded weights via convert.py.  This is a genuine parity check (our forward vs
    real HF model), not a self-circular check.  Tolerance: rtol=1e-5, atol=1e-4.
    """
    ref = np.load(_REAL_REF)
    weights = dict(np.load(str(_WEIGHTS_PATH)))
    cfg = Qwen3Config(
        dim=int(ref["dim"]),
        n_layers=int(ref["n_layers"]),
        n_heads=int(ref["n_heads"]),
        n_kv_heads=int(ref["n_kv_heads"]),
        head_dim=int(ref["head_dim"]),
        vocab_size=int(ref["vocab_size"]),
        max_seq_len=int(ref["max_seq_len"]),
        norm_eps=float(ref["norm_eps"]),
        qk_norm_eps=float(ref["qk_norm_eps"]),
        rope_base=float(ref["rope_base"]),
    )
    params = load_qwen3(weights, cfg)
    out = qwen3_forward(ref["input_ids"], params, cfg)
    np.testing.assert_allclose(out, ref["logits"], rtol=1e-5, atol=1e-4)
