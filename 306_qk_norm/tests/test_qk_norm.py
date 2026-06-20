"""306 — tests for ``qk_norm``.

Two categories:
  1. Real-fixture parity — the output must match the float64 reference
     computed from genuine ``Qwen3ForCausalLM`` Q/K projections (pre-norm
     values captured from a real forward pass, expected outputs computed via
     pure float64 numpy RMSNorm).
  2. Random / property tests — shape preserved; identity weight reduces to
     plain RMSNorm; per-head independence; Q and K are normalised
     independently.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from leet_llm import rms_norm
from leet_llm.grader import load

_m = load(__file__)
qk_norm = _m.qk_norm

FIX = pathlib.Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# 1. Real-fixture parity
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
    np.testing.assert_allclose(q_out, q_post, rtol=1e-9, atol=1e-9,
                               err_msg="Q parity failed vs Qwen3 fixture")
    np.testing.assert_allclose(k_out, k_post, rtol=1e-9, atol=1e-9,
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
    q_ref = rms_norm(q, ones, eps=eps)  # rms_norm also acts over last axis

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

    # Perturb one head heavily
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
    q_perturbed[1, 3] = rng.standard_normal(d) * 100.0  # head 1, pos 3

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

    # Large perturbation to K — Q output must be unchanged
    k2 = rng.standard_normal((2, 5, 8)) * 1000.0
    q_out2, _ = qk_norm(q, k2, q_w, k_w)
    np.testing.assert_allclose(
        q_out2, q_out, rtol=1e-12, atol=1e-12,
        err_msg="Q output changed when only K was perturbed",
    )

    # Large perturbation to Q — K output must be unchanged
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

    # Each batch element should equal the unbatched result
    for b in range(B):
        q_single, k_single = qk_norm(q[b], k[b], q_w, k_w)
        np.testing.assert_allclose(q_out[b], q_single, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(k_out[b], k_single, rtol=1e-12, atol=1e-12)
