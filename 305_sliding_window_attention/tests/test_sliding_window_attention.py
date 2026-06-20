"""305 — tests for ``sliding_window_mask`` and the Mistral whole-model.

Three categories:
  1. Band-mask invariants — ``sliding_window_mask`` shape/dtype/attended-set (original).
  2. Whole-model parity (A) — ``mistral_forward`` vs the composed float64 oracle in
     ``tiny_mistral.npz`` at ``rtol=1e-9``.
  3. Real-weights parity (B, skippable) — ``mistral_forward`` vs ``real_ref.npz`` logits
     produced by a genuine ``MistralForCausalLM`` (SiLU-forced, float64) on the downloaded
     ``hf-internal-testing/tiny-random-MistralForCausalLM`` weights.
     The random-init checkpoint has ``hidden_act="gelu"`` in its config; convert.py
     forces ``hidden_act="silu"`` so both sides use real Mistral's activation.
     Test B is a genuine parity check at ``rtol=1e-5``, NOT a self-circular check.
     Run ``305_sliding_window_attention/download.sh`` to populate the weights.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from leet_llm import sdpa, triangular_mask
from leet_llm.grader import load

_m = load(__file__)
sliding_window_mask = _m.sliding_window_mask
MistralConfig = _m.MistralConfig
load_mistral = _m.load_mistral
mistral_forward = _m.mistral_forward

FIX = pathlib.Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# 1. Real-fixture parity — band mask
# ---------------------------------------------------------------------------


def test_matches_hf_mistral_band():
    """``sliding_window_mask`` must exactly equal the HF Mistral additive mask."""
    d = np.load(FIX / "band.npz")
    L = int(d["seq_len"])
    W = int(d["window"])
    expected = d["mask"]  # (L, L) float64, 0.0 attended / -inf masked

    result = sliding_window_mask(L, W)

    assert result.shape == (L, L), f"expected shape ({L},{L}), got {result.shape}"
    np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# 2. Shape and dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L,W", [(1, 1), (4, 2), (8, 3), (10, 10)])
def test_shape(L, W):
    out = sliding_window_mask(L, W)
    assert out.shape == (L, L)
    assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# 3. Attended-set invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L,W", [(6, 3), (8, 4), (5, 2), (10, 1)])
def test_attended_set(L, W):
    """Attended positions are exactly those with i-W < j <= i (no more, no less)."""
    mask = sliding_window_mask(L, W)
    for i in range(L):
        for j in range(L):
            expected_attended = (i - W < j <= i)
            if expected_attended:
                assert mask[i, j] == 0.0, (
                    f"position ({i},{j}) should be attended (0.0) but got {mask[i,j]}"
                )
            else:
                assert mask[i, j] == -np.inf, (
                    f"position ({i},{j}) should be masked (-inf) but got {mask[i,j]}"
                )


def test_causal_upper_triangle_always_masked():
    """Future positions (j > i) must always be -inf regardless of window size."""
    L, W = 8, 6
    mask = sliding_window_mask(L, W)
    # upper triangle (j > i)
    i_idx, j_idx = np.triu_indices(L, k=1)
    assert np.all(mask[i_idx, j_idx] == -np.inf), "future positions must be -inf"


def test_diagonal_always_attended():
    """Self-attention (j == i) must always be 0.0 (inside any window >= 1)."""
    L, W = 7, 2
    mask = sliding_window_mask(L, W)
    assert np.all(np.diag(mask) == 0.0), "diagonal (self-attention) must be 0.0"


# ---------------------------------------------------------------------------
# 4. Degenerate case: window >= L reduces to triangular_mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L,W", [(5, 5), (5, 10), (6, 6), (8, 100)])
def test_window_ge_L_is_causal_mask(L, W):
    """When window >= seq_len, the mask is the standard causal (lower-triangle) mask."""
    result = sliding_window_mask(L, W)
    # triangular_mask(L) is bool True = masked (upper triangle)
    # Convert to additive form for comparison
    causal_additive = np.where(triangular_mask(L), -np.inf, 0.0)
    np.testing.assert_array_equal(result, causal_additive)


# ---------------------------------------------------------------------------
# 5. Window = 1: only self-attention allowed
# ---------------------------------------------------------------------------


def test_window_1_only_self_attention():
    """Window=1 means query i may only attend to j=i (no past context)."""
    L = 5
    mask = sliding_window_mask(L, 1)
    expected = np.full((L, L), -np.inf, dtype=np.float64)
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_array_equal(mask, expected)


# ---------------------------------------------------------------------------
# 6. Attention-parity check via sdpa
# ---------------------------------------------------------------------------


def test_sdpa_zeroes_out_of_band_contributions():
    """Swapping the sliding-window mask into sdpa zeroes contributions outside the band.

    We construct Q, K, V where V is a one-hot indicator for each key position.
    With a sliding-window mask the output at query i is a weighted sum of
    V[j] for j in (i-W, i].  So the output row for query i must have zero weight
    on every V column outside the band.
    """
    rng = np.random.default_rng(0)
    L = 8
    W = 3
    d_k = 4

    mask = sliding_window_mask(L, W)

    # Use the additive mask as a boolean mask for sdpa
    # sdpa takes bool mask (True = masked), so invert
    bool_mask = np.isinf(mask)  # True where masked (-inf), False where attended

    # Q, K are random; V is identity (V[j] = one-hot(j)) so we can read off weights
    Q = rng.standard_normal((1, 1, L, d_k))
    K = rng.standard_normal((1, 1, L, d_k))
    V_identity = np.eye(L, dtype=np.float64)[np.newaxis, np.newaxis, :, :]  # (1,1,L,L)

    out = sdpa(Q, K, V_identity, mask=bool_mask)  # (1, 1, L, L)
    weights = out[0, 0]  # (L, L): weights[i, j] = how much query i used key j

    for i in range(L):
        for j in range(L):
            in_band = (i - W < j <= i)
            if not in_band:
                assert weights[i, j] == pytest.approx(0.0, abs=1e-12), (
                    f"query {i} should have zero weight on key {j} (outside band), "
                    f"got {weights[i,j]}"
                )


def test_additive_mask_format_compatible_with_sdpa():
    """The additive mask can be applied directly to scores before softmax.

    Use the mask as additive offset (adding to QK^T/sqrt(d_k)) and verify
    that positions outside the band get softmax weight ~0.
    """
    rng = np.random.default_rng(1)
    L = 6
    W = 2
    d_k = 8

    sw_mask = sliding_window_mask(L, W)  # (L, L) additive

    Q = rng.standard_normal((L, d_k))
    K = rng.standard_normal((L, d_k))
    V = np.eye(L, dtype=np.float64)

    # Compute attention manually with additive mask
    scores = (Q @ K.T) / np.sqrt(d_k) + sw_mask  # (L, L)
    # Softmax row-wise
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    # positions with -inf → exp(-inf) = 0
    exp_scores = np.where(np.isinf(sw_mask), 0.0, exp_scores)
    attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)  # (L, L)

    for i in range(L):
        for j in range(L):
            in_band = (i - W < j <= i)
            if not in_band:
                assert attn_weights[i, j] == pytest.approx(0.0, abs=1e-12), (
                    f"additive mask: query {i}, key {j} outside band but weight={attn_weights[i,j]}"
                )
        # Attended weights must be positive and sum to 1
        attended = np.array([j for j in range(L) if i - W < j <= i])
        assert attn_weights[i, attended].sum() == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# A. Whole-model parity — tiny hermetic fixture (always-on)
# ---------------------------------------------------------------------------

_TINY = np.load(FIX / "tiny_mistral.npz")


def _tiny_cfg():
    return MistralConfig(
        dim=int(_TINY["dim"]),
        n_layers=int(_TINY["n_layers"]),
        n_heads=int(_TINY["n_heads"]),
        n_kv_heads=int(_TINY["n_kv_heads"]),
        vocab_size=int(_TINY["vocab_size"]),
        sliding_window=int(_TINY["sliding_window"]),
        max_seq_len=int(_TINY["max_seq_len"]),
        norm_eps=float(_TINY["norm_eps"]),
        rope_base=float(_TINY["rope_base"]),
    )


def _tiny_params():
    return load_mistral({k: _TINY[k] for k in _TINY.files}, _tiny_cfg())


def test_mistral_logits_match_oracle():
    """mistral_forward must reproduce the composed float64 oracle logits at rtol=1e-9."""
    out = mistral_forward(_TINY["input_ids"], _tiny_params(), _tiny_cfg())
    np.testing.assert_allclose(out, _TINY["logits"], rtol=1e-9, atol=1e-9)


def test_mistral_logits_shape():
    out = mistral_forward(_TINY["input_ids"], _tiny_params(), _tiny_cfg())
    B, L = _TINY["input_ids"].shape
    assert out.shape == (B, L, int(_TINY["vocab_size"]))


def test_mistral_causal():
    """Changing the last token must NOT affect earlier logits (causal masking)."""
    p, cfg = _tiny_params(), _tiny_cfg()
    base = mistral_forward(_TINY["input_ids"], p, cfg)
    ids2 = _TINY["input_ids"].copy()
    ids2[0, -1] = (ids2[0, -1] + 1) % int(_TINY["vocab_size"])
    pert = mistral_forward(ids2, p, cfg)
    np.testing.assert_allclose(base[0, :-1], pert[0, :-1], atol=1e-9)


# ---------------------------------------------------------------------------
# B. Real-weights parity — skippable (run download.sh first)
# ---------------------------------------------------------------------------

_WEIGHTS_PATH = pathlib.Path(__file__).resolve().parents[1] / "mistral_tiny.npz"
_REAL_REF = FIX / "real_ref.npz"


@pytest.mark.skipif(
    not _WEIGHTS_PATH.exists(),
    reason="run 305_sliding_window_attention/download.sh to fetch real weights",
)
def test_mistral_real_weights_logits():
    """mistral_forward on the real tiny-random-MistralForCausalLM weights must
    match the committed real_ref.npz logits.

    real_ref.npz logits were produced by a genuine MistralForCausalLM
    (hidden_act="silu" forced, float64) on the downloaded weights via convert.py.
    This is a genuine parity check (our forward vs real HF model), not a
    self-circular check. Both sides use SiLU on the same weights; tolerance rtol=1e-5.
    """
    ref = np.load(_REAL_REF)
    weights = dict(np.load(str(_WEIGHTS_PATH)))
    cfg = MistralConfig(
        dim=int(ref["dim"]),
        n_layers=int(ref["n_layers"]),
        n_heads=int(ref["n_heads"]),
        n_kv_heads=int(ref["n_kv_heads"]),
        vocab_size=int(ref["vocab_size"]),
        sliding_window=int(ref["sliding_window"]),
        max_seq_len=int(ref["max_seq_len"]),
        norm_eps=float(ref["norm_eps"]),
        rope_base=float(ref["rope_base"]),
    )
    params = load_mistral(weights, cfg)
    out = mistral_forward(ref["input_ids"], params, cfg)
    np.testing.assert_allclose(out, ref["logits"], rtol=1e-5, atol=1e-4)
