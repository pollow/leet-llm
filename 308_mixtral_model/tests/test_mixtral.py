"""308 — tests for ``moe_ffn`` and the Mixtral whole-model.

Three categories:
  1. moe_ffn operator unit tests — direct float64 oracle slice, routing invariants:
     (a) routing depends only on selected top-k experts,
     (b) zeroing a NON-selected expert is a no-op,
     (c) selected gate weights sum to 1 per token.
  2. Whole-model parity (A) — ``mixtral_forward`` vs the composed float64 oracle in
     ``tiny_mixtral.npz`` at ``rtol=1e-9``.
  3. Real-weights parity (B, skippable) — ``mixtral_forward`` vs ``real_ref.npz`` logits
     produced by a genuine ``MixtralForCausalLM`` on the downloaded
     ``hf-internal-testing/tiny-random-MixtralForCausalLM`` weights.
     Run ``308_mixtral_model/download.sh`` to populate the weights.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from leet_llm.grader import load

_m = load(__file__)
moe_ffn = _m.moe_ffn
MixtralConfig = _m.MixtralConfig
load_mixtral = _m.load_mixtral
mixtral_forward = _m.mixtral_forward

FIX = pathlib.Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Helpers for MoE oracle (float64)
# ---------------------------------------------------------------------------

def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _moe_oracle(x_np, router_weight_np, gate_up_proj_np, down_proj_np, top_k):
    """Float64 reference MoE FFN.

    x_np: (T, d)
    router_weight_np: (num_experts, d)
    gate_up_proj_np: (num_experts, 2*Fd, d)
    down_proj_np: (num_experts, d, Fd)
    Returns: (T, d)
    """
    x = x_np.astype(np.float64)
    router_weight = router_weight_np.astype(np.float64)
    gate_up = gate_up_proj_np.astype(np.float64)
    down = down_proj_np.astype(np.float64)

    T = x.shape[0]
    num_experts = router_weight.shape[0]

    # Router
    logits = x @ router_weight.T                     # (T, num_experts)
    probs = _softmax_np(logits, axis=-1)             # softmax over ALL experts

    # Top-k
    idx = np.argsort(probs, axis=-1)[:, ::-1][:, :top_k]   # (T, top_k) descending
    weights = probs[np.arange(T)[:, None], idx]             # (T, top_k)
    weights = weights / weights.sum(axis=-1, keepdims=True)  # renormalise

    out = np.zeros_like(x)
    Fd = gate_up.shape[1] // 2
    for k in range(top_k):
        for t in range(T):
            e = idx[t, k]
            x_t = x[t]                                      # (d,)
            gu = x_t @ gate_up[e].T                         # (2*Fd,)
            gate_v, up_v = gu[:Fd], gu[Fd:]
            h = (gate_v / (1 + np.exp(-gate_v))) * up_v    # SiLU gate * up
            y = h @ down[e].T                               # (d,)
            out[t] += weights[t, k] * y

    return out


# Retrieve SwiGLUParams from leet_llm for constructing fake experts
try:
    from leet_llm import SwiGLUParams
    _HAS_SWIGLU = True
except Exception:
    _HAS_SWIGLU = False


def _make_fake_experts(gate_up_proj_np, down_proj_np):
    """Build list[SwiGLUParams] from packed gate_up_proj/down_proj arrays."""
    if not _HAS_SWIGLU:
        return None
    n = gate_up_proj_np.shape[0]
    Fd = gate_up_proj_np.shape[1] // 2
    experts = []
    for e in range(n):
        W1 = gate_up_proj_np[e, :Fd, :]    # gate_proj (Fd, d)
        W3 = gate_up_proj_np[e, Fd:, :]    # up_proj   (Fd, d)
        W2 = down_proj_np[e]               # down_proj (d, Fd)
        experts.append(SwiGLUParams(W1=W1, W3=W3, W2=W2))
    return experts


# ---------------------------------------------------------------------------
# 1. moe_ffn operator — direct unit test vs float64 oracle
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_SWIGLU, reason="SwiGLUParams not available (earlier task not solved)")
def test_moe_ffn_matches_oracle():
    """moe_ffn must match the float64 reference MoE oracle on a small input."""
    rng = np.random.default_rng(7)
    T, d, Fd, NE, NK = 6, 8, 16, 4, 2
    x = rng.standard_normal((T, d))
    router_weight = rng.standard_normal((NE, d))
    gate_up_proj = rng.standard_normal((NE, 2 * Fd, d))
    down_proj = rng.standard_normal((NE, d, Fd))

    experts = _make_fake_experts(gate_up_proj, down_proj)
    out = moe_ffn(x, router_weight, experts, NK)
    ref = _moe_oracle(x, router_weight, gate_up_proj, down_proj, NK)

    assert out.shape == x.shape, f"shape: expected {x.shape}, got {out.shape}"
    np.testing.assert_allclose(out, ref, rtol=1e-9, atol=0,
                               err_msg="moe_ffn vs float64 oracle mismatch")


@pytest.mark.skipif(not _HAS_SWIGLU, reason="SwiGLUParams not available")
def test_moe_ffn_shape():
    """moe_ffn output shape must equal input shape."""
    rng = np.random.default_rng(8)
    B, L, d, Fd, NE, NK = 2, 4, 8, 16, 4, 2
    x = rng.standard_normal((B * L, d))
    router_weight = rng.standard_normal((NE, d))
    gate_up_proj = rng.standard_normal((NE, 2 * Fd, d))
    down_proj = rng.standard_normal((NE, d, Fd))

    experts = _make_fake_experts(gate_up_proj, down_proj)
    out = moe_ffn(x, router_weight, experts, NK)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 2. MoE routing invariants
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_SWIGLU, reason="SwiGLUParams not available")
def test_moe_routing_depends_only_on_top_k():
    """Routing output depends only on the selected top-k experts.

    We zero out ALL non-selected expert weights and the output must be identical.
    """
    rng = np.random.default_rng(6)
    T, d, Fd, NE, NK = 5, 8, 16, 4, 2
    x = rng.standard_normal((T, d))
    router_weight = rng.standard_normal((NE, d))
    gate_up_proj = rng.standard_normal((NE, 2 * Fd, d))
    down_proj = rng.standard_normal((NE, d, Fd))

    experts_orig = _make_fake_experts(gate_up_proj, down_proj)
    out_orig = moe_ffn(x.copy(), router_weight, experts_orig, NK)

    # Determine which experts are selected for each token
    logits = x @ router_weight.T
    probs = _softmax_np(logits, axis=-1)
    idx = np.argsort(probs, axis=-1)[:, ::-1][:, :NK]    # (T, NK) selected experts
    selected_experts = set(idx.flatten().tolist())
    all_experts = set(range(NE))
    non_selected = all_experts - selected_experts

    if not non_selected:
        pytest.skip("All experts selected in this random seed — invariant vacuously true")

    # Zero the weights of non-selected experts
    gate_up_zeroed = gate_up_proj.copy()
    down_zeroed = down_proj.copy()
    for e in non_selected:
        gate_up_zeroed[e] = 0.0
        down_zeroed[e] = 0.0

    experts_zeroed = _make_fake_experts(gate_up_zeroed, down_zeroed)
    out_zeroed = moe_ffn(x.copy(), router_weight, experts_zeroed, NK)

    np.testing.assert_allclose(
        out_zeroed, out_orig, rtol=1e-9, atol=0,
        err_msg="Zeroing non-selected expert weights changed the output",
    )


@pytest.mark.skipif(not _HAS_SWIGLU, reason="SwiGLUParams not available")
def test_moe_non_selected_expert_is_noop():
    """Zeroing a specific non-selected expert's weights must not change the output."""
    rng = np.random.default_rng(11)
    T, d, Fd, NE, NK = 4, 8, 16, 4, 2
    x = rng.standard_normal((T, d))
    router_weight = rng.standard_normal((NE, d))
    gate_up_proj = rng.standard_normal((NE, 2 * Fd, d))
    down_proj = rng.standard_normal((NE, d, Fd))

    experts_orig = _make_fake_experts(gate_up_proj, down_proj)
    out_orig = moe_ffn(x.copy(), router_weight, experts_orig, NK)

    # Find at least one token where expert 0 is NOT selected
    logits = x @ router_weight.T
    probs = _softmax_np(logits, axis=-1)
    idx = np.argsort(probs, axis=-1)[:, ::-1][:, :NK]

    # Identify a non-selected expert for at least one token
    selected_for_all = set(idx.flatten().tolist())
    non_selected_global = set(range(NE)) - selected_for_all

    if not non_selected_global:
        pytest.skip("All experts selected across tokens — no non-selected expert to test")

    target_e = next(iter(non_selected_global))

    gate_up_mod = gate_up_proj.copy()
    down_mod = down_proj.copy()
    gate_up_mod[target_e] = 0.0
    down_mod[target_e] = 0.0

    experts_mod = _make_fake_experts(gate_up_mod, down_mod)
    out_mod = moe_ffn(x.copy(), router_weight, experts_mod, NK)

    np.testing.assert_allclose(
        out_mod, out_orig, rtol=1e-9, atol=0,
        err_msg=f"Zeroing non-selected expert {target_e} changed the output",
    )


@pytest.mark.skipif(not _HAS_SWIGLU, reason="SwiGLUParams not available")
def test_moe_gate_weights_sum_to_one():
    """The selected gate weights must renormalise to sum to 1 per token.

    Observed THROUGH moe_ffn (not recomputed in numpy): make every expert identical,
    so the routed output collapses to (Σ_k gate_k) · SwiGLU(x). That equals a single
    SwiGLU pass iff the selected gate weights sum to 1. A missing or wrong
    renormalisation would scale the output differently, so this exercises the
    student's routing rather than testing numpy against itself.
    """
    rng = np.random.default_rng(12)
    T, d, Fd, NE, NK = 7, 8, 16, 4, 2
    x = rng.standard_normal((T, d))
    router_weight = rng.standard_normal((NE, d))

    # All experts share one weight set.
    gate_up_one = rng.standard_normal((2 * Fd, d))
    down_one = rng.standard_normal((d, Fd))
    gate_up_proj = np.broadcast_to(gate_up_one, (NE, 2 * Fd, d)).copy()
    down_proj = np.broadcast_to(down_one, (NE, d, Fd)).copy()

    experts = _make_fake_experts(gate_up_proj, down_proj)
    out = moe_ffn(x, router_weight, experts, NK)

    # Single shared-expert SwiGLU (float64); equals moe_ffn iff gate weights sum to 1.
    gu = x @ gate_up_one.T                       # (T, 2*Fd)
    gate_v, up_v = gu[:, :Fd], gu[:, Fd:]
    h = (gate_v / (1 + np.exp(-gate_v))) * up_v  # SiLU gate * up
    ref = h @ down_one.T                          # (T, d)

    np.testing.assert_allclose(
        out, ref, rtol=1e-9, atol=0,
        err_msg="moe_ffn != single-expert SwiGLU — selected gate weights don't sum to 1",
    )


# ---------------------------------------------------------------------------
# A. Whole-model parity — tiny hermetic fixture (always-on)
# ---------------------------------------------------------------------------

_TINY = np.load(FIX / "tiny_mixtral.npz")


def _tiny_cfg():
    return MixtralConfig(
        dim=int(_TINY["dim"]),
        n_layers=int(_TINY["n_layers"]),
        n_heads=int(_TINY["n_heads"]),
        n_kv_heads=int(_TINY["n_kv_heads"]),
        vocab_size=int(_TINY["vocab_size"]),
        num_local_experts=int(_TINY["num_local_experts"]),
        num_experts_per_tok=int(_TINY["num_experts_per_tok"]),
        max_seq_len=int(_TINY["max_seq_len"]),
        norm_eps=float(_TINY["norm_eps"]),
        rope_base=float(_TINY["rope_base"]),
    )


def _tiny_params():
    return load_mixtral({k: _TINY[k] for k in _TINY.files}, _tiny_cfg())


def test_mixtral_logits_match_oracle():
    """mixtral_forward must reproduce the composed float64 oracle logits at rtol=1e-9."""
    out = mixtral_forward(_TINY["input_ids"], _tiny_params(), _tiny_cfg())
    np.testing.assert_allclose(out, _TINY["logits"], rtol=1e-9, atol=1e-9)


def test_mixtral_logits_shape():
    out = mixtral_forward(_TINY["input_ids"], _tiny_params(), _tiny_cfg())
    B, L = _TINY["input_ids"].shape
    assert out.shape == (B, L, int(_TINY["vocab_size"]))


def test_mixtral_causal():
    """Changing the last token must NOT affect earlier logits (causal masking)."""
    p, cfg = _tiny_params(), _tiny_cfg()
    base = mixtral_forward(_TINY["input_ids"], p, cfg)
    ids2 = _TINY["input_ids"].copy()
    ids2[0, -1] = (ids2[0, -1] + 1) % int(_TINY["vocab_size"])
    pert = mixtral_forward(ids2, p, cfg)
    np.testing.assert_allclose(base[0, :-1], pert[0, :-1], atol=1e-9)


# ---------------------------------------------------------------------------
# B. Real-weights parity — skippable (run download.sh first)
# ---------------------------------------------------------------------------

_WEIGHTS_PATH = pathlib.Path(__file__).resolve().parents[1] / "mixtral_tiny.npz"
_REAL_REF = FIX / "real_ref.npz"


@pytest.mark.skipif(
    not _WEIGHTS_PATH.exists(),
    reason="run 308_mixtral_model/download.sh to fetch real weights",
)
def test_mixtral_real_weights_logits():
    """mixtral_forward on the real tiny-random-MixtralForCausalLM weights must
    match the committed real_ref.npz logits.

    real_ref.npz logits were produced by a genuine MixtralForCausalLM (SiLU-forced,
    float32) on the downloaded weights via convert.py.  Our forward runs in float64 on
    the same weights, so the comparison is float64 vs float32: tolerance rtol=1e-3.
    This is a genuine parity check (our forward vs real HF model), not self-circular.
    """
    ref = np.load(_REAL_REF)
    weights = dict(np.load(str(_WEIGHTS_PATH)))
    cfg = MixtralConfig(
        dim=int(ref["dim"]),
        n_layers=int(ref["n_layers"]),
        n_heads=int(ref["n_heads"]),
        n_kv_heads=int(ref["n_kv_heads"]),
        vocab_size=int(ref["vocab_size"]),
        num_local_experts=int(ref["num_local_experts"]),
        num_experts_per_tok=int(ref["num_experts_per_tok"]),
        max_seq_len=int(ref["max_seq_len"]),
        norm_eps=float(ref["norm_eps"]),
        rope_base=float(ref["rope_base"]),
    )
    params = load_mixtral(weights, cfg)
    out = mixtral_forward(ref["input_ids"], params, cfg)
    np.testing.assert_allclose(out, ref["logits"], rtol=1e-3, atol=1e-2)
