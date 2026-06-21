"""309 — tests for ``softcap`` / ``geglu_ffn`` and the Gemma-2 whole-model.

Categories:
  1. Operator unit tests — ``softcap`` (vs float64 oracle + saturation) and
     ``geglu_ffn`` (vs float64 oracle; must use GELU-tanh, not SiLU).
  2. Whole-model parity (A) — ``gemma_forward`` vs the composed float64 oracle in
     ``tiny_gemma.npz`` at ``rtol=1e-9``.
  3. Wrinkle isolations observed THROUGH ``gemma_forward``:
     (a) ``(1+w)`` RMSNorm — zeroing every norm weight is NOT a no-op (would zero
         the residual stream under a plain ``w*`` norm),
     (b) √d embedding scale — the embed scale is applied (forward is sensitive to it),
     (c) per-layer alternating sliding/full masks — the even layer's window matters,
     (d) attention/final logit soft-cap saturate large logits.
  4. Real-weights parity (B, skippable) — ``gemma_forward`` vs ``real_ref.npz`` logits
     from a genuine ``Gemma2ForCausalLM``. Run ``309_gemma_model/download.sh`` first.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from leet_llm.grader import load

_m = load(__file__)
softcap = _m.softcap
geglu_ffn = _m.geglu_ffn
GeGLUParams = _m.GeGLUParams
GemmaConfig = _m.GemmaConfig
GemmaParams = _m.GemmaParams
load_gemma = _m.load_gemma
gemma_forward = _m.gemma_forward

FIX = pathlib.Path(__file__).parent / "fixtures"
_TINY = np.load(FIX / "tiny_gemma.npz")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_cfg() -> "GemmaConfig":
    return GemmaConfig(
        dim=int(_TINY["dim"]),
        n_layers=int(_TINY["n_layers"]),
        n_heads=int(_TINY["n_heads"]),
        n_kv_heads=int(_TINY["n_kv_heads"]),
        head_dim=int(_TINY["head_dim"]),
        vocab_size=int(_TINY["vocab_size"]),
        intermediate_size=int(_TINY["intermediate_size"]),
        norm_eps=float(_TINY["norm_eps"]),
        rope_base=float(_TINY["rope_base"]),
        query_pre_attn_scalar=int(_TINY["query_pre_attn_scalar"]),
        final_logit_softcapping=float(_TINY["final_logit_softcapping"]),
        attn_logit_softcapping=float(_TINY["attn_logit_softcapping"]),
        sliding_window=int(_TINY["sliding_window"]),
        max_seq_len=int(_TINY["max_seq_len"]),
    )


def _weights() -> dict:
    return {k: _TINY[k] for k in _TINY.files}


def _tiny_params() -> "GemmaParams":
    return load_gemma(_weights(), _tiny_cfg())


def _gelu_tanh_np(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


# ---------------------------------------------------------------------------
# 1. Operator unit tests
# ---------------------------------------------------------------------------

def test_softcap_matches_oracle():
    """softcap must equal cap * tanh(x / cap) elementwise (float64)."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 5)) * 10.0
    cap = 30.0
    out = softcap(x, cap)
    ref = cap * np.tanh(x / cap)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=0)


def test_softcap_saturates():
    """Large |x| saturate toward ±cap; output is always in (-cap, cap)."""
    cap = 5.0
    big = np.array([1e3, -1e3, 1e6, -1e6])
    out = softcap(big, cap)
    assert np.all(np.abs(out) <= cap)
    np.testing.assert_allclose(out, np.array([cap, -cap, cap, -cap]), atol=1e-6)


def test_softcap_near_linear_small():
    """For |x| << cap, softcap is approximately the identity."""
    cap = 100.0
    x = np.array([0.0, 0.1, -0.2, 0.05])
    out = softcap(x, cap)
    np.testing.assert_allclose(out, x, rtol=0, atol=1e-3)


def test_geglu_matches_oracle():
    """geglu_ffn must equal down(gelu_tanh(x@gate.T) * (x@up.T))."""
    rng = np.random.default_rng(2)
    B, L, d, F = 2, 3, 8, 16
    x = rng.standard_normal((B, L, d))
    gate = rng.standard_normal((F, d))
    up = rng.standard_normal((F, d))
    down = rng.standard_normal((d, F))
    out = geglu_ffn(x, GeGLUParams(gate=gate, up=up, down=down))

    ref = (_gelu_tanh_np(x @ gate.T) * (x @ up.T)) @ down.T
    assert out.shape == (B, L, d)
    np.testing.assert_allclose(out, ref, rtol=1e-9, atol=0)


def test_geglu_uses_gelu_not_silu():
    """The activation must be GELU-tanh, NOT SiLU.

    Build a single-element gate input and compare to both activations; the output
    must track GELU-tanh and differ from SiLU.
    """
    d, F = 1, 1
    # gate weight makes (x@gate.T) = z for a chosen z
    z = 2.0
    x = np.array([[[1.0]]])           # (1,1,1)
    gate = np.array([[z]])            # (F=1, d=1) → x@gate.T = z
    up = np.array([[1.0]])            # up = x → 1.0
    down = np.array([[1.0]])          # identity down
    out = geglu_ffn(x, GeGLUParams(gate=gate, up=up, down=down)).item()

    gelu = float(_gelu_tanh_np(np.array(z)))   # expected (× up=1 × down=1)
    silu = float(z / (1.0 + np.exp(-z)))
    assert abs(out - gelu) < 1e-9, f"geglu not using GELU-tanh (got {out}, gelu {gelu})"
    assert abs(out - silu) > 1e-3, "geglu appears to use SiLU, not GELU-tanh"


# ---------------------------------------------------------------------------
# 2. Whole-model parity — tiny hermetic fixture (always-on)
# ---------------------------------------------------------------------------

def test_gemma_logits_match_oracle():
    """gemma_forward must reproduce the composed float64 oracle logits at rtol=1e-9.

    This is the exact check for every wrinkle that cannot be isolated independently —
    in particular the constant ``* sqrt(dim)`` embedding scale and the
    ``query_pre_attn_scalar ** -0.5`` attention scale: the oracle bakes both in, so any
    omitted or mis-sized constant fails here at rtol=1e-9.
    """
    out = gemma_forward(_TINY["input_ids"], _tiny_params(), _tiny_cfg())
    np.testing.assert_allclose(out, _TINY["logits"], rtol=1e-9, atol=1e-9)


def test_gemma_logits_shape():
    out = gemma_forward(_TINY["input_ids"], _tiny_params(), _tiny_cfg())
    B, L = _TINY["input_ids"].shape
    assert out.shape == (B, L, int(_TINY["vocab_size"]))


def test_gemma_causal():
    """Changing the last token must NOT affect earlier logits (causal masking)."""
    p, cfg = _tiny_params(), _tiny_cfg()
    base = gemma_forward(_TINY["input_ids"], p, cfg)
    ids2 = _TINY["input_ids"].copy()
    ids2[0, -1] = (ids2[0, -1] + 1) % int(_TINY["vocab_size"])
    pert = gemma_forward(ids2, p, cfg)
    np.testing.assert_allclose(base[0, :-1], pert[0, :-1], atol=1e-9)


def test_gemma_final_logits_softcapped():
    """The final logit soft-cap must actually compress the output.

    A bare "logits are within ±cap" bound is necessary but not sufficient — on this
    fixture the *uncapped* logits already sit below the default cap (≈27 < 30), so a
    forward that simply forgot the final cap would pass a bound check. We instead set a
    *tiny* cap (0.5) that the raw logits (~27) cannot satisfy unless the cap is applied,
    AND require the tiny cap to change the output versus an effectively-disabled cap.
    Both conditions hold only if ``softcap(logits, final_logit_softcapping)`` is run.
    """
    cfg = _tiny_cfg()
    small_cap = GemmaConfig(**{**cfg.__dict__, "final_logit_softcapping": 0.5})
    huge_cap = GemmaConfig(**{**cfg.__dict__, "final_logit_softcapping": 1e9})

    out_small = gemma_forward(_TINY["input_ids"], _tiny_params(), small_cap)
    out_huge = gemma_forward(_TINY["input_ids"], _tiny_params(), huge_cap)

    assert np.max(np.abs(out_small)) <= 0.5 + 1e-6, (
        "logits exceed a tiny final soft-cap — final_logit_softcapping not applied"
    )
    assert not np.allclose(out_small, out_huge, atol=1e-3), (
        "tiny vs disabled final cap produced identical logits — cap has no effect"
    )


# ---------------------------------------------------------------------------
# 3. Wrinkle isolations observed THROUGH gemma_forward
# ---------------------------------------------------------------------------

def test_one_plus_w_rmsnorm():
    """Gemma uses ``(1+w)`` RMSNorm, not the plain ``w*`` Llama norm.

    Zeroing EVERY norm weight makes the gain ``(1+0)=1`` (identity scale) under the
    correct convention, so the forward still produces input-dependent, non-zero
    logits. Under the wrong plain ``w*`` convention, every normalized activation
    collapses to zero and the logits would be all zero. We assert the output is NOT
    all-zero — which can only hold for the ``(1+w)`` convention.
    """
    cfg = _tiny_cfg()
    W = _weights()
    for k in list(W.keys()):
        if k.endswith("layernorm.weight") or k == "model.norm.weight":
            W[k] = np.zeros_like(W[k])
    out = gemma_forward(_TINY["input_ids"], load_gemma(W, cfg), cfg)
    assert np.max(np.abs(out)) > 1e-6, (
        "Zeroing all norm weights produced ~zero logits — norm is plain w*, "
        "not Gemma's (1+w)."
    )


# NOTE on the √d embedding scale: there is deliberately no standalone isolation test
# for it. The constant `* sqrt(dim)` scale cannot be probed independently through the
# public forward — both RMSNorms that bracket the residual stream (and especially the
# final norm before lm_head) are scale-invariant, so no simple input construction can
# expose the constant alone. It IS verified *exactly* by `test_gemma_logits_match_oracle`:
# the committed float64 oracle applies `h = h * sqrt(dim)`, so omitting or mis-sizing the
# scale fails whole-model parity at rtol=1e-9. A weaker "does the output change when I
# edit the embedding" probe would pass any model and give false confidence, so it is
# intentionally omitted here.


def test_alternating_sliding_full_masks():
    """Even-indexed layers use sliding-window attention; odd-indexed use full causal.

    With a sliding window smaller than the sequence length, the even (sliding) layers
    drop far-past keys. Shrinking the window must therefore change the logits — but
    only because at least one layer consults the window. We assert sensitivity to the
    window size (a model that used full causal everywhere would be window-invariant).
    """
    cfg = _tiny_cfg()
    L = int(_TINY["input_ids"].shape[1])

    big_win = GemmaConfig(**{**cfg.__dict__, "sliding_window": L})        # window covers all
    small_win = GemmaConfig(**{**cfg.__dict__, "sliding_window": 2})      # window drops far past

    out_big = gemma_forward(_TINY["input_ids"], _tiny_params(), big_win)
    out_small = gemma_forward(_TINY["input_ids"], _tiny_params(), small_win)

    assert not np.allclose(out_big, out_small, atol=1e-6), (
        "Changing the sliding window had no effect — the alternating sliding/full "
        "masks are not applied (even layers should consult sliding_window)."
    )


def test_attention_softcap_active():
    """The attention logit soft-cap changes the forward output.

    Compare a tiny cap (heavy saturation) vs a huge cap (≈ no capping). The outputs
    must differ, proving the cap is applied to the pre-softmax scores.
    """
    cfg = _tiny_cfg()
    tiny_cap = GemmaConfig(**{**cfg.__dict__, "attn_logit_softcapping": 0.5})
    huge_cap = GemmaConfig(**{**cfg.__dict__, "attn_logit_softcapping": 1e9})

    out_tiny = gemma_forward(_TINY["input_ids"], _tiny_params(), tiny_cap)
    out_huge = gemma_forward(_TINY["input_ids"], _tiny_params(), huge_cap)
    assert not np.allclose(out_tiny, out_huge, atol=1e-6), (
        "attn_logit_softcapping had no effect — not applied before softmax."
    )


# ---------------------------------------------------------------------------
# B. Real-weights parity — skippable (run download.sh first)
# ---------------------------------------------------------------------------

_WEIGHTS_PATH = pathlib.Path(__file__).resolve().parents[1] / "gemma_tiny.npz"
_REAL_REF = FIX / "real_ref.npz"


@pytest.mark.skipif(
    not (_WEIGHTS_PATH.exists() and _REAL_REF.exists()),
    reason="run 309_gemma_model/download.sh to fetch real weights",
)
def test_gemma_real_weights_logits():
    """gemma_forward on the real Gemma-2 weights must match the committed real_ref.npz.

    real_ref.npz logits were produced by a genuine Gemma2ForCausalLM (eager, float32)
    on the downloaded weights via convert.py. Our forward runs in float64 on the same
    weights, so the comparison is float64 vs float32: tolerance rtol=1e-2/atol=1e-2.
    """
    ref = np.load(_REAL_REF)
    weights = dict(np.load(str(_WEIGHTS_PATH)))
    cfg = GemmaConfig(
        dim=int(ref["dim"]),
        n_layers=int(ref["n_layers"]),
        n_heads=int(ref["n_heads"]),
        n_kv_heads=int(ref["n_kv_heads"]),
        head_dim=int(ref["head_dim"]),
        vocab_size=int(ref["vocab_size"]),
        intermediate_size=int(ref["intermediate_size"]),
        norm_eps=float(ref["norm_eps"]),
        rope_base=float(ref["rope_base"]),
        query_pre_attn_scalar=int(ref["query_pre_attn_scalar"]),
        final_logit_softcapping=float(ref["final_logit_softcapping"]),
        attn_logit_softcapping=float(ref["attn_logit_softcapping"]),
        sliding_window=int(ref["sliding_window"]),
        max_seq_len=int(ref["max_seq_len"]),
    )
    params = load_gemma(weights, cfg)
    out = gemma_forward(ref["input_ids"], params, cfg)
    np.testing.assert_allclose(out, ref["logits"], rtol=1e-2, atol=1e-2)
