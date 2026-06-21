"""311 — tests for ``rope_scaled_freqs`` / ``rope_from_freqs`` and the Llama-3.1 whole-model.

Categories:
  1. Operator unit tests:
     - ``rope_scaled_freqs`` vs frozen genuine-HF ``inv_freq`` goldens (per ``rope_type``);
       ``default`` equals the plain 213 schedule; ``linear`` divides the frequencies by
       ``factor``.
     - ``rope_from_freqs`` equals 213's ``rope_half`` when fed the default frequencies
       (the apply step is unchanged — only the frequencies are rescaled).
  2. Whole-model parity (A) — ``llama31_forward`` vs the composed float64 oracle in
     ``tiny_llama31.npz`` at ``rtol=1e-9``.
  3. Wrinkle isolations observed THROUGH ``llama31_forward`` (the scaling actually bends
     the logits; causal masking).
  4. Real-weights parity (B, skippable) — ``llama31_forward`` vs ``real_ref.npz`` logits
     from a genuine ``LlamaForCausalLM`` (the ungated tiny ``llamafactory/tiny-random-Llama-3``
     ships an active ``rope_type=llama3`` schedule). Run ``311_llama31_model/download.sh`` first.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from leet_llm import rope_half
from leet_llm.grader import load

_m = load(__file__)
rope_scaled_freqs = _m.rope_scaled_freqs
rope_from_freqs = _m.rope_from_freqs
Llama31Config = _m.Llama31Config
load_llama31 = _m.load_llama31
llama31_forward = _m.llama31_forward

FIX = pathlib.Path(__file__).parent / "fixtures"
_TINY = np.load(FIX / "tiny_llama31.npz")
_ROPE = np.load(FIX / "rope_freqs.npz", allow_pickle=False)

_HEAD_DIM = int(_ROPE["head_dim"])
_BASE = float(_ROPE["rope_base"])


# ---------------------------------------------------------------------------
# 1a. rope_scaled_freqs
# ---------------------------------------------------------------------------

def test_rope_default_matches_base_freqs():
    """default schedule == 1 / base**(arange(0, d, 2) / d) (the 213 frequencies)."""
    inv = rope_scaled_freqs(_HEAD_DIM, _BASE, {"rope_type": "default"})
    ref = 1.0 / (_BASE ** (np.arange(0, _HEAD_DIM, 2, dtype=np.float64) / _HEAD_DIM))
    assert inv.shape == (_HEAD_DIM // 2,)
    np.testing.assert_allclose(inv, ref, rtol=1e-12, atol=0)


def test_rope_none_is_default():
    """scaling=None is the default schedule."""
    a = rope_scaled_freqs(_HEAD_DIM, _BASE, None)
    b = rope_scaled_freqs(_HEAD_DIM, _BASE, {"rope_type": "default"})
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=0)


def test_rope_linear_divides_by_factor():
    """linear scaling divides every inverse frequency by ``factor`` (positions stretch)."""
    base_inv = rope_scaled_freqs(_HEAD_DIM, _BASE, None)
    inv = rope_scaled_freqs(_HEAD_DIM, _BASE, {"rope_type": "linear", "factor": 2.0})
    np.testing.assert_allclose(inv, base_inv / 2.0, rtol=1e-12, atol=0)


@pytest.mark.parametrize("rope_type", ["linear", "dynamic", "llama3", "yarn"])
def test_rope_scaled_matches_hf_golden(rope_type):
    """Per-rope_type inv_freq matches the frozen genuine-HF golden (ROPE_INIT_FUNCTIONS)."""
    scaling = json.loads(str(_ROPE[f"{rope_type}_scaling"]))
    inv = rope_scaled_freqs(_HEAD_DIM, _BASE, scaling)
    golden = _ROPE[f"{rope_type}_inv_freq"]
    assert inv.shape == golden.shape
    np.testing.assert_allclose(inv, golden, rtol=1e-5, atol=1e-8)


def test_rope_llama3_bends_low_frequencies():
    """The llama3 schedule leaves the top frequency alone but shrinks the low ones."""
    base_inv = rope_scaled_freqs(_HEAD_DIM, _BASE, None)
    scaling = json.loads(str(_ROPE["llama3_scaling"]))
    inv = rope_scaled_freqs(_HEAD_DIM, _BASE, scaling)
    # highest frequency (short wavelength) is preserved; the lowest is divided down
    np.testing.assert_allclose(inv[0], base_inv[0], rtol=1e-9)
    assert inv[-1] < base_inv[-1]


# ---------------------------------------------------------------------------
# 1b. rope_from_freqs
# ---------------------------------------------------------------------------

def test_rope_from_freqs_equals_rope_half_on_default():
    """rope_from_freqs with the default frequencies reproduces 213's rope_half exactly."""
    rng = np.random.default_rng(1)
    B, Hh, L = 2, 3, 5
    x = rng.standard_normal((B, Hh, L, _HEAD_DIM))
    positions = np.arange(L)
    inv = rope_scaled_freqs(_HEAD_DIM, _BASE, None)
    out = rope_from_freqs(x, positions, inv)
    ref = rope_half(x, positions, _BASE)
    assert out.shape == x.shape
    np.testing.assert_allclose(out, ref, rtol=1e-9, atol=1e-12)


def test_rope_from_freqs_zero_position_is_identity():
    """Position 0 has zero angle → rotation is the identity."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((1, 1, 1, _HEAD_DIM))
    inv = rope_scaled_freqs(_HEAD_DIM, _BASE, None)
    out = rope_from_freqs(x, np.array([0]), inv)
    np.testing.assert_allclose(out, x, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Whole-model parity — tiny hermetic fixture (always-on)
# ---------------------------------------------------------------------------

def _scaling():
    return json.loads(str(_TINY["rope_scaling"]))


def _tiny_cfg(rope_scaling="__keep__"):
    return Llama31Config(
        dim=int(_TINY["dim"]),
        n_layers=int(_TINY["n_layers"]),
        n_heads=int(_TINY["n_heads"]),
        n_kv_heads=int(_TINY["n_kv_heads"]),
        vocab_size=int(_TINY["vocab_size"]),
        max_seq_len=int(_TINY["max_seq_len"]),
        norm_eps=float(_TINY["norm_eps"]),
        rope_base=float(_TINY["rope_base"]),
        rope_scaling=_scaling() if rope_scaling == "__keep__" else rope_scaling,
    )


def _weights():
    return {k: _TINY[k] for k in _TINY.files}


def _tiny_params():
    return load_llama31(_weights(), _tiny_cfg())


def test_llama31_logits_match_oracle():
    """llama31_forward must reproduce the composed float64 oracle logits at rtol=1e-9."""
    out = llama31_forward(_TINY["input_ids"], _tiny_params(), _tiny_cfg())
    np.testing.assert_allclose(out, _TINY["logits"], rtol=1e-9, atol=1e-9)


def test_llama31_logits_shape():
    out = llama31_forward(_TINY["input_ids"], _tiny_params(), _tiny_cfg())
    B, L = _TINY["input_ids"].shape
    assert out.shape == (B, L, int(_TINY["vocab_size"]))


def test_llama31_causal():
    """Changing the last token must NOT affect earlier logits (causal masking)."""
    p, cfg = _tiny_params(), _tiny_cfg()
    base = llama31_forward(_TINY["input_ids"], p, cfg)
    ids2 = _TINY["input_ids"].copy()
    ids2[0, -1] = (ids2[0, -1] + 1) % int(_TINY["vocab_size"])
    pert = llama31_forward(ids2, p, cfg)
    np.testing.assert_allclose(base[0, :-1], pert[0, :-1], atol=1e-9)


def test_llama31_scaling_changes_logits():
    """The rope_scaling schedule is actually wired in: llama3 vs default differ."""
    p = _tiny_params()
    out_scaled = llama31_forward(_TINY["input_ids"], p, _tiny_cfg())
    out_default = llama31_forward(_TINY["input_ids"], p, _tiny_cfg(rope_scaling=None))
    assert not np.allclose(out_scaled, out_default, atol=1e-6), (
        "rope_scaling had no effect — scaled inv_freq not used in the forward"
    )


# ---------------------------------------------------------------------------
# B. Real-weights parity — skippable (run download.sh first)
# ---------------------------------------------------------------------------

_WEIGHTS_PATH = pathlib.Path(__file__).resolve().parents[1] / "llama31_tiny.npz"
_REAL_REF = FIX / "real_ref.npz"


@pytest.mark.skipif(
    not (_WEIGHTS_PATH.exists() and _REAL_REF.exists()),
    reason="run 311_llama31_model/download.sh to fetch real weights",
)
def test_llama31_real_weights_logits():
    """llama31_forward on the real tiny-random-Llama-3 weights must match real_ref.npz.

    real_ref.npz logits were produced by a genuine LlamaForCausalLM (eager, float32)
    with its native ``rope_type=llama3`` schedule active. Our forward runs in float64
    on the same weights: tolerance rtol=1e-2/atol=1e-2.
    """
    ref = np.load(_REAL_REF)
    weights = dict(np.load(str(_WEIGHTS_PATH)))
    cfg = Llama31Config(
        dim=int(ref["dim"]),
        n_layers=int(ref["n_layers"]),
        n_heads=int(ref["n_heads"]),
        n_kv_heads=int(ref["n_kv_heads"]),
        vocab_size=int(ref["vocab_size"]),
        max_seq_len=int(ref["max_seq_len"]),
        norm_eps=float(ref["norm_eps"]),
        rope_base=float(ref["rope_base"]),
        rope_scaling=json.loads(str(ref["rope_scaling"])),
    )
    params = load_llama31(weights, cfg)
    out = llama31_forward(ref["input_ids"], params, cfg)
    np.testing.assert_allclose(out, ref["logits"], rtol=1e-2, atol=1e-2)
