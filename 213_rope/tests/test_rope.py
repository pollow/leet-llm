import pathlib

import numpy as np
import pytest

from leet_llm.grader import load

_m = load(__file__)
rope_interleaved = _m.rope_interleaved
rope_half = _m.rope_half
rope_qk_dot = _m.rope_qk_dot

FIX = pathlib.Path(__file__).parent / "fixtures"
_INTERLEAVED = sorted(FIX.glob("interleaved_*.npz"))
_HALF = sorted(FIX.glob("half_*.npz"))


@pytest.mark.parametrize("path", _INTERLEAVED, ids=[p.stem for p in _INTERLEAVED])
def test_interleaved_matches_torch_fixture(path):
    """Goldens from official torch complex rotation (Meta / llama3.np)."""
    d = np.load(path)
    np.testing.assert_allclose(
        rope_interleaved(d["x"], d["positions"]), d["out"], rtol=1e-9, atol=1e-9
    )


@pytest.mark.parametrize("path", _HALF, ids=[p.stem for p in _HALF])
def test_half_matches_hf_fixture(path):
    """Goldens from HuggingFace transformers' official rotate_half."""
    d = np.load(path)
    np.testing.assert_allclose(rope_half(d["x"], d["positions"]), d["out"], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("rope", [rope_interleaved, rope_half], ids=["interleaved", "half"])
def test_position_zero_is_identity(rope):
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 1, 6))
    np.testing.assert_allclose(rope(x, np.array([0])), x, atol=1e-12)


@pytest.mark.parametrize("rope", [rope_interleaved, rope_half], ids=["interleaved", "half"])
def test_norm_preserved(rope):
    rng = np.random.default_rng(5)
    x = rng.standard_normal((2, 3, 4, 8))
    out = rope(x, np.arange(4))
    np.testing.assert_allclose(
        np.linalg.norm(out, axis=-1), np.linalg.norm(x, axis=-1), atol=1e-9
    )


def test_qk_dot_translation_invariance():
    # <RoPE(q,m), RoPE(k,n)> depends only on (m - n): shifting both by s is invariant.
    rng = np.random.default_rng(3)
    q = rng.standard_normal(8)
    k = rng.standard_normal(8)
    np.testing.assert_allclose(rope_qk_dot(q, k, 2, 5), rope_qk_dot(q, k, 2 + 4, 5 + 4), atol=1e-9)


def test_qk_dot_equal_positions_recovers_plain_dot():
    # at m == n the rotation cancels: <RoPE(q,m), RoPE(k,m)> == <q, k>
    rng = np.random.default_rng(6)
    q = rng.standard_normal(8)
    k = rng.standard_normal(8)
    np.testing.assert_allclose(rope_qk_dot(q, k, 7, 7), np.dot(q, k), atol=1e-9)
