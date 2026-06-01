import pathlib

import numpy as np
import pytest

from leet_llm.grader import load

_m = load(__file__)
rope = _m.rope

FIX = pathlib.Path(__file__).parent / "fixtures"
_FIXTURES = sorted(FIX.glob("*.npz"))


@pytest.mark.parametrize("path", _FIXTURES, ids=[p.stem for p in _FIXTURES])
def test_matches_rotate_half_fixture(path):
    """Frozen goldens pinning the rotate-half (HF/Llama) convention."""
    d = np.load(path)
    np.testing.assert_allclose(rope(d["x"], d["positions"]), d["out"], rtol=1e-9, atol=1e-9)


def test_position_zero_is_identity():
    # angle(0) = 0 -> cos=1, sin=0 -> RoPE is the identity at position 0
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 1, 6))
    np.testing.assert_allclose(rope(x, np.array([0])), x, atol=1e-12)


def test_norm_preserved():
    # rotation preserves the norm of each vector (convention-independent)
    rng = np.random.default_rng(5)
    x = rng.standard_normal((2, 3, 4, 8))
    out = rope(x, np.arange(4))
    np.testing.assert_allclose(
        np.linalg.norm(out, axis=-1), np.linalg.norm(x, axis=-1), atol=1e-9
    )


def test_relative_position_dot_product():
    # <RoPE(q, m), RoPE(k, n)> depends only on (m - n): shifting both by s is invariant.
    rng = np.random.default_rng(3)
    q = rng.standard_normal((1, 8))
    k = rng.standard_normal((1, 8))

    def dot(m, n):
        return float(np.sum(rope(q, np.array([m])) * rope(k, np.array([n]))))

    np.testing.assert_allclose(dot(2, 5), dot(2 + 3, 5 + 3), atol=1e-9)
