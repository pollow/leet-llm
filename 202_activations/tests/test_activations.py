import pathlib

import numpy as np
import pytest

from leet_llm.grader import load

_m = load(__file__)
gelu = _m.gelu
silu = _m.silu

FIX = pathlib.Path(__file__).parent / "fixtures"
_FIXTURES = sorted(FIX.glob("*.npz"))


@pytest.mark.parametrize("path", _FIXTURES, ids=[p.stem for p in _FIXTURES])
def test_matches_torch_fixture(path):
    """Frozen goldens from float64 torch F.gelu (exact) / F.silu."""
    d = np.load(path)
    np.testing.assert_allclose(gelu(d["x"]), d["gelu"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(silu(d["x"]), d["silu"], rtol=1e-9, atol=1e-9)


def test_zero_maps_to_zero():
    z = np.zeros(5)
    np.testing.assert_allclose(gelu(z), 0.0, atol=1e-12)
    np.testing.assert_allclose(silu(z), 0.0, atol=1e-12)


def test_asymptotes():
    big = np.array([20.0, -20.0])
    # large positive -> ~x ; large negative -> ~0
    np.testing.assert_allclose(gelu(big), [20.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(silu(big), [20.0, 0.0], atol=1e-5)


def test_shape_preserved():
    x = np.zeros((3, 4))
    assert gelu(x).shape == (3, 4)
    assert silu(x).shape == (3, 4)
