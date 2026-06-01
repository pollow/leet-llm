import pathlib

import numpy as np
import pytest

from leet_llm.grader import load

_m = load(__file__)
rms_norm = _m.rms_norm

FIX = pathlib.Path(__file__).parent / "fixtures"
_FIXTURES = sorted(FIX.glob("*.npz"))


@pytest.mark.parametrize("path", _FIXTURES, ids=[p.stem for p in _FIXTURES])
def test_matches_torch_fixture(path):
    """Frozen goldens from float64 torch F.rms_norm (eps=1e-5)."""
    d = np.load(path)
    np.testing.assert_allclose(rms_norm(d["x"], d["weight"]), d["out"], rtol=1e-9, atol=1e-9)


def test_output_rms_is_one_with_unit_weight():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 16))
    out = rms_norm(x, np.ones(16))
    rms = np.sqrt(np.mean(out**2, axis=-1))
    np.testing.assert_allclose(rms, 1.0, atol=1e-3)


def test_weight_scales_output():
    # rms_norm(x, w) == w * rms_norm(x, 1)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 8))
    w = rng.standard_normal(8)
    base = rms_norm(x, np.ones(8))
    np.testing.assert_allclose(rms_norm(x, w), w * base, atol=1e-6)


def test_scale_equivariance():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 8))
    w = np.ones(8)
    np.testing.assert_allclose(rms_norm(3.5 * x, w), rms_norm(x, w), atol=1e-4)


def test_no_recentering():
    # Unlike LayerNorm, RMSNorm does NOT subtract the mean: a constant input
    # normalizes to ~1 (its own sign), not to 0.
    x = np.full((1, 4), 5.0)
    np.testing.assert_allclose(rms_norm(x, np.ones(4)), np.ones((1, 4)), atol=1e-3)


def test_shape_preserved():
    x = np.zeros((2, 3, 5))
    assert rms_norm(x, np.ones(5)).shape == (2, 3, 5)
