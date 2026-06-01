import pathlib

import numpy as np
import pytest

from leet_llm.grader import load

_m = load(__file__)
ffn = _m.ffn
FFNParams = _m.FFNParams

FIX = pathlib.Path(__file__).parent / "fixtures"
_FIXTURES = sorted(FIX.glob("*.npz"))


@pytest.mark.parametrize("path", _FIXTURES, ids=[p.stem for p in _FIXTURES])
def test_matches_torch_fixture(path):
    """Frozen goldens from float64 torch: gelu(x@W1.T + b1) @ W2.T + b2."""
    d = np.load(path)
    p = FFNParams(W1=d["W1"], b1=d["b1"], W2=d["W2"], b2=d["b2"])
    np.testing.assert_allclose(ffn(d["x"], p), d["out"], rtol=1e-9, atol=1e-9)


def test_zero_first_layer_returns_b2():
    # W1=0, b1=0 -> gelu(0)=0 -> out = 0 @ W2.T + b2 == b2
    rng = np.random.default_rng(0)
    d, d_ff = 5, 7
    p = FFNParams(
        W1=np.zeros((d_ff, d)),
        b1=np.zeros(d_ff),
        W2=rng.standard_normal((d, d_ff)),
        b2=rng.standard_normal(d),
    )
    x = rng.standard_normal((3, d))
    np.testing.assert_allclose(ffn(x, p), np.broadcast_to(p.b2, (3, d)), atol=1e-12)


def test_shape_preserved():
    rng = np.random.default_rng(1)
    d, d_ff = 6, 13
    p = FFNParams(
        W1=rng.standard_normal((d_ff, d)),
        b1=rng.standard_normal(d_ff),
        W2=rng.standard_normal((d, d_ff)),
        b2=rng.standard_normal(d),
    )
    assert ffn(rng.standard_normal((2, 4, d)), p).shape == (2, 4, d)
