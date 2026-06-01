import pathlib

import numpy as np
import pytest

from leet_llm.grader import load

_m = load(__file__)
swiglu_ffn = _m.swiglu_ffn
SwiGLUParams = _m.SwiGLUParams

FIX = pathlib.Path(__file__).parent / "fixtures"
_FIXTURES = sorted(FIX.glob("*.npz"))


@pytest.mark.parametrize("path", _FIXTURES, ids=[p.stem for p in _FIXTURES])
def test_matches_torch_fixture(path):
    """Frozen goldens from float64 torch (SiLU(x@W1.T) * (x@W3.T)) @ W2.T."""
    d = np.load(path)
    p = SwiGLUParams(W1=d["W1"], W3=d["W3"], W2=d["W2"])
    np.testing.assert_allclose(swiglu_ffn(d["x"], p), d["out"], rtol=1e-9, atol=1e-9)


def test_zero_gate_kills_output():
    # SiLU(0) = 0, so a zero gate projection (W1=0) forces an all-zero output.
    rng = np.random.default_rng(0)
    d, d_ff = 4, 6
    p = SwiGLUParams(
        W1=np.zeros((d_ff, d)),
        W3=rng.standard_normal((d_ff, d)),
        W2=rng.standard_normal((d, d_ff)),
    )
    x = rng.standard_normal((2, d))
    np.testing.assert_allclose(swiglu_ffn(x, p), 0.0, atol=1e-12)


def test_shape_preserved():
    rng = np.random.default_rng(1)
    d, d_ff = 5, 11
    p = SwiGLUParams(
        W1=rng.standard_normal((d_ff, d)),
        W3=rng.standard_normal((d_ff, d)),
        W2=rng.standard_normal((d, d_ff)),
    )
    assert swiglu_ffn(rng.standard_normal((3, d)), p).shape == (3, d)
