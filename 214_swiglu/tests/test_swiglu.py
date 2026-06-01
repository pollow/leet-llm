import numpy as np

from leet_llm.grader import load

_m = load(__file__)
swiglu_ffn = _m.swiglu_ffn
SwiGLUParams = _m.SwiGLUParams


def _golden_params():
    W1 = np.array([[0.001, 0.299], [-0.274, -0.891], [-0.455, -0.992]])
    W3 = np.array([[0.06, 1.34], [-0.492, -0.62], [0.49, 0.357]])
    W2 = np.array([[0.105, -0.93, -0.029], [0.695, -1.344, -0.458]])
    return SwiGLUParams(W1=W1, W3=W3, W2=W2)


def test_golden():
    # Frozen golden computed offline from (SiLU(x@W1.T) * (x@W3.T)) @ W2.T.
    x = np.array([[-1.901, -1.29], [-1.842, -0.235]])
    expected = np.array(
        [[-2.160488203032, -1.853419973602], [-0.444606291160, -0.306993603868]]
    )
    np.testing.assert_allclose(swiglu_ffn(x, _golden_params()), expected, rtol=1e-9, atol=1e-12)


def test_zero_gate_kills_output():
    # SiLU(0) = 0, so a zero gate projection (W1=0) forces an all-zero output
    # regardless of the up/down weights.
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
