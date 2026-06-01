import numpy as np

from leet_llm.grader import load

_m = load(__file__)
gelu = _m.gelu
silu = _m.silu

# Frozen anchors (computed once from the definitions via math.erf / sigmoid).
XS = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
GELU = np.array(
    [-0.0455002639, -0.1586552539, -0.1542687694, 0.0,
     0.3457312306, 0.8413447461, 1.9544997361]
)
SILU = np.array(
    [-0.2384058440, -0.2689414214, -0.1887703344, 0.0,
     0.3112296656, 0.7310585786, 1.7615941560]
)


def test_gelu_anchors():
    np.testing.assert_allclose(gelu(XS), GELU, atol=1e-8)


def test_silu_anchors():
    np.testing.assert_allclose(silu(XS), SILU, atol=1e-8)


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
