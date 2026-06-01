import numpy as np

from leet_llm.grader import load

_m = load(__file__)
layer_norm = _m.layer_norm


def test_mean0_var1_with_identity_affine():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 8))
    out = layer_norm(x, np.ones(8), np.zeros(8))
    np.testing.assert_allclose(out.mean(-1), 0.0, atol=1e-6)
    np.testing.assert_allclose(out.var(-1), 1.0, atol=1e-3)


def test_affine_decomposition():
    # layer_norm(x, g, b) == g * layer_norm(x, 1, 0) + b
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 5))
    g = rng.standard_normal(5)
    b = rng.standard_normal(5)
    base = layer_norm(x, np.ones(5), np.zeros(5))
    np.testing.assert_allclose(layer_norm(x, g, b), g * base + b, atol=1e-6)


def test_shift_invariance():
    # adding a constant to every feature doesn't change the (mean-subtracted) output
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 5))
    g, b = np.ones(5), np.zeros(5)
    np.testing.assert_allclose(layer_norm(x + 3.0, g, b), layer_norm(x, g, b), atol=1e-6)


def test_constant_row_returns_beta():
    # x constant along the last axis -> (x - mean) = 0 -> output == beta (exercises eps)
    x = np.full((2, 6), 4.2)
    g = np.arange(6, dtype=float) + 1.0
    b = np.linspace(-1.0, 1.0, 6)
    np.testing.assert_allclose(layer_norm(x, g, b), np.broadcast_to(b, (2, 6)), atol=1e-4)


def test_shape_preserved():
    x = np.zeros((2, 3, 7))
    assert layer_norm(x, np.ones(7), np.zeros(7)).shape == (2, 3, 7)
