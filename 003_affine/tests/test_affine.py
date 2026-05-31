import numpy as np

from leet_llm.grader import load

affine = load(__file__).affine


def test_shape():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 4))
    W = rng.standard_normal((5, 4))
    b = rng.standard_normal((5,))
    assert affine(x, W, b).shape == (2, 3, 5)


def test_values_with_bias():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((6, 4))
    W = rng.standard_normal((3, 4))
    b = rng.standard_normal((3,))
    np.testing.assert_allclose(affine(x, W, b), x @ W.T + b)


def test_no_bias_defaults_to_zero():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 3, 4))
    W = rng.standard_normal((5, 4))
    np.testing.assert_allclose(affine(x, W), x @ W.T)


def test_broadcasts_over_many_leading_axes():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 3, 4, 8))
    W = rng.standard_normal((6, 8))
    b = rng.standard_normal((6,))
    out = affine(x, W, b)
    assert out.shape == (2, 3, 4, 6)
    np.testing.assert_allclose(out, x @ W.T + b)
