import numpy as np

from leet_llm.grader import load

_m = load(__file__)
add_bias = _m.add_bias
standardize = _m.standardize


def test_add_bias_broadcasts_over_leading_axes():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 4))
    b = rng.standard_normal((4,))
    out = add_bias(x, b)
    assert out.shape == (2, 3, 4)
    np.testing.assert_allclose(out, x + b)


def test_add_bias_does_not_mutate_input():
    x = np.zeros((2, 4))
    add_bias(x, np.ones((4,)))
    np.testing.assert_array_equal(x, np.zeros((2, 4)))


def test_standardize_matches_formula():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 5, 7))
    eps = 1e-5
    expected = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)
    np.testing.assert_allclose(standardize(x), expected, rtol=1e-6, atol=1e-6)


def test_standardize_gives_zero_mean_unit_var():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 512)) * 5.0 + 3.0  # arbitrary scale/shift
    out = standardize(x)
    np.testing.assert_allclose(out.mean(-1), 0.0, atol=1e-6)
    np.testing.assert_allclose(out.std(-1), 1.0, atol=1e-3)
