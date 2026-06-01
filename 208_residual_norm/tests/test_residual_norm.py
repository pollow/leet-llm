import numpy as np

from leet_llm.grader import load

_m = load(__file__)
add_residual = _m.add_residual


def test_explicit_sum():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    s = np.array([[10.0, 20.0], [30.0, 40.0]])
    np.testing.assert_allclose(add_residual(x, s), [[11.0, 22.0], [33.0, 44.0]])


def test_shape_and_no_mutation():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 4))
    s = rng.standard_normal((3, 4))
    x0 = x.copy()
    out = add_residual(x, s)
    assert out.shape == (3, 4)
    np.testing.assert_allclose(out, x0 + s)
    np.testing.assert_allclose(x, x0)  # input left unmodified
