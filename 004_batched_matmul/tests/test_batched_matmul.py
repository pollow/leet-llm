import numpy as np

from leet_llm.grader import load

_m = load(__file__)
batched_matmul = _m.batched_matmul
outer_product = _m.outer_product
batched_trace = _m.batched_trace


def test_batched_matmul_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2, 3, 4, 5))
    b = rng.standard_normal((2, 3, 5, 6))
    out = batched_matmul(a, b)
    assert out.shape == (2, 3, 4, 6)
    np.testing.assert_allclose(out, a @ b)


def test_outer_product():
    rng = np.random.default_rng(1)
    u = rng.standard_normal((2, 3))
    v = rng.standard_normal((2, 4))
    out = outer_product(u, v)
    assert out.shape == (2, 3, 4)
    np.testing.assert_allclose(out, u[..., :, None] * v[..., None, :])


def test_batched_trace():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((2, 3, 5, 5))
    out = batched_trace(a)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, np.trace(a, axis1=-2, axis2=-1))


def test_matmul_2d_is_plain_matrix_product():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((4, 5))
    b = rng.standard_normal((5, 6))
    np.testing.assert_allclose(batched_matmul(a, b), a @ b)
