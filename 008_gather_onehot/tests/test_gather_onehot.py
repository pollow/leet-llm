import numpy as np

from leet_llm.grader import load

_m = load(__file__)
gather_rows = _m.gather_rows
one_hot = _m.one_hot


def test_gather_rows_1d_index():
    rng = np.random.default_rng(0)
    table = rng.standard_normal((10, 4))
    idx = np.array([0, 3, 9, 1])
    out = gather_rows(table, idx)
    assert out.shape == (4, 4)
    np.testing.assert_array_equal(out, table[idx])


def test_gather_rows_2d_index_preserves_shape():
    rng = np.random.default_rng(1)
    table = rng.standard_normal((6, 5))
    idx = np.array([[0, 1], [2, 3]])
    out = gather_rows(table, idx)
    assert out.shape == (2, 2, 5)
    np.testing.assert_array_equal(out, table[idx])


def test_one_hot_basic():
    out = one_hot(np.array([0, 2, 1]), 4)
    expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=float)
    np.testing.assert_array_equal(out, expected)


def test_one_hot_is_float_and_sums_to_one():
    out = one_hot(np.array([0, 3, 3, 1]), 4)
    assert out.dtype == np.float64 or np.issubdtype(out.dtype, np.floating)
    np.testing.assert_allclose(out.sum(-1), 1.0)


def test_one_hot_multidim():
    idx = np.array([[0, 1], [2, 3]])
    assert one_hot(idx, 4).shape == (2, 2, 4)
